"""Job runner: phases of work units run by a pool of spawned worker processes.

Run as ``python -m src.prep_jobs.runner <job> --workers N [--resume] [--deadline-at EPOCH]``,
normally through ``scripts/launch_job.py`` (which detaches it and runs it from a snapshot).

A job module in ``src/prep_jobs/jobs/<job>.py`` provides ``phases(cfg)``, a list of dicts::

    {"name": "months", "fn": "src.prep_jobs.jobs.x:month_unit",
     "units": [{"id": "2016-01", "weight": 123456, ...}], "workers": None, "needs_all_prior": False}

``fn(ctx, unit)`` returns ``{"rows": int, "bytes": int, ...}``. A unit writes temporary files and
renames them; the runner writes ``checkpoints/<phase>__<unit>.done`` after it returns. Failed units are
retried with backoff and a smaller batch size, then marked failed; the job continues.
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from src.prep_jobs.config import data_path, job_cfg, job_dir, load_config, project_root, runs_dir
from src.prep_jobs.governor import Governor
from src.prep_jobs.util import atomic_path, dir_size, peak_memory_gb, read_json, rss_gb, write_json

STATES = ("queued", "running", "paused", "stopped", "done", "done_with_failures", "failed", "crashed")


def iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, timezone.utc).astimezone().isoformat(timespec="seconds") if ts else None


# ------------------------------------------------------------------ worker side
class UnitContext:
    """Handed to a unit function: config, batch size, memory guard and progress reporting."""

    def __init__(self, job: str, cfg: dict, run_dir: Path, unit_key: str, attempt: int,
                 batch_rows: int, rss_limit_gb: float) -> None:
        self.job, self.cfg, self.run_dir = job, cfg, run_dir
        self.jcfg = cfg["jobs"][job]
        self.unit_key, self.attempt = unit_key, attempt
        self.batch_rows, self.rss_limit_gb = batch_rows, rss_limit_gb
        self._last_report = 0.0
        self.t0 = time.time()

    def log(self, msg: str) -> None:
        with open(self.run_dir / "job.log", "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} [{self.unit_key}] {msg}\n")

    def report(self, rows: int | None = None, fraction: float | None = None, force: bool = False) -> None:
        """Tell the runner how far this unit is (rows read so far, fraction of the unit done)."""
        now = time.time()
        if not force and now - self._last_report < 15:
            return
        self._last_report = now
        write_json({"rows": rows or 0, "fraction": fraction or 0.0, "t": now, "rss_gb": rss_gb()},
                   self.run_dir / "progress" / f"{self.unit_key}.json")

    def over_limit(self) -> bool:
        return rss_gb() > self.rss_limit_gb

    def shrink(self) -> None:
        self.batch_rows = max(1000, self.batch_rows // 2)
        self.log(f"memory above {self.rss_limit_gb} GB: batch size now {self.batch_rows}")


def _low_priority() -> None:
    try:
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:  # pragma: no cover
        pass


def unit_entry(job: str, cfg_path: str, phase_name: str, fn_path: str, unit: dict, attempt: int,
               batch_rows: int, run_dir: str, rss_limit_gb: float, low_priority: bool) -> None:
    """Process target: run one unit and write its checkpoint."""
    if low_priority:
        _low_priority()
    run_dir_p = Path(run_dir)
    key = f"{phase_name}__{unit['id']}"
    try:
        cfg = load_config(cfg_path)
        ctx = UnitContext(job, cfg, run_dir_p, key, attempt, batch_rows, rss_limit_gb)
        module, func = fn_path.split(":")
        result = getattr(importlib.import_module(module), func)(ctx, unit) or {}
        ck = {"unit": unit["id"], "phase": phase_name, "rows": int(result.pop("rows", 0)),
              "bytes": int(result.pop("bytes", 0)), "seconds": round(time.time() - ctx.t0, 1),
              "peak_gb": round(peak_memory_gb(), 3), "work": unit["weight"], "attempt": attempt,
              "finished_at": iso(time.time()), **result}
        write_json(ck, run_dir_p / "checkpoints" / f"{key}.done")
    except BaseException:  # noqa: BLE001 - report everything, the parent decides about retries
        err = traceback.format_exc()
        (run_dir_p / "errors").mkdir(exist_ok=True)
        with open(run_dir_p / "errors" / f"{key}.txt", "a", encoding="utf-8") as fh:
            fh.write(f"--- attempt {attempt} at {datetime.now():%H:%M:%S}\n{err}\n")
        sys.exit(1)


# ------------------------------------------------------------------ parent side
class JobRun:
    def __init__(self, job: str, workers: int, resume: bool, deadline_at: float | None, cfg_path: str) -> None:
        self.job, self.cfg_path = job, cfg_path
        self.cfg = load_config(cfg_path)
        self.fw = self.cfg["framework"]
        self.guard = self.cfg["load_guard"]
        self.jcfg = job_cfg(self.cfg, job)
        self.run_dir = job_dir(self.cfg, job)
        self.runs = runs_dir(self.cfg)
        self.workers = max(1, min(workers, (os.cpu_count() or 2) - self.guard["reserved_cores"]))
        self.resume = resume
        self.mod = importlib.import_module(f"src.prep_jobs.jobs.{job}")
        self.started = time.time()
        self.deadline_at = deadline_at or self.started + self.fw["deadline_hours"] * 3600
        self.state, self.phase, self.last_error, self.waiting_for = "queued", "", "", ""
        self.hint = "if at risk, cut in this order: " + "; ".join(getattr(self.mod, "DEADLINE_CUTS", []))
        self.pause_reason = ""
        self.lock = threading.Lock()
        self.phases: list[dict] = []
        self.units_total = 0
        self.work_total = 0.0
        self.done: dict[str, dict] = {}       # unit_key -> checkpoint
        self.running: dict[str, dict] = {}    # unit_key -> info
        self.failed: dict[str, str] = {}
        self.session_start_work = 0.0
        self.samples: deque = deque()          # (time, rows) for the rolling rate
        self.peak_gb = 0.0
        self.mem_kills = 0
        self.eta_seconds: float | None = None
        self.governor = Governor(self.guard, data_path(self.cfg, "results").parent, self.runs, self.run_dir)
        self._stop_hb = threading.Event()

    # ---- bookkeeping
    def ck_path(self, key: str) -> Path:
        return self.run_dir / "checkpoints" / f"{key}.done"

    def load_checkpoints(self) -> None:
        self.done = {}
        for f in (self.run_dir / "checkpoints").glob("*.done"):
            ck = read_json(f)
            if ck:
                self.done[f.stem] = ck

    def work_done(self) -> tuple[float, int]:
        with self.lock:
            w = sum(c["work"] for c in self.done.values())
            rows = sum(c["rows"] for c in self.done.values())
            for key, info in self.running.items():
                prog = read_json(self.run_dir / "progress" / f"{key}.json", {})
                w += info["weight"] * min(0.99, prog.get("fraction", 0.0))
                rows += prog.get("rows", 0)
            return w, rows

    def build_status(self) -> dict:
        work, rows = self.work_done()
        now = time.time()
        self.samples.append((now, rows))
        while self.samples and now - self.samples[0][0] > 300:
            self.samples.popleft()
        rps = 0.0
        if len(self.samples) > 1 and self.samples[-1][0] > self.samples[0][0]:
            rps = (self.samples[-1][1] - self.samples[0][1]) / (self.samples[-1][0] - self.samples[0][0])
        session_work = work - self.session_start_work
        elapsed = now - self.started
        eta = None
        if session_work > 0 and self.units_done_count() > 0 and elapsed > 0 and self.work_total > work:
            eta = (self.work_total - work) / (session_work / elapsed)
        elif self.work_total and work >= self.work_total:
            eta = 0.0
        self.eta_seconds = eta
        eta_clock = now + eta if eta is not None else None
        at_risk = bool(eta_clock and eta_clock > self.deadline_at)
        gov = self.governor.snapshot()
        state = self.state
        if state == "running" and gov["state"] == "throttled":
            state = "throttled"
        if state == "running" and gov["state"] == "paused" and not self.running:
            state = "paused"
        with self.lock:
            return {
                "job": self.job, "state": state, "phase": self.phase, "pid": os.getpid(),
                "started_at": iso(self.started), "updated_at": iso(now), "updated_epoch": now,
                "started_epoch": self.started,
                "units_done": self.units_done_count(), "units_total": self.units_total,
                "work_done": round(work, 1), "work_total": round(self.work_total, 1),
                "percent": round(100 * work / self.work_total, 2) if self.work_total else 0.0,
                "workers": self.workers, "allowed_workers": self.governor.allowed(self.workers),
                "current_units": sorted(self.running),
                "rows_per_second": round(rps, 1),
                "output_bytes_so_far": int(sum(c["bytes"] for c in self.done.values())),
                "peak_memory_gb": round(self.peak_gb, 2),
                "eta_seconds": round(eta) if eta is not None else None, "eta_clock_time": iso(eta_clock),
                "deadline_at": iso(self.deadline_at), "at_risk": at_risk,
                "hint": self.hint if at_risk else "",
                "last_error": self.last_error, "failed_units": sorted(self.failed),
                "waiting_for": self.waiting_for, "pause_reason": self.pause_reason or gov["reason"],
                "system": gov,
            }

    def units_done_count(self) -> int:
        return len(self.done)

    def write_status(self) -> None:
        write_json(self.build_status(), self.run_dir / "status.json")

    def heartbeat(self) -> None:
        while not self._stop_hb.wait(self.fw["heartbeat_seconds"]):
            try:
                self.write_status()
            except Exception:  # pragma: no cover - never let the heartbeat kill the job
                pass

    # ---- control files
    def stop_requested(self) -> bool:
        return (self.run_dir / "STOP").exists()

    # ---- main
    def prepare(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for sub in ("checkpoints", "progress", "errors"):
            (self.run_dir / sub).mkdir(exist_ok=True)
        if self.resume:
            self.load_checkpoints()
        else:
            self.load_checkpoints()
            if self.done:
                raise SystemExit(f"{self.job}: {len(self.done)} units already done; use --resume (or --fresh)")

    def log(self, msg: str) -> None:
        with open(self.run_dir / "job.log", "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}\n")

    def wait_for_dependencies(self) -> bool:
        deps = getattr(self.mod, "DEPENDS_ON", None)
        if deps is None:
            deps = self.jcfg.get("depends_on", [])
        while True:
            waiting = []
            for dep in deps:
                st = read_json(self.runs / dep / "status.json", {})
                if st.get("state") != "done":
                    waiting.append(f"{dep} ({st.get('state', 'not launched')})")
            if not waiting:
                self.waiting_for = ""
                return True
            self.waiting_for = "waiting for " + ", ".join(waiting)
            if self.stop_requested():
                return False
            time.sleep(15)

    def run(self) -> str:
        self.prepare()
        self.governor.start()
        threading.Thread(target=self.heartbeat, daemon=True).start()
        self.write_status()
        self.log(f"start pid={os.getpid()} workers={self.workers} resume={self.resume}")
        if not self.wait_for_dependencies():
            self.state = "stopped"
            return self.state
        self.state = "running"
        try:
            self.phases = self.mod.phases(self.cfg)
        except Exception:
            self.last_error = traceback.format_exc(limit=3)
            self.state = "failed"
            self.log(self.last_error)
            return self.state
        self.units_total = sum(len(p["units"]) for p in self.phases)
        self.work_total = sum(u["weight"] for p in self.phases for u in p["units"])
        self.load_checkpoints()
        self.session_start_work = self.work_done()[0]
        self.started = time.time()  # ETA is per session
        self.write_status()
        for phase in self.phases:
            if self.stop_requested():
                break
            if phase.get("needs_all_prior") and self.failed:
                self.log(f"phase {phase['name']} skipped: earlier units failed {sorted(self.failed)}")
                self.last_error = f"phase {phase['name']} skipped because earlier units failed"
                for u in phase["units"]:
                    self.failed[f"{phase['name']}__{u['id']}"] = "skipped"
                continue
            outcome = self.run_phase(phase)
            if outcome in ("paused", "stopped"):
                self.state = outcome
                break
        else:
            self.state = "done_with_failures" if self.failed else "done"
        if self.state == "running":  # broke out via STOP before finishing
            self.state = "stopped"
        return self.state

    def run_phase(self, phase: dict) -> str:
        name, fn = phase["name"], phase["fn"]
        self.phase = name
        cap = phase.get("workers") or self.workers
        cap = min(cap, self.workers)
        ctx = mp.get_context("spawn")
        todo = [u for u in phase["units"] if f"{name}__{u['id']}" not in self.done]
        attempts: dict[str, int] = {}
        not_before: dict[str, float] = {}
        batch = {u["id"]: self.fw["batch_rows"] for u in todo}
        procs: dict[str, Any] = {}
        queue = list(todo)
        limit, kill_factor = self.guard["worker_rss_limit_gb"], self.guard["worker_rss_kill_factor"]
        while queue or procs:
            now = time.time()
            # finish
            for key in list(procs):
                info = procs[key]
                proc = info["proc"]
                try:
                    rss = psutil.Process(proc.pid).memory_info().rss / 1e9
                    info["max_rss"] = max(info["max_rss"], rss)
                    self.peak_gb = max(self.peak_gb, rss)
                    if rss > limit * kill_factor and proc.is_alive():
                        proc.kill()
                        info["killed_memory"] = True
                except psutil.Error:
                    pass
                if proc.is_alive():
                    continue
                proc.join()
                unit = info["unit"]
                uid = unit["id"]
                with self.lock:
                    del procs[key]
                    self.running.pop(key, None)
                if self.ck_path(key).exists():
                    ck = read_json(self.ck_path(key))
                    with self.lock:
                        self.done[key] = ck
                    self.peak_gb = max(self.peak_gb, ck.get("peak_gb", 0))
                    (self.run_dir / "progress" / f"{key}.json").unlink(missing_ok=True)
                    self.log(f"done {key}: {ck['rows']:,} rows, {ck['bytes'] / 1e6:.0f} MB, {ck['seconds']} s, "
                             f"peak {ck['peak_gb']} GB")
                    reason = getattr(self.mod, "after_unit", lambda *a: None)(self.cfg, name, unit, ck)
                    if reason:
                        self.pause_reason = reason
                        self.last_error = reason
                        self.log(f"PAUSED by job: {reason}")
                        for p in procs.values():
                            p["proc"].join()
                        return "paused"
                else:
                    attempts[uid] = attempts.get(uid, 0) + 1
                    why = "memory (killed)" if info.get("killed_memory") else f"exit code {proc.exitcode}"
                    err = read_json(self.run_dir / "errors" / f"{key}.txt", None)
                    self.last_error = f"{key}: {why}"
                    self.log(f"FAILED {key} attempt {attempts[uid]}: {why}")
                    if info.get("killed_memory"):  # memory-killed twice: halve this job's workers
                        self.mem_kills += 1
                        if self.mem_kills % 2 == 0 and self.workers > 1:
                            self.workers = max(1, self.workers // 2)
                            note = f"- {datetime.now():%Y-%m-%d %H:%M} memory-killed {self.mem_kills} times: workers halved to {self.workers}\n"
                            with open(self.run_dir / "NOTES.md", "a", encoding="utf-8") as fh:
                                fh.write(note)
                            self.log(note.strip())
                    if attempts[uid] >= self.fw["max_attempts"]:
                        with self.lock:
                            self.failed[key] = why
                        self.log(f"giving up on {key}")
                    else:
                        batch[uid] = max(1000, batch[uid] // 2)
                        not_before[uid] = time.time() + self.fw["backoff_seconds"] * 2 ** (attempts[uid] - 1)
                        queue.insert(0, unit)
            # dispatch
            if not self.stop_requested():
                allowed = min(cap, self.governor.allowed(self.workers))
                ready = [u for u in queue if not_before.get(u["id"], 0) <= time.time()]
                while ready and len(procs) < allowed:
                    unit = ready.pop(0)
                    queue.remove(unit)
                    key = f"{name}__{unit['id']}"
                    attempt = attempts.get(unit["id"], 0) + 1
                    proc = ctx.Process(target=unit_entry, args=(
                        self.job, self.cfg_path, name, fn, unit, attempt, batch[unit["id"]],
                        str(self.run_dir), limit, self.fw["low_priority"]), name=key)
                    proc.start()
                    with self.lock:
                        procs[key] = {"proc": proc, "unit": unit, "max_rss": 0.0}
                        self.running[key] = {"weight": unit["weight"], "started": time.time()}
                    self.log(f"start {key} (attempt {attempt}, batch {batch[unit['id']]})")
            elif not procs:
                return "stopped"
            time.sleep(2.0)
        return "ok"


def acquire_lock(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = run_dir / "lock"
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            (run_dir / "pid").write_text(str(os.getpid()))
            return lock
        except FileExistsError:
            try:
                pid = int(lock.read_text().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and psutil.pid_exists(pid) and pid != os.getpid():
                raise SystemExit(f"{run_dir.name} is already running (pid {pid}); lock file {lock}")
            lock.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--deadline-at", type=float, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg_path = args.config or str(Path(__file__).resolve().parents[2] / "config" / "prep_jobs.yaml")
    cfg = load_config(cfg_path)
    rd = job_dir(cfg, args.job)
    lock = acquire_lock(rd)
    if cfg["framework"]["low_priority"]:
        _low_priority()
    if cfg["framework"]["prevent_sleep"] and os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED
        except Exception:  # pragma: no cover
            pass
    run = JobRun(args.job, args.workers, args.resume, args.deadline_at, cfg_path)
    try:
        state = run.run()
    except SystemExit as exc:
        run.state, run.last_error = "failed", str(exc)
        state = "failed"
    except BaseException:  # noqa: BLE001
        run.state, run.last_error = "failed", traceback.format_exc(limit=5)
        run.log(run.last_error)
        state = "failed"
    run.state = state
    run._stop_hb.set()
    run.governor.stop()
    run.write_status()
    run.log(f"end state={state}")
    lock.unlink(missing_ok=True)
    return 0 if state in ("done", "stopped", "paused") else 1


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())

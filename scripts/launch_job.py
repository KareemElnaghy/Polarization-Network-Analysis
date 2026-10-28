"""Launch a prep job as a fully detached background process and return immediately.

    python scripts/launch_job.py <job> [--workers N] [--resume | --fresh] [--deadline-hours H]

The job runs from a snapshot of ``src/`` and the config in ``runs/<job>/snapshot/``, so later edits and
commits cannot change it. Closing the terminal, VS Code or the chat does not stop it. Stop it cleanly
with an empty file ``runs/<job>/STOP``; check on it with ``python scripts/jobs_status.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def trampoline(payload_json: str) -> int:
    """Start the runner detached, record its PID, and exit, so the runner has no living parent."""
    p = json.loads(payload_json)
    run_dir = Path(p["run_dir"])
    env = dict(os.environ, PREP_PROJECT_ROOT=p["root"], PYTHONHASHSEED="0", PYTHONUNBUFFERED="1",
               PYTHONPATH=p["snapshot"])
    out = open(run_dir / "stdout.log", "ab")
    err = open(run_dir / "stderr.log", "ab")
    kwargs = dict(cwd=p["snapshot"], env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, close_fds=True)
    if os.name == "nt":
        base = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        try:  # leave any job object the terminal put us in, if it allows that
            proc = subprocess.Popen(p["cmd"], creationflags=base | CREATE_BREAKAWAY_FROM_JOB, **kwargs)
        except OSError:
            proc = subprocess.Popen(p["cmd"], creationflags=base, **kwargs)
    else:
        proc = subprocess.Popen(p["cmd"], start_new_session=True, **kwargs)
    (run_dir / "launch.pid").write_text(str(proc.pid))
    return 0


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--trampoline":
        return trampoline(sys.argv[2])
    import psutil
    from src.prep_jobs.allocation import plan
    from src.prep_jobs.config import job_dir, load_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job")
    ap.add_argument("--workers", type=int)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--resume", action="store_true", help="skip units that have a checkpoint")
    g.add_argument("--fresh", action="store_true", help="discard checkpoints and start over")
    ap.add_argument("--deadline-hours", type=float)
    args = ap.parse_args()

    cfg = load_config()
    if args.job not in cfg["jobs"]:
        print(f"unknown job {args.job!r}; known: {', '.join(cfg['jobs'])}")
        return 2
    run_dir = job_dir(cfg, args.job)
    lock = run_dir / "lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip() or 0)
        except (OSError, ValueError):
            pid = 0
        if pid and psutil.pid_exists(pid):
            print(f"{args.job} is already running (pid {pid}). Not starting a second copy.")
            return 3
    p = plan(cfg)
    workers = args.workers or p["alloc"].get(args.job) or 1
    print("NOTICE: this PC is running heavy background jobs at low priority; it stays usable but may be slower.")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for sub in ("checkpoints", "progress", "errors"):
            shutil.rmtree(run_dir / sub, ignore_errors=True)
        (run_dir / "status.json").unlink(missing_ok=True)
    (run_dir / "STOP").unlink(missing_ok=True)

    snapshot = run_dir / "snapshot"
    shutil.rmtree(snapshot, ignore_errors=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "src", snapshot / "src", ignore=ignore)
    shutil.copytree(ROOT / "config", snapshot / "config")  # prep_jobs.yaml, gt_runs.yaml, labels.yaml, ...

    deadline_hours = args.deadline_hours or cfg["framework"]["deadline_hours"]
    cmd = [sys.executable, "-m", "src.prep_jobs.runner", args.job, "--workers", str(workers),
           "--deadline-at", str(time.time() + deadline_hours * 3600)]
    if args.resume:
        cmd.append("--resume")
    payload = json.dumps({"cmd": cmd, "run_dir": str(run_dir), "snapshot": str(snapshot), "root": str(ROOT)})
    (run_dir / "launch.pid").unlink(missing_ok=True)
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--trampoline", payload],
                     close_fds=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, **kwargs)
    for _ in range(100):
        time.sleep(0.2)
        if (run_dir / "launch.pid").exists():
            break
    pid = (run_dir / "launch.pid").read_text() if (run_dir / "launch.pid").exists() else "?"
    print(f"launched {args.job}: pid {pid}, {workers} workers, deadline {deadline_hours:g} h, "
          f"{'resuming' if args.resume else 'fresh'}")
    print(f"  status: python scripts/jobs_status.py   logs: {run_dir / 'job.log'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Show the status of every prep job (reads runs/*/status.json).

    python scripts/jobs_status.py [--watch SECONDS] [--json]

Flags: STALE (heartbeat older than 90 s while running), CRASHED (process gone while running; the state is
rewritten to ``crashed``), AT RISK (ETA is past the deadline), THROTTLED/PAUSED with the reason.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import psutil  # noqa: E402

from src.prep_jobs.config import load_config, runs_dir  # noqa: E402
from src.prep_jobs.util import read_json, write_json  # noqa: E402

ACTIVE = ("running", "queued", "throttled", "paused")


def fmt_dur(s) -> str:
    if s is None:
        return "-"
    s = int(s)
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def fmt_size(b) -> str:
    return f"{b / 1e9:.1f} GB" if b >= 1e9 else f"{b / 1e6:.0f} MB"


def pid_alive(pid) -> bool:
    return bool(pid) and psutil.pid_exists(int(pid))


def collect(runs: Path) -> list[dict]:
    rows = []
    now = time.time()
    for f in sorted(runs.glob("*/status.json")):
        st = read_json(f)
        if not st:
            continue
        flags = []
        age = now - st.get("updated_epoch", now)
        if st["state"] in ACTIVE and not pid_alive(st.get("pid")):
            st["state"] = "crashed"
            write_json(st, f)
            flags.append("CRASHED")
        elif st["state"] == "running" and age > 90:
            flags.append("STALE")
        if st.get("at_risk"):
            flags.append("AT RISK" + (f" (cut: {st['hint']})" if st.get("hint") else ""))
        if st["state"] == "throttled" or (st["state"] in ("running", "paused") and st.get("pause_reason")):
            flags.append(f"{st['state'].upper()}: {st.get('pause_reason') or st['system'].get('reason', '')}")
        if st.get("failed_units"):
            flags.append(f"{len(st['failed_units'])} failed")
        if st.get("waiting_for"):
            flags.append(st["waiting_for"])
        st["_flags"], st["_age"] = flags, age
        rows.append(st)
    return rows


def render(runs: Path) -> str:
    rows = collect(runs)
    now = time.time()
    head = (f"{'job':26s} {'state':12s} {'pct':>6s} {'units':>7s} {'wk':>3s} {'elapsed':>8s} {'ETA':>8s} "
            f"{'ETA at':>7s} {'beat':>6s} {'output':>9s}  flags")
    lines = [head, "-" * len(head)]
    etas = []
    for st in rows:
        live = st["state"] in ACTIVE
        end = now if live else st.get("updated_epoch", now)
        eta_at = (st.get("eta_clock_time") or "")[11:16] if live else ""
        if live and st.get("eta_seconds") is not None:
            etas.append(now + st["eta_seconds"])
        lines.append(f"{st['job']:26s} {st['state']:12s} {st['percent']:>5.1f}% {st['units_done']:>3d}/{st['units_total']:<3d} "
                     f"{st['workers']:>3d} {fmt_dur(end - st.get('started_epoch', end)):>8s} "
                     f"{fmt_dur(st.get('eta_seconds')) if live else '-':>8s} {eta_at:>7s} "
                     f"{fmt_dur(st['_age']):>6s} {fmt_size(st.get('output_bytes_so_far', 0)):>9s}  {'; '.join(st['_flags'])}")
    if not rows:
        lines.append("(no jobs found under runs/)")
    vm = psutil.virtual_memory()
    du = shutil.disk_usage(runs.parent if runs.exists() else ROOT)
    overall = time.strftime("%H:%M", time.localtime(max(etas))) if etas else "-"
    lines += ["", f"overall ETA (latest job): {overall}",
              f"system: CPU {psutil.cpu_percent(interval=0.3):.0f}%  RAM used {vm.percent:.0f}%  "
              f"free disk {du.free / 1e9:.0f} GB ({100 * du.free / du.total:.0f}%)"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=int, metavar="SECONDS")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    runs = runs_dir(load_config())
    while True:
        if args.json:
            print(json.dumps([{k: v for k, v in s.items() if not k.startswith("_")} | {"flags": s["_flags"]}
                              for s in collect(runs)], indent=2))
        else:
            if args.watch:
                print("\033[2J\033[H", end="")
            print(render(runs))
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())

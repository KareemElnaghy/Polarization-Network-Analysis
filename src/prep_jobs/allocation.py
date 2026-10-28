"""How many worker processes each job gets.

Total workers across all jobs = min(cpu_count - reserved_cores, floor(headroom * free_RAM_GB / peak_GB_per_worker)).
Peak memory per worker comes from units already measured (checkpoints), else the job's config value, else 4 GB.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import psutil

from src.prep_jobs.config import data_path, runs_dir
from src.prep_jobs.util import read_json

LAUNCH_ORDER = ["phrase_counts", "clean_edges_and_sampling", "thread_user", "label_versions",
                "within_camp_setup", "label_coverage"]


def measured_peak_gb(cfg: dict, job: str) -> float | None:
    peaks = [read_json(f, {}).get("peak_gb", 0) for f in (runs_dir(cfg) / job / "checkpoints").glob("*.done")]
    return max(peaks) if peaks else None


def peak_for(cfg: dict, job: str) -> tuple[float, str]:
    m = measured_peak_gb(cfg, job)
    if m:
        return max(m * 1.2, 0.5), "measured"
    return cfg["framework"]["peak_gb_per_worker_default"], "default (unmeasured)"


def plan(cfg: dict) -> dict:
    guard, fw = cfg["load_guard"], cfg["framework"]
    cpu = os.cpu_count() or 2
    mem = psutil.virtual_memory()
    free_gb = mem.available / 1e9
    peaks = {j: peak_for(cfg, j) for j in LAUNCH_ORDER}
    worst = max(p for p, _ in peaks.values())
    # one global figure: sized for the heaviest job's per-worker peak
    total = max(1, min(cpu - guard["reserved_cores"], int(fw["ram_headroom_fraction"] * free_gb // worst)))
    share = cfg["allocation"]
    jobs = [j for j in LAUNCH_ORDER if share.get(j, 0) > 0]
    alloc = {j: max(1, int(total * share[j])) for j in jobs}
    # hand leftover workers to the biggest shares first
    left = total - sum(alloc.values())
    for j in sorted(jobs, key=lambda x: -share[x]):
        if left <= 0:
            break
        alloc[j] += 1
        left -= 1
    alloc["comment_flags"] = cfg["comment_flags_workers"]
    alloc["label_coverage"] = cfg["label_coverage_workers"]
    usage = shutil.disk_usage(data_path(cfg, "runs").parent)
    return {"cpu_count": cpu, "ram_total_gb": mem.total / 1e9, "ram_free_gb": free_gb, "peak_gb_per_worker": worst,
            "peaks": peaks, "total_workers": total, "alloc": alloc, "free_disk_gb": usage.free / 1e9}


def format_plan(p: dict) -> str:
    lines = [f"CPU cores {p['cpu_count']}, RAM {p['ram_total_gb']:.0f} GB total / {p['ram_free_gb']:.0f} GB free, "
             f"disk free {p['free_disk_gb']:.0f} GB",
             f"peak GB per worker (heaviest job): {p['peak_gb_per_worker']:.1f}  ->  "
             f"total workers = min({p['cpu_count']} - reserved, floor(0.8 x free / peak)) = {p['total_workers']}",
             "", f"{'job':28s} {'workers':>7s}  {'peak GB/worker':>14s}  note"]
    notes = {"comment_flags": "short prerequisite scan, runs first",
             "label_coverage": "waits for thread_user, clean_edges_and_sampling, label_versions"}
    for j in ["comment_flags"] + LAUNCH_ORDER:
        peak = p["peaks"].get(j, (p["peak_gb_per_worker"], ""))
        lines.append(f"{j:28s} {p['alloc'][j]:>7d}  {peak[0]:>9.1f} ({peak[1]:8s})  {notes.get(j, '')}")
    return "\n".join(lines)

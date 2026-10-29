"""Print the worker plan for the ground-truth jobs from measured peak memory (runs/gt_measure.json).

Total workers = min(cpu_count - reserved_cores, floor(budget_fraction x free RAM / heaviest measured peak)),
split among T1 / T2 / T3 / T4 by ``split`` in config/gt_runs.yaml (minimum 1 each).
Measure a unit first:  python scripts/measure_unit.py <job> <week>
"""

import json
import os
import sys
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.gt_runs.config import gt_cfg  # noqa: E402
from src.prep_jobs.config import load_config, runs_dir  # noqa: E402

if __name__ == "__main__":
    cfg, g = load_config(), gt_cfg()
    measured = {}
    f = runs_dir(cfg) / "gt_measure.json"
    if f.exists():
        measured = json.loads(f.read_text())
    free = psutil.virtual_memory().available / 1e9
    peaks = {j: m["peak_gb"] for j, m in measured.items()}
    heavy = max(peaks.values(), default=g["memory"]["default_peak_gb"])
    total = max(1, min((os.cpu_count() or 2) - cfg["load_guard"]["reserved_cores"],
                       int(g["memory"]["ram_budget_fraction"] * free // heavy)))
    print(f"free RAM {free:.0f} GB, heaviest measured peak {heavy:.2f} GB "
          f"({', '.join(f'{j} {p}' for j, p in peaks.items()) or 'none measured'})")
    print(f"total workers = min({os.cpu_count()} - {cfg['load_guard']['reserved_cores']}, "
          f"floor({g['memory']['ram_budget_fraction']} x {free:.0f} / {heavy:.2f})) = {total}")
    alloc = {j: max(1, int(total * s)) for j, s in g["split"].items()}
    for j in sorted(alloc, key=lambda j: -g["split"][j]):  # hand out the remainder, largest share first
        if sum(alloc.values()) >= total:
            break
        alloc[j] += 1
    alloc["gt_variants"], alloc["gt_summary"], alloc["labels"] = 1, 1, 1
    for j, w in alloc.items():
        print(f"  {j:26s} {w:>2d} workers   peak {peaks.get(j, '-')}")
    print(json.dumps(alloc))

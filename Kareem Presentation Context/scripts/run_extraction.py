"""Detached, resumable extraction run (config configs/llm/llm.yaml). Selects whole threads per week in a fixed random
order until ~budget edges/week (the crossing thread is included; weeks with fewer edges take everything), then runs the
shared extraction. Stop it by creating <run_dir>/STOP; rerun with the same --budget to resume (cache) or a larger one to extend.
usage: python scripts/llm/run_extraction.py --rate 0.01 --budget 230 --name tier0 [--plan-only]"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import signal
import threading

sys.path.insert(0, ".")
from src.algorithms.llm import extract, provenance
from src.algorithms.llm.backends import OllamaBackend
from src.algorithms.llm.data import load_shared
from src.algorithms.llm.schema import pair_hash
from src.algorithms.llm.store import Store

ap = argparse.ArgumentParser()
ap.add_argument("--rate", type=float, default=0.01)
ap.add_argument("--budget", type=int, default=230)
ap.add_argument("--name", default="tier0")
ap.add_argument("--plan-only", action="store_true")
a = ap.parse_args()
cfg = yaml.safe_load(open("configs/llm/llm.yaml"))

for f in glob.glob("runs/*/status.json"):  # hard rule 4
    if json.load(open(f)).get("state") in ("running", "throttled"):
        sys.exit(f"REFUSED: prep job running ({f})")

e = load_shared(a.rate)
rng = np.random.default_rng(cfg["seed"])
sel = []
for wk, g in e.groupby("week"):
    sizes = g.groupby("link_id").size()
    order = sizes.index.to_numpy()[np.random.default_rng([cfg["seed"], int(wk.replace("-", ""))]).permutation(len(sizes))]
    cum, keep, ends = 0, [], {}
    for t in order:
        if cum >= a.budget:
            break
        keep.append(t)
        cum += sizes[t]
        ends[t] = cum
    gk = g[g.link_id.isin(keep)]
    # position of each thread along its week's random thread order, as a fraction of the week's selected edges: processing in
    # ascending order gives a thread-complete subsample at about the same fraction in every week at any stopping point
    sel.append(gk.assign(frac=gk.link_id.map(ends) / cum))
import pandas as pd
sel = pd.concat(sel)
sel["h"] = [pair_hash(p, b, cfg["max_chars"]) for p, b in zip(sel.parent, sel.body)]
n_unique = sel.h.nunique()
per_week = sel.groupby("week").size()
print(f"selected {len(sel)} edges, {n_unique} unique pairs, {sel.link_id.nunique()} threads; per week min/med/max "
      f"{per_week.min()}/{int(per_week.median())}/{per_week.max()}; at 0.7 edges/s ~ {n_unique / 0.7 / 3600:.1f} h")
if a.plan_only:
    sys.exit()

run = provenance.new_run(a.name, cfg, ["data/processed/edges_sampled/sampled_threads.parquet", "configs/llm/llm.yaml"],
                         sample_rate=a.rate, budget_per_week=a.budget, weeks=[per_week.index.min(), per_week.index.max()])
sel[["comment_id", "link_id", "week", "h", "frac"]].to_parquet(run / "selected_edges.parquet")
store = Store(cfg["store"])
backend = OllamaBackend(cfg)
t0 = time.time()
log = run / "status.json"


stop_flag = threading.Event()
signal.signal(signal.SIGINT, lambda *_: (stop_flag.set(), print("\nCtrl-C: finishing in-flight requests, then saving as partial (Ctrl-C again to force quit)")) if not stop_flag.is_set() else sys.exit(130))


def log_status(c):
    if c["done"] % 25 == 0:
        el = time.time() - t0
        log.write_text(json.dumps({**c, "state": "running", "elapsed_h": el / 3600,
                                   "eta_h": (c["to_run"] - c["done"]) * el / max(c["done"], 1) / 3600}))


print(f"run folder {run}; {len(sel)} edges, {n_unique} unique pairs; Ctrl-C stops safely and the run can be resumed by rerunning the same command")
c = extract.run(sel, backend, store, cfg, stop=lambda: stop_flag.is_set() or (run / "STOP").exists(), progress=extract.bar_progress(log_status), order_col="frac")
status = "done" if c["done"] + c["failed"] == c["to_run"] and not (run / "STOP").exists() and not stop_flag.is_set() else "partial"
log.write_text(json.dumps({**c, "state": status, "elapsed_h": (time.time() - t0) / 3600}))
provenance.finish_run(run, status, {k: c[k] for k in ("edges", "unique", "done", "failed", "truncation_rate")}, f"{a.name} extraction")

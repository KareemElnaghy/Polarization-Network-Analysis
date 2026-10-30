"""Throughput + validity benchmark on the pilot slice. Refuses to run while a prep job is alive (hard rule 4).
usage: python scripts/llm/bench.py MODEL [--parallel 1,2,4] [--n 150]"""
import argparse
import glob
import json
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import psutil
import yaml

sys.path.insert(0, ".")
from src.algorithms.llm import extract
from src.algorithms.llm.backends import OllamaBackend
from src.algorithms.llm.data import load_shared
from src.algorithms.llm.schema import pair_hash
from src.algorithms.llm.store import Store

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--parallel", default="1,4")
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--base-url", default=None)
ap.add_argument("--tag", default="")
a = ap.parse_args()

for f in glob.glob("runs/*/status.json"):  # rule 4: no benchmark while a prep job is running
    if json.load(open(f)).get("state") in ("running", "throttled"):
        sys.exit(f"REFUSED: prep job running ({f})")

cfg = yaml.safe_load(open("configs/llm/llm.yaml")) | {"model": a.model}
if a.base_url:
    cfg["base_url"] = a.base_url
key = pd.read_csv("data/llm/pilot/pilot_key_PRIVATE.csv")
edges = load_shared(0.05)
pilot = edges[edges.comment_id.isin(key.comment_id)].head(a.n)

peak = [0.0]
stop = threading.Event()


def watch():  # peak RSS of all ollama processes
    while not stop.is_set():
        peak[0] = max(peak[0], sum(p.info["memory_info"].rss for p in psutil.process_iter(["name", "memory_info"])
                                    if any(k in (p.info["name"] or "").lower() for k in ("ollama", "llama-server"))) / 1e9)
        time.sleep(1)


threading.Thread(target=watch, daemon=True).start()
rows = []
for par in map(int, a.parallel.split(",")):
    c = cfg | {"parallel": par}
    b = OllamaBackend(c)
    store = Store(f"data/llm/bench/{a.model.replace(':', '_')}{a.tag}_p{par}.sqlite")  # fresh store per setting (no cache hits)
    t0 = time.time()
    cnt = extract.run(pilot, b, store, c)
    wall = time.time() - t0
    t = store.table(extract.context(c, b))
    rows.append({"model": a.model, "parallel": par, "edges": len(pilot), "done": cnt["done"], "failed": cnt["failed"],
                 "edges_per_s": cnt["done"] / wall, "tokens_per_s": t.n_tokens.sum() / wall,
                 "first_try_valid": (t.retries == 0).mean(), "valid_final": cnt["done"] / len(pilot),
                 "probs_present": t.p_agree.notna().mean(), "peak_ollama_gb": peak[0], "wall_s": wall,
                 "truncation": cnt["truncation_rate"]})
    print(rows[-1], flush=True)
stop.set()
res = pd.DataFrame(rows)
# projection: unique shared-venue pairs per rate / best edges_per_s
best = res.edges_per_s.max()
for r in (0.01, 0.05, 0.2):
    e = load_shared(r)
    u = e.assign(h=[pair_hash(p, b_, cfg["max_chars"]) for p, b_ in zip(e.parent, e.body)]).h.nunique()
    res[f"proj_hours_{r}"] = u / res.edges_per_s / 3600
    print(f"rate {r}: {u} unique pairs -> {u / best / 3600:.1f} h at best setting")
Path("results/llm/bench").mkdir(parents=True, exist_ok=True)
res.to_csv(f"results/llm/bench/{a.model.replace(':', '_')}{a.tag}.csv", index=False)

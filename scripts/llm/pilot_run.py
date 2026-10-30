"""Run a prompt version on the pilot edges and score it against the consensus labels (data/llm/pilot/labels_kareem.csv).
Dev split freely; the held-out split needs --final and is logged (score it once, on the final prompt).
usage: python scripts/llm/pilot_run.py MODEL --prompt v2 --split dev|heldout [--final]"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, ".")
from src.algorithms.llm import extract
from src.algorithms.llm.backends import OllamaBackend
from src.algorithms.llm.data import load_shared
from src.algorithms.llm.schema import pair_hash
from src.algorithms.llm.store import Store

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--prompt", required=True)
ap.add_argument("--split", choices=["dev", "heldout"], default="dev")
ap.add_argument("--final", action="store_true")
a = ap.parse_args()
if a.split == "heldout" and not a.final:
    sys.exit("held-out split needs --final (score it once, on the frozen prompt)")

P = Path("data/llm/pilot")
cfg = yaml.safe_load(open("configs/llm/llm.yaml")) | {"model": a.model, "prompt_version": a.prompt, "parallel": 1}
key = pd.read_csv(P / "pilot_key_PRIVATE.csv").merge(pd.read_csv(P / "pilot_split.csv"))
key = key[key.split == a.split]
lab = pd.read_csv(P / "labels_kareem.csv").set_index("pilot_id").label
pair = load_shared(0.05).set_index("comment_id")
edges = pair.loc[key.comment_id].reset_index().assign(week="w")
b = OllamaBackend(cfg)
store = Store(f"data/llm/bench/pilot_{a.model.replace(':', '_')}_{a.prompt}.sqlite")
t0 = time.time()
c = extract.run(edges, b, store, cfg)
wall = time.time() - t0
t = store.table(extract.context(cfg, b)).set_index("pair_hash")
h = {pid: pair_hash(pair.loc[cid, "parent"], pair.loc[cid, "body"], cfg["max_chars"]) for pid, cid in zip(key.pilot_id, key.comment_id)}
d = pd.DataFrame({"human": lab[key.pilot_id], "model": pd.Series({p: t.loc[h[p], "agreement"] for p in key.pilot_id})})
out = {"model": a.model, "prompt": a.prompt, "split": a.split, "n": len(d), "accuracy": float((d.human == d.model).mean()),
       "kappa": float(cohen_kappa_score(d.human, d.model)), "first_try_valid": float((t.retries == 0).mean()),
       "failed": c["failed"], "mean_output_tokens": float(t.n_tokens.mean()), "edges_per_s": c["done"] / wall if c["done"] else None}
print(json.dumps(out))
print(pd.crosstab(d.human, d.model))
if a.split == "heldout":
    open(P / "heldout_evals.log", "a").write(json.dumps(out | {"time": time.strftime("%FT%T")}) + "\n")

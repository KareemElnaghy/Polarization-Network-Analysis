"""P2.2 (SPEC_ENCODER section 9): encoder inference on the full cleaned shared venue, week by week, into an encoder store (store schema).
The 1%/5%/20% tiers are subsets of the full venue and share the truncated-text pair hash, so encoding the full venue covers every tier.
Resumable: hashes already in the target store for this model are skipped. A STOP file in the repo root ends the run after the current chunk.
usage: python scripts/final/encode_edges.py --checkpoint data/llm/encoder/E3/best.pt [--weeks 1-27 | --weeks 2016-01-04,...] [--out-store ...]"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, ".")
from src.algorithms.encoder import model as em
from src.algorithms.encoder import outputs as eo
from src.algorithms.llm import provenance
from src.algorithms.llm.data import load_full, load_shared
from src.algorithms.llm.schema import clip, pair_hash
from src.algorithms.llm.store import Store

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--weeks", default=None, help="'a-b' (1-based index into the sorted week list) or comma list of week strings; default all")
ap.add_argument("--out-store", default="data/llm/encoder_outputs.sqlite")
ap.add_argument("--rate", type=float, default=None, help="encode a sampled tier (data/processed/edges_sampled/rate=R) instead of the full venue")
ap.add_argument("--chunk", type=int, default=10000, help="rows written per commit")
a = ap.parse_args()
from transformers import AutoTokenizer

all_weeks = sorted(p.name.split("=", 1)[1] for p in Path("data/processed/edges_clean").glob("week=*"))
if a.weeks is None:
    weeks = all_weeks
elif "-" in a.weeks and not a.weeks.startswith("20"):
    lo, hi = map(int, a.weeks.split("-"))
    weeks = all_weeks[lo - 1: hi]
else:
    weeks = a.weeks.split(",")

c = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
m = em.DistillEncoder(c["backbone"])
m.load_state_dict(c["state_dict"])
m = m.cuda().eval()
tok = AutoTokenizer.from_pretrained(c["backbone"])
L = c["max_length"]
sha = hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
ctx = eo.ctx_for(c["backbone"])
mc = yaml.safe_load(open("configs/llm/llm.yaml"))["max_chars"]
store = Store(a.out_store)
run = provenance.new_run("encode_edges", vars(a) | {"sha256": sha, "store_ctx": ctx, "weeks_resolved": weeks}, [a.checkpoint, "data/processed/edges_clean"], sample_rate=a.rate or 1.0)
print(f"model {ctx['model']} sha256 {sha[:12]} weeks {weeks[0]}..{weeks[-1]} ({len(weeks)})", flush=True)

t_all, n_all, last_log = time.time(), 0, time.time()
TIER = load_shared(a.rate) if a.rate else None
if TIER is not None:
    TIER["week"] = TIER.week.astype(str)
for wk in weeks:
    if Path("STOP").exists():
        print("STOP file found, exiting", flush=True)
        break
    e = load_full([wk]) if TIER is None else TIER[TIER.week == wk].copy()
    e["pair_hash"] = [pair_hash(p, b, mc) for p, b in zip(e.parent, e.body)]
    U = e.drop_duplicates("pair_hash")
    have = store.have(ctx)
    U = U[~U.pair_hash.isin(have)]
    if not len(U):
        print(f"{wk}: all {len(e)} edges already encoded", flush=True)
        continue
    P = [clip(s, mc["parent"]) for s in U.parent]
    R = [clip(s, mc["body"]) for s in U.body]
    H = U.pair_hash.tolist()
    order = np.argsort([len(p) + len(r) for p, r in zip(P, R)], kind="stable")
    tw = time.time()
    for i in range(0, len(order), a.chunk):
        idx = order[i: i + a.chunk]
        pred = em.predict(m, tok, [P[j] for j in idx], [R[j] for j in idx], L, batch=256)
        eo.write(store, [H[j] for j in idx], pred, ctx)
        n_all += len(idx)
        if time.time() - last_log > 60:
            print(f"  {wk} {i + len(idx)}/{len(order)}  total {n_all}  {n_all / (time.time() - t_all):.0f} pairs/s", flush=True)
            last_log = time.time()
    print(f"{wk}: {len(e)} edges, {len(U)} new unique pairs, {len(U) / (time.time() - tw):.0f} pairs/s, elapsed {(time.time() - t_all) / 3600:.2f} h", flush=True)

summary = {"pairs_encoded": n_all, "seconds": time.time() - t_all, "failures": 0, "sha256": sha, "weeks": len(weeks)}
provenance.finish_run(run, "done", summary, purpose="P2.2 encoder inference on the full shared venue")
print("DONE", summary, flush=True)

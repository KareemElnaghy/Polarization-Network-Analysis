"""G1.1: export the distillation dataset (SPEC_ENCODER section 2) to data/llm/distill/distill_v1.parquet.
Reads the extraction store READ-ONLY. Pilot pairs are removed. No labels are touched.
usage: python scripts/final/export_distill.py [--rate 0.05] [--out data/llm/distill]"""
import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, ".")
from src.algorithms.encoder.distill import split_of
from src.algorithms.llm.data import load_shared
from src.algorithms.llm.schema import clip, pair_hash

ap = argparse.ArgumentParser()
ap.add_argument("--rate", type=float, default=0.05)
ap.add_argument("--out", default="data/llm/distill")
a = ap.parse_args()
cfg = yaml.safe_load(open("configs/llm/llm.yaml"))
mc = cfg["max_chars"]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

e = load_shared(a.rate)
e["pair_hash"] = [pair_hash(p, b, mc) for p, b in zip(e.parent, e.body)]
e["parent_text"] = [clip(p, mc["parent"]) for p in e.parent]
e["reply_text"] = [clip(b, mc["body"]) for b in e.body]

# pilot exclusion: by comment_id -> hash, and by exact truncated text pair (to be safe)
key = pd.read_csv("data/llm/pilot/pilot_key_PRIVATE.csv", usecols=["comment_id"])
pil = e[e.comment_id.isin(key.comment_id)]
pil_hash = set(pil.pair_hash)
pil_text = set(zip(pil.parent_text, pil.reply_text))
assert len(key) == 150 and len(pil) >= 150 - 5, (len(key), len(pil))
is_pil = e.pair_hash.isin(pil_hash) | pd.Series([t in pil_text for t in zip(e.parent_text, e.reply_text)], index=e.index)

# every occurrence of every pair (the full list is recorded; the dataset keeps the first)
e = e.reset_index(drop=True).assign(occ=lambda d: d.groupby("pair_hash").cumcount())
e[["pair_hash", "comment_id", "link_id", "week", "occ"]].assign(is_pilot=is_pil.values).to_parquet(out / "edge_occurrences.parquet")
first = e[(e.occ == 0) & ~is_pil.values]

db = sqlite3.connect(f"file:{cfg['store']}?mode=ro", uri=True)
t = pd.read_sql("SELECT pair_hash,fields,p_agree,p_neutral,p_disagree FROM extractions WHERE model=? AND quant='Q4_K_M' "
                "AND prompt_version=? AND temperature=? AND seed=?", db, params=(cfg["model"], cfg["prompt_version"], cfg["temperature"], cfg["seed"]))
t = t.join(pd.json_normalize(t.pop("fields").map(json.loads)))
d = first.merge(t, on="pair_hash", how="inner", validate="1:1")
assert len(d) == len(first), f"{len(first) - len(d)} pairs missing from the cache"
d = d.rename(columns={"link_id": "thread"})
d["split"] = d.thread.map(split_of)
cols = ["pair_hash", "parent_text", "reply_text", "p_agree", "p_neutral", "p_disagree", "agreement", "emotional_charge",
        "hostility", "personal_attack", "week", "thread", "split"]
d = d[cols].reset_index(drop=True)

# integrity: the stored hash must be reproducible from the stored texts
chk = d.sample(2000, random_state=0)
assert all(pair_hash(p, b, mc) == h for p, b, h in zip(chk.parent_text, chk.reply_text, chk.pair_hash))
assert d.pair_hash.is_unique and d[["p_agree", "p_neutral", "p_disagree"]].notna().all().all()
assert not d.pair_hash.isin(pil_hash).any()

f = out / "distill_v1.parquet"
d.to_parquet(f, index=False)
man = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "file": f.name, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
       "rows": len(d), "rate": a.rate, "model": cfg["model"], "prompt_version": cfg["prompt_version"],
       "split_rule": "int(sha1(thread).hexdigest(),16)%20: 0 test, 1 val, else train",
       "pilot_removed_unique_pairs": len(pil_hash), "pilot_removed_rows_by_text_or_hash": int(is_pil.sum()),
       "edges_total": len(e), "unique_pairs_in_cache_for_tier": int(e.pair_hash.nunique()),
       "pairs_with_multiple_edges": int((e.groupby("pair_hash").size() > 1).sum()),
       "weeks": int(d.week.nunique()), "threads": int(d.thread.nunique()),
       "splits": {s: {"rows": int(g.shape[0]), "threads": int(g.thread.nunique()),
                      **{c: g[c].astype(str).value_counts(normalize=True).round(4).to_dict() for c in ["agreement", "emotional_charge", "hostility", "personal_attack"]}}
                  for s, g in d.groupby("split")}}
(out / "manifest.json").write_text(json.dumps(man, indent=1))
print(json.dumps({k: man[k] for k in ["rows", "sha256", "pilot_removed_unique_pairs", "threads", "weeks"]}))
print({s: v["rows"] for s, v in man["splits"].items()})

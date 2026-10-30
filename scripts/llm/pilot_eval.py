"""Cohen's kappa between the two human raters and each model's agreement with the consensus (rows where both raters agree).
Expects data/llm/pilot/labels_kareem.csv and labels_yasmina.csv (columns pilot_id,label). Model outputs come from the
benchmark stores in data/llm/bench/. usage: python scripts/llm/pilot_eval.py"""
import glob
import json
import sqlite3
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, ".")
key = pd.read_csv("data/llm/pilot/pilot_key_PRIVATE.csv")
k = pd.read_csv("data/llm/pilot/labels_kareem.csv").set_index("pilot_id").label.str.lower().str.strip()
y = pd.read_csv("data/llm/pilot/labels_yasmina.csv").set_index("pilot_id").label.str.lower().str.strip()
j = pd.concat([k, y], axis=1, keys=["k", "y"]).dropna()
print(f"n={len(j)} kappa(Kareem,Yasmina)={cohen_kappa_score(j.k, j.y):.3f} raw agreement={(j.k == j.y).mean():.3f}")
cons = j[j.k == j.y].k
print("consensus rows:", len(cons), cons.value_counts().to_dict())
from src.algorithms.llm.data import load_shared
pair = load_shared(0.05).set_index("comment_id")
for f in sorted(glob.glob("data/llm/bench/*_p1.sqlite")):
    db = sqlite3.connect(f)
    m = pd.read_sql("SELECT pair_hash, fields, p_agree,p_neutral,p_disagree FROM extractions", db)
    m["pred"] = m.fields.map(lambda s: json.loads(s)["agreement"])
    from src.algorithms.llm.schema import pair_hash
    import yaml
    mc = yaml.safe_load(open("configs/llm/llm.yaml"))["max_chars"]
    h = {pid: pair_hash(pair.loc[cid, "parent"], pair.loc[cid, "body"], mc) for pid, cid in zip(key.pilot_id, key.comment_id)}
    pred = m.set_index("pair_hash").pred
    c = pd.DataFrame({"h": cons.index.map(h), "lab": cons.values}).assign(pred=lambda d: d.h.map(pred)).dropna()
    print(f"{f}: acc vs consensus={(c.pred == c.lab).mean():.3f} (n={len(c)}) kappa={cohen_kappa_score(c.lab, c.pred):.3f}")

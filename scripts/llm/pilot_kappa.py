"""Human-vs-human report: Cohen's kappa and the pilot_ids where the two raters disagree. No model labels are read.
usage: python scripts/llm/pilot_kappa.py [--show]   (--show also prints the texts of the disagreements)"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data/llm/pilot")
ap.add_argument("--show", action="store_true")
a = ap.parse_args()
P = Path(a.dir)
paths = [P / "labels_kareem.csv", P / "labels_yasmina.csv"]
if not all(p.exists() for p in paths):
    sys.exit("missing: " + ", ".join(str(p) for p in paths if not p.exists()))
k, y = (pd.read_csv(p).set_index("pilot_id").label.str.lower().str.strip() for p in paths)
j = pd.concat([k, y], axis=1, keys=["kareem", "yasmina"])
both = j.dropna()
print(f"labelled: kareem {k.size}, yasmina {y.size}; both {len(both)}; one rater only {len(j) - len(both)}")
print(f"Cohen's kappa = {cohen_kappa_score(both.kareem, both.yasmina):.3f}   raw agreement = {(both.kareem == both.yasmina).mean():.3f}")
print("\nconfusion (rows kareem, columns yasmina):\n", pd.crosstab(both.kareem, both.yasmina))
d = both[both.kareem != both.yasmina]
print(f"\n{len(d)} disagreements:")
print(d.to_string())
if a.show:
    sheet = pd.read_csv(P / "pilot_sheet_blank.csv").set_index("pilot_id")
    for pid in d.index:
        print(f"\n--- {pid}: kareem={d.loc[pid, 'kareem']} yasmina={d.loc[pid, 'yasmina']}\nPARENT: {sheet.loc[pid, 'parent_comment']}\nREPLY: {sheet.loc[pid, 'reply']}")

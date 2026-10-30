"""Camp recovery of a Pipeline B run (reads user labels; the only script that does). usage: python scripts/llm/eval_camp_recovery.py PIPEB_RUN_DIR"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from src.algorithms.eval import camp_recovery as cr

run = Path(sys.argv[1])
camps = pd.read_parquet(run / "camp_assignments.parquet")
out = cr.evaluate(camps, cr.load_labels())
out.to_csv(run / "camp_recovery.csv", index=False)
out.to_parquet(run / "camp_recovery.parquet")
s = out.groupby("sign_source").apply(lambda g: pd.Series({
    "weeks": g.balanced_acc.notna().sum(), "mean_balanced_acc": g.balanced_acc.mean(), "mean_null": g.null_mean.mean(),
    "weeks_above_null_p95": (g.balanced_acc > g.null_p95).sum(), "mean_labelled_users": g.n_labelled.mean()}))
print(s.round(3).to_string())

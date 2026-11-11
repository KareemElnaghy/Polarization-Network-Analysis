"""Exports the slim per-edge input that the label-free analyses need (P1.2 noise ceilings), so another machine can run them without the data tree.
One row per extracted 5%-tier edge: comment_id, thread, week, source, target (pseudonymised), LLM probabilities and the Pipeline A edge score.
No text, no camp labels. Also copies the weekly ground truth. Writes to --out (default exports/analysis_inputs/).
usage: python scripts/final/export_analysis_inputs.py RUN_A [--out exports/analysis_inputs]"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from src.algorithms.llm.data import load_users

ap = argparse.ArgumentParser()
ap.add_argument("run_a")
ap.add_argument("--out", default="exports/analysis_inputs")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

e = pd.read_parquet(Path(a.run_a) / "edge_scores.parquet")
e = e.merge(load_users(0.05), on="comment_id", how="inner", validate="1:1").rename(columns={"link_id": "thread"})
e = e[["comment_id", "thread", "week", "source", "target", "p_agree", "p_neutral", "p_disagree", "agreement", "emotional_charge", "hostility", "personal_attack", "score"]]
e.to_parquet(out / "edges_5pct.parquet", index=False)
shutil.copy("results/ground_truth/ground_truth_weekly.csv", out / "ground_truth_weekly.csv")
shutil.copy(Path(a.run_a) / "weekly_A.csv", out / "weekly_A_reference.csv")
shutil.copy("configs/llm/pipeline_b.yaml", out / "pipeline_b.yaml")
sums = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir()) if f.name != "SHA256SUMS.txt"}
(out / "SHA256SUMS.txt").write_text("\n".join(f"{v}  {k}" for k, v in sums.items()) + "\n")
print(len(e), "edges", e.week.nunique(), "weeks", e.thread.nunique(), "threads ->", out)

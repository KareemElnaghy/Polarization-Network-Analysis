"""Pipeline A from stored extractions (no LLM calls). usage: python scripts/llm/run_pipeline_a.py EXTRACTION_RUN_DIR [--name pipeA]
Writes runs/llm/<ts>_<name>/{edge_scores.parquet,weekly_A.csv,weekly_A.parquet,pipelineA_weekly.png/.pdf/.csv} and a registry entry."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.algorithms.llm import provenance
from src.algorithms.llm.store import Store
from src.algorithms.pipeline_a import aggregate as pa

ap = argparse.ArgumentParser()
ap.add_argument("extraction_run")
ap.add_argument("--name", default="pipeA")
a = ap.parse_args()
cfg = yaml.safe_load(open("configs/llm/llm.yaml"))
src_man = json.load(open(Path(a.extraction_run) / "manifest.json"))
cfg = src_man["config"]  # use the model/prompt/seed the extraction run used
sel = pd.read_parquet(Path(a.extraction_run) / "selected_edges.parquet")
store = Store(cfg["store"])
quant = store.db.execute("SELECT DISTINCT quant FROM extractions WHERE model=?", (cfg["model"],)).fetchone()[0]
ctx = {"model": cfg["model"], "quant": quant, "prompt_version": cfg["prompt_version"], "temperature": cfg["temperature"], "seed": cfg["seed"]}
ext = store.table(ctx).drop(columns=["model", "quant", "prompt_version", "temperature", "seed"])
rules = pa.load_rules()
edges = sel.merge(ext, left_on="h", right_on="pair_hash", how="inner")  # only edges already extracted
edges = pa.score_edges(edges, rules)
weekly = pa.aggregate_weekly(edges, rules)
weekly["coverage"] = weekly.week.map(edges.groupby("week").size() / sel.groupby("week").size())

run = provenance.new_run(a.name, cfg, [str(Path(a.extraction_run) / "selected_edges.parquet")], sample_rate=src_man.get("sample_rate"),
                         source_run=src_man["run_id"], rules_version=rules["version"])
edges[["comment_id", "link_id", "week", "agreement", "emotional_charge", "hostility", "personal_attack", "score",
       "p_agree", "p_neutral", "p_disagree"]].to_parquet(run / "edge_scores.parquet")
weekly.to_parquet(run / "weekly_A.parquet")
weekly.to_csv(run / "weekly_A.csv", index=False)
weekly.to_csv(run / "pipelineA_weekly.csv", index=False)  # data behind the plot
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
x = pd.to_datetime(weekly.week)
ax.plot(x, weekly.P_A, marker="o", ms=3, color="#2a6fb0", label="P_A (mean edge score, 0-10)")
ax.plot(x[weekly.low_n], weekly.P_A[weekly.low_n], "o", ms=6, mfc="none", color="#c0392b", label=f"n < {rules['min_edges_per_week']}")
ax.axvline(pd.Timestamp("2016-06-27"), ls="--", color="gray", lw=1)
ax.set_ylabel("P_A"); ax.legend(frameon=False); ax.set_title(f"Pipeline A, weekly ({cfg['model']}, prompt {cfg['prompt_version']}, {len(edges)} edges)")
ax2.bar(x, weekly.n, width=5, color="#9aa7b4"); ax2.set_ylabel("edges")
fig.tight_layout(); fig.savefig(run / "pipelineA_weekly.png", dpi=150); fig.savefig(run / "pipelineA_weekly.pdf")
head = {"weeks": int(len(weekly)), "edges": int(len(edges)), "P_A_min": float(weekly.P_A.min()), "P_A_max": float(weekly.P_A.max()),
        "P_A_mean": float(weekly.P_A.mean()), "low_n_weeks": int(weekly.low_n.sum()), "min_coverage": float(weekly.coverage.min())}
provenance.finish_run(run, "done", head, "Pipeline A weekly scores")
print(run, head)

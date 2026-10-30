"""Calibration targets for the synthetic generator: weekly pair-graph statistics of the sampled shared-venue edges
(5% and 1% samples) and a pooled empirical degree sequence. No LLM, no labels."""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.algorithms.llm.data import load_shared, load_users
from src.algorithms.pipeline_b import structural as st

rows, pool = [], []
for rate in (0.01, 0.05):
    e = load_shared(rate)[["comment_id", "week"]].merge(load_users(rate), on="comment_id")
    for wk, g in e.groupby("week"):
        gr = st.build_graph(g.assign(w=1.0))
        n = len(gr["nodes"])
        if n < 50:
            continue
        deg = np.bincount(np.r_[gr["u"], gr["v"]], minlength=n)
        gi = st.giant(gr)
        rows.append({"rate": rate, "week": wk, "nodes": n, "pairs": len(gr["w"]), "mean_degree": deg.mean(), "median_degree": np.median(deg),
                     "p90": np.quantile(deg, .9), "p99": np.quantile(deg, .99), "max_degree": deg.max(), "giant_share": len(gi["nodes"]) / n})
        if rate == 0.05 and len(pool) < 40:
            pool.append(deg)
d = pd.DataFrame(rows)
d.to_csv("results/llm/synthetic/calibration_weekly.csv", index=False)
np.save("results/llm/synthetic/degree_pool.npy", np.concatenate(pool))
print(d.groupby("rate")[["nodes", "pairs", "mean_degree", "median_degree", "p90", "p99", "max_degree", "giant_share"]].describe().T.loc[(slice(None), ["min", "50%", "max"]), :].round(2).to_string())

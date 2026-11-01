"""Stability check (not a new test): repeats the rq1_compare delta-rho block bootstrap with several seeds and more draws, to see whether the
CI bounds in the headline verdict are stable. usage: python scripts/llm/rq1_bootstrap_stability.py RUN_A RUN_B"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

A = pd.read_csv(Path(sys.argv[1]) / "weekly_A.csv").set_index("week")["P_A"]
B = pd.read_csv(Path(sys.argv[2]) / "weekly_B.csv")
B = B[B.sign_source == "continuous_llm"].set_index("week")["P_B"]
GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start")
for col in ("gt1_excess", "gt2"):
    d = pd.concat([A, B, GT[col]], axis=1, keys=["a", "b", "g"]).dropna()
    xa, xb, g, n = d.a.to_numpy(), d.b.to_numpy(), d.g.to_numpy(), len(d)
    for seed in range(5):
        rng, ds = np.random.default_rng(seed), []
        for _ in range(10000):
            idx = np.concatenate([np.arange(s, s + 4) for s in rng.integers(0, n - 3, int(np.ceil(n / 4)))])[:n]
            ds.append(spearmanr(xb[idx], g[idx])[0] - spearmanr(xa[idx], g[idx])[0])
        ds = np.array(ds)
        print(f"{col} seed {seed}: delta {spearmanr(xb, g)[0] - spearmanr(xa, g)[0]:.3f} CI [{np.nanpercentile(ds, 2.5):.3f}, {np.nanpercentile(ds, 97.5):.3f}]  P(delta<=0)={np.mean(ds <= 0):.3f}")

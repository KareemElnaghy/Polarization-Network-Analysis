"""EXPLORATORY sensitivity (not part of eval-spec-v1): do the first and last (partial) weeks drive the correlations? usage: RUN_A RUN_B"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

A = pd.read_csv(Path(sys.argv[1]) / "weekly_A.csv").set_index("week")
B = pd.read_csv(Path(sys.argv[2]) / "weekly_B.csv")
B = B[B.sign_source == "continuous_llm"].set_index("week")
GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start")
d = pd.concat([A["P_A"], B["P_B"], GT[["gt1_excess", "gt2"]], A["n"].rename("nA")], axis=1).dropna(subset=["P_A", "P_B", "gt2"])
print("weeks", len(d), "first", d.index[0], "n edges first/last/median", int(d.nA.iloc[0]), int(d.nA.iloc[-1]), int(d.nA.median()))
for name, dd in (("all 53 weeks", d), ("drop first week", d.iloc[1:]), ("drop first and last", d.iloc[1:-1])):
    out = [f"{name:22s} n={len(dd)}"]
    for g in ("gt1_excess", "gt2"):
        ra, rb = spearmanr(dd.P_A, dd[g])[0], spearmanr(dd.P_B, dd[g])[0]
        out.append(f"{g}: rho_A {ra:+.2f} rho_B {rb:+.2f} delta {rb - ra:+.2f}")
    print(" | ".join(out))

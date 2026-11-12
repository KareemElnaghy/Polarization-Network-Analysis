"""P1.2 / A2 summary: split-half reliability r_s = Spearman(H1 series, H2 series) over weeks, Spearman-Brown 2r/(1+r), median and 5-95% over splits,
disattenuated rho = rho_full / sqrt(rel_pipeline * rel_GT) with rel_GT1 = 0.93 (handbook 6.1) and rel_GT2 = 0.79 (5% sample stability, a stand-in).
usage: python scripts/final/noise_ceiling_summary.py [--dir runs/noise_ceiling] [--out results/llm/final/noise_ceiling]"""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from src.algorithms.eval import metrics as M
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="runs/noise_ceiling"); ap.add_argument("--out", default="results/llm/final/noise_ceiling")
ap.add_argument("--A", default="runs/llm/20261107T121812_pipeA_5pct_full"); ap.add_argument("--B", default="runs/llm/20261107T121822_pipeB_5pct_full")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
D = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(f"{a.dir}/split_*_R20.csv")))
GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start"); GT.index = GT.index.astype(str)
weeks = sorted(GT.index)
full = {"P_A": pd.read_csv(f"{a.A}/weekly_A.csv").set_index("week").reindex(weeks).P_A,
        "P_B": pd.read_csv(f"{a.B}/weekly_B.csv").query("sign_source=='continuous_llm'").set_index("week").reindex(weeks).P_B}
REL_GT = {"gt1_excess": 0.93, "gt2": 0.79}
rows, rel = [], []
for s, g in D.groupby("split"):
    for col in ("P_A", "P_B"):
        p = g.pivot(index="week", columns="half", values=col)
        r = M.rho(p[1], p[2]); rel.append({"split": s, "series": col, "r_halves": r, "spearman_brown": 2 * r / (1 + r), "n_weeks": int(p.dropna().shape[0])})
rel = pd.DataFrame(rel); rel.to_csv(out / "reliability_by_split.csv", index=False)
summ = {}
for col in ("P_A", "P_B"):
    x = rel[rel.series == col].spearman_brown
    med, lo, hi = float(x.median()), float(x.quantile(.05)), float(x.quantile(.95))
    summ[col] = {"rel_median": med, "rel_p05": lo, "rel_p95": hi, "r_half_median": float(rel[rel.series == col].r_halves.median())}
    for t in REL_GT:
        rho = M.rho(full[col], GT[t])
        d = rho / np.sqrt(med * REL_GT[t]) if med > 0 else None
        summ[col][t] = {"rho_full": rho, "rel_GT": REL_GT[t], "disattenuated_rho": d, "note": None if d is not None and abs(d) <= 1 else "undefined or outside [-1,1]: reliability too low to disattenuate meaningfully"}
summ["label"] = "A2 split-half noise ceilings, 20 thread splits, R=20 nulls; rel_GT2 is a stand-in (5% sample stability)"
summ["rule"] = "if rel_B < 0.3 the observed rho_B is bounded by noise and the density ladder is the decisive test"
summ["rel_B_below_0.3"] = bool(summ["P_B"]["rel_median"] < 0.3)
(out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
print(json.dumps(summ, indent=1, default=float))

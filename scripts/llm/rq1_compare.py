"""RQ1 comparison exactly as specified in docs/eval_spec.md (tag eval-spec-v1). Does not change the spec.
Series: P_A (weekly_A.csv) and P_B continuous_llm (weekly_B.csv; NaN weeks dropped). Truths: full-week GT (gt1_excess, gt2) and the
5% sample-matched GT columns (gt1_excess_0.05, gt2_0.05, gt2_alt_auc_0.05). Per (pipeline x truth): Spearman rho on levels, circular-shift
permutation p, Spearman on first differences, partial rho controlling for weekly edge volume; volume-only baseline; delta rho_B - rho_A
with a 4-week moving-block bootstrap (2,000 draws) on the weeks where both series exist. Decision rule at the end.
usage: python scripts/llm/rq1_compare.py RUN_A RUN_B [--out results/llm/rq1_5pct] [--label 5pct]"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("run_a")
ap.add_argument("run_b")
ap.add_argument("--out", default="results/llm/rq1_5pct")
ap.add_argument("--label", default="5pct")
ap.add_argument("--sign-source", default="continuous_llm")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--block", type=int, default=4)
a = ap.parse_args()

GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start")
GT.index = GT.index.astype(str)
# PRIMARY: full-week GT. SECONDARY: 5%-matched columns, computed on all 5% threads, not on the processed prefix, so they do not match a
# partial run's subset (same caveat as Tier 0); report them labelled as secondary only.
TRUTHS = {"PRIMARY-full": ["gt1_excess", "gt2"], "SECONDARY-5pct-matched": ["gt1_excess_0.05", "gt2_0.05", "gt2_alt_auc_0.05"]}

A = pd.read_csv(Path(a.run_a) / "weekly_A.csv").set_index("week")
B = pd.read_csv(Path(a.run_b) / "weekly_B.csv")
B = B[B.sign_source == a.sign_source].set_index("week")
weeks = sorted(GT.index)
A, B = A.reindex(weeks), B.reindex(weeks)
pA = A["P_A"]
pB = B["P_B"]
volA = A["n"] if "n" in A else A.filter(like="n_").iloc[:, 0]
volB = B["n_edges_in"]


def rho(x, y):
    m = x.notna() & y.notna()
    return spearmanr(x[m], y[m])[0] if m.sum() >= 5 else np.nan


def circ_p(x, y):
    """Circular-shift permutation p: shift x against y over every distinct shift (T-1 of them; 10,000 would only repeat these)."""
    obs, T = rho(x, y), len(x)
    null = [rho(pd.Series(np.roll(x.to_numpy(), k), index=x.index), y) for k in range(1, T)]
    null = np.array([v for v in null if not np.isnan(v)])
    return (1 + (np.abs(null) >= abs(obs)).sum()) / (1 + len(null))


def partial(x, y, z):
    m = x.notna() & y.notna() & z.notna()
    rx, ry, rz = (rankdata(v[m]) for v in (x, y, z))
    res = lambda u: u - np.polyval(np.polyfit(rz, u, 1), rz)
    return np.corrcoef(res(rx), res(ry))[0, 1] if m.sum() >= 5 else np.nan


def delta_boot(gt):
    m = pA.notna() & pB.notna() & gt.notna()
    xa, xb, g = pA[m].to_numpy(), pB[m].to_numpy(), gt[m].to_numpy()
    n, rng = len(g), np.random.default_rng(0)
    nb = int(np.ceil(n / a.block))
    d = []
    for _ in range(a.boot):
        idx = np.concatenate([np.arange(s, s + a.block) for s in rng.integers(0, max(n - a.block + 1, 1), nb)])[:n]
        ra, rb = spearmanr(xa[idx], g[idx])[0], spearmanr(xb[idx], g[idx])[0]
        if not (np.isnan(ra) or np.isnan(rb)):
            d.append(rb - ra)
    point = spearmanr(xb, g)[0] - spearmanr(xa, g)[0]
    return point, np.percentile(d, 2.5), np.percentile(d, 97.5), n


rows = []
for kind, cols in TRUTHS.items():
    for c in cols:
        gt = GT[c]
        for name, s, vol in (("P_A", pA, volA), ("P_B", pB, volB)):
            m = s.notna() & gt.notna()
            rows.append({"truth_set": kind, "truth": c, "series": name, "n_weeks": int(m.sum()), "rho": rho(s, gt), "perm_p": circ_p(s, gt),
                         "rho_diff": rho(s.diff(), gt.diff()), "partial_rho_vol": partial(s, gt, vol)})
        for name, vol in (("volume_A", volA), ("volume_B", volB)):
            rows.append({"truth_set": kind, "truth": c, "series": name, "n_weeks": int((vol.notna() & gt.notna()).sum()), "rho": rho(vol, gt),
                         "perm_p": circ_p(vol, gt), "rho_diff": rho(vol.diff(), gt.diff()), "partial_rho_vol": np.nan})
res = pd.DataFrame(rows)

drows = []
for kind, cols in TRUTHS.items():
    for c in cols:
        d, lo, hi, n = delta_boot(GT[c])
        drows.append({"truth_set": kind, "truth": c, "delta_rho_B_minus_A": d, "ci_lo": lo, "ci_hi": hi, "n_weeks_both": n})
dl = pd.DataFrame(drows)
ab = rho(pA, pB)

# decision rule (eval spec v1 + clarification D-E3 logged before 5% numbers): B must beat A on GT#2 (leans against B) with the delta CI excluding 0 and B itself positive
def verdict(gt2col):
    r = dl[dl.truth == gt2col].iloc[0]
    rb = res[(res.truth == gt2col) & (res.series == "P_B")].iloc[0]
    if r.ci_lo > 0 and rb.rho > 0:
        return "B wins on GT#2 (delta CI excludes 0 and rho_B > 0)"
    if r.ci_lo > 0:
        return "NO WINNER: delta CI excludes 0 but only because A is negative or B is not positive (merely less negative)"
    return "NO WINNER (delta CI includes 0)"


cov = {"weeks_A": int(pA.notna().sum()), "weeks_B": int(pB.notna().sum()),
       "edges_A": int(A["n"].sum()) if "n" in A else None, "edges_B_input": int(volB.sum(skipna=True)),
       "min_coverage_A": float(A["coverage"].min()) if "coverage" in A else None}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
res.to_csv(out / f"rq1_{a.label}_correlations.csv", index=False)
dl.to_csv(out / f"rq1_{a.label}_delta.csv", index=False)
json.dump({"run_a": a.run_a, "run_b": a.run_b, "A_vs_B_rho": ab, "coverage": cov,
           "PRIMARY_verdict_full_gt2": verdict("gt2"), "SECONDARY_verdict_matched_gt2": verdict("gt2_0.05")}, open(out / f"rq1_{a.label}_summary.json", "w"), indent=1)
pd.set_option("display.width", 200)
print(res.round(3).to_string(index=False))
print("\nDelta rho_B - rho_A (4-week block bootstrap)\n", dl.round(3).to_string(index=False))
print(f"\nA vs B rho: {ab:.3f}\nCoverage: {cov}\nFull-week GT#2: {verdict('gt2')}\nSample-matched GT#2: {verdict('gt2_0.05')}")

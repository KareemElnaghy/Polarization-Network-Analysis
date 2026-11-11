"""P1.1 / D-E6 section A (confirmatory A1 Holm, A2 baselines, A3 06-27 split) + robustness checks, on the full-5% runs.
Statistics are those of eval-spec-v1 (Spearman, circular-shift p, 4-week moving-block bootstrap, 2,000 draws, seed 0).
usage: python scripts/final/rq1_spec_completion.py RUN_A RUN_B [--out results/llm/final/rq1_spec_completion]"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.algorithms.eval import metrics as M
from src.algorithms.llm import provenance

ap = argparse.ArgumentParser()
ap.add_argument("run_a")
ap.add_argument("run_b")
ap.add_argument("--out", default="results/llm/final/rq1_spec_completion")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
TRUTHS = ["gt1_excess", "gt2"]
COVS = ["share_politics", "ratio_rep_dem", "share_new_users"]  # D-E6 A2 (names in drivers_weekly.csv)
SPLIT = "2016-06-27"

run = provenance.new_run("final_rq1_spec_completion", {"seed": 0, "boot": a.boot, "block": 4},
                         [str(Path(a.run_a) / "weekly_A.csv"), str(Path(a.run_b) / "weekly_B.csv"),
                          "results/ground_truth/ground_truth_weekly.csv", "results/gt1_drivers/drivers_weekly.csv"],
                         sample_rate=0.05, run_a=a.run_a, run_b=a.run_b)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start")
GT.index = GT.index.astype(str)
weeks = sorted(GT.index)
A = pd.read_csv(Path(a.run_a) / "weekly_A.csv").set_index("week").reindex(weeks)
B = pd.read_csv(Path(a.run_b) / "weekly_B.csv")
B = B[B.sign_source == "continuous_llm"].set_index("week").reindex(weeks)
D = pd.read_csv("results/gt1_drivers/drivers_weekly.csv").set_index("week_start")
D.index = D.index.astype(str)
D = D.reindex(weeks)
S = {"P_A": A.P_A, "P_B": B.P_B}
assert A.P_A.notna().sum() == 53 and B.P_B.notna().sum() == 53

# ---- A1.1 Holm over family A ({A,B} x {GT#1,GT#2}) ----
fam = {f"rho_{s[2]}_{t}": M.circ_p(S[s], GT[t]) for s in S for t in TRUTHS}
rhos = {f"rho_{s[2]}_{t}": M.rho(S[s], GT[t]) for s in S for t in TRUTHS}
h = M.holm(fam).assign(rho=lambda d: d.test.map(rhos))
h.insert(0, "family", "A (D-E6 section A)")
h.to_csv(out / "holm.csv", index=False)
n_sig = int(h.significant_at_alpha.sum())
v_holm = (f"{n_sig} of 4 family-A tests significant after Holm at 0.05 (smallest adjusted p = {h.p_holm.min():.3f}). "
          f"With 52 distinct circular shifts the smallest attainable raw p is {1 / 53:.3f}, so Holm over 4 tests cannot go below {4 / 53:.3f}.")

# ---- A1.2 composition baselines ----
rows = []
for cov in COVS:
    c = D[cov]
    for t in TRUTHS:
        r = {"covariate": cov, "truth": t, "n_weeks": int((c.notna() & GT[t].notna()).sum()),
             "rho_cov_truth": M.rho(c, GT[t]), "p_cov_truth": M.circ_p(c, GT[t])}
        for s in S:
            m = (c.notna() & S[s].notna() & GT[t].notna()).to_numpy()
            x, y, z = S[s].to_numpy()[m], GT[t].to_numpy()[m], c.to_numpy()[m]
            r[f"rho_{s}"] = M.rho(pd.Series(x), pd.Series(y))
            pt, lo, hi, _ = M.block_boot(lambda u, v, w: M.partial(pd.Series(u), pd.Series(v), pd.Series(w)), [x, y, z], boot=a.boot)
            r.update({f"partial_rho_{s}": pt, f"partial_rho_{s}_lo": lo, f"partial_rho_{s}_hi": hi})
        rows.append(r)
base = pd.DataFrame(rows)
base.to_csv(out / "composition_baselines.csv", index=False)

# ---- A1.3 06-27 split ----
rows = []
for seg, wk in (("before_2016-06-27", [w for w in weeks if w < SPLIT]), ("from_2016-06-27", [w for w in weeks if w >= SPLIT])):
    for t in TRUTHS:
        for s in S:
            x, y = S[s].loc[wk], GT[t].loc[wk]
            pt, lo, hi, _ = M.block_boot(lambda u, v: M.rho(pd.Series(u), pd.Series(v)), [x.to_numpy(), y.to_numpy()], boot=a.boot)
            rows.append({"segment": seg, "n_weeks": len(wk), "truth": t, "series": s, "rho": pt, "ci_lo": lo, "ci_hi": hi,
                         "mean_P": float(x.mean()), "mean_truth": float(y.mean())})
seg = pd.DataFrame(rows)
seg.to_csv(out / "segments_0627.csv", index=False)

# ---- A1.4 robustness (existing scripts, unchanged) ----
rob = {}
for name, script in (("bootstrap_stability", "rq1_bootstrap_stability.py"), ("partial_week_sensitivity_EXPLORATORY", "rq1_partial_week_sensitivity.py")):
    p = subprocess.run([sys.executable, f"scripts/llm/{script}", a.run_a, a.run_b], capture_output=True, text=True)
    (out / f"{name}.txt").write_text(p.stdout + p.stderr)
    rob[name] = p.returncode
assert all(v == 0 for v in rob.values()), rob

b = base.set_index(["covariate", "truth"])
summary = {
    "labels": "A1.1-A1.3 CONFIRMATORY (D-E6 section A); A1.4 robustness, partial-week check EXPLORATORY",
    "weeks": 53, "run_a": a.run_a, "run_b": a.run_b,
    "holm_verdict": v_holm,
    "baseline_verdict": {f"{c}|{t}": {"rho_cov_truth": round(float(b.loc[(c, t), "rho_cov_truth"]), 3),
                                      "rho_A": round(float(b.loc[(c, t), "rho_P_A"]), 3), "partial_A": round(float(b.loc[(c, t), "partial_rho_P_A"]), 3),
                                      "rho_B": round(float(b.loc[(c, t), "rho_P_B"]), 3), "partial_B": round(float(b.loc[(c, t), "partial_rho_P_B"]), 3)}
                         for c in COVS for t in TRUTHS},
    "segments": seg.round(3).to_dict("records"),
    "robustness_exit_codes": rob,
}
(out / "summary.json").write_text(json.dumps(summary, indent=1))
for f in out.iterdir():
    (run / f.name).write_bytes(f.read_bytes())
provenance.finish_run(run, "done", {"holm_n_significant": n_sig, "min_p_holm": round(float(h.p_holm.min()), 4)}, purpose="P1.1 spec completion (Holm, baselines, 06-27 split) on full 5%")
pd.set_option("display.width", 220)
print(h.round(3).to_string(index=False)); print(v_holm)
print(base.round(3).to_string(index=False)); print(seg.round(3).to_string(index=False))
print((out / "bootstrap_stability.txt").read_text()); print((out / "partial_week_sensitivity_EXPLORATORY.txt").read_text())

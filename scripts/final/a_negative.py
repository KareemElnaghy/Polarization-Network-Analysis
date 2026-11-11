"""P1.4 / H-AN: why Pipeline A runs negative (SPEC_ANALYSES A4; D-E6 section B).
Confirmatory (family H-AN, 2 tests, defined on GT#1 as in D-E6): (i) rho(s_t, GT#1) < 0; (ii) |partial rho_A(GT#1 | s_t)| < |rho_A(GT#1)| with
a 4-week block-bootstrap CI on the difference excluding 0. GT#2 versions and the decomposition are secondary/descriptive.
usage: python scripts/final/a_negative.py RUN_A [--out results/llm/final/a_negative]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.algorithms.analysis import alignment as al
from src.algorithms.eval import metrics as M
from src.algorithms.llm import provenance

ap = argparse.ArgumentParser()
ap.add_argument("run_a")
ap.add_argument("--out", default="results/llm/final/a_negative")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
run = provenance.new_run("final_a_negative", {"seed": 0, "boot": a.boot, "block": 4}, [str(Path(a.run_a) / "edge_scores.parquet")],
                         sample_rate=0.05, run_a=a.run_a)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

E = al.build_edge_table(a.run_a, with_vader=False)
E["etype"] = np.where(~E.lb, "other", np.where(E.cross == 1, "cross", "same"))
GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start")
GT.index = GT.index.astype(str)
weeks = sorted(GT.index)
WA = pd.read_csv(Path(a.run_a) / "weekly_A.csv").set_index("week").reindex(weeks)
PA = WA.P_A
assert abs(E.groupby("week").score.mean().reindex(weeks) - PA).max() < 1e-9  # edge table reproduces the run's P_A

lb = E[E.lb]
s = lb.groupby("week").cross.mean().reindex(weeks).rename("s_t")
TR = {"gt1_excess": "PRIMARY", "gt2": "secondary"}

# ---- (i) and (ii) ----
rows = []
for t, kind in TR.items():
    r_s, p_s = M.rho(s, GT[t]), M.circ_p(s, GT[t])
    rA, pr = M.rho(PA, GT[t]), M.partial(PA, GT[t], s)
    stat = lambda x, y, z: abs(M.rho(pd.Series(x), pd.Series(y))) - abs(M.partial(pd.Series(x), pd.Series(y), pd.Series(z)))
    pt, lo, hi, d = M.block_boot(stat, [PA.to_numpy(), GT[t].to_numpy(), s.to_numpy()], boot=a.boot)
    rows.append({"truth": t, "status": kind, "rho_s_truth": r_s, "p_s_truth": p_s, "rho_A": rA, "partial_rho_A_given_s": pr,
                 "abs_diff": pt, "abs_diff_lo": lo, "abs_diff_hi": hi, "boot_p_two_sided": min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))})
test = pd.DataFrame(rows)
test.to_csv(out / "han_tests.csv", index=False)
p1 = test[test.truth == "gt1_excess"].iloc[0]
holm = M.holm({"H-AN(i) rho(s_t,GT1)": p1.p_s_truth, "H-AN(ii) |rho_A|>|partial|": max(p1.boot_p_two_sided, 1 / a.boot)})
holm.to_csv(out / "han_holm.csv", index=False)
v_i = f"(i) {'holds' if p1.rho_s_truth < 0 else 'fails'}: rho(s_t, GT#1) = {p1.rho_s_truth:+.3f} (circular-shift p = {p1.p_s_truth:.3f}, Holm-adjusted {holm.p_holm[0]:.3f})"
v_ii = (f"(ii) {'holds' if p1.abs_diff_lo > 0 else 'fails'}: |rho_A| - |partial rho_A| = {p1.abs_diff:+.3f}, 95% CI [{p1.abs_diff_lo:+.3f}, {p1.abs_diff_hi:+.3f}] "
        f"(rho_A = {p1.rho_A:+.3f} -> partial {p1.partial_rho_A_given_s:+.3f})")

# ---- decomposition by edge type (descriptive) ----
types = ["same", "cross", "other"]
ws = E.groupby(["week", "etype"]).score.agg(["mean", "size"]).unstack("etype").reindex(weeks)
mean_t, n_t = ws["mean"][types], ws["size"][types]
share = n_t.div(n_t.sum(axis=1), axis=0)
sub = []
for k in types:
    for t in TR:
        sub.append({"edge_type": k, "truth": t, "rho_P_A_subset": M.rho(mean_t[k], GT[t]), "perm_p": M.circ_p(mean_t[k], GT[t]),
                    "mean_score": float(E[E.etype == k].score.mean()), "mean_share": float(share[k].mean()),
                    "rho_share_truth": M.rho(share[k], GT[t])})
pd.DataFrame(sub).to_csv(out / "A_by_edge_type.csv", index=False)
pd.concat({"mean_score": mean_t, "share": share}, axis=1).to_csv(out / "weekly_by_edge_type.csv")

# midpoint shift-share against the period means: P_t - Pbar = sum_k (share_kt - share_k) * score_k_bar  [mix]  +  sum_k share_k_bar * (score_kt - score_k) [within]
sbar, mbar = share.mean(), mean_t.mean()
mix = ((share - sbar) * (mean_t + mbar) / 2).sum(axis=1)
within = (((share + sbar) / 2) * (mean_t - mbar)).sum(axis=1)
tot = PA - PA.mean()
chk = (mix + within - tot).abs().max()
d_share = share.diff()
dmix, dwithin = (d_share * (mean_t + mean_t.shift()) / 2).sum(axis=1), (((share + share.shift()) / 2) * mean_t.diff()).sum(axis=1)
dtot = PA.diff()
sf = pd.DataFrame({"week": weeks, "P_A": PA.values, "mix_component": mix.values, "within_component": within.values,
                   "d_mix": dmix.values, "d_within": dwithin.values, "d_P_A": dtot.values})
sf.to_csv(out / "shift_share.csv", index=False)
var_share = {"levels_mix": float(np.cov(mix, tot)[0, 1] / tot.var()), "levels_within": float(np.cov(within, tot)[0, 1] / tot.var()),
             "diffs_mix": float(dmix[1:].cov(dtot[1:]) / dtot[1:].var()), "diffs_within": float(dwithin[1:].cov(dtot[1:]) / dtot[1:].var()),
             "decomposition_max_abs_error_levels": float(chk)}

# ---- A components ----
comp = []
for c in ["A_disagree", "A_hostility", "share_high", "P_A"]:
    x = WA[c]
    comp.append({"component": c, **{f"rho_{t}": M.rho(x, GT[t]) for t in TR}, "rho_s_t": M.rho(x, s)})
comp = pd.DataFrame(comp)
comp.to_csv(out / "A_components.csv", index=False)

mix_dom = var_share["levels_mix"]
met_i, met_ii = bool(p1.rho_s_truth < 0), bool(p1.abs_diff_lo > 0)
sentence = (f"Pre-registered conditions as written: (i) {'met' if met_i else 'not met'}, (ii) {'met' if met_ii else 'not met'}. {v_i}; {v_ii}. "
            f"Caveats stated by the script, not resolved by it: (i) is not significant after Holm (adjusted p = {holm.p_holm[0]:.3f}); the descriptive shift-share "
            f"attributes only {var_share['levels_mix']:.0%} of the variance of P_A to the changing mix of same/cross/other edge types "
            f"({var_share['levels_within']:.0%} to within-type score changes). Which CLAIMS.md wording applies is a CP1 judgment call.")
summary = {"label": "H-AN CONFIRMATORY on GT#1 (2 tests, Holm); GT#2 rows, decomposition and components are descriptive", "run_a": a.run_a,
           "n_lb_edges": int(len(lb)), "mean_cross_share": float(s.mean()), "verdict_i": v_i, "verdict_ii": v_ii, "holm": holm.to_dict("records"),
           "shift_share_variance_shares": var_share, "sentence": sentence, "tests": test.to_dict("records")}
(out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
for f in out.iterdir():
    (run / f.name).write_bytes(f.read_bytes())
provenance.finish_run(run, "done", {"rho_s_gt1": round(float(p1.rho_s_truth), 3), "partial_rho_A_gt1": round(float(p1.partial_rho_A_given_s), 3)}, purpose="P1.4 why A runs negative (H-AN)")
pd.set_option("display.width", 220)
print(test.round(3).to_string(index=False)); print(holm.round(3).to_string(index=False)); print(pd.DataFrame(sub).round(3).to_string(index=False))
print(var_share); print(comp.round(3).to_string(index=False)); print(sentence)

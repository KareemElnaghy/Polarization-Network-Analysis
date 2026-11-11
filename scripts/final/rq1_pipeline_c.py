"""P1.5 / D-E6 section C: Pipeline C through the RQ1 statistics. Confirmatory family C = {C vs B, C vs A} x {GT#1, GT#2} (4 tests, Holm).
"C beats B": delta(rho_C - rho_B) CI > 0 on GT#1 and rho_C > 0.  "C beats A": D-E3 form on GT#2 (delta CI > 0 and rho_C > 0).
usage: python scripts/final/rq1_pipeline_c.py RUN_A RUN_B RUN_C [--variant C1] [--out results/llm/final/pipeline_c]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.algorithms.analysis import alignment as al
from src.algorithms.eval import camp_recovery as cr
from src.algorithms.eval import metrics as M
from src.algorithms.llm import provenance
from src.algorithms.pipeline_c import weights as wc

ap = argparse.ArgumentParser()
ap.add_argument("run_a")
ap.add_argument("run_b")
ap.add_argument("run_c")
ap.add_argument("--variant", default="C1")
ap.add_argument("--out", default="results/llm/final/pipeline_c")
ap.add_argument("--weights", default=None, help="C2: parquet pair_hash,w_C2")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
V = a.variant
run = provenance.new_run(f"final_rq1_{V}", {"seed": 0, "boot": a.boot, "block": 4}, [str(Path(a.run_c) / "weekly_C.csv")], sample_rate=0.05,
                         run_a=a.run_a, run_b=a.run_b, run_c=a.run_c)
out = Path(a.out) / V
out.mkdir(parents=True, exist_ok=True)

GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start")
GT.index = GT.index.astype(str)
weeks = sorted(GT.index)
PA = pd.read_csv(Path(a.run_a) / "weekly_A.csv").set_index("week").reindex(weeks).P_A
PB = pd.read_csv(Path(a.run_b) / "weekly_B.csv").query("sign_source == 'continuous_llm'").set_index("week").reindex(weeks).P_B
WC = pd.read_csv(Path(a.run_c) / "weekly_C.csv").set_index("week").reindex(weeks)
PC = WC.P_B
n_c = int(PC.notna().sum())
S = {"A": PA, "B": PB, V: PC}
TR = ["gt1_excess", "gt2"]

lev = pd.DataFrame([{"series": k, "truth": t, "n_weeks": int((s.notna() & GT[t].notna()).sum()), "rho": M.rho(s, GT[t]), "perm_p": M.circ_p(s, GT[t]),
                     "rho_diff": M.rho(s.diff(), GT[t].diff())} for k, s in S.items() for t in TR])
lev.to_csv(out / "levels.csv", index=False)

rows = []
for base in ("B", "A"):
    for t in TR:
        m = (PC.notna() & S[base].notna() & GT[t].notna()).to_numpy()
        stat = lambda c, b, g: M.rho(pd.Series(c), pd.Series(g)) - M.rho(pd.Series(b), pd.Series(g))
        pt, lo, hi, d = M.block_boot(stat, [PC.to_numpy()[m], S[base].to_numpy()[m], GT[t].to_numpy()[m]], boot=a.boot)
        rows.append({"comparison": f"{V} vs {base}", "truth": t, "n_weeks": int(m.sum()), "delta": pt, "ci_lo": lo, "ci_hi": hi,
                     "boot_p_two_sided": max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / a.boot),
                     "rho_C": M.rho(PC, GT[t]), "pre_registered_truth_for_rule": (t == "gt1_excess") if base == "B" else (t == "gt2")})
dl = pd.DataFrame(rows)
h = M.holm({f"{r.comparison}|{r.truth}": r.boot_p_two_sided for r in dl.itertuples()})
dl["p_holm"] = h.p_holm.values
dl["significant_holm"] = h.significant_at_alpha.values
dl.to_csv(out / "deltas_holm.csv", index=False)


def rule(base, truth):
    r = dl[(dl.comparison == f"{V} vs {base}") & (dl.truth == truth)].iloc[0]
    ok = r.ci_lo > 0 and r.rho_C > 0
    return (f"{V} beats {base} on {truth}: {'YES' if ok else 'NO'} (delta {r.delta:+.3f}, 95% CI [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}], rho_{V} {r.rho_C:+.3f}, "
            f"Holm-adjusted p {r.p_holm:.3f})")


# camp recovery of C vs B (reads labels; this script is an evaluation script)
camps = pd.read_parquet(Path(a.run_c) / "camp_assignments.parquet")
cp = cr.evaluate(camps, cr.load_labels())
cp.to_csv(out / "camp_recovery.csv", index=False)
cb = pd.read_csv(Path(a.run_b) / "camp_recovery.csv").query("sign_source == 'continuous_llm'")
camp = {"mean_BA_C": float(cp.balanced_acc.mean()), "mean_BA_B": float(cb.balanced_acc.mean()), "mean_null_C": float(cp.null_mean.mean()),
        "weeks_above_null_p95_C": int((cp.balanced_acc > cp.null_p95).sum()), "weeks_above_null_p95_B": int((cb.balanced_acc > cb.null_p95).sum())}

# alignment of the C weights on labelled-both edges (A3 style)
E = al.build_edge_table(a.run_a, with_vader=False)
if V == "C2":
    hmap = pd.read_parquet("runs/llm/20261102T150142_tier5pct/selected_edges.parquet").drop_duplicates("comment_id").set_index("comment_id").h
    E["w_C"] = E.comment_id.map(hmap).map(pd.read_parquet(a.weights).drop_duplicates("pair_hash").set_index("pair_hash").w_C2)
else:
    E["w_C"] = wc.VARIANTS[V](E)
lb = E[E.lb]
dw = {k: al.cluster_boot_diff(lb[c].to_numpy(), lb.cross.to_numpy(), lb.thread.to_numpy(), 1000) for k, c in (("C", "w_C"), ("B", "w_continuous_llm"))}
dw_std = {k: {"delta": v["delta"], "delta_over_sd": v["delta"] / float(lb["w_C" if k == "C" else "w_continuous_llm"].std()), "ci": [v["ci_lo"], v["ci_hi"]]} for k, v in dw.items()}

verdicts = {"C_beats_B_GT1_(rule)": rule("B", "gt1_excess"), "C_beats_B_GT2_(secondary)": rule("B", "gt2"),
            "C_beats_A_GT2_(rule)": rule("A", "gt2"), "C_beats_A_GT1_(secondary)": rule("A", "gt1_excess")}
summary = {"label": f"D-E6 section C CONFIRMATORY (family C, 4 tests, Holm); variant {V}; camp recovery and alignment are secondary",
           "weeks_scored_C": n_c, "run_c": a.run_c, "verdicts": verdicts, "camp_recovery": camp, "alignment_dw_cross_minus_same": dw_std,
           "leakage_diagnostic": "not computed: GT#2 phrase-leaning extremity per reply is not available at edge level",
           "levels": lev.round(4).to_dict("records")}
(out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
for f in out.iterdir():
    (run / f.name).write_bytes(f.read_bytes())
provenance.finish_run(run, "done" if n_c == 53 else "partial", {"weeks_scored": n_c, "rho_C_gt1": round(float(M.rho(PC, GT['gt1_excess'])), 3),
                      "rho_C_gt2": round(float(M.rho(PC, GT['gt2'])), 3)}, purpose=f"P1.5 Pipeline {V} through RQ1 statistics")
pd.set_option("display.width", 220)
print(lev.round(3).to_string(index=False)); print(dl.round(3).to_string(index=False)); print(camp); print(dw_std)
for v in verdicts.values():
    print(v)

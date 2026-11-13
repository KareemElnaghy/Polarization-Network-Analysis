"""P2.4 + P2.5 (SPEC_ANALYSES A8, D-E6 §D): statistics over the density ladder from runs/final_ladder/d<density>/<pipe>/<week>.csv.
All encoder-based numbers carry the D-E6a deviation label. Re-runnable as densities complete; a density counts as complete when every pipeline has all
weeks it was run for (partial densities are reported with n_weeks and excluded from H1 tests).
- Per density x pipeline x truth: rho, circular-shift p, 4-week block-bootstrap CI; camp recovery (graph pipelines); mean degree of the giant component.
- H1a (family D-trend, 2 tests): Spearman(density rank, rho_B) = +1, or rho_B(100%) - rho_B(5%) > 0 with CI excluding 0. Per truth.
- H1b (2 truths, Holm): at 100%, rho_B > 0 and Holm-adjusted circular-shift p < 0.05.  H1c: D-E3 rule (B vs A on GT#2) at 100%.
- P2.5 anchor: encoder 5% vs LLM 5% weekly series (A, B, C1), Spearman and Pearson.
usage: python scripts/final/ladder_stats.py [--in runs/final_ladder] [--out results/llm/final/ladder]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from src.algorithms.eval import camp_recovery as cr
from src.algorithms.eval import metrics as M

LABEL = "deviation: encoder failed G-ENC check 1 (D-E6a)"
TR = ["gt1_excess", "gt2"]
ANCHOR = {"A": ("runs/llm/20261107T121812_pipeA_5pct_full/weekly_A.csv", None, "P_A"),
          "B": ("runs/llm/20261107T121822_pipeB_5pct_full/weekly_B.csv", "continuous_llm", "P_B"),
          "B_bin": ("runs/llm/20261107T121822_pipeB_5pct_full/weekly_B.csv", "binary_llm", "P_B"),
          "C1": ("runs/llm/20261111T125450_pipeC1_5pct_full/weekly_C.csv", None, "P_B")}

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", default="runs/final_ladder")
ap.add_argument("--out", default="results/llm/final/ladder")
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--no-camps", action="store_true")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start")
GT.index = GT.index.astype(str)
weeks = sorted(GT.index)

rows = []
for d in sorted(Path(a.inp).glob("d*")):
    for p in sorted(x for x in d.iterdir() if x.is_dir()):
        fs = sorted(p.glob("*.csv"))
        if fs:
            rows.append(pd.concat([pd.read_csv(f) for f in fs]))
W = pd.concat(rows)
W["week"] = W.week.astype(str)
W["value"] = np.where(W.pipeline == "A", W.get("P_A"), W.get("P_B"))
W.to_csv(out / "weekly_all.csv", index=False)

lev, cov = [], []
for (dens, pipe), g in W.groupby(["density", "pipeline"]):
    s = g.set_index("week").value.reindex(weeks)
    c = {"density": dens, "pipeline": pipe, "n_weeks": int(s.notna().sum())}
    if pipe != "A":
        c |= {"mean_degree_giant": float(g.mean_degree_giant.mean()), "mean_nodes_giant": float(g.n_nodes_giant.mean()), "mean_P_B": float(g.P_B.mean())}
        if not a.no_camps:
            cf = sorted((Path(a.inp) / f"d{dens:g}" / pipe).glob("*.camps.parquet"))
            camps = pd.concat([pd.read_parquet(f) for f in cf]).assign(sign_source=pipe)
            ev = cr.evaluate(camps, cr.load_labels())
            c |= {"camp_BA_mean": float(ev.balanced_acc.mean()), "camp_weeks_above_null95": int((ev.balanced_acc > ev.null_p95).sum())}
    cov.append(c)
    for t in TR:
        m = s.notna() & GT[t].notna()
        pt, lo, hi, _ = M.block_boot(lambda x, y: M.rho(pd.Series(x), pd.Series(y)), [s[m].to_numpy(), GT[t][m].to_numpy()], boot=a.boot)
        lev.append({"density": dens, "pipeline": pipe, "truth": t, "n_weeks": int(m.sum()), "rho": M.rho(s, GT[t]), "perm_p": M.circ_p(s, GT[t]),
                    "ci_lo": lo, "ci_hi": hi, "label": LABEL})
lev, cov = pd.DataFrame(lev), pd.DataFrame(cov)
lev.to_csv(out / "levels.csv", index=False)
cov.to_csv(out / "coverage_degree_camps.csv", index=False)

# ---- H1 tests (only densities with all 53 weeks for B) ----
res = {"label": LABEL}
complete = sorted(cov[(cov.pipeline == "B") & (cov.n_weeks >= len(weeks) - 1)].density)
res["densities_complete_for_B"] = complete
ser = lambda dens, pipe: W[(W.density == dens) & (W.pipeline == pipe)].set_index("week").value.reindex(weeks)
if len(complete) >= 3:
    h1a = {}
    for t in TR:
        r = [float(lev[(lev.density == dd) & (lev.pipeline == "B") & (lev.truth == t)].rho.iloc[0]) for dd in complete]
        tr = float(spearmanr(range(len(complete)), r)[0])
        h1a[t] = {"densities": complete, "rho_B": r, "spearman_density_rho": tr, "monotone_pass": bool(tr == 1.0)}
        if 1.0 in complete and 0.05 in complete:
            st_ = lambda f, b, g: M.rho(pd.Series(f), pd.Series(g)) - M.rho(pd.Series(b), pd.Series(g))
            pt, lo, hi, dd = M.block_boot(st_, [ser(1.0, "B").to_numpy(), ser(0.05, "B").to_numpy(), GT[t].to_numpy()], boot=a.boot)
            h1a[t] |= {"delta_full_minus_5pct": pt, "ci": [lo, hi], "delta_pass": bool(pt > 0 and lo > 0)}
        h1a[t]["H1a_pass"] = bool(h1a[t]["monotone_pass"] or h1a[t].get("delta_pass", False))
    res["H1a"] = h1a
if 1.0 in complete:
    pv = {t: float(lev[(lev.density == 1.0) & (lev.pipeline == "B") & (lev.truth == t)].perm_p.iloc[0]) for t in TR}
    hb = M.holm(pv).set_index("test").p_holm
    res["H1b"] = {t: {"rho_B": float(lev[(lev.density == 1.0) & (lev.pipeline == "B") & (lev.truth == t)].rho.iloc[0]), "perm_p": pv[t],
                      "p_holm": float(hb[t])} for t in TR}
    for t in TR:
        res["H1b"][t]["tracks"] = bool(res["H1b"][t]["rho_B"] > 0 and res["H1b"][t]["p_holm"] < 0.05)
    st_ = lambda b, x, g: M.rho(pd.Series(b), pd.Series(g)) - M.rho(pd.Series(x), pd.Series(g))
    pt, lo, hi, _ = M.block_boot(st_, [ser(1.0, "B").to_numpy(), ser(1.0, "A").to_numpy(), GT["gt2"].to_numpy()], boot=a.boot)
    rb = float(lev[(lev.density == 1.0) & (lev.pipeline == "B") & (lev.truth == "gt2")].rho.iloc[0])
    res["H1c"] = {"delta_B_minus_A_gt2": pt, "ci": [lo, hi], "rho_B_gt2": rb, "B_wins": bool(lo > 0 and rb > 0)}

# ---- P2.5 anchor: encoder 5% vs LLM 5% ----
anc = []
for pipe, (f, src, col) in ANCHOR.items():
    if not Path(f).exists() or not len(W[(W.density == 0.05) & (W.pipeline == pipe)]):
        continue
    L = pd.read_csv(f)
    if src:
        L = L[L.sign_source == src]
    L = L.set_index("week")[col].reindex(weeks)
    E = ser(0.05, pipe)
    m = L.notna() & E.notna()
    anc.append({"pipeline": pipe, "n_weeks": int(m.sum()), "spearman": float(spearmanr(L[m], E[m])[0]), "pearson": float(np.corrcoef(L[m], E[m])[0, 1]),
                "mean_llm": float(L[m].mean()), "mean_encoder": float(E[m].mean())})
anc = pd.DataFrame(anc)
anc.to_csv(out / "anchor_5pct_encoder_vs_llm.csv", index=False)
res["anchor"] = anc.to_dict("records")
(out / "h1_tests.json").write_text(json.dumps(res, indent=1, default=float))
pd.set_option("display.width", 220)
print(cov.round(3).to_string(index=False)); print(lev.round(3).to_string(index=False)); print(anc.round(3).to_string(index=False))
print(json.dumps({k: v for k, v in res.items() if k != "anchor"}, indent=1, default=float))

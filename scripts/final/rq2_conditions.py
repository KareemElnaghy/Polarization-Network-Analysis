"""P1.7 / A6 parts 3-4 (SECONDARY, exploratory): high- vs low-GT quartile weeks (A-B agreement, P_B level, camp recovery) and the discordant-allies share
(share of negative labelled-both edges that are within-camp) against the per-week A/B rank divergence.
usage: python scripts/final/rq2_conditions.py RUN_A RUN_B [--out results/llm/final/rq2]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from src.algorithms.analysis import alignment as al
from src.algorithms.eval import metrics as M
ap = argparse.ArgumentParser(); ap.add_argument("run_a"); ap.add_argument("run_b"); ap.add_argument("--out", default="results/llm/final/rq2")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start"); GT.index = GT.index.astype(str)
weeks = sorted(GT.index)
PA = pd.read_csv(Path(a.run_a) / "weekly_A.csv").set_index("week").reindex(weeks).P_A
WB = pd.read_csv(Path(a.run_b) / "weekly_B.csv").query("sign_source=='continuous_llm'").set_index("week").reindex(weeks)
PB = WB.P_B
cr = pd.read_csv(Path(a.run_b) / "camp_recovery.csv").query("sign_source=='continuous_llm'").set_index("week").reindex(weeks)
rows = []
for t in ("gt1_excess", "gt2"):
    q1, q3 = GT[t].quantile(.25), GT[t].quantile(.75)
    for name, m in (("bottom_quartile", GT[t] <= q1), ("top_quartile", GT[t] >= q3)):
        rows.append({"truth": t, "group": name, "n_weeks": int(m.sum()), "rho_PA_PB": M.rho(PA[m], PB[m]), "mean_P_A": PA[m].mean(), "mean_P_B": PB[m].mean(),
                     "mean_camp_BA": cr.balanced_acc[m].mean(), "mean_camp_BA_minus_null": (cr.balanced_acc - cr.null_mean)[m].mean()})
q = pd.DataFrame(rows); q.to_csv(out / "quartiles.csv", index=False)
E = al.build_edge_table(a.run_a, with_vader=False)
lb = E[E.lb & E.w_continuous_llm.notna()]
neg = lb[lb.w_continuous_llm < 0]
d = pd.DataFrame({"within_camp_share_of_negative": 1 - neg.groupby("week").cross.mean(), "n_neg": neg.groupby("week").size()}).reindex(weeks)
d["rank_divergence"] = (PA.rank() - PB.rank()).abs()
d["P_A"], d["P_B"] = PA, PB
d.to_csv(out / "discordant_allies_weekly.csv")
res = {"label": "A6 parts 3-4, exploratory", "rho_discordant_vs_divergence": M.rho(d.within_camp_share_of_negative, d.rank_divergence),
       "perm_p": M.circ_p(d.within_camp_share_of_negative, d.rank_divergence), "mean_within_camp_share_of_negative": float(d.within_camp_share_of_negative.mean())}
for t in ("gt1_excess", "gt2"):
    res[f"rho_discordant_vs_{t}"] = M.rho(d.within_camp_share_of_negative, GT[t])
(out / "summary.json").write_text(json.dumps(res, indent=1, default=float))
pd.set_option("display.width", 200); print(q.round(3).to_string(index=False)); print(json.dumps(res, indent=1, default=float))

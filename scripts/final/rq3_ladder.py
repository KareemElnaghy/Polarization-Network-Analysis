"""P1.6 / A5 steps 2-4 (SECONDARY, exploratory per D-E6): all sign sources through the RQ1 statistics, pilot accuracy per rung, camp recovery per rung.
Rungs: unsigned, sentiment (VADER), binary_llm, continuous_llm (B), C1, C2. Series from the existing full-5% runs (R=50).
usage: python scripts/final/rq3_ladder.py RUN_A RUN_B RUN_C1 RUN_C2 [--out results/llm/final/rq3]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.algorithms.eval import camp_recovery as cr
from src.algorithms.eval import metrics as M
from src.algorithms.llm.data import load_shared
from src.algorithms.llm.schema import pair_hash

ap = argparse.ArgumentParser()
for k in ("run_a", "run_b", "run_c1", "run_c2"):
    ap.add_argument(k)
ap.add_argument("--out", default="results/llm/final/rq3")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start")
GT.index = GT.index.astype(str)
weeks = sorted(GT.index)
TR = ["gt1_excess", "gt2"]
WB = pd.read_csv(Path(a.run_b) / "weekly_B.csv")
S = {"A": pd.read_csv(Path(a.run_a) / "weekly_A.csv").set_index("week").reindex(weeks).P_A}
for src in ("unsigned", "sentiment", "binary_llm", "continuous_llm"):
    S[src] = WB[WB.sign_source == src].set_index("week").reindex(weeks).P_B
S["C1"] = pd.read_csv(Path(a.run_c1) / "weekly_C.csv").set_index("week").reindex(weeks).P_B
S["C2"] = pd.read_csv(Path(a.run_c2) / "weekly_C.csv").set_index("week").reindex(weeks).P_B

# ---- RQ1 statistics per rung ----
lev, rows = [], []
for k, s in S.items():
    for t in TR:
        lev.append({"series": k, "truth": t, "rho": M.rho(s, GT[t]), "perm_p": M.circ_p(s, GT[t]), "rho_diff": M.rho(s.diff(), GT[t].diff())})
        if k in ("A", "continuous_llm"):
            continue
        stat = lambda c, b, g: M.rho(pd.Series(c), pd.Series(g)) - M.rho(pd.Series(b), pd.Series(g))
        pt, lo, hi, d = M.block_boot(stat, [s.to_numpy(), S["continuous_llm"].to_numpy(), GT[t].to_numpy()], boot=a.boot)
        rows.append({"series": k, "truth": t, "delta_vs_B": pt, "ci_lo": lo, "ci_hi": hi, "p": max(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())), 1 / a.boot)})
lev = pd.DataFrame(lev)
dl = pd.DataFrame(rows).dropna(subset=["delta_vs_B"])
h = M.holm({f"{r.series}|{r.truth}": r.p for r in dl.itertuples()})
dl["p_holm"] = h.p_holm.values
lev.to_csv(out / "levels_by_rung.csv", index=False)
dl.to_csv(out / "delta_vs_B_by_rung.csv", index=False)

# ---- pilot accuracy per rung ----
mc = json.load(open(Path(a.run_b) / "manifest.json"))["config"]["max_chars"]
key = pd.read_csv("data/llm/pilot/pilot_key_PRIVATE.csv", usecols=["pilot_id", "comment_id"])
lab = pd.read_csv("data/llm/pilot/labels_kareem.csv").merge(key, on="pilot_id")
sh = load_shared(0.05).set_index("comment_id")
w2 = pd.read_parquet("data/llm/c2/c2_weights_5pct.parquet").drop_duplicates("pair_hash").set_index("pair_hash")
ext = pd.read_parquet(Path(a.run_a) / "edge_scores.parquet").set_index("comment_id")
L = lab[lab.label.isin(["agree", "disagree"])].copy()
truth = np.where(L.label == "agree", 1, -1)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
an = SentimentIntensityAnalyzer()
comp = np.array([an.polarity_scores(sh.loc[c, "body"])["compound"] for c in L.comment_id])
vader = np.select([comp > 0.05, comp < -0.05], [1, -1], 0)
P = ext.loc[L.comment_id, ["p_agree", "p_neutral", "p_disagree"]].to_numpy()
am = P.argmax(1)
binary = np.select([am == 0, am == 2], [1, -1], 0)
cont = np.sign(P[:, 0] - P[:, 2]).astype(int)
ph = [pair_hash(sh.loc[c, "parent"], sh.loc[c, "body"], mc) for c in L.comment_id]
c2 = np.sign(w2.loc[ph, "yhat_oof"].to_numpy()).astype(int)
pil = []
for name, pred in (("sentiment", vader), ("binary_llm", binary), ("continuous_llm", cont), ("C1", cont), ("C2", c2)):
    pil.append({"rung": name, "n": len(L), "accuracy_agree_disagree": float((pred == truth).mean()), "no_call_share": float((pred == 0).mean())})
pil = pd.DataFrame(pil)
pil.to_csv(out / "pilot_accuracy_by_rung.csv", index=False)

# ---- camp recovery per rung ----
labels = cr.load_labels()
camp = []
cb = pd.read_csv(Path(a.run_b) / "camp_recovery.csv")
for src in ("unsigned", "sentiment", "binary_llm", "continuous_llm"):
    c = cb[cb.sign_source == src]
    camp.append({"rung": src, "mean_balanced_acc": float(c.balanced_acc.mean()), "weeks_above_null_p95": int((c.balanced_acc > c.null_p95).sum()), "weeks": int(len(c))})
for name, run in (("C1", a.run_c1), ("C2", a.run_c2)):
    c = cr.evaluate(pd.read_parquet(Path(run) / "camp_assignments.parquet"), labels)
    camp.append({"rung": name, "mean_balanced_acc": float(c.balanced_acc.mean()), "weeks_above_null_p95": int((c.balanced_acc > c.null_p95).sum()), "weeks": int(len(c))})
camp = pd.DataFrame(camp)
camp.to_csv(out / "camp_recovery_by_rung.csv", index=False)

# ---- ladder figure: pilot accuracy vs rho_B and camp recovery ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
m = pil.merge(camp, on="rung")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for i, t in enumerate(TR):
    r = lev[lev.truth == t].set_index("series").rho
    ax[i].scatter(m.accuracy_agree_disagree, [r[x] for x in m.rung])
    for x, xx in zip(m.rung, m.accuracy_agree_disagree):
        ax[i].annotate(x, (xx, r[x]), fontsize=8)
    ax[i].set(xlabel="pilot accuracy (agree/disagree)", ylabel=f"rho vs {t}")
ax[2].scatter(m.accuracy_agree_disagree, m.mean_balanced_acc)
for x, xx, yy in zip(m.rung, m.accuracy_agree_disagree, m.mean_balanced_acc):
    ax[2].annotate(x, (xx, yy), fontsize=8)
ax[2].set(xlabel="pilot accuracy (agree/disagree)", ylabel="mean camp-recovery balanced acc")
plt.tight_layout()
plt.savefig(out / "ladder.png", dpi=150)

pd.set_option("display.width", 200)
print(lev.round(3).to_string(index=False)); print(dl.round(3).to_string(index=False)); print(pil.round(3).to_string(index=False)); print(camp.round(3).to_string(index=False))

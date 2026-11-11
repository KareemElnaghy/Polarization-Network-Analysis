"""Slide figures for the RQ1 5% result (light surface, thin marks, colour = pipeline, ground truths in neutral ink).
usage: python scripts/llm/make_slide_figures.py RUN_A RUN_B OUT_DIR"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RA, RB, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#f0efec"
CA, CB = "#eb6834", "#2a78d6"   # Pipeline A orange, Pipeline B blue (fixed for every figure)
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})
z = lambda s: (s - s.mean()) / s.std()
GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start")
GT.index = pd.to_datetime(GT.index)
A = pd.read_csv(RA / "weekly_A.csv").set_index("week")["P_A"]
B = pd.read_csv(RB / "weekly_B.csv")
B = B[B.sign_source == "continuous_llm"].set_index("week")["P_B"]
A.index, B.index = pd.to_datetime(A.index), pd.to_datetime(B.index)

# 1. weekly series, z-scored, three small multiples sharing the x axis
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
ax[0].plot(GT.index, z(GT.gt1_excess), color=INK, lw=2, label="GT#1 (camps in different threads)")
ax[0].plot(GT.index, z(GT.gt2), color=INK2, lw=2, ls=(0, (4, 3)), label="GT#2 (language predicts camp)")
ax[0].set_title("Ground truths (LLM-free), z-scored", loc="left", fontsize=12)
ax[0].legend(frameon=False, loc="upper right", ncol=1, fontsize=9)
for a_, s, c, name in ((ax[1], A, CA, "Pipeline A (mean hostility score)"), (ax[2], B, CB, "Pipeline B (signed-network P_B)")):
    a_.plot(s.index, z(s), color=c, lw=2)
    a_.set_title(name + ", z-scored", loc="left", fontsize=12)
    a_.axhline(0, color=GRID, lw=1)
for a_ in ax:
    a_.axvline(pd.Timestamp("2016-06-27"), color=INK2, lw=1, ls=":")
ax[0].text(pd.Timestamp("2016-06-29"), ax[0].get_ylim()[1] * 0.85, "2016-06-27", color=INK2, fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig1_weekly_series.png", dpi=200)
plt.close(fig)

# 2. rho per pipeline per truth (primary, full-week GT) with permutation p in the labels
c = pd.read_csv(sorted(glob.glob("results/llm/rq1_5pct/5pct_final/rq1_5pct_final_correlations.csv"))[0])
c = c[(c.truth_set == "PRIMARY-full") & c.series.isin(["P_A", "P_B"])]
fig, ax = plt.subplots(figsize=(8, 4.2))
w = 0.34
for i, (truth, lab) in enumerate((("gt1_excess", "GT#1"), ("gt2", "GT#2"))):
    for j, (se, col) in enumerate((("P_A", CA), ("P_B", CB))):
        r = c[(c.truth == truth) & (c.series == se)].iloc[0]
        x = i + (j - 0.5) * w
        ax.bar(x, r.rho, width=w - 0.04, color=col)
        ax.text(x, r.rho + (0.03 if r.rho >= 0 else -0.03), f"{r.rho:+.2f}\np={r.perm_p:.2f}", ha="center", va="bottom" if r.rho >= 0 else "top", fontsize=9, color=INK2)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks([0, 1]); ax.set_xticklabels(["vs GT#1", "vs GT#2"])
ax.set_ylim(-0.65, 0.35); ax.set_ylabel("Spearman rho (53 weeks)")
ax.set_title("Neither pipeline tracks either truth", loc="left", fontsize=12)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=CA), plt.Rectangle((0, 0), 1, 1, color=CB)], labels=["Pipeline A", "Pipeline B"], frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig2_rho_by_pipeline.png", dpi=200); plt.close(fig)

# 3. delta rho (B - A) with CI across checkpoints, GT#2 (primary)
rows = []
for d in sorted(glob.glob("results/llm/rq1_5pct/5pct_*")):
    t = d.split("5pct_")[-1]
    dl = pd.read_csv(f"{d}/rq1_5pct_{t}_delta.csv")
    s = json.load(open(f"{d}/rq1_5pct_{t}_summary.json"))
    r = dl[dl.truth == "gt2"].iloc[0]
    rows.append((s["coverage"]["edges_A"], r.delta_rho_B_minus_A, r.ci_lo, r.ci_hi))
rows.sort()
fig, ax = plt.subplots(figsize=(8, 4.2))
xs = np.arange(len(rows))
for x, (n, d, lo, hi) in zip(xs, rows):
    ax.plot([x, x], [lo, hi], color=INK2, lw=2)
    ax.plot(x, d, "o", color=CB, ms=8, mec=SURF, mew=2)
ax.axhline(0, color=INK2, lw=1)
ax.set_xticks(xs); ax.set_xticklabels([f"{n/1000:.0f}k" for n, *_ in rows])
ax.set_xlabel("edges processed (thread-prefix sample of the 5% tier)"); ax.set_ylabel("delta rho = rho_B - rho_A (GT#2)")
ax.set_title("The B-over-A gap on GT#2 is positive but borderline", loc="left", fontsize=12)
fig.tight_layout(); fig.savefig(OUT / "fig3_delta_over_checkpoints.png", dpi=200); plt.close(fig)

# 4. camp recovery by sign source vs label-shuffle null (final)
cr = pd.read_csv(RB / "camp_recovery.csv")
order = ["continuous_llm", "binary_llm", "sentiment", "unsigned"]
names = {"continuous_llm": "LLM signs\n(continuous)", "binary_llm": "LLM signs\n(binary)", "sentiment": "VADER\n(no LLM)", "unsigned": "Unsigned\n(control)"}
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, k in enumerate(order):
    d = cr[cr.sign_source == k]
    acc, null, above = d.balanced_acc.mean(), d.null_mean.mean(), int((d.balanced_acc > d.null_p95).sum())
    ax.bar(i, acc, width=0.5, color=CB if "llm" in k else INK2)
    ax.plot([i - 0.3, i + 0.3], [null] * 2, color=INK, lw=2)
    ax.text(i, acc + 0.004, f"{acc:.3f}\n{above}/53 wks > p95", ha="center", va="bottom", fontsize=9, color=INK2)
ax.set_xticks(range(4))
ax.set_xticklabels([names[k] for k in order])
ax.set_ylim(0.48, 0.58)
ax.set_ylabel("balanced accuracy vs real Dem/Rep labels")
ax.set_title("Inferred camps barely beat the shuffle null (black line)", loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig4_camp_recovery.png", dpi=200)
plt.close(fig)
print("figures written to", OUT)

"""Phase 4 synthetic validation; criteria are fixed in docs/synthetic_spec.md. usage: python scripts/llm/run_synthetic.py [--quick]"""
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.algorithms.eval.camp_recovery import balanced_accuracy
from src.algorithms.llm import provenance
from src.algorithms.pipeline_b import structural as st
from src.algorithms.pipeline_b.frustration import exact_frustration
from src.algorithms.synthetic.generator import generate


def main():

    QUICK = "--quick" in sys.argv
    REPS = 2 if QUICK else 5
    POOL = np.load("results/llm/synthetic/degree_pool.npy")
    OUT = Path("results/llm/synthetic")
    run = provenance.new_run("synthetic" + ("_quick" if QUICK else ""), {"seed": 0, "quick": QUICK, "reps": REPS}, ["docs/synthetic_spec.md"])
    rows = []
    EX = ProcessPoolExecutor(6)


    def one(check, seed, n=4000, deg=3.2, lam=0.5, R=20, Rr=0, **kw):
        g = st.giant(generate(n, deg, lam, POOL, seed=seed, **kw))
        camp = g.pop("camp", None) if "camp" in g else None
        res = st.normalised_frustration(g, R, 10, seed, Rr, EX)
        side = res.pop("side")
        y = None
        full = generate(n, deg, lam, POOL, seed=seed, **kw)  # same seed -> same graph; used only for planted camps
        ok = full["camp"] >= 0
        gi_nodes = g["nodes"]
        lab = full["camp"][gi_nodes]
        m = lab >= 0
        ba = balanced_accuracy(side[m], lab[m]) if m.sum() > 10 and len(set(lab[m])) == 2 else np.nan
        rows.append({"check": check, "seed": seed, "n": n, "mean_degree": deg, "lam": lam, **kw, "giant_nodes": len(g["nodes"]),
                     "giant_pairs": len(g["w"]), **res, "F_obs_per_weight": res["F_obs"] / res["A"], "camp_balanced_acc": ba,
                     "z_vs_shuffle": (res["F_null_mean"] - res["F_obs"]) / res["F_null_sd"] if res["F_null_sd"] > 0 else np.nan})


    t0 = time.time()
    lams = np.round(np.arange(0, 1.01, 0.1), 2)
    for lam in lams:
        for s in range(REPS):
            one("1_lam", s, lam=lam)
    print("check1", round(time.time() - t0), flush=True)
    for n in ([500, 2000, 10000] if QUICK else [500, 2000, 10000, 40000]):
        for s in range(REPS if n < 40000 else 2):
            one("2_size", s, n=n, lam=0.5)
    for deg in [2.5, 5.0, 7.0]:
        for s in range(REPS):
            one("2_density", s, deg=deg, lam=0.5)
    print("check2", round(time.time() - t0), flush=True)
    for lam in (0.0, 0.6):
        for s in range(REPS):
            one("3_null", s, lam=lam, Rr=5)
    print("check3", round(time.time() - t0), flush=True)
    for d in (0, .1, .2, .3, .5):
        for s in range(REPS):
            one("5_discordant", s, lam=0.8, discordant_share=d)
    for rs in (0.5, 0.3, 0.14, 0.05):
        for s in range(REPS):
            one("6_imbalance", s, lam=0.8, rep_share=rs)
    for lam in (0.3, 0.6, 0.9):
        for s in range(REPS):
            one("7_recovery", s, lam=lam)
    print("check5-7", round(time.time() - t0), flush=True)

    gaps = []   # check 4: heuristic vs exact on calibrated small graphs
    for s in range(10 if not QUICK else 4):
        g = st.giant(generate(int(80 + 20 * (s % 3)), 3.2, 0.6, POOL, seed=100 + s))
        from src.algorithms.pipeline_b.frustration import csr_from_edges, frustration
        Fh, _ = frustration(csr_from_edges(len(g["nodes"]), g["u"], g["v"], g["w"]), 10, s)
        ex = exact_frustration(len(g["nodes"]), g["u"], g["v"], g["w"], time_limit=60, detail=True)
        best = min(Fh, ex["F"])
        gaps.append({"nodes": len(g["nodes"]), "pairs": len(g["w"]), "F_heur": Fh, "F_exact": ex["F"], "proved": ex["optimal"],
                     "rel_gap": (Fh - best) / best if best > 0 else 0.0})
    gap = pd.DataFrame(gaps)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic_replicates.csv", index=False)
    gap.to_csv(OUT / "heuristic_vs_exact_calibrated.csv", index=False)

    mean = lambda c, col, **q: df[(df.check == c) & np.all([df[k] == v for k, v in q.items()], axis=0)][col].mean()
    checks = []
    c1 = df[df.check == "1_lam"].groupby("lam").P_B.mean()
    rho = spearmanr(c1.index, c1.values)[0]
    checks.append(("1 monotone in lam", f"Spearman {rho:.3f}; max adjacent drop {max(0, -np.diff(c1.values).min()):.3f}", rho >= 0.95 and -np.diff(c1.values).min() <= 0.03))
    sz = df[(df.check == "2_size") | ((df.check == "1_lam") & (df.lam == 0.5) & (df.n == 4000))].groupby("n").agg(P_B=("P_B", "mean"), F=("F_obs", "mean"))
    cv = sz.P_B.std() / sz.P_B.mean()
    checks.append(("2 size flatness (lam .5)", f"P_B by n {sz.P_B.round(3).to_dict()}; CV {cv:.3f}; raw F ratio max/min {sz.F.max() / sz.F.min():.1f}", cv < 0.15 and sz.F.max() / sz.F.min() > 10))
    dn = df[(df.check == "2_density") | ((df.check == "1_lam") & (df.lam == 0.5))].groupby("mean_degree").P_B.mean()
    checks.append(("2 density (reported only)", f"P_B by mean degree {dn.round(3).to_dict()}", True))
    n0, n6 = df[(df.check == "3_null") & (df.lam == 0)], df[(df.check == "3_null") & (df.lam == 0.6)]
    z_rw6 = ((n6.F_rewire_mean - n6.F_obs) / n6.F_null_sd).mean()
    checks.append(("3 separation from nulls", f"lam0 P_B {n0.P_B.mean():.3f}; lam.6 P_B {n6.P_B.mean():.3f}, z vs shuffle {n6.z_vs_shuffle.mean():.1f}, P_B_rewire {n6.P_B_rewire.mean():.3f}",
                   abs(n0.P_B.mean()) < 0.05 and n6.P_B.mean() > 0.2 and n6.z_vs_shuffle.mean() > 5))
    checks.append(("4 heuristic vs exact (calibrated)", f"proved {int(gap.proved.sum())}/{len(gap)}; mean gap {gap.rel_gap.mean():.4f}; max {gap.rel_gap.max():.4f}", gap.rel_gap.mean() < 0.01 and gap.rel_gap.max() < 0.05))
    dd = df[df.check == "5_discordant"].groupby("discordant_share").P_B.mean()
    r5 = spearmanr(dd.index, dd.values)[0]
    checks.append(("5 discordant allies", f"P_B by share {dd.round(3).to_dict()}; Spearman {r5:.3f}", r5 <= -0.9))
    im = df[df.check == "6_imbalance"].groupby("rep_share").agg(P_B=("P_B", "mean"), BA=("camp_balanced_acc", "mean"))
    checks.append(("6 imbalance", f"{im.round(3).to_dict('index')}", im.loc[0.14, "P_B"] > 0.5 * im.loc[0.5, "P_B"]))
    rc = df[df.check == "7_recovery"].groupby("lam").camp_balanced_acc.mean()
    checks.append(("7 planted-camp recovery", f"BA by lam {rc.round(3).to_dict()}", rc.loc[0.9] > 0.8))
    md = "# Synthetic validation checks (criteria: docs/synthetic_spec.md)\n\n| check | observed | pass |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {'PASS' if c else 'FAIL'} |" for a, b, c in checks)
    (OUT / "checks.md").write_text(md + f"\n\nreplicates per cell: {REPS}; runtime {time.time() - t0:.0f} s\n")
    print(md)

    fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
    g1 = df[df.check == "1_lam"].groupby("lam").P_B.agg(["mean", "std"])
    ax[0, 0].errorbar(g1.index, g1["mean"], g1["std"], marker="o", ms=3); ax[0, 0].set_title("1  P_B vs polarization lam"); ax[0, 0].set_xlabel("lam")
    ax[0, 1].plot(sz.index, sz.P_B, marker="o", label="P_B"); ax[0, 1].set_xscale("log"); ax[0, 1].set_ylim(0, 1); ax[0, 1].set_xlabel("nodes")
    ax2 = ax[0, 1].twinx(); ax2.plot(sz.index, sz.F, color="gray", marker="s", label="raw F"); ax2.set_yscale("log"); ax[0, 1].set_title("2  normalised (blue) vs raw (grey), lam .5")
    ax[0, 2].plot(dn.index, dn.values, marker="o"); ax[0, 2].set_title("2  P_B vs mean degree"); ax[0, 2].set_xlabel("mean degree")
    ax[1, 0].plot(dd.index, dd.values, marker="o"); ax[1, 0].set_title("5  discordant allies"); ax[1, 0].set_xlabel("discordant_share")
    ax[1, 1].plot(im.index, im.P_B, marker="o", label="P_B"); ax[1, 1].plot(im.index, im.BA, marker="s", label="camp BA"); ax[1, 1].legend(); ax[1, 1].set_title("6  camp balance"); ax[1, 1].set_xlabel("rep_share")
    ax[1, 2].plot(rc.index, rc.values, marker="o"); ax[1, 2].set_title("7  planted-camp recovery"); ax[1, 2].set_xlabel("lam"); ax[1, 2].set_ylim(0.4, 1)
    fig.tight_layout(); fig.savefig(OUT / "synthetic_checks.png", dpi=140); fig.savefig(OUT / "synthetic_checks.pdf")
    pd.concat([g1.assign(panel="1_lam"), dd.to_frame("P_B").assign(panel="5_discordant"), im.assign(panel="6_imbalance")]).to_csv(OUT / "synthetic_checks_plot_data.csv")
    for f in ("synthetic_replicates.csv", "heuristic_vs_exact_calibrated.csv", "checks.md", "synthetic_checks.png", "synthetic_checks.pdf", "synthetic_checks_plot_data.csv"):
        (run / f).write_bytes((OUT / f).read_bytes())
    provenance.finish_run(run, "done", {c[0]: bool(c[2]) for c in checks}, "Phase 4 synthetic validation")


if __name__ == "__main__":
    main()

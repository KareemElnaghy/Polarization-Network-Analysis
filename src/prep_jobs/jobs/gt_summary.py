"""Job S, gt_summary: plots, null checks, agreement tables and the Kareem-ready weekly file.

Reads the outputs of T1 to T5 (whatever exists) and writes ``results/ground_truth/summary.md``, plots, and
``ground_truth_weekly.csv`` (main / shared, full week plus one set of columns per sample rate).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.ground_truth.dissimilarity import dissimilarity_index  # noqa: E402
from src.gt_runs.cells import read_thread_user  # noqa: E402
from src.gt_runs.config import gt_cfg, gt_path, labels_cfg, venues  # noqa: E402
from src.gt_runs.labels import heavy_set, load_side_map  # noqa: E402
from src.prep_jobs.jobs.clean_edges_and_sampling import weeks_of  # noqa: E402
from src.prep_jobs.jobs.label_versions import md_table  # noqa: E402

NAME = "gt_summary"
DEADLINE_CUTS: list[str] = []

COLUMN_GUIDE = """## Column guide

`ground_truth_weekly.csv` (one row per week, main label set, shared venue = r/politics + r/PoliticalDiscussion):

| column | meaning |
|---|---|
| week_start | Monday 00:00 UTC |
| gt1_excess | GT#1: thread dissimilarity index minus its shuffled-label null mean (0 = no segregation, 1 = camps never share a thread) |
| gt2_pi | GT#2: leave-out partisanship of language (0.5 = camps use phrases alike, 1 = disjoint) |
| gt2_null_mean | mean of GT#2 under shuffled labels (should be about 0.5) |
| gt2_boot_lo, gt2_boot_hi | 95% bootstrap band for gt2_pi |
| n_dem, n_rep | labelled users per side entering GT#1 (active in the venue, heavy posters excluded) |
| gt2_n_dem, gt2_n_rep | labelled users per side entering GT#2 (also at least 20 in-vocabulary phrases) |
| low_n | True when either GT is NaN because a side has fewer than `min_users_per_side` users (the row is kept) |
| gt1_excess_<rate>, gt2_pi_<rate>, low_n_<rate> | the same inside the thread sample at that rate (only the sampled threads) |
"""


def spearman(x: pd.Series, y: pd.Series) -> float:
    ok = x.notna() & y.notna()
    return float(spearmanr(x[ok], y[ok])[0]) if ok.sum() >= 3 else float("nan")


def phases(cfg):
    mod = "src.prep_jobs.jobs.gt_summary"
    return [{"name": "summary", "fn": f"{mod}:summary", "units": [{"id": "summary", "weight": 100}], "workers": 1}]


def event_lines(ax, events, text=False):
    for name, day in events.items():
        ax.axvline(pd.Timestamp(day), color="grey", ls=":", lw=0.8)
    if text:
        top = ax.get_ylim()[1]
        for name, day in events.items():
            ax.text(pd.Timestamp(day), top, name, rotation=90, va="top", ha="right", fontsize=6.5, color="dimgrey")


def shuffled_gt1_check(cfg, g, lab, weeks) -> pd.Series:
    """GT#1 `excess` after shuffling the labels across users once per week: should average about 0."""
    root = gt_path(g, "labels")
    side = load_side_map(root, "shared", "main")
    authors = np.array(list(side))
    values = np.array([side[a] for a in authors])
    rng = np.random.default_rng(g["seed"] + 1)
    shuf = dict(zip(authors, rng.permutation(values)))
    out = {}
    for w in weeks:
        tu = read_thread_user(cfg, w)
        tu = tu[tu["subreddit"].isin(venues(g)["shared"])]
        heavy = heavy_set(root, w) if lab["exclude_heavy_posters"] else set()
        tu = tu[~tu["author"].isin(heavy)]
        lbl = tu["author"].map(shuf)
        rows = pd.DataFrame({"thread": tu["link_id"], "user": tu["author"], "label": lbl}).dropna()
        out[w] = dissimilarity_index(rows, n_perm=50, seed=g["seed"], min_users=lab["min_users_per_side"])["excess"]
    return pd.Series(out)


def summary(ctx, unit):
    cfg, g, lab = ctx.cfg, gt_cfg(), labels_cfg()
    res = gt_path(g, "results")
    g1 = pd.read_parquet(res / "gt1_weekly.parquet")
    g2 = pd.read_parquet(res / "gt2_weekly_full.parquet")
    g2s = pd.read_parquet(res / "gt2_weekly_samples.parquet") if (res / "gt2_weekly_samples.parquet").exists() else None
    events = g["events"]
    weeks = weeks_of(cfg)
    main1 = g1[(g1.venue == "shared") & (g1.label_set == "main") & (g1["sample"] == "full")].set_index("week_start").reindex(weeks)
    main2 = g2[(g2.venue == "shared") & (g2.label_set == "main")].set_index("week_start").reindex(weeks)
    x = pd.to_datetime(pd.Series(weeks))

    # ---- main figure
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    ax = axes[0]
    ax.plot(x, main1["excess"], color="#7d3c98", label="GT#1 excess")
    ax.fill_between(x, -2 * main1["null_sd"].fillna(0), 2 * main1["null_sd"].fillna(0), color="grey", alpha=0.25, label="null band (+-2 sd)")
    ax.set_ylabel("dissimilarity excess")
    ax = axes[1]
    ax.plot(x, main2["pi"], color="#117a65", label="GT#2 pi")
    ax.fill_between(x, main2["boot_lo"], main2["boot_hi"], color="#117a65", alpha=0.25, label="bootstrap 95%")
    ax.fill_between(x, main2["null_mean"] - 2 * main2["null_sd"], main2["null_mean"] + 2 * main2["null_sd"], color="grey", alpha=0.25,
                    label="shuffled-label null (+-2 sd)")
    ax.axhline(0.5, color="grey", ls="--", lw=0.7)
    ax.set_ylabel("leave-out pi")
    low = main2["low_n"].fillna(True).astype(bool).to_numpy()
    for ax, ser in ((axes[0], main1["excess"]), (axes[1], main2["pi"])):
        if low.any():
            ax.scatter(x[low], np.zeros(low.sum()) + np.nanmin(ser) if ser.notna().any() else [], marker="x", color="red", label="low_n week")
    axes[2].plot(x, main1["n_dem"], color="#2471a3", label="dem (GT#1)")
    axes[2].plot(x, main1["n_rep"], color="#c0392b", label="rep (GT#1)")
    axes[2].plot(x, main2["n_dem"], color="#2471a3", ls="--", label="dem (GT#2)")
    axes[2].plot(x, main2["n_rep"], color="#c0392b", ls="--", label="rep (GT#2)")
    axes[2].axhline(lab["min_users_per_side"], color="grey", ls=":", lw=0.8)
    axes[2].set_ylabel("labelled users per side")
    for i, ax in enumerate(axes):
        event_lines(ax, events, text=(i == 0))
        ax.legend(fontsize=7, loc="upper left", ncol=3)
    axes[0].set_title("Ground truth, main labels, r/politics + r/PoliticalDiscussion, full week")
    fig.tight_layout()
    fig.savefig(res / "gt_main_shared_full.png", dpi=120)
    plt.close(fig)

    # ---- null checks
    tol = g["null_check_tolerance"]
    null2 = float(main2["null_mean"].mean())
    shuf1 = shuffled_gt1_check(cfg, g, lab, [w for w in weeks if not np.isnan(main1.loc[w, "observed"])])
    null1 = float(shuf1.mean())
    checks = [("GT#2 shuffled-label mean (main / shared / full), mean over weeks", null2, abs(null2 - 0.5) <= tol, "0.5"),
              ("GT#1 excess under one label shuffle per week (main / shared / full), mean over weeks", null1, abs(null1) <= tol, "0")]
    # ---- agreement and stability
    rows = []
    rows.append(("GT#1 excess vs GT#2 pi, main / shared / full", spearman(main1["excess"], main2["pi"])))
    for ls in sorted(set(g1.label_set) - {"main"}):
        a = g1[(g1.venue == "shared") & (g1.label_set == ls) & (g1["sample"] == "full")].set_index("week_start")["excess"]
        rows.append((f"GT#1: main vs {ls} (shared, full)", spearman(main1["excess"], a.reindex(weeks))))
    for ls in sorted(set(g2.label_set) - {"main"}):
        a = g2[(g2.venue == "shared") & (g2.label_set == ls)].set_index("week_start")["pi"]
        rows.append((f"GT#2: main vs {ls} (shared, full)", spearman(main2["pi"], a.reindex(weeks))))
    for r in sorted(set(g1["sample"]) - {"full"}, key=float):
        a = g1[(g1.venue == "shared") & (g1.label_set == "main") & (g1["sample"] == r)].set_index("week_start")["excess"]
        rows.append((f"GT#1: full vs sample {r}", spearman(main1["excess"], a.reindex(weeks))))
    if g2s is not None:
        for r in sorted(set(g2s["sample"]), key=float):
            a = g2s[(g2s.venue == "shared") & (g2s.label_set == "main") & (g2s["sample"] == r)].set_index("week_start")["pi"]
            rows.append((f"GT#2: full vs sample {r}", spearman(main2["pi"], a.reindex(weeks))))
    stab = pd.DataFrame(rows, columns=["comparison (Spearman across weeks)", "rho"])

    # ---- venues available
    ven = g1[(g1.label_set == "main") & (g1["sample"] == "full")].groupby("venue").agg(
        weeks_with_gt1=("low_n", lambda s: int((~s).sum())), median_n_dem=("n_dem", "median"), median_n_rep=("n_rep", "median"))
    ven2 = g2[g2.label_set == "main"].groupby("venue")["low_n"].apply(lambda s: int((~s).sum())).rename("weeks_with_gt2")
    ven = ven.join(ven2).reset_index()

    # ---- Kareem file
    k = pd.DataFrame({"week_start": weeks, "gt1_excess": main1["excess"].to_numpy(), "gt2_pi": main2["pi"].to_numpy(),
                      "gt2_null_mean": main2["null_mean"].to_numpy(), "gt2_boot_lo": main2["boot_lo"].to_numpy(),
                      "gt2_boot_hi": main2["boot_hi"].to_numpy(), "n_dem": main1["n_dem"].to_numpy(), "n_rep": main1["n_rep"].to_numpy(),
                      "gt2_n_dem": main2["n_dem"].to_numpy(), "gt2_n_rep": main2["n_rep"].to_numpy()})
    k["low_n"] = (main1["low_n"].fillna(True) | main2["low_n"].fillna(True)).to_numpy()
    for r in sorted(set(g1["sample"]) - {"full"}, key=float):
        a = g1[(g1.venue == "shared") & (g1.label_set == "main") & (g1["sample"] == r)].set_index("week_start").reindex(weeks)
        k[f"gt1_excess_{r}"] = a["excess"].to_numpy()
        low_r = a["low_n"].fillna(True)
        if g2s is not None:
            b = g2s[(g2s.venue == "shared") & (g2s.label_set == "main") & (g2s["sample"] == r)].set_index("week_start").reindex(weeks)
            k[f"gt2_pi_{r}"] = b["pi"].to_numpy()
            low_r = low_r | b["low_n"].fillna(True)
        k[f"low_n_{r}"] = low_r.to_numpy()
    k.to_csv(res / "ground_truth_weekly.csv", index=False)

    # ---- summary.md
    md = ["# Ground truth summary (generated by the gt_summary job)", "", COLUMN_GUIDE, "",
          "## Null checks", ""]
    for name, val, ok, target in checks:
        md.append(f"- {'OK' if ok else '**FAIL: CHECK THIS**'}: {name} = {val:.4f} (expected about {target}, tolerance {tol})")
    md += ["", "## Series (main / shared / full)", "",
           f"- GT#1 excess: mean {main1['excess'].mean():.3f}, range {main1['excess'].min():.3f} to {main1['excess'].max():.3f}",
           f"- GT#2 pi: mean {main2['pi'].mean():.3f}, range {main2['pi'].min():.3f} to {main2['pi'].max():.3f}; weeks with the bootstrap "
           f"band entirely below 0.5: {int((main2['boot_hi'] < 0.5).sum())}, entirely above: {int((main2['boot_lo'] > 0.5).sum())}",
           f"- weeks flagged low_n: {int(k['low_n'].sum())} of {len(k)}",
           "- Plot: `gt_main_shared_full.png` (events: " + ", ".join(events) + ").", "",
           "## Venues and label sets", "", md_table(ven), "",
           "## Agreement and stability", "", md_table(stab, "{:.3f}"), ""]
    var = res / "variants_summary.csv"
    if var.exists():
        md += ["## Variants (T5)", "", md_table(pd.read_csv(var), "{:.3f}"), ""]
    wc = res / "within_camp_weekly.parquet"
    if wc.exists():
        w = pd.read_parquet(wc)
        md += ["## Within camp (Sanders vs Clinton, T4)", "",
               f"GT#1 excess mean {w['gt1_excess'].mean():.3f}; GT#2 pi mean {w['gt2_pi'].mean():.3f}. Plot: `within_camp.png`.", ""]
    (res / "summary.md").write_text("\n".join(md), encoding="utf-8")
    for name, val, ok, _ in checks:
        if not ok:
            ctx.log(f"NULL CHECK FAILED: {name} = {val}")
    return {"rows": len(k), "bytes": 0}

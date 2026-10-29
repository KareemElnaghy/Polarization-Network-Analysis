"""Job S, gt_summary: plots, null checks, agreement tables and the Kareem-ready weekly file.

Reads the outputs of T1 to T5 (whatever exists) and of ``gt2_transfer`` and writes ``results/ground_truth/summary.md``,
plots, ``ground_truth_weekly.csv`` (main / shared, full week plus one set of columns per sample rate) and
``user_scores_gt2.parquet``. Ground truth #2 is the transfer design: phrase estimator with leanings frozen from the supporter
subs, training weighting cap 100 (``gt2``); the logistic AUC with equal weighting is the secondary set (``gt2_alt_auc``);
the original volume-weighted leave-out value is kept as ``gt2_pi_volume_weighted``.
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
| gt2 | **GT#2 (chosen): phrase estimator.** Phrase leanings are learned in the party-tagged supporter subs (training users outside the user's own pile, per-user-week cap of 100 phrases) and frozen; a user's estimate is the average leaning of the phrases they used that week in the neutral subs; `gt2 = 0.5 * mean over Republicans + 0.5 * mean of (1 - estimate) over Democrats`. 0.5 = no side information, higher = the camps talk differently. Levels are small (about 0.51): use the null and the ranking, not the size |
| gt2_null_mean | the same statistic with labels shuffled among the scored users (should be 0.5) |
| gt2_excess | gt2 minus gt2_null_mean |
| gt2_boot_lo, gt2_boot_hi | 95% bootstrap band for gt2 (resampling users) |
| gt2_n_dem, gt2_n_rep | users scored per side that week (at least 20 neutral-sub phrases, main label) |
| gt2_alt_auc | secondary GT#2: logistic regression on phrases (equal-weight users in training); weekly AUC of the predicted probability of Republican against the main labels (0.5 = no information) |
| gt2_alt_auc_null_mean, gt2_alt_auc_excess, gt2_alt_auc_boot_lo, gt2_alt_auc_boot_hi | its shuffled-label null, AUC minus null, and 95% bootstrap band |
| gt2_pi_volume_weighted (+ _null_mean, _boot_lo, _boot_hi, _n_dem, _n_rep) | the original GT#2, kept: leave-out estimator in the neutral subs with users weighted by volume. It sits below its null in most weeks (see docs/decisions.md); not recommended |
| n_dem, n_rep | labelled users per side entering GT#1 (active in the venue, heavy posters excluded) |
| low_n | True when GT#1 or the chosen GT#2 is NaN that week because a side has too few users (the row is kept) |
| gt1_excess_<rate>, low_n_<rate> | GT#1 inside the thread sample at that rate (only the sampled threads); low_n_<rate> is True if GT#1 or gt2 is missing there |
| gt2_<rate>, gt2_null_mean_<rate>, gt2_excess_<rate>, gt2_boot_lo_<rate>, gt2_boot_hi_<rate>, gt2_n_dem_<rate>, gt2_n_rep_<rate> | chosen GT#2 computed only on comments in the sampled threads (same frozen model) |
| gt2_alt_auc_<rate>, gt2_alt_auc_null_mean_<rate>, gt2_alt_auc_excess_<rate> | secondary AUC inside the sample |
| gt2_pi_volume_weighted_<rate> | the original volume-weighted value inside the sample |

**The 1% sample columns are weak.** At 1% most weeks have too few scored users (11 of 53 weeks are NaN for gt2_0.01, and the rank agreement with the full week is only about 0.68; the AUC alternative about 0.54). Use the 5% and 20% samples for anything that needs to be stable.

`user_scores_gt2.parquet`: one row per user per week (full week, neutral subs): `gt2_score` (the user's average phrase leaning, chosen instrument), `gt2_alt_auc_score` (logistic probability of Republican), `n_phrases`, `side_main`, `side_gold`, `pile`.
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
    old2 = g2[(g2.venue == "shared") & (g2.label_set == "main")].set_index("week_start").reindex(weeks)   # volume-weighted leave-out
    tr = pd.read_parquet(res.parent / "gt2_transfer" / "weekly_results.parquet")
    tr = tr[tr.labels == "main"]

    def tseries(inst, wt, src="full"):
        t = tr[(tr.instrument == inst) & (tr.weighting == wt) & (tr.source == src)].set_index("week_start").reindex(weeks)
        t["pi"] = t["value"]
        t["low_n"] = t["value"].isna()
        t["excess"] = t["value"] - t["null_mean"]
        return t

    main2 = tseries("phrase", "cap100")          # chosen GT#2
    alt = tseries("logit", "equal")              # secondary: AUC
    x = pd.to_datetime(pd.Series(weeks))

    # ---- main figure
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    ax = axes[0]
    ax.plot(x, main1["excess"], color="#7d3c98", label="GT#1 excess")
    ax.fill_between(x, -2 * main1["null_sd"].fillna(0), 2 * main1["null_sd"].fillna(0), color="grey", alpha=0.25, label="null band (+-2 sd)")
    ax.set_ylabel("dissimilarity excess")
    ax = axes[1]
    ax.plot(x, main2["pi"], color="#117a65", label="GT#2 (phrase estimator, transfer)")
    ax.fill_between(x, main2["boot_lo"], main2["boot_hi"], color="#117a65", alpha=0.25, label="bootstrap 95%")
    ax.fill_between(x, main2["null_mean"] - 2 * main2["null_sd"], main2["null_mean"] + 2 * main2["null_sd"], color="grey", alpha=0.25,
                    label="shuffled-label null (+-2 sd)")
    ax.axhline(0.5, color="grey", ls="--", lw=0.7)
    ax.set_ylabel("GT#2 pi")
    low = main2["low_n"].fillna(True).astype(bool).to_numpy()
    for ax, ser in ((axes[0], main1["excess"]), (axes[1], main2["pi"])):
        if low.any():
            ax.scatter(x[low], np.zeros(low.sum()) + np.nanmin(ser) if ser.notna().any() else [], marker="x", color="red", label="low_n week")
    axes[2].plot(x, alt["pi"], color="#b9770e", label="secondary: logistic AUC")
    axes[2].fill_between(x, alt["boot_lo"], alt["boot_hi"], color="#b9770e", alpha=0.25, label="bootstrap 95%")
    axes[2].fill_between(x, alt["null_mean"] - 2 * alt["null_sd"], alt["null_mean"] + 2 * alt["null_sd"], color="grey", alpha=0.25,
                         label="shuffled-label null (+-2 sd)")
    axes[2].axhline(0.5, color="grey", ls="--", lw=0.7)
    axes[2].set_ylabel("AUC")
    axes[3].plot(x, main1["n_dem"], color="#2471a3", label="dem (GT#1)")
    axes[3].plot(x, main1["n_rep"], color="#c0392b", label="rep (GT#1)")
    axes[3].plot(x, main2["n_dem"], color="#2471a3", ls="--", label="dem (GT#2)")
    axes[3].plot(x, main2["n_rep"], color="#c0392b", ls="--", label="rep (GT#2)")
    axes[3].axhline(lab["min_users_per_side"], color="grey", ls=":", lw=0.8)
    axes[3].set_ylabel("labelled users per side")
    for i, ax in enumerate(axes):
        event_lines(ax, events, text=(i == 0))
        ax.legend(fontsize=7, loc="upper left", ncol=3)
    axes[0].set_title("Ground truth, main labels, r/politics + r/PoliticalDiscussion, full week (GT#2 = transfer design)")
    fig.tight_layout()
    fig.savefig(res / "gt_main_shared_full.png", dpi=120)
    plt.close(fig)

    # ---- null checks
    tol = g["null_check_tolerance"]
    null2 = float(main2["null_mean"].mean())
    null_alt = float(alt["null_mean"].mean())
    shuf1 = shuffled_gt1_check(cfg, g, lab, [w for w in weeks if not np.isnan(main1.loc[w, "observed"])])
    null1 = float(shuf1.mean())
    tr_all = pd.read_parquet(res.parent / "gt2_transfer" / "weekly_results.parquet")
    checks = [("GT#2 (chosen) shuffled-label mean (main / shared / full), mean over weeks", null2, abs(null2 - 0.5) <= tol, "0.5"),
              ("GT#2 secondary AUC shuffled-label mean, mean over weeks", null_alt, abs(null_alt - 0.5) <= tol, "0.5"),
              ("GT#1 excess under one label shuffle per week (main / shared / full), mean over weeks", null1, abs(null1) <= tol, "0")]
    # ---- agreement and stability
    rows = []
    rows.append(("GT#1 excess vs GT#2 (chosen)", spearman(main1["excess"], main2["pi"])))
    rows.append(("GT#1 excess vs GT#2 secondary AUC", spearman(main1["excess"], alt["pi"])))
    rows.append(("GT#2 chosen vs secondary AUC", spearman(main2["pi"], alt["pi"])))
    rows.append(("GT#2 chosen vs original volume-weighted", spearman(main2["pi"], old2["pi"])))
    gold = tr_all[(tr_all.labels == "gold") & (tr_all.instrument == "phrase") & (tr_all.weighting == "cap100") & (tr_all.source == "full")]
    rows.append(("GT#2 chosen: main labels vs gold labels", spearman(main2["pi"], gold.set_index("week_start")["value"].reindex(weeks))))
    for ls in sorted(set(g1.label_set) - {"main"}):
        a = g1[(g1.venue == "shared") & (g1.label_set == ls) & (g1["sample"] == "full")].set_index("week_start")["excess"]
        rows.append((f"GT#1: main vs {ls} (shared, full)", spearman(main1["excess"], a.reindex(weeks))))
    for r in sorted(set(g1["sample"]) - {"full"}, key=float):
        a = g1[(g1.venue == "shared") & (g1.label_set == "main") & (g1["sample"] == r)].set_index("week_start")["excess"]
        rows.append((f"GT#1: full vs sample {r}", spearman(main1["excess"], a.reindex(weeks))))
    for r in sorted(set(g1["sample"]) - {"full"}, key=float):
        rows.append((f"GT#2 chosen: full vs sample {r}", spearman(main2["pi"], tseries("phrase", "cap100", f"r{r}")["pi"])))
        rows.append((f"GT#2 secondary AUC: full vs sample {r}", spearman(alt["pi"], tseries("logit", "equal", f"r{r}")["pi"])))
    if g2s is not None:
        for r in sorted(set(g2s["sample"]), key=float):
            a = g2s[(g2s.venue == "shared") & (g2s.label_set == "main") & (g2s["sample"] == r)].set_index("week_start")["pi"]
            rows.append((f"GT#2 original volume-weighted: full vs sample {r}", spearman(old2["pi"], a.reindex(weeks))))
    stab = pd.DataFrame(rows, columns=["comparison (Spearman across weeks)", "rho"])

    # ---- venues available
    ven = g1[(g1.label_set == "main") & (g1["sample"] == "full")].groupby("venue").agg(
        weeks_with_gt1=("low_n", lambda s: int((~s).sum())), median_n_dem=("n_dem", "median"), median_n_rep=("n_rep", "median"))
    ven2 = g2[g2.label_set == "main"].groupby("venue")["low_n"].apply(lambda s: int((~s).sum())).rename("weeks_with_gt2_volume_weighted")
    ven = ven.join(ven2).reset_index()

    # ---- Kareem file
    def tcols(t, prefix, suffix=""):
        d = {f"{prefix}{suffix}": t["pi"].to_numpy(), f"{prefix}_null_mean{suffix}": t["null_mean"].to_numpy(),
             f"{prefix}_excess{suffix}": t["excess"].to_numpy()}
        return d

    k = pd.DataFrame({"week_start": weeks, "gt1_excess": main1["excess"].to_numpy()})
    for c, v in {**tcols(main2, "gt2"), "gt2_boot_lo": main2["boot_lo"].to_numpy(), "gt2_boot_hi": main2["boot_hi"].to_numpy(),
                 "gt2_n_dem": main2["n_dem"].where(main2["pi"].notna()).to_numpy(), "gt2_n_rep": main2["n_rep"].where(main2["pi"].notna()).to_numpy()}.items():
        k[c] = v
    for c, v in {"gt2_alt_auc": alt["pi"], "gt2_alt_auc_null_mean": alt["null_mean"], "gt2_alt_auc_excess": alt["excess"],
                 "gt2_alt_auc_boot_lo": alt["boot_lo"], "gt2_alt_auc_boot_hi": alt["boot_hi"]}.items():
        k[c] = v.to_numpy()
    for c, v in {"gt2_pi_volume_weighted": old2["pi"], "gt2_pi_volume_weighted_null_mean": old2["null_mean"],
                 "gt2_pi_volume_weighted_boot_lo": old2["boot_lo"], "gt2_pi_volume_weighted_boot_hi": old2["boot_hi"],
                 "gt2_pi_volume_weighted_n_dem": old2["n_dem"], "gt2_pi_volume_weighted_n_rep": old2["n_rep"]}.items():
        k[c] = v.to_numpy()
    k["n_dem"], k["n_rep"] = main1["n_dem"].to_numpy(), main1["n_rep"].to_numpy()
    k["low_n"] = (main1["low_n"].fillna(True) | main2["low_n"].fillna(True)).to_numpy()
    for r in sorted(set(g1["sample"]) - {"full"}, key=float):
        a = g1[(g1.venue == "shared") & (g1.label_set == "main") & (g1["sample"] == r)].set_index("week_start").reindex(weeks)
        k[f"gt1_excess_{r}"] = a["excess"].to_numpy()
        t, ta = tseries("phrase", "cap100", f"r{r}"), tseries("logit", "equal", f"r{r}")
        k[f"gt2_{r}"] = t["pi"].to_numpy()
        k[f"gt2_null_mean_{r}"] = t["null_mean"].to_numpy()
        k[f"gt2_excess_{r}"] = t["excess"].to_numpy()
        k[f"gt2_boot_lo_{r}"] = t["boot_lo"].to_numpy()
        k[f"gt2_boot_hi_{r}"] = t["boot_hi"].to_numpy()
        k[f"gt2_n_dem_{r}"] = t["n_dem"].where(t["pi"].notna()).to_numpy()
        k[f"gt2_n_rep_{r}"] = t["n_rep"].where(t["pi"].notna()).to_numpy()
        k[f"gt2_alt_auc_{r}"] = ta["pi"].to_numpy()
        k[f"gt2_alt_auc_null_mean_{r}"] = ta["null_mean"].to_numpy()
        k[f"gt2_alt_auc_excess_{r}"] = ta["excess"].to_numpy()
        if g2s is not None:
            b = g2s[(g2s.venue == "shared") & (g2s.label_set == "main") & (g2s["sample"] == r)].set_index("week_start").reindex(weeks)
            k[f"gt2_pi_volume_weighted_{r}"] = b["pi"].to_numpy()
        k[f"low_n_{r}"] = (a["low_n"].fillna(True) | t["low_n"]).to_numpy()
    k.to_csv(res / "ground_truth_weekly.csv", index=False)
    us = pd.read_parquet(res.parent / "gt2_transfer" / "user_scores.parquet")
    us = us.rename(columns={"phrase_cap100": "gt2_score", "logit_equal": "gt2_alt_auc_score"})[
        ["week_start", "author", "gt2_score", "gt2_alt_auc_score", "n_phrases", "side_main", "side_gold", "pile"]]
    us.to_parquet(res / "user_scores_gt2.parquet", index=False)

    # ---- summary.md
    md = ["# Ground truth summary (generated by the gt_summary job)", "", COLUMN_GUIDE, "",
          "## Null checks", ""]
    for name, val, ok, target in checks:
        md.append(f"- {'OK' if ok else '**FAIL: CHECK THIS**'}: {name} = {val:.4f} (expected about {target}, tolerance {tol})")
    md += ["", "## Series (main / shared / full)", "",
           f"- GT#1 excess: mean {main1['excess'].mean():.3f}, range {main1['excess'].min():.3f} to {main1['excess'].max():.3f}",
           f"- GT#2 (chosen, phrase estimator): mean {main2['pi'].mean():.4f} (null {null2:.4f}), range {main2['pi'].min():.4f} to {main2['pi'].max():.4f}; "
           f"weeks above null + 2 sd: {int((main2['pi'] > main2['null_mean'] + 2 * main2['null_sd']).sum())} of {int(main2['pi'].notna().sum())} usable; "
           f"weeks NaN (too few users): {int(main2['pi'].isna().sum())}",
           f"- GT#2 secondary AUC: mean {alt['pi'].mean():.3f} (null {null_alt:.3f}), range {alt['pi'].min():.3f} to {alt['pi'].max():.3f}; "
           f"weeks above null + 2 sd: {int((alt['pi'] > alt['null_mean'] + 2 * alt['null_sd']).sum())} of {int(alt['pi'].notna().sum())}",
           f"- GT#2 original volume-weighted pi (kept as gt2_pi_volume_weighted): mean {old2['pi'].mean():.3f}, below its own null mean in "
           f"{int((old2['pi'] < old2['null_mean']).sum())} of {int(old2['pi'].notna().sum())} weeks",
           f"- weeks flagged low_n: {int(k['low_n'].sum())} of {len(k)}",
           "- Plot: `gt_main_shared_full.png` (events: " + ", ".join(events) + ").", "",
           "## Venues and label sets", "",
           "(The chosen GT#2 exists for the shared venue only. The venue table below counts weeks of GT#1 and of the original volume-weighted GT#2.)", "",
           md_table(ven), "",
           "## Agreement and stability", "", md_table(stab, "{:.3f}"), ""]
    var = res / "variants_summary.csv"
    if var.exists():
        md += ["## Variants (T5; these use the original volume-weighted GT#2)", "", md_table(pd.read_csv(var), "{:.3f}"), ""]
    wc = res / "within_camp_weekly.parquet"
    if wc.exists():
        w = pd.read_parquet(wc)
        md += ["## Within camp (Sanders vs Clinton, T4)", "",
               f"GT#1 excess mean {w['gt1_excess'].mean():.3f}; original volume-weighted GT#2 pi mean {w['gt2_pi'].mean():.3f} (not recomputed with the transfer design). Plot: `within_camp.png`.", ""]
    (res / "summary.md").write_text("\n".join(md), encoding="utf-8")
    for name, val, ok, _ in checks:
        if not ok:
            ctx.log(f"NULL CHECK FAILED: {name} = {val}")
    return {"rows": len(k), "bytes": 0}

"""Tables, plots and summary.md for the gt1_panel job."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.gt_runs import gt1_robust as R  # noqa: E402
from src.gt_runs.gt1_panel import mean_ci  # noqa: E402
from src.prep_jobs.config import project_root  # noqa: E402
from src.prep_jobs.jobs.gt1_robustness_report import md, week_of  # noqa: E402

LABEL = {
    "main": "Main (all labelled users)",
    "panel_a": "1. Panel A (>= 12 active weeks)",
    "panel_b": "2. Panel B (>= 5 weeks each half)",
    "balanced": "3. Balanced sides (all users)",
    "pol_balanced": "4. r/politics only, balanced",
    "strict": "5. Panel B + balanced + r/politics",
    "reweight": "6. Reweighting counterfactual (after weeks)",
}
PARTIAL = ["2015-12-28", "2016-12-26"]


def verdict(share: float, lo: float, hi: float) -> str:
    if not np.isfinite(share):
        return "n/a"
    if lo <= 0 <= hi or share < 1 / 3:
        return "vanishes"
    return "survives" if share >= 2 / 3 else "shrinks a lot"


def make_report(weeks: list[str], files: dict, tgt: dict, g: dict, out, log) -> None:
    p = g["gt1_panel"]
    split = p["split_week"]
    n_boot = p["n_boot"]
    x = pd.to_datetime(weeks)
    before = np.array([w < split for w in weeks])
    ser, boot, nd, nr = {}, {}, {}, {}
    for v in LABEL:
        s = np.full(len(weeks), np.nan); b = np.full((len(weeks), n_boot), np.nan)
        d = np.full(len(weeks), np.nan); r = np.full(len(weeks), np.nan)
        for i, w in enumerate(weeks):
            e = files[w].get(v)
            if e is None:
                continue
            s[i] = np.nan if e["excess"] is None else e["excess"]
            b[i] = [np.nan if q is None else q for q in e["boot"]]
            d[i], r[i] = e["n_dem"], e["n_rep"]
        ser[v], boot[v], nd[v], nr[v] = s, b, d, r
    # the counterfactual keeps the main values before the split
    ser["reweight"] = np.where(before, ser["main"], ser["reweight"])
    boot["reweight"] = np.where(before[:, None], boot["main"], boot["reweight"])
    nd["reweight"] = np.where(before, nd["main"], nd["reweight"]); nr["reweight"] = np.where(before, nr["main"], nr["reweight"])

    official = pd.read_parquet(project_root() / "results" / "ground_truth" / "gt1_weekly.parquet")
    official = official[(official.venue == "shared") & (official.label_set == "main") & (official["sample"] == "full")].set_index("week_start")["excess"]
    chk = float(spearmanr(pd.Series(ser["main"], index=weeks).dropna(), official.reindex(weeks)[~np.isnan(ser["main"])])[0])
    log(f"recomputed main vs gt1_weekly Spearman {chk:.4f}")

    ev_dates = {k: v for k, v in g["events"].items() if k in g["gt1_robust"]["events"]}
    event_weeks = sorted({w for w in (week_of(d, weeks) for d in ev_dates.values()) if w})

    def table(mask: np.ndarray) -> pd.DataFrame:
        rows = {}
        main_drop = None
        for v in LABEL:
            s = pd.Series(ser[v], index=weeks)
            bm, am = mask & before, mask & ~before
            bb = np.nanmean(boot[v][bm], axis=0); ab = np.nanmean(boot[v][am], axis=0)
            b0, a0 = np.nanmean(ser[v][bm]), np.nanmean(ser[v][am])
            drop = b0 - a0
            if v == "main":
                main_drop = drop
            lo, hi = mean_ci(bb - ab)
            both = pd.DataFrame({"a": ser["main"], "b": ser[v]})[mask].dropna()
            rho = float(spearmanr(both["a"], both["b"])[0]) if len(both) > 4 else np.nan
            ev = R.event_test(s[mask], [w for w in event_weeks if w in np.array(weeks)[mask]], p["event_perm"], g["seed"])
            share = drop / main_drop
            rows[LABEL[v]] = {
                "weeks used": int(np.isfinite(ser[v][mask]).sum()),
                "mean before": b0, "before lo": mean_ci(bb)[0], "before hi": mean_ci(bb)[1],
                "mean after": a0, "after lo": mean_ci(ab)[0], "after hi": mean_ci(ab)[1],
                "drop": drop, "drop lo": lo, "drop hi": hi, "drop / main drop": share,
                "Spearman with main": rho, "event diff": ev["diff"], "event p (one-sided)": ev["p_one_sided"],
                "reading": verdict(share, lo, hi),
            }
        return pd.DataFrame(rows).T

    full = table(np.ones(len(weeks), dtype=bool))
    nopart = table(np.array([w not in PARTIAL for w in weeks]))
    full.to_csv(out / "variants_table_all_weeks.csv"); nopart.to_csv(out / "variants_table_no_partial_weeks.csv")
    pd.DataFrame({f"{v}_excess": ser[v] for v in LABEL} | {f"{v}_n_dem": nd[v] for v in LABEL} | {f"{v}_n_rep": nr[v] for v in LABEL},
                 index=pd.Index(weeks, name="week_start")).to_csv(out / "variants_weekly.csv")

    def short(t: pd.DataFrame) -> pd.DataFrame:
        o = pd.DataFrame(index=t.index)
        o["weeks"] = t["weeks used"]
        o["before [95% CI]"] = [f"{r['mean before']:.4f} [{r['before lo']:.4f}, {r['before hi']:.4f}]" for _, r in t.iterrows()]
        o["after [95% CI]"] = [f"{r['mean after']:.4f} [{r['after lo']:.4f}, {r['after hi']:.4f}]" for _, r in t.iterrows()]
        o["drop [95% CI]"] = [f"{r['drop']:.4f} [{r['drop lo']:.4f}, {r['drop hi']:.4f}]" for _, r in t.iterrows()]
        o["drop / main"] = t["drop / main drop"]
        o["Spearman w/ main"] = t["Spearman with main"]
        o["events diff (p)"] = [f"{r['event diff']:.4f} ({r['event p (one-sided)']:.3f})" for _, r in t.iterrows()]
        o["reading"] = t["reading"]
        return o

    # ---- users per side
    us = []
    for v in LABEL:
        for nm, m in (("before", before), ("after", ~before)):
            if v == "reweight" and nm == "before":
                continue
            us.append({"variant": LABEL[v], "period": nm, "weeks": int(np.isfinite(nd[v][m]).sum()),
                       "dem mean": np.nanmean(nd[v][m]), "dem min": np.nanmin(nd[v][m]), "dem max": np.nanmax(nd[v][m]),
                       "rep mean": np.nanmean(nr[v][m]), "rep min": np.nanmin(nr[v][m]), "rep max": np.nanmax(nr[v][m])})
    users = pd.DataFrame(us).set_index("variant")
    users.to_csv(out / "users_per_side.csv")

    # ---- plots
    fig, axs = plt.subplots(4, 2, figsize=(12, 11), sharex=True, sharey=True)
    for ax, v in zip(axs.ravel(), LABEL):
        ax.plot(x, ser["main"], color="0.75", label="main")
        ax.plot(x, ser[v], color="C0" if v != "main" else "k", label=LABEL[v])
        ax.axvline(pd.Timestamp(split), color="C3", lw=0.8, ls="--")
        ax.set_title(LABEL[v], fontsize=9)
    axs.ravel()[-1].axis("off")
    fig.tight_layout(); fig.savefig(out / "series_by_variant.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    t = full
    ypos = np.arange(len(t))[::-1]
    ax.errorbar(t["drop"], ypos, xerr=[t["drop"] - t["drop lo"], t["drop hi"] - t["drop"]], fmt="o", color="C0", capsize=3)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(ypos); ax.set_yticklabels(t.index, fontsize=8)
    ax.set_xlabel("mean excess before minus after 2016-06-27 (95% thread bootstrap)")
    fig.tight_layout(); fig.savefig(out / "drop_by_variant.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for v in LABEL:
        ax[0].plot(x, nd[v], label=LABEL[v]); ax[1].plot(x, nr[v], label=LABEL[v])
    ax[0].set_title("Democrat users per week"); ax[1].set_title("Republican users per week")
    for a in ax:
        a.axvline(pd.Timestamp(split), color="C3", lw=0.8, ls="--"); a.set_yscale("log")
    ax[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(out / "users_per_side.png", dpi=110); plt.close(fig)

    txt = f"""# GT#1 fixed-panel check: is the fall around {split} behaviour or composition?

Series: main labels, shared venue (r/politics + r/PoliticalDiscussion), full week, heavy posters excluded, {len(weeks)} weeks;
excess = observed thread dissimilarity minus the mean of the shuffled-label null (as in the main GT#1). The recomputed main series
matches `gt1_weekly.parquet` (Spearman {chk:.3f}). "Before" = {int(before.sum())} weeks starting before {split}; "after" = {int((~before).sum())} weeks from {split}.
Intervals: {n_boot} thread-bootstrap resamples per week (null mean from {p['boot_perm']} shuffles per resample); resamples are independent between weeks, so the
before and after means and their difference are computed replicate by replicate. For the balanced and reweighted variants every replicate also redraws the
user subsample. Event test: mean in event weeks minus other weeks, date-shuffle p ({p['event_perm']} shuffles), events as in `gt1_robustness`.
Reading rule (drop in the variant as a share of the main drop): >= 2/3 and interval excludes 0 = survives; 1/3 to 2/3 = shrinks a lot; below 1/3 or interval contains 0 = vanishes.

## All variants, all weeks
{md(short(full))}

## Same, without the partial first and last weeks ({', '.join(PARTIAL)})
{md(short(nopart))}

## Users kept per side per week
Panel sizes: {tgt['panel_a_users']:,} users in Panel A and {tgt['panel_b_users']:,} in Panel B, out of {tgt['n_users']:,} labelled users that appear in the year.
Reweighting target (average of the weeks before the split): Republican/Democrat user ratio {tgt['rep_dem_ratio']:.3f}, share of users whose home subreddit is r/politics {tgt['politics_share']:.3f}.
{md(users)}

Plots: `series_by_variant.png`, `drop_by_variant.png`, `users_per_side.png`. Tables: `variants_table_all_weeks.csv`, `variants_table_no_partial_weeks.csv`, `variants_weekly.csv`, `users_per_side.csv`.
"""
    (out / "summary.md").write_text(txt, encoding="utf-8")
    log("summary written")

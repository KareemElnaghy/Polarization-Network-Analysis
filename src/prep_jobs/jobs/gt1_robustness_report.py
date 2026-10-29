"""Tables, plots and summary.md for the gt1_robustness job."""

from __future__ import annotations

from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from src.gt_runs import gt1_robust as R  # noqa: E402


def md(df: pd.DataFrame, nd: int = 3) -> str:
    df = df.reset_index()
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)


def week_of(d: str, weeks: list[str]) -> str | None:
    x = date.fromisoformat(d)
    for w in weeks:
        s = date.fromisoformat(w)
        if s <= x < s + timedelta(days=7):
            return w
    return None


def corr(a: pd.Series, b: pd.Series, how="spearman") -> float:
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 5:
        return np.nan
    return float((spearmanr if how == "spearman" else pearsonr)(d.iloc[:, 0], d.iloc[:, 1])[0])


def make_report(weekly: pd.DataFrame, halves: dict, g: dict, out, log) -> None:
    p = g["gt1_robust"]
    weeks = list(weekly.index)
    x = pd.to_datetime(weekly.index)
    ev_dates = {k: v for k, v in g["events"].items() if k in p["events"]}
    ev_weeks = {k: week_of(v, weeks) for k, v in ev_dates.items()}
    event_weeks = sorted({w for w in ev_weeks.values() if w})
    main = weekly["main_excess"]

    def vlines(ax):
        for name, w in ev_weeks.items():
            if w:
                ax.axvline(pd.Timestamp(w), color="grey", lw=0.6, ls=":")

    # ---- (a) events
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, main, color="C0", label="excess")
    ax.fill_between(x, weekly["boot_lo"], weekly["boot_hi"], color="C0", alpha=0.25, label="95% thread bootstrap")
    vlines(ax)
    ax.set_ylabel("GT#1 excess"); ax.legend(); ax.set_title("GT#1 weekly excess with bootstrap bands (dotted: events)")
    fig.tight_layout(); fig.savefig(out / "a_events.png", dpi=110); plt.close(fig)
    tests = {}
    for name, col in [("main", "main_excess"), ("stratified", "strat_excess")]:
        tests[name] = R.event_test(weekly[col], event_weeks, p["event_perm"], g["seed"])
    ev_tab = pd.DataFrame(tests).T
    per_ev = pd.DataFrame({"week": ev_weeks, "excess": {k: (main.get(w) if w else np.nan) for k, w in ev_weeks.items()}}).rename_axis("event")
    z = (main - main.mean()) / main.std()
    band = float((weekly["boot_hi"] - weekly["boot_lo"]).mean())

    # ---- (b) topic control
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(x, main, label="excess (plain null)")
    ax[0].plot(x, weekly["strat_excess"], label="excess over subreddit x day x size null")
    ax[0].legend(); vlines(ax[0])
    for s, st in [("politics", "-"), ("PoliticalDiscussion", "--")]:
        ax[1].plot(x, weekly[f"sub_{s}_excess"], st, label=s)
    ax[1].plot(x, weekly["big_excess"], ":", color="k", label="threads with >= N users per side")
    ax[1].legend(); vlines(ax[1])
    fig.tight_layout(); fig.savefig(out / "b_topic_control.png", dpi=110); plt.close(fig)
    topic = pd.DataFrame({
        "mean excess": [main.mean(), weekly["strat_excess"].mean(), weekly["sub_politics_excess"].mean(),
                        weekly["sub_PoliticalDiscussion_excess"].mean(), weekly["big_excess"].mean()],
        "weeks usable": [int(main.notna().sum()), int(weekly["strat_excess"].notna().sum()), int(weekly["sub_politics_excess"].notna().sum()),
                         int(weekly["sub_PoliticalDiscussion_excess"].notna().sum()), int(weekly["big_excess"].notna().sum())],
        "Spearman with main": [1.0, corr(main, weekly["strat_excess"]), corr(main, weekly["sub_politics_excess"]),
                               corr(main, weekly["sub_PoliticalDiscussion_excess"]), corr(main, weekly["big_excess"])],
    }, index=["main (plain null)", "stratified null", "r/politics only", "r/PoliticalDiscussion only", "threads with >= 10 users per side"])
    topic["share of main kept"] = topic["mean excess"] / main.mean()
    ev_strat = R.event_test(weekly["strat_excess"], event_weeks, p["event_perm"], g["seed"])

    # ---- (c) label noise
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, main, label="main labels"); ax.plot(x, weekly["gold_excess"], label="gold labels")
    ax.plot(x, weekly["gold_flip_excess"], label=f"gold, {int(p['flip_frac'] * 100)}% flipped"); ax.legend(); vlines(ax)
    fig.tight_layout(); fig.savefig(out / "c_label_noise.png", dpi=110); plt.close(fig)
    shrink = (weekly["gold_flip_excess"] / weekly["gold_excess"]).replace([np.inf, -np.inf], np.nan)
    noise = pd.DataFrame({"mean excess": [main.mean(), weekly["gold_excess"].mean(), weekly["gold_flip_excess"].mean()],
                          "weeks usable": [int(main.notna().sum()), int(weekly["gold_excess"].notna().sum()), int(weekly["gold_flip_excess"].notna().sum())]},
                         index=["main labels", "gold labels", f"gold, {int(p['flip_frac'] * 100)}% flipped"])

    # ---- (d) reliability
    rs, sbs = [], []
    arr = {w: np.array([[np.nan if v is None else v for v in h] for h in hs]) for w, hs in halves.items()}
    for k in range(p["n_splits"]):
        a = pd.Series({w: arr[w][k, 0] for w in weeks})
        b = pd.Series({w: arr[w][k, 1] for w in weeks})
        r = corr(a, b, "pearson")
        rs.append(r); sbs.append(2 * r / (1 + r) if r == r else np.nan)
    rs, sbs = np.array(rs), np.array(sbs)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    a0 = pd.Series({w: arr[w][0, 0] for w in weeks}); b0 = pd.Series({w: arr[w][0, 1] for w in weeks})
    ax[0].scatter(a0, b0, s=12); ax[0].set_xlabel("half A excess"); ax[0].set_ylabel("half B excess"); ax[0].set_title("one split")
    ax[1].hist(rs[~np.isnan(rs)], bins=10); ax[1].set_title("split-half r over splits")
    fig.tight_layout(); fig.savefig(out / "d_reliability.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, main, label="all labelled users"); ax.plot(x, weekly["notop_excess"], label="top 1% most active removed"); ax.legend(); vlines(ax)
    fig.tight_layout(); fig.savefig(out / "d_no_top_users.png", dpi=110); plt.close(fig)

    txt = f"""# GT#1 robustness (main labels, shared venue, full week, {len(weeks)} weeks)

Existing GT outputs are untouched. Per-week values are in `robustness_weekly.parquet`; the checks use the same labels, heavy-poster exclusion and
index as `gt1_weekly.parquet` (the `main_excess` here reproduces its `excess`).

## a. Event weeks
Event weeks (the week containing each date): {', '.join(f'{k} {w}' for k, w in ev_weeks.items())}.
Mean 95% thread-bootstrap band width: {band:.3f} (about {band / main.std():.2f} of the sd of the weekly series, {main.std():.3f}).

{md(ev_tab)}

(`diff` = event-week mean minus other-week mean; the p-values come from {p['event_perm']} shuffles of which weeks are event weeks.)

{md(per_ev)}

Plot: `a_events.png`.

## b. Topic control
{md(topic)}

Stratified null: users are grouped by the (subreddit, day, thread-size bin) of their most-commented thread and labels are shuffled only among users
in the same cell (thread day = day of the thread's first reply edge that week; threads without a reply edge have day unknown; size = labelled users
in the thread, log2 bins). Event-week test on the stratified excess: diff {ev_strat['diff']:.4f}, p one-sided {ev_strat['p_one_sided']:.3f}.
Plot: `b_topic_control.png`.

## c. Label noise
{md(noise)}

Mean ratio of flipped to unflipped gold excess: {np.nanmean(shrink):.2f} (expected about {(1 - 2 * p['flip_frac']) ** 2:.2f} if excess scaled with the
square of the signal, {1 - 2 * p['flip_frac']:.2f} if linearly). Spearman of the gold and main series: {corr(main, weekly['gold_excess']):.2f}.
Plot: `c_label_noise.png`.

## d. Reliability
Split-half over {p['n_splits']} random halves of threads (each half recomputed with its own null): mean Pearson r {np.nanmean(rs):.2f}
(range {np.nanmin(rs):.2f} to {np.nanmax(rs):.2f}), Spearman-Brown corrected {np.nanmean(sbs):.2f}.
Without the top 1% most active users: mean excess {weekly['notop_excess'].mean():.3f} (all users {main.mean():.3f}), Spearman with the full series
{corr(main, weekly['notop_excess']):.2f}, Pearson {corr(main, weekly['notop_excess'], 'pearson'):.2f}.
Plots: `d_reliability.png`, `d_no_top_users.png`.
"""
    (out / "summary.md").write_text(txt, encoding="utf-8")
    log(f"summary written to {out / 'summary.md'}")

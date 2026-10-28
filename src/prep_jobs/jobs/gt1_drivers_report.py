"""Analysis, plots and summary.md for the gt1_drivers job."""

from __future__ import annotations

from datetime import date, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.gt_runs import gt1_robust as R  # noqa: E402
from src.prep_jobs.config import project_root  # noqa: E402

VARS = ["n_dem", "n_rep", "ratio_rep_dem", "n_threads", "share_threads_both", "mean_thread_size", "median_thread_size",
        "share_politics", "total_comments_all", "share_new_users", "share_continuing_users"]


def md(df: pd.DataFrame, nd: int = 3) -> str:
    df = df.reset_index()
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def ols(X: np.ndarray, y: np.ndarray) -> dict:
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    rss, tss = float(res @ res), float(((y - y.mean()) ** 2).sum())
    H = A @ np.linalg.pinv(A.T @ A) @ A.T
    loo = res / (1 - np.diag(H))
    n, k = A.shape
    return {"beta": beta, "resid": res, "rss": rss, "r2": 1 - rss / tss, "adj_r2": 1 - (rss / (n - k)) / (tss / (n - 1)),
            "loo_r2": 1 - float(loo @ loo) / tss, "loo_resid": loo, "n": n, "k": k - 1}


def zs(X: pd.DataFrame) -> np.ndarray:
    return ((X - X.mean()) / X.std()).to_numpy()


def vif(Z: np.ndarray) -> np.ndarray:
    out = []
    for j in range(Z.shape[1]):
        o = ols(np.delete(Z, j, axis=1), Z[:, j])
        out.append(1 / max(1 - o["r2"], 1e-9))
    return np.array(out)


def week_of(d: str, weeks: list[str]):
    x = date.fromisoformat(d)
    for w in weeks:
        s = date.fromisoformat(w)
        if s <= x < s + timedelta(days=7):
            return w


def autocorr(x: np.ndarray) -> float:
    x = x - x.mean()
    return float((x[1:] @ x[:-1]) / (x @ x))


def changepoint(y: np.ndarray, min_seg: int = 8, n_perm: int = 2000, seed: int = 0):
    def best(v):
        tss = ((v - v.mean()) ** 2).sum()
        out = (-1.0, 0)
        for k in range(min_seg, len(v) - min_seg + 1):
            rss = ((v[:k] - v[:k].mean()) ** 2).sum() + ((v[k:] - v[k:].mean()) ** 2).sum()
            out = max(out, (1 - rss / tss, k))
        return out

    r2, k = best(y)
    rng = np.random.default_rng(seed)
    perm = [best(rng.permutation(y))[0] for _ in range(n_perm)]
    return {"k": k, "r2": r2, "p": float((np.array(perm) >= r2).mean()), "mean_before": float(y[:k].mean()), "mean_after": float(y[k:].mean())}


def analyse(df: pd.DataFrame, events: list[str], n_perm: int, seed: int) -> dict:
    """All numbers for one set of weeks (index = week_start, columns: excess + VARS)."""
    d = df.dropna(subset=["excess"] + VARS)
    y = d["excess"].to_numpy()
    t = np.arange(len(d), dtype=float)
    out = {"n": len(d), "weeks": d}
    sp = pd.DataFrame({v: stats.spearmanr(d[v], y) for v in VARS}, index=["rho", "p"]).T
    out["spearman"] = sp
    Z = zs(d[VARS])
    full = ols(Z, y)
    pr = {}
    for j, v in enumerate(VARS):
        o = ols(np.delete(Z, j, axis=1), y)
        pr[v] = (o["rss"] - full["rss"]) / o["rss"]
    out["partial"] = pd.Series(pr)
    out["full"] = full
    out["vif"] = pd.Series(vif(Z), index=VARS)
    zt = np.column_stack([Z, (t - t.mean()) / t.std()])
    full_t = ols(zt, y)
    out["full_trend"] = full_t
    # time structure
    lr = stats.linregress(t, y)
    mon = pd.Series(pd.to_datetime(d.index).month, index=d.index)
    dummies = pd.get_dummies(mon, drop_first=True).astype(float).to_numpy()
    om = ols(dummies, y)
    f = ((1 - (om["rss"] / ((y - y.mean()) ** 2).sum())) / om["k"]) / ((om["rss"] / ((y - y.mean()) ** 2).sum()) / (len(y) - om["k"] - 1))
    rng = np.random.default_rng(seed)
    ac = autocorr(y)
    ac_perm = np.array([autocorr(rng.permutation(y)) for _ in range(n_perm)])
    out["time"] = {"slope_per_week": float(lr.slope), "trend_r2": float(lr.rvalue ** 2), "trend_p": float(lr.pvalue),
                   "month_r2": om["r2"], "month_adj_r2": om["adj_r2"], "month_p": float(1 - stats.f.cdf(f, om["k"], len(y) - om["k"] - 1)),
                   "lag1_autocorr": ac, "lag1_p": float((ac_perm >= ac).mean()),
                   "lag1_autocorr_resid": autocorr(full_t["resid"])}
    out["change"] = changepoint(y, seed=seed)
    out["change"]["week"] = d.index[out["change"]["k"]]
    # events on raw series and on residuals
    tests = {"raw excess": y, "residual: composition + trend": full_t["resid"], "residual: composition + trend (leave-one-out)": full_t["loo_resid"],
             "residual: trend only": y - (lr.intercept + lr.slope * t)}
    out["events"] = pd.DataFrame({k: R.event_test(pd.Series(v, index=d.index), events, n_perm, seed) for k, v in tests.items()}).T
    out["resid_trend_comp"] = pd.Series(full_t["resid"], index=d.index)
    return out


def make_report(df: pd.DataFrame, g: dict, out_dir, log) -> None:
    p = g["gt1_robust"]
    gt1 = pd.read_parquet(project_root() / "results" / "ground_truth" / "gt1_weekly.parquet")
    gt1 = gt1[(gt1.venue == "shared") & (gt1.label_set == "main") & (gt1["sample"] == "full")].set_index("week_start")["excess"]
    df = df.join(gt1.rename("excess"))
    df.to_csv(out_dir / "drivers_weekly.csv")
    weeks = list(df.index)
    ev = {k: week_of(v, weeks) for k, v in g["events"].items() if k in p["events"]}
    event_weeks = sorted({w for w in ev.values() if w})
    seed, nperm = g["seed"], p["event_perm"]
    A = analyse(df, event_weeks, nperm, seed)
    B = analyse(df.iloc[1:-1], event_weeks, nperm, seed)

    def block(X: dict, title: str) -> str:
        t, ch = X["time"], X["change"]
        reg = pd.DataFrame({"Spearman rho": X["spearman"]["rho"], "p": X["spearman"]["p"], "partial R2": X["partial"], "VIF": X["vif"]})
        overall = pd.DataFrame({"composition (11 variables)": [X["full"]["r2"], X["full"]["adj_r2"], X["full"]["loo_r2"]],
                                "composition + linear trend": [X["full_trend"]["r2"], X["full_trend"]["adj_r2"], X["full_trend"]["loo_r2"]]},
                               index=["R2", "adjusted R2", "leave-one-out R2"]).T
        time = pd.DataFrame({"value": {
            "linear trend: slope per week": t["slope_per_week"], "linear trend: R2": t["trend_r2"], "linear trend: p": t["trend_p"],
            "month effects: R2 (11 dummies)": t["month_r2"], "month effects: adjusted R2": t["month_adj_r2"], "month effects: F-test p": t["month_p"],
            "lag-1 autocorrelation": t["lag1_autocorr"], "lag-1 permutation p (one-sided)": t["lag1_p"],
            "lag-1 autocorrelation of the composition+trend residual": t["lag1_autocorr_resid"],
            f"best single mean shift: before {ch['week']}": ch["mean_before"], "best single mean shift: after": ch["mean_after"],
            "changepoint R2 / permutation p": f"{ch['r2']:.3f} / {ch['p']:.3f}"}})
        return (f"## {title} ({X['n']} weeks)\n\n### Each variable alone and in the joint regression\n{md(reg)}\n\n"
                f"Joint regression of excess on all variables (standardised):\n\n{md(overall)}\n\n### Time structure\n{md(time)}\n\n"
                f"### Event weeks, same date-shuffle test ({p['event_perm']} shuffles)\n{md(X['events'])}\n")

    # plots
    d = A["weeks"]
    x = pd.to_datetime(d.index)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7))
    ax[0].plot(x, d["excess"], label="excess")
    ax[0].plot(x, d["excess"] - A["resid_trend_comp"], label="fitted: composition + trend")
    for w in event_weeks:
        ax[0].axvline(pd.Timestamp(w), color="grey", lw=0.6, ls=":")
    ax[0].legend()
    ax[1].bar(x, A["resid_trend_comp"], width=5); ax[1].set_title("residual after composition + trend (dotted lines in top panel: events)")
    for w in event_weeks:
        ax[1].axvline(pd.Timestamp(w), color="grey", lw=0.6, ls=":")
    fig.tight_layout(); fig.savefig(out_dir / "fit_and_residual.png", dpi=110); plt.close(fig)
    fig, axs = plt.subplots(3, 4, figsize=(13, 8))
    for a, v in zip(axs.ravel(), VARS):
        a.scatter(d[v], d["excess"], s=10); a.set_title(f"{v} (rho {A['spearman'].loc[v, 'rho']:.2f})", fontsize=8)
    fig.tight_layout(); fig.savefig(out_dir / "scatter_vs_excess.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    A["partial"].sort_values().plot.barh(ax=ax); ax.set_title("partial R2 (drop-one) in the joint regression")
    fig.tight_layout(); fig.savefig(out_dir / "partial_r2.png", dpi=110); plt.close(fig)

    keys = pd.DataFrame({
        "all weeks": [A["n"], A["full"]["r2"], A["full"]["adj_r2"], A["full"]["loo_r2"], A["full_trend"]["r2"], A["time"]["trend_r2"],
                      A["time"]["month_r2"], A["time"]["lag1_autocorr"], A["events"].loc["raw excess", "diff"], A["events"].loc["raw excess", "p_one_sided"],
                      A["events"].iloc[1]["diff"], A["events"].iloc[1]["p_one_sided"]],
        "without partial first and last week": [B["n"], B["full"]["r2"], B["full"]["adj_r2"], B["full"]["loo_r2"], B["full_trend"]["r2"], B["time"]["trend_r2"],
                                                 B["time"]["month_r2"], B["time"]["lag1_autocorr"], B["events"].loc["raw excess", "diff"],
                                                 B["events"].loc["raw excess", "p_one_sided"], B["events"].iloc[1]["diff"], B["events"].iloc[1]["p_one_sided"]],
    }, index=["weeks", "composition R2", "composition adjusted R2", "composition leave-one-out R2", "composition + trend R2", "trend R2", "month R2",
              "lag-1 autocorrelation", "event minus other weeks (raw)", "p one-sided (raw)", "event minus other weeks (residual)", "p one-sided (residual)"])
    txt = f"""# What drives the week-to-week variation of GT#1?

Series: main labels, shared venue (r/politics + r/PoliticalDiscussion), full week, heavy posters excluded (`gt1_weekly.parquet`, read only).
Composition variables come from `thread_user` for the same cell: `n_dem`, `n_rep`, `ratio_rep_dem` (labelled users per side that week);
`n_threads` (threads with at least one labelled user), `share_threads_both` (of those, threads with both camps), `mean_thread_size` and
`median_thread_size` (labelled users per thread); `share_politics` (share of the labelled users' comments in r/politics, the rest is
r/PoliticalDiscussion); `total_comments_all` (all comments in the two subs, labelled or not); `share_new_users` (labelled users active that week and
not in any earlier week; undefined in the first week) and `share_continuing_users` (labelled users also active the week before; undefined in the first
week). Weeks used where all are defined. Partial R2 = relative increase in residual sum of squares when the variable is dropped from the joint model.
Weeks of 2015-12-28 and 2016-12-26 are partial (data start on 1 Jan and end on 31 Dec).

## Main numbers
{md(keys)}

{block(A, 'All weeks')}
{block(B, 'Without the partial first and last week')}
Plots: `fit_and_residual.png`, `scatter_vs_excess.png`, `partial_r2.png`. Weekly table: `drivers_weekly.csv`.
"""
    (out_dir / "summary.md").write_text(txt, encoding="utf-8")
    log(f"summary written to {out_dir / 'summary.md'}")

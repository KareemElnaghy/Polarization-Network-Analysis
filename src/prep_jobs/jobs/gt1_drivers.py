"""Job R3, gt1_drivers: what drives the week-to-week variation of ground truth #1?

Same cell as the main GT#1 series (main labels, shared venue = r/politics + r/PoliticalDiscussion, full week, heavy posters
excluded). One unit per week computes composition measures from ``thread_user``; ``finalise`` relates them to the existing
``gt1_weekly.parquet`` excess (read only) and writes ``results/gt1_drivers/``: Spearman table, regression with partial R^2,
trend / month effects / autocorrelation / changepoint, the event test on residuals, and the same with the partial first
and last weeks dropped. Existing outputs are not touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.gt_runs.config import gt_cfg, gt_path, labels_cfg, venues
from src.gt_runs.labels import heavy_set, load_side_map
from src.prep_jobs.config import data_path, project_root, runs_dir
from src.prep_jobs.jobs.clean_edges_and_sampling import weeks_of
from src.prep_jobs.jobs.gt1_robustness import week_rows
from src.prep_jobs.util import read_json, write_json

NAME = "gt1_drivers"
DEADLINE_CUTS: list[str] = []


def phases(cfg):
    mod = "src.prep_jobs.jobs.gt1_drivers"
    return [
        {"name": "weeks", "fn": f"{mod}:week_unit", "units": [{"id": w, "weight": 100} for w in weeks_of(cfg)]},
        {"name": "finalise", "fn": f"{mod}:finalise", "units": [{"id": "finalise", "weight": 50}], "workers": 1,
         "needs_all_prior": True},
    ]


def week_unit(ctx, unit):
    cfg, g, lab, week = ctx.cfg, gt_cfg(), labels_cfg(), unit["id"]
    root = gt_path(g, "labels")
    subs = venues(g)["shared"]
    heavy = heavy_set(root, week) if lab["exclude_heavy_posters"] else None
    side_map = load_side_map(root, "shared", "main")
    rows = week_rows(cfg, week, subs, side_map, heavy)  # one row per (thread, labelled user), with n_comments
    f = data_path(cfg, "thread_user") / f"week={week}" / "part.parquet"
    allr = pq.read_table(f, columns=["subreddit", "author", "n_comments"]).to_pandas()
    allr = allr[allr["subreddit"].isin(subs)]
    per_thread = rows.groupby("thread")["side"].agg(["size", "sum"])  # size = labelled users, sum = republicans
    both = ((per_thread["sum"] > 0) & (per_thread["sum"] < per_thread["size"])).mean()
    users = rows.drop_duplicates("user")
    by_sub = rows.groupby("sub")["n_comments"].sum()
    out = {
        "week_start": week, "n_dem": int((users["side"] == 0).sum()), "n_rep": int((users["side"] == 1).sum()),
        "n_threads": int(len(per_thread)), "share_threads_both": float(both),
        "mean_thread_size": float(per_thread["size"].mean()), "median_thread_size": float(per_thread["size"].median()),
        "share_politics": float(by_sub.get("politics", 0) / by_sub.sum()),
        "total_comments_all": int(allr["n_comments"].sum()), "total_comments_labelled": int(rows["n_comments"].sum()),
    }
    out["ratio_rep_dem"] = out["n_rep"] / out["n_dem"]
    write_json(out, runs_dir(cfg) / NAME / "work" / f"{week}.json")
    users[["user"]].to_parquet(runs_dir(cfg) / NAME / "work" / f"users_{week}.parquet")
    return {"rows": 1, "bytes": 0}


def finalise(ctx, unit):
    from src.prep_jobs.jobs import gt1_drivers_report as rep

    cfg, g = ctx.cfg, gt_cfg()
    work = runs_dir(cfg) / NAME / "work"
    weeks = weeks_of(cfg)
    df = pd.DataFrame([read_json(work / f"{w}.json") for w in weeks]).set_index("week_start")
    seen: set = set()
    prev: set = set()
    new, cont = [], []
    for w in weeks:
        u = set(pd.read_parquet(work / f"users_{w}.parquet")["user"])
        new.append(np.nan if not seen else len(u - seen) / len(u))
        cont.append(np.nan if not prev else len(u & prev) / len(u))
        seen |= u
        prev = u
    df["share_new_users"], df["share_continuing_users"] = new, cont
    out = project_root() / "results" / "gt1_drivers"
    out.mkdir(parents=True, exist_ok=True)
    rep.make_report(df, g, out, ctx.log)
    return {"rows": len(df), "bytes": 0}

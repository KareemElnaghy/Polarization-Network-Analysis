"""Job R1, gt1_robustness: robustness of ground truth #1 (main labels, shared venue, full week, all weeks).

One unit per week computes: the excess with thread-bootstrap bands; a stratified null (labels shuffled among users
with the same home ``subreddit x day x thread-size`` cell); separate r/politics and r/PoliticalDiscussion values;
threads with at least N labelled users per side; gold labels, gold with 20% of labels flipped; 20 random thread
halves; and the series without the top 1% most active users. The finalise step makes the event tests, the
split-half reliability, plots and ``results/gt1_robustness/summary.md``. Existing outputs are never overwritten.
"""

from __future__ import annotations

import gc
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.gt_runs import gt1_robust as R
from src.gt_runs.config import gt_cfg, gt_path, labels_cfg, venues
from src.gt_runs.labels import heavy_set, load_side_map
from src.prep_jobs.config import data_path, runs_dir
from src.prep_jobs.jobs.clean_edges_and_sampling import weeks_of
from src.prep_jobs.util import read_json, write_json, write_parquet

NAME = "gt1_robustness"
DEADLINE_CUTS: list[str] = []
OUT_NAME = "gt1_robustness"


def phases(cfg):
    mod = "src.prep_jobs.jobs.gt1_robustness"
    return [
        {"name": "weeks", "fn": f"{mod}:week_unit", "units": [{"id": w, "weight": 100} for w in weeks_of(cfg)]},
        {"name": "finalise", "fn": f"{mod}:finalise", "units": [{"id": "finalise", "weight": 50}], "workers": 1,
         "needs_all_prior": True},
    ]


def out_dir(g):
    return gt_path(g, "results").parent / OUT_NAME


def week_rows(cfg, week, subs, side_map, heavy):
    f = data_path(cfg, "thread_user") / f"week={week}" / "part.parquet"
    tu = pq.read_table(f, columns=["subreddit", "link_id", "author", "n_comments"]).to_pandas()
    tu = tu[tu["subreddit"].isin(subs)]
    tu = tu.assign(side=tu["author"].map(side_map))
    tu = tu[tu["side"].notna()]
    if heavy:
        tu = tu[~tu["author"].isin(heavy)]
    rows = pd.DataFrame({"thread": tu["link_id"].astype(str), "user": tu["author"].astype(str), "side": tu["side"].map({"dem": 0, "rep": 1}),
                         "sub": tu["subreddit"].astype(str), "n_comments": tu["n_comments"]})
    return rows.drop_duplicates(["thread", "user"])


def thread_days(cfg, week) -> pd.Series:
    """Day (days since epoch) of the earliest reply edge of each thread this week; threads without edges are missing."""
    f = data_path(cfg, "edges_clean") / f"week={week}" / "part.parquet"
    e = pq.read_table(f, columns=["link_id", "created_utc"]).to_pandas()
    day = pd.to_datetime(e["created_utc"], utc=True).dt.strftime("%Y%m%d").astype(int)
    return day.groupby(e["link_id"].astype(str)).min()


def week_unit(ctx, unit):
    cfg, g, lab, week = ctx.cfg, gt_cfg(), labels_cfg(), unit["id"]
    p = g["gt1_robust"]
    min_users = lab["min_users_per_side"]
    root = gt_path(g, "labels")
    subs = venues(g)["shared"]
    heavy = heavy_set(root, week) if lab["exclude_heavy_posters"] else None
    main_map, gold_map = load_side_map(root, "shared", "main"), load_side_map(root, "shared", "gold")
    rng = np.random.default_rng(g["seed"] + zlib.crc32(week.encode()))
    rows = week_rows(cfg, week, subs, main_map, heavy)
    out: dict = {"week_start": week}
    cell = R.Cell(rows)
    out.update({"n_dem": cell.n_dem, "n_rep": cell.n_rep, "n_threads": len(cell.threads)})
    out.update({f"main_{k}": v for k, v in cell.excess(rng, p["n_perm"], min_users).items()})
    ctx.report(fraction=0.1)
    out.update(R.bootstrap(cell, rng, p["n_boot"], p["boot_perm"], min_users))
    ctx.report(fraction=0.2)

    # (b i) null that shuffles labels within subreddit x day x thread-size cells
    tstrata = R.thread_strata(rows, thread_days(cfg, week))
    out["n_strata"] = int(tstrata.nunique())
    strata = R.home_strata(rows, tstrata, cell)
    out.update({f"strat_{k}": v for k, v in cell.excess(rng, p["n_perm"], min_users, strata=strata).items()})
    # (b ii) one subreddit at a time
    for s in subs:
        c = R.Cell(rows[rows["sub"] == s])
        out.update({f"sub_{s}_{k}": v for k, v in c.excess(rng, p["n_perm_small"], min_users).items() if k in ("excess", "observed")})
        out[f"sub_{s}_n_dem"], out[f"sub_{s}_n_rep"] = c.n_dem, c.n_rep
    # (b iii) threads with at least N labelled users on each side
    per = rows.groupby(["thread", "side"]).size().unstack(fill_value=0).reindex(columns=[0, 1], fill_value=0)
    big = per.index[(per[0] >= p["min_side_thread"]) & (per[1] >= p["min_side_thread"])]
    c = R.Cell(rows[rows["thread"].isin(big)])
    out.update({f"big_{k}": v for k, v in c.excess(rng, p["n_perm_small"], min_users).items() if k in ("excess", "observed")})
    out["big_threads"], out["big_n_dem"], out["big_n_rep"] = len(big), c.n_dem, c.n_rep
    ctx.report(fraction=0.4)

    # (c) gold labels and gold with a fraction flipped
    grows = week_rows(cfg, week, subs, gold_map, heavy)
    gc_ = R.Cell(grows)
    out.update({f"gold_{k}": v for k, v in gc_.excess(rng, p["n_perm_small"], min_users).items() if k in ("excess", "observed", "null_mean")})
    out["gold_n_dem"], out["gold_n_rep"] = gc_.n_dem, gc_.n_rep
    flips = []
    for _ in range(p["flips"]):
        fc = R.Cell(R.flip_sides(grows, rng, p["flip_frac"]))
        flips.append(fc.excess(rng, p["flip_perm"], min_users)["excess"])
    out["gold_flip_excess"] = float(np.nanmean(flips)) if not np.all(np.isnan(flips)) else np.nan
    ctx.report(fraction=0.6)

    # (d) split-half reliability (random halves of threads) and the series without the top users
    halves = []
    th = rows["thread"].to_numpy()
    uniq = np.unique(th)
    for _ in range(p["n_splits"]):
        in_a = dict(zip(uniq, rng.random(len(uniq)) < 0.5))
        m = np.fromiter((in_a[t] for t in th), dtype=bool, count=len(th))
        halves.append([R.Cell(rows[m]).excess(rng, p["split_perm"], min_users)["excess"],
                       R.Cell(rows[~m]).excess(rng, p["split_perm"], min_users)["excess"]])
    out["halves"] = [[None if np.isnan(v) else v for v in h] for h in halves]
    act = rows.groupby("user")["n_comments"].sum()
    cut = act.quantile(1 - p["top_frac"])
    top = set(act.index[act > cut])
    c = R.Cell(rows[~rows["user"].isin(top)])
    out.update({f"notop_{k}": v for k, v in c.excess(rng, p["n_perm_small"], min_users).items() if k in ("excess", "observed")})
    out["notop_removed"] = len(top)

    clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in out.items()}
    write_json(clean, runs_dir(cfg) / NAME / "work" / f"{week}.json")
    del rows, grows, cell
    gc.collect()
    return {"rows": 1, "bytes": 0}


def finalise(ctx, unit):
    from src.prep_jobs.jobs import gt1_robustness_report as rep

    cfg, g = ctx.cfg, gt_cfg()
    files = [read_json(f) for f in sorted((runs_dir(cfg) / NAME / "work").glob("*.json"))]
    out = out_dir(g)
    out.mkdir(parents=True, exist_ok=True)
    weekly = pd.DataFrame([{k: v for k, v in f.items() if k != "halves"} for f in files]).set_index("week_start").sort_index()
    halves = {f["week_start"]: f["halves"] for f in files}
    df = weekly.reset_index()
    size = write_parquet(pa.Table.from_pandas(df, preserve_index=False), out / "robustness_weekly.parquet")
    rep.make_report(weekly, halves, g, out, ctx.log)
    return {"rows": len(df), "bytes": size}

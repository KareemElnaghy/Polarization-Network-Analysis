"""Job T1, gt1_dissimilarity: the weekly thread dissimilarity index (ground truth #1).

One unit per week. For every (venue, label set, sample version) the labelled users active in the venue's
subreddits that week (from ``thread_user``), minus heavy posters, are compared across threads with
``src.ground_truth.dissimilarity``; sample versions keep only the sampled threads. Cells with too few
labelled users on a side are NaN with ``low_n`` set and are kept. Output ``results/ground_truth/gt1_weekly.parquet``.
"""

from __future__ import annotations

import gc

import pandas as pd
import pyarrow as pa

from src.gt_runs.cells import gt1_cell, read_thread_user, sampled_links
from src.gt_runs.config import gt_cfg, gt_path, label_sets, labels_cfg, venues
from src.gt_runs.labels import heavy_set, load_side_map
from src.prep_jobs.config import runs_dir
from src.prep_jobs.jobs.clean_edges_and_sampling import weeks_of
from src.prep_jobs.util import read_json, write_json, write_parquet

NAME = "gt1_dissimilarity"
DEADLINE_CUTS: list[str] = []
COLUMNS = ["week_start", "venue", "label_set", "sample", "n_dem", "n_rep", "n_threads", "n_threads_both",
           "observed", "null_mean", "null_sd", "excess", "low_n"]


def phases(cfg):
    mod = "src.prep_jobs.jobs.gt1_dissimilarity"
    return [
        {"name": "weeks", "fn": f"{mod}:week_unit", "units": [{"id": w, "weight": 100} for w in weeks_of(cfg)]},
        {"name": "finalise", "fn": f"{mod}:finalise", "units": [{"id": "finalise", "weight": 50}], "workers": 1,
         "needs_all_prior": True},
    ]


def week_cells(cfg, g, lab, week, tu, heavy, links) -> list[dict]:
    root = gt_path(g, "labels")
    rows = []
    for venue, subs in venues(g).items():
        for label_set in label_sets(g, "gt1"):
            side_map = load_side_map(root, venue, label_set)
            samples = ["full"] + ([r for r in g["sample_rates"]] if label_set == "main" else [])
            for sample in samples:
                r = gt1_cell(tu, subs, side_map, heavy, None if sample == "full" else links[sample],
                             g["gt1"]["n_perm"], g["seed"], lab["min_users_per_side"])
                rows.append({"week_start": week, "venue": venue, "label_set": label_set, "sample": str(sample), **r})
    return rows


def week_unit(ctx, unit):
    cfg, g, lab, week = ctx.cfg, gt_cfg(), labels_cfg(), unit["id"]
    tu = read_thread_user(cfg, week)
    heavy = heavy_set(gt_path(g, "labels"), week) if lab["exclude_heavy_posters"] else None
    links = sampled_links(cfg, week, g["sample_rates"])
    rows = week_cells(cfg, g, lab, week, tu, heavy, links)
    write_json(rows, runs_dir(cfg) / NAME / "work" / f"{week}.json")
    del tu, links
    gc.collect()
    return {"rows": len(rows), "bytes": 0}


def finalise(ctx, unit):
    cfg, g = ctx.cfg, gt_cfg()
    rows = [r for f in sorted((runs_dir(cfg) / NAME / "work").glob("*.json")) for r in read_json(f)]
    df = pd.DataFrame(rows)[COLUMNS]
    size = write_parquet(pa.Table.from_pandas(df, preserve_index=False), gt_path(g, "results") / "gt1_weekly.parquet")
    ctx.log(f"{len(df)} cells; low_n {int(df['low_n'].sum())}")
    return {"rows": len(df), "bytes": size}

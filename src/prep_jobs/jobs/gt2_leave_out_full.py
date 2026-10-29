"""Job T2, gt2_leave_out_full: the weekly leave-out estimator on the full week (ground truth #2).

One unit per week; only one week's phrase counts are in memory. For every (venue, label set) the users x
phrases matrix is built from ``phrase_counts`` (summed over the venue's subreddits, labelled users with at
least ``min_phrases`` in-vocabulary phrases that week, heavy posters excluded) after dropping phrases with a
contraction fragment (``exclude_tokens``), then ``leave_out_estimator`` runs with a shuffled-label null and,
for main / shared only, bootstrap bands. Output ``results/ground_truth/gt2_weekly_full.parquet``.
"""

from __future__ import annotations

import gc

import pandas as pd
import pyarrow as pa

from src.gt_runs.cells import PhraseWeek, gt2_cell, keep_phrase_mask, read_phrase_week
from src.gt_runs.config import gt_cfg, gt_path, label_sets, labels_cfg, venues
from src.gt_runs.labels import AuthorIndex, heavy_set, load_side_map
from src.prep_jobs.config import runs_dir
from src.prep_jobs.jobs.clean_edges_and_sampling import weeks_of
from src.prep_jobs.jobs.phrase_counts import load_authors
from src.prep_jobs.util import read_json, write_json, write_parquet

NAME = "gt2_leave_out_full"
DEADLINE_CUTS = ["n_perm 100 -> 50 (never for main / shared / full)"]
COLUMNS = ["week_start", "venue", "label_set", "n_dem", "n_rep", "pi", "null_mean", "null_sd", "boot_lo", "boot_hi", "low_n"]


def phases(cfg):
    mod = "src.prep_jobs.jobs.gt2_leave_out_full"
    return [
        {"name": "weeks", "fn": f"{mod}:week_unit", "units": [{"id": w, "weight": 100} for w in weeks_of(cfg)]},
        {"name": "finalise", "fn": f"{mod}:finalise", "units": [{"id": "finalise", "weight": 50}], "workers": 1,
         "needs_all_prior": True},
    ]


def week_cells(cfg, g, lab, week, pw: PhraseWeek, aidx: AuthorIndex, keep, heavy_vec, n_phrases, filtered=True) -> list[dict]:
    root = gt_path(g, "labels")
    p = g["gt2"]
    rows = []
    for venue, subs in venues(g).items():
        a, ph, c = pw.rows(subs)
        for label_set in label_sets(g, "gt2_full"):
            side_vec = aidx.side_vector(load_side_map(root, venue, label_set))
            boot = p["n_boot"] if (venue == "shared" and label_set == "main") else 0
            r = gt2_cell(a, ph, c, side_vec, keep if filtered else None, heavy_vec, n_phrases, p["min_phrases"],
                         p["n_perm"], boot, g["seed"], lab["min_users_per_side"])
            rows.append({"week_start": week, "venue": venue, "label_set": label_set, **r})
    return rows


def week_unit(ctx, unit):
    cfg, g, lab, week = ctx.cfg, gt_cfg(), labels_cfg(), unit["id"]
    keep, dropped = keep_phrase_mask(cfg, g["gt2"]["exclude_tokens"])
    aidx = AuthorIndex(load_authors(cfg)[0])
    heavy = heavy_set(gt_path(g, "labels"), week) if lab["exclude_heavy_posters"] else None
    heavy_vec = aidx.flag_vector(heavy) if heavy else None
    pw = PhraseWeek(read_phrase_week(cfg, week), aidx)
    ctx.report(fraction=0.3)
    rows = week_cells(cfg, g, lab, week, pw, aidx, keep, heavy_vec, len(keep), True)
    write_json({"rows": rows, "phrases_dropped": dropped, "vocab": int(len(keep))}, runs_dir(cfg) / NAME / "work" / f"{week}.json")
    del pw
    gc.collect()
    return {"rows": len(rows), "bytes": 0}


def finalise(ctx, unit):
    cfg, g = ctx.cfg, gt_cfg()
    files = [read_json(f) for f in sorted((runs_dir(cfg) / NAME / "work").glob("*.json"))]
    df = pd.DataFrame([r for f in files for r in f["rows"]])[COLUMNS]
    size = write_parquet(pa.Table.from_pandas(df, preserve_index=False), gt_path(g, "results") / "gt2_weekly_full.parquet")
    ctx.log(f"{len(df)} cells; low_n {int(df['low_n'].sum())}; contraction filter dropped {files[0]['phrases_dropped']:,} "
            f"of {files[0]['vocab']:,} vocabulary phrases")
    return {"rows": len(df), "bytes": size, "phrases_dropped": files[0]["phrases_dropped"]}

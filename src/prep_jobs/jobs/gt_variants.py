"""Job T5, gt_variants: how much the main / shared / full ground truths move under changes to the pipeline.

Per week, GT#1 and GT#2 for main / shared / full with: heavy posters kept (``exclude_heavy_posters: false``);
for GT#2 also no contraction filter, and (an addition to the spec, flagged in NOTES) a per-user cap on
in-vocabulary phrases, because very active users dominate the pooled frequencies. Then Spearman between each
variant series and the main run, mean absolute difference, and the same on weeks with above-median n.
Output ``results/ground_truth/variants_weekly.parquet`` and ``variants_summary.csv``.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.stats import spearmanr

from src.gt_runs.cells import PhraseWeek, gt1_cell, gt2_cell, keep_phrase_mask, read_phrase_week, read_thread_user
from src.gt_runs.config import gt_cfg, gt_path, labels_cfg, venues
from src.gt_runs.labels import AuthorIndex, heavy_set, load_side_map
from src.prep_jobs.config import runs_dir
from src.prep_jobs.jobs.clean_edges_and_sampling import weeks_of
from src.prep_jobs.jobs.phrase_counts import load_authors
from src.prep_jobs.util import read_json, write_json, write_parquet

NAME = "gt_variants"
DEADLINE_CUTS = ["drop this job first"]


def phases(cfg):
    mod = "src.prep_jobs.jobs.gt_variants"
    return [
        {"name": "weeks", "fn": f"{mod}:week_unit", "units": [{"id": w, "weight": 100} for w in weeks_of(cfg)]},
        {"name": "finalise", "fn": f"{mod}:finalise", "units": [{"id": "finalise", "weight": 50}], "workers": 1,
         "needs_all_prior": True},
    ]


def week_unit(ctx, unit):
    cfg, g, lab, week = ctx.cfg, gt_cfg(), labels_cfg(), unit["id"]
    subs = venues(g)["shared"]
    root = gt_path(g, "labels")
    side = load_side_map(root, "shared", "main")
    heavy = heavy_set(root, week)
    rows = []

    tu = read_thread_user(cfg, week)
    r = gt1_cell(tu, subs, side, None, None, g["gt1"]["n_perm"], g["seed"], lab["min_users_per_side"])
    rows.append({"week_start": week, "variant": "heavy_posters_kept", "gt": "gt1", "n_dem": r["n_dem"], "n_rep": r["n_rep"],
                 "value": r["excess"], "null_mean": r["null_mean"], "low_n": r["low_n"]})
    del tu

    aidx = AuthorIndex(load_authors(cfg)[0])
    keep, _ = keep_phrase_mask(cfg, g["gt2"]["exclude_tokens"])
    pw = PhraseWeek(read_phrase_week(cfg, week), aidx)
    a, p, c = pw.rows(subs)
    side_vec, heavy_vec = aidx.side_vector(side), aidx.flag_vector(heavy)
    p2 = g["gt2"]
    runs = [("heavy_posters_kept", keep, None, None), ("no_contraction_filter", None, heavy_vec, None)]
    runs += [(f"user_cap_{cap}", keep, heavy_vec, cap) for cap in g["variants"].get("user_caps", [])]
    for name, kp, hv, cap in runs:
        r = gt2_cell(a, p, c, side_vec, kp, hv, len(keep), p2["min_phrases"], p2["n_perm"], 0, g["seed"],
                     lab["min_users_per_side"], cap=cap)
        rows.append({"week_start": week, "variant": name, "gt": "gt2", "n_dem": r["n_dem"], "n_rep": r["n_rep"], "value": r["pi"],
                     "null_mean": r["null_mean"], "low_n": r["low_n"]})
    write_json(rows, runs_dir(cfg) / NAME / "work" / f"{week}.json")
    del pw
    gc.collect()
    return {"rows": len(rows), "bytes": 0}


def compare(x: pd.Series, y: pd.Series) -> dict:
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return {"weeks": int(ok.sum()), "spearman": np.nan, "mean_abs_diff": np.nan, "mean_diff": np.nan}
    return {"weeks": int(ok.sum()), "spearman": float(spearmanr(x[ok], y[ok])[0]),
            "mean_abs_diff": float((x[ok] - y[ok]).abs().mean()), "mean_diff": float((x[ok] - y[ok]).mean())}


def finalise(ctx, unit):
    cfg, g = ctx.cfg, gt_cfg()
    res = gt_path(g, "results")
    var = pd.DataFrame([r for f in sorted((runs_dir(cfg) / NAME / "work").glob("*.json")) for r in read_json(f)])
    write_parquet(pa.Table.from_pandas(var, preserve_index=False), res / "variants_weekly.parquet")
    g1 = pd.read_parquet(res / "gt1_weekly.parquet")
    g1 = g1[(g1.venue == "shared") & (g1.label_set == "main") & (g1["sample"] == "full")].set_index("week_start")
    g2 = pd.read_parquet(res / "gt2_weekly_full.parquet")
    g2 = g2[(g2.venue == "shared") & (g2.label_set == "main")].set_index("week_start")
    main = {"gt1": g1["excess"], "gt2": g2["pi"]}
    n_min = {"gt1": g1[["n_dem", "n_rep"]].min(axis=1), "gt2": g2[["n_dem", "n_rep"]].min(axis=1)}
    out = []
    for (name, gt), grp in var.groupby(["variant", "gt"]):
        s = grp.set_index("week_start")["value"]
        row = {"variant": name, "gt": gt, **compare(s, main[gt].reindex(s.index))}
        above = n_min[gt][n_min[gt] > n_min[gt].median()].index
        ab = compare(s.reindex(above), main[gt].reindex(above))
        row.update({"spearman_above_median_n": ab["spearman"], "mean_abs_diff_above_median_n": ab["mean_abs_diff"]})
        out.append(row)
    summ = pd.DataFrame(out)
    summ.to_csv(res / "variants_summary.csv", index=False)
    ctx.log("\n" + summ.round(3).to_string(index=False))
    return {"rows": len(var), "bytes": 0}

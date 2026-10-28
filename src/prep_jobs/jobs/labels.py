"""Job L, labels: apply config/labels.yaml to Job G's per-user counts and write the per-venue label sets.

Output ``data/processed/labels/venue=<venue>/set=<set>.parquet`` (author, side, share, n_comments),
``summary.csv`` (users per side per venue per set) and ``heavy_posters.parquet`` (week_start, author: users
who sent more than ``heavy_poster_threshold`` replies that week, recomputed from the raw edge table
because Job C did not save the list). ``data/processed/camp_recovery/labels.parquet`` holds the shared-venue
sides side by side for evaluating Pipeline B only; nothing the pipelines read may use it.
Only the sets that run (main, gold, and ``label_sets_extra``) are built; broad and strict are not.
"""

from __future__ import annotations

import shutil

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.gt_runs.config import CONFIG_DIR, gt_cfg, gt_path, label_sets, labels_cfg, venues
from src.gt_runs.labels import gold_sides, label_path, posting_sides
from src.prep_jobs.config import data_path
from src.prep_jobs.data import party_map
from src.prep_jobs.util import week_label, write_parquet

NAME = "labels"
DEADLINE_CUTS: list[str] = []


def phases(cfg):
    return [{"name": "build", "fn": "src.prep_jobs.jobs.labels:build", "units": [{"id": "all", "weight": 100}], "workers": 1}]


def heavy_posters(cfg, threshold: int) -> pd.DataFrame:
    parts = []
    for m in cfg["months"]:
        t = pq.read_table(data_path(cfg, "edges") / f"edges_{m}.parquet", columns=["source", "week_start", "author_week_replies"])
        t = t.filter(pc.greater(t["author_week_replies"], threshold))
        parts.append(t.group_by(["source", "week_start"]).aggregate([]))
    t = pa.concat_tables(parts).group_by(["source", "week_start"]).aggregate([])
    df = t.to_pandas()
    return pd.DataFrame({"week_start": df["week_start"].map(week_label), "author": df["source"].astype(str)})


def build(ctx, unit):
    cfg, g, lab = ctx.cfg, gt_cfg(), labels_cfg()
    root = gt_path(g, "labels")
    counts = pq.read_table(data_path(cfg, "label_versions") / "user_counts.parquet").to_pandas()
    party = party_map(cfg)
    decl = pd.read_parquet(data_path(cfg, "user_profiles") / "user_subreddit_declarations.parquet",
                           columns=["author", "kind", "label"])
    gold = gold_sides(decl)
    rows, notes = [], []
    for venue, subs in venues(g).items():
        in_counts = [s for s in subs if s in set(counts["subreddit"])]
        for name in sorted({s for st in ("gt1", "gt2_full", "gt2_samples") for s in label_sets(g, st)}):
            if name == "gold":
                sides = gold
            else:
                sides = posting_sides(counts, party, lab[name], venue_subs=in_counts)
            sides = sides.reset_index(drop=True)
            assert sides["author"].is_unique, f"a user has two sides in venue={venue} set={name}"
            write_parquet(pa.Table.from_pandas(sides, preserve_index=False), label_path(root, venue, name))
            c = sides["side"].value_counts()
            rows.append({"venue": venue, "label_set": name, "dem": int(c.get("dem", 0)), "rep": int(c.get("rep", 0)),
                         "venue_subs_in_job_G_counts": ",".join(in_counts) or "none (labels already exclude the venue)"})
            print(f"{venue:22s} {name:6s} dem {rows[-1]['dem']:>8,} rep {rows[-1]['rep']:>8,}")
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "summary.csv", index=False)

    # shared-venue sides side by side, for evaluating Pipeline B only
    shared = None
    for name in sorted({s for st in ("gt1", "gt2_full", "gt2_samples") for s in label_sets(g, st)}):
        t = pd.read_parquet(label_path(root, "shared", name), columns=["author", "side"]).rename(columns={"side": f"side_{name}"})
        shared = t if shared is None else shared.merge(t, on="author", how="outer")
    write_parquet(pa.Table.from_pandas(shared, preserve_index=False), gt_path(g, "camp_recovery") / "labels.parquet")

    heavy = heavy_posters(cfg, g["heavy_poster_threshold"])
    write_parquet(pa.Table.from_pandas(heavy, preserve_index=False), root / "heavy_posters.parquet")
    out = gt_path(g, "results")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_DIR / "labels.yaml", out / "labels_used.yaml")
    shutil.copy2(CONFIG_DIR / "gt_runs.yaml", out / "gt_runs_used.yaml")
    print(f"heavy-poster user-weeks: {len(heavy):,} ({heavy['author'].nunique():,} users)")
    return {"rows": len(summary), "bytes": 0}

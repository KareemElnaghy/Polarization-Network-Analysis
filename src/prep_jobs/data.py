"""Readers shared by the prep jobs: filtered comment batches, party maps, declarations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.common.text_filters import privacy_overwritten_mask, removed_mask
from src.prep_jobs.config import data_path


def comment_file(cfg: dict, month: str) -> Path:
    return data_path(cfg, "comments") / f"comments_{month}.parquet"


def flags_file(cfg: dict, month: str) -> Path:
    return data_path(cfg, "comment_flags") / f"distinguished_{month}.parquet"


def distinguished_ids(cfg: dict, month: str) -> pa.Array:
    """Ids of comments by moderators/admins in ``month`` (from the ``comment_flags`` job)."""
    path = flags_file(cfg, month)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run the comment_flags job first")
    return pq.read_table(path, columns=["id"])["id"].combine_chunks()


def exclusion_keep_mask(cfg: dict, table: pa.Table, dist_ids: pa.Array | None, stats: Counter) -> pa.ChunkedArray:
    """Rows to keep: not removed/deleted, not a privacy-script overwrite, not distinguished."""
    flt = cfg["text_filters"]
    body = table["body"]
    removed = removed_mask(body)
    privacy = privacy_overwritten_mask(body, flt["privacy_overwrite_regex"])
    bad = pc.or_(removed, privacy)
    stats["removed_or_deleted"] += pc.sum(removed).as_py() or 0
    stats["privacy_overwritten"] += pc.sum(privacy).as_py() or 0
    if flt["exclude_distinguished"] and dist_ids is not None:
        dist = pc.is_in(table["id"], dist_ids)
        stats["distinguished"] += pc.sum(dist).as_py() or 0
        bad = pc.or_(bad, dist)
    return pc.invert(bad)


def iter_comments(cfg: dict, month: str, columns: list[str], batch_rows: int, stats: Counter,
                  exclusions: bool = True) -> Iterator[pa.Table]:
    """Batches of ``columns`` from one month's comments, with the shared exclusions applied.

    ``stats`` accumulates ``rows`` (read), ``kept`` and the excluded counts. ``body`` and ``id`` are read
    for the exclusion test even when not requested.
    """
    need = list(dict.fromkeys(columns + (["id", "body"] if exclusions else [])))
    dist = distinguished_ids(cfg, month) if exclusions and cfg["text_filters"]["exclude_distinguished"] else None
    pf = pq.ParquetFile(comment_file(cfg, month))
    for batch in pf.iter_batches(batch_size=batch_rows, columns=need):
        t = pa.Table.from_batches([batch])
        stats["rows"] += t.num_rows
        if exclusions:
            t = t.filter(exclusion_keep_mask(cfg, t, dist, stats))
        stats["kept"] += t.num_rows
        yield t.select(columns)


def month_rows(cfg: dict, month: str) -> int:
    return pq.ParquetFile(comment_file(cfg, month)).metadata.num_rows


def party_map(cfg: dict) -> dict[str, str]:
    """``subreddit -> 'dem' | 'rep'`` from subreddits_metadata.json."""
    out = {}
    for line in (data_path(cfg, "raw") / "subreddits_metadata.json").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["party"]:
                out[row["subreddit"]] = row["party"]
    return out


def supporter_subs(cfg: dict, jcfg: dict) -> list[str]:
    """Supporter-community subs: the configured list, or all party-tagged subs minus the exclusions."""
    if jcfg.get("supporter_subs"):
        return sorted(jcfg["supporter_subs"])
    return sorted(s for s in party_map(cfg) if s not in set(jcfg["supporter_exclude"]))


def declared_lean(cfg: dict) -> pa.Table | None:
    """Per author: single declared label (left/right/third) or 'conflict', from the earlier declaration table."""
    path = data_path(cfg, "user_profiles") / "user_subreddit_declarations.parquet"
    if not path.exists():
        return None
    import pandas as pd
    d = pd.read_parquet(path, columns=["author", "label"])
    s = d.groupby("author")["label"].agg(lambda x: next(iter(set(x))) if len(set(x)) == 1 else "conflict")
    return pa.table({"author": s.index.astype(str).tolist(), "declared": s.tolist()})


# ------------------------------------------------------------------ score versions
def score_histogram(cfg: dict, month: str, subs: list[str], batch_rows: int, stats: Counter) -> dict[str, Counter]:
    """``{sub: Counter(score -> comments)}`` for ``subs`` in one month, after the shared exclusions."""
    hist: dict[str, Counter] = {s: Counter() for s in subs}
    for t in iter_comments(cfg, month, ["subreddit", "score"], batch_rows, stats):
        t = t.set_column(0, "subreddit", pc.cast(t["subreddit"], pa.string()))
        t = t.filter(pc.is_in(t["subreddit"], pa.array(subs)))
        if t.num_rows:
            g = t.group_by(["subreddit", "score"]).aggregate([("score", "count")])
            for s, sc, n in zip(g["subreddit"].to_pylist(), g["score"].to_pylist(), g["score_count"].to_pylist()):
                hist[s][sc] += n
    return hist


def median_from_histogram(counter: Counter) -> float:
    total = sum(counter.values())
    if not total:
        return float("nan")
    items = sorted(counter.items())
    def at(rank):  # value at 0-based rank
        run = 0
        for value, n in items:
            run += n
            if rank < run:
                return value
    return (at((total - 1) // 2) + at(total // 2)) / 2


def version_columns(score: pa.ChunkedArray, sub: pa.ChunkedArray, medians: dict[str, float]) -> dict[str, pa.Array]:
    """Per-comment contribution to each count version (see jobs.within_camp_setup / label_versions)."""
    s = pc.cast(score, pa.int64())
    names = list(medians)
    med = pa.array([medians[n] for n in names], pa.float64())
    row_med = pc.take(med, pc.fill_null(pc.index_in(sub, pa.array(names)), 0))
    return {
        "all": pa.array([1] * len(s), pa.int64()),
        "score_ge_1": pc.cast(pc.greater_equal(s, 1), pa.int64()),
        "score_ge_2": pc.cast(pc.greater_equal(s, 2), pa.int64()),
        "above_sub_median": pc.cast(pc.greater(pc.cast(s, pa.float64()), row_med), pa.int64()),
        "weighted": pc.ln(pc.add(pc.cast(pc.max_element_wise(s, 0), pa.float64()), 1.0)),
    }


def label_counts(rep: "pd.Series", dem: "pd.Series", min_comments: float, share_hi: float) -> "pd.DataFrame":
    """Label from two count series: 'right' if rep share >= share_hi, 'left' if <= 1 - share_hi, else 'mixed'."""
    import pandas as pd
    n = rep + dem
    share = rep / n.where(n > 0)
    lab = pd.Series("mixed", index=n.index)
    lab[share >= share_hi] = "right"
    lab[share <= 1 - share_hi] = "left"
    lab[~(n >= min_comments)] = "unlabelled"
    return pd.DataFrame({"n": n, "share": share, "label": lab})

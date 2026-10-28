"""Build interaction edges from parsed Reddit comments.

An edge is one reply to a comment (``parent_id`` starting with ``t1_``), directed
from the replier (``source``) to the parent comment's author (``target``).
Self-replies and replies whose parent comment is not among the kept comments are
dropped and counted by reason. Nothing else is dropped; questionable edges are
flagged instead:

    reply_removed / parent_removed  reply or parent text is ``[removed]``/``[deleted]``
    author_week_replies             replies by ``source`` in the edge's weekly window
    body_repeat_count               edges in the whole run with an identical reply text

Usage:
    python src/dataset/build_edges.py 2016-01 2016-02 ... 2016-12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if __package__ in (None, ""):  # run as a script: make ``src`` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.common.text_filters import REMOVED_TEXTS  # noqa: E402
from src.dataset.windowing import assign_windows  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMENTS_DIR = PROJECT_ROOT / "data" / "interim" / "comments"
EDGES_DIR = PROJECT_ROOT / "data" / "interim" / "edges"

PARENT_NOT_IN_DATA = "parent_not_in_data"  # posted before the first parsed month
SELF_REPLY = "self_reply"


def parent_key(parent_id: pa.ChunkedArray) -> pa.ChunkedArray:
    """Strip the ``t1_`` prefix so a parent id matches a comment ``id``."""
    return pc.utf8_slice_codeunits(parent_id, 3)


def text_hash(body: pa.ChunkedArray) -> pa.Array:
    """64-bit hash of each text, used to count identical replies without holding all texts."""
    return pa.array(pd.util.hash_array(body.to_numpy(zero_copy_only=False).astype(object)), pa.uint64())


def build_reply_edges(
    comments: pa.Table,
    parents: pa.Table,
    dropped: pa.Table | None = None,
) -> tuple[pa.Table, Counter]:
    """Convert parsed comments into ``replier -> parent author`` edges.

    ``comments`` holds parsed comments (``parse_comments`` schema); only its replies to
    comments become edges. ``parents`` is the lookup of candidate parent comments
    (``id``, ``author``, ``body``). ``dropped`` (``id``, ``reason``) explains parents that
    were filtered out during parsing; a parent found in neither table is counted as
    ``parent_not_in_data``.

    Returns the edge table and a counter with ``replies``, ``edges``, ``self_reply`` and
    one ``parent_missing:<reason>`` key per missing-parent reason.
    """
    replies = comments.filter(pc.equal(comments["is_reply"], True))
    if pa.types.is_dictionary(replies.schema.field("subreddit").type):  # joins can't carry dictionaries
        replies = replies.set_column(replies.schema.get_field_index("subreddit"), "subreddit",
                                     pc.cast(replies["subreddit"], pa.string()))
    replies = replies.append_column("parent_key", parent_key(replies["parent_id"]))
    lookup = parents.select(["id", "author", "body"]).rename_columns(
        ["parent_key", "target", "parent_body"])
    joined = replies.join(lookup, "parent_key", join_type="left outer")

    losses = Counter(replies=replies.num_rows)
    missing = pc.is_null(joined["target"])
    if pc.any(missing).as_py():
        lost = joined.filter(missing).select(["parent_key"])
        if dropped is not None and dropped.num_rows:
            why = pa.table({"parent_key": dropped["id"], "reason": pc.cast(dropped["reason"], pa.string())})
            lost = lost.join(why, "parent_key", join_type="left outer")
            reasons = pc.cast(lost["reason"], pa.string()).fill_null(PARENT_NOT_IN_DATA)
        else:
            reasons = pa.array([PARENT_NOT_IN_DATA] * lost.num_rows)
        for item in pc.value_counts(reasons).to_pylist():
            losses[f"parent_missing:{item['values']}"] += item["counts"]
        joined = joined.filter(pc.invert(missing))

    self_reply = pc.equal(joined["author"], joined["target"])
    losses[SELF_REPLY] = pc.sum(self_reply).as_py() or 0
    edges = joined.filter(pc.invert(self_reply))
    losses["edges"] = edges.num_rows

    edges = pa.table({
        "comment_id": edges["id"],
        "parent_comment_id": edges["parent_key"],
        "link_id": edges["link_id"],
        "source": edges["author"],
        "target": edges["target"],
        "created_utc": edges["created_utc"],
        "subreddit": edges["subreddit"],
        "body": edges["body"],
        "parent_body": edges["parent_body"],
        "reply_removed": pc.is_in(edges["body"], pa.array(REMOVED_TEXTS)),
        "parent_removed": pc.is_in(edges["parent_body"], pa.array(REMOVED_TEXTS)),
        "body_hash": text_hash(edges["body"]),
    })
    return edges.sort_by("created_utc"), losses


def add_repeat_flags(edges: pa.Table, author_week: pa.Table, body_counts: pa.Table) -> pa.Table:
    """Attach ``author_week_replies`` and ``body_repeat_count`` from global count tables."""
    order = pa.array(range(edges.num_rows), pa.int64())
    out = edges.append_column("_row", order)
    week_type = edges.schema.field("week_start").type  # Parquet stores seconds as ms; align keys
    author_week = author_week.set_column(1, "week_start", author_week["week_start"].cast(week_type))
    out = out.join(author_week, ["source", "week_start"], join_type="left outer")
    out = out.join(body_counts, "body_hash", join_type="left outer")
    return out.sort_by("_row").drop_columns(["_row"])


def count_repeats(edge_files: list[Path]) -> tuple[pa.Table, pa.Table]:
    """Global per-(author, week) reply counts and per-text repeat counts across ``edge_files``."""
    slim = pa.concat_tables(pq.read_table(f, columns=["source", "week_start", "body_hash"]) for f in edge_files)
    author_week = slim.group_by(["source", "week_start"]).aggregate([("body_hash", "count")])
    author_week = author_week.rename_columns(["source", "week_start", "author_week_replies"])
    body_counts = slim.group_by("body_hash").aggregate([("source", "count")])
    body_counts = body_counts.rename_columns(["body_hash", "body_repeat_count"])
    return author_week, body_counts


def _load_parent_index(comment_files: dict[str, Path]) -> pa.Table:
    """``id``, ``author``, ``month`` for every parsed comment (no text, so it fits in memory)."""
    parts = []
    for month, path in comment_files.items():
        t = pq.read_table(path, columns=["id", "author"])
        parts.append(t.append_column("month", pa.array([month] * t.num_rows, pa.string()).dictionary_encode()))
    return pa.concat_tables(parts)


def _parents_with_text(needed: pa.Table, comment_files: dict[str, Path]) -> pa.Table:
    """Fetch ``body`` for the parent ids in ``needed`` (``id``, ``author``, ``month``)."""
    parts = []
    months = pc.unique(pc.cast(needed["month"], pa.string())).to_pylist()
    for month in sorted(months):
        ids = needed.filter(pc.equal(pc.cast(needed["month"], pa.string()), month))
        text = pq.read_table(comment_files[month], columns=["id", "body"],
                             filters=pc.field("id").isin(ids["id"].combine_chunks()))
        parts.append(ids.select(["id", "author"]).join(text, "id"))
    return pa.concat_tables(parts) if parts else pa.table({"id": [], "author": [], "body": []},
                                                          schema=pa.schema([("id", pa.string()),
                                                                            ("author", pa.string()),
                                                                            ("body", pa.string())]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("months", nargs="+", metavar="YYYY-MM")
    ap.add_argument("--comments-dir", type=Path, default=COMMENTS_DIR)
    ap.add_argument("--out-dir", type=Path, default=EDGES_DIR)
    args = ap.parse_args()

    t0 = time.perf_counter()
    comment_files = {m: args.comments_dir / f"comments_{m}.parquet" for m in sorted(args.months)}
    index = _load_parent_index(comment_files)
    dropped_files = [args.comments_dir / "dropped" / f"comments_{m}.dropped.parquet" for m in comment_files]
    dropped = pa.concat_tables(pq.read_table(f, columns=["id", "reason"]) for f in dropped_files if f.exists())
    print(f"parent index: {index.num_rows:,} comments, {dropped.num_rows:,} dropped ids "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "stats").mkdir(exist_ok=True)
    totals: Counter = Counter()
    per_month: dict[str, dict] = {}
    for month, path in comment_files.items():
        t1 = time.perf_counter()
        comments = pq.read_table(path)
        keys = pc.unique(parent_key(comments.filter(pc.equal(comments["is_reply"], True))["parent_id"]))
        needed = index.filter(pc.is_in(index["id"], keys))
        edges, losses = build_reply_edges(comments, _parents_with_text(needed, comment_files), dropped)
        edges = assign_windows(edges)
        pq.write_table(edges, args.out_dir / f"edges_{month}.parquet", compression="zstd")
        per_month[month] = dict(losses)
        totals.update(losses)
        print(f"{month}: {losses['replies']:,} replies -> {losses['edges']:,} edges "
              f"({time.perf_counter() - t1:.0f}s)", flush=True)
        del comments, edges

    # Second pass: counts that span months (weeks cross month boundaries; texts repeat all year).
    edge_files = [args.out_dir / f"edges_{m}.parquet" for m in comment_files]
    author_week, body_counts = count_repeats(edge_files)
    for f in edge_files:
        pq.write_table(add_repeat_flags(pq.read_table(f), author_week, body_counts), f, compression="zstd")

    summary = {"months": list(comment_files), "total": dict(totals), "per_month": per_month}
    (args.out_dir / "stats" / "edges_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"total: {totals['replies']:,} replies -> {totals['edges']:,} edges; "
          f"lost {dict((k, v) for k, v in totals.items() if k not in ('replies', 'edges'))}")
    print(f"wall time {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

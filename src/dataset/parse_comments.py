"""Parsing utilities for raw Reddit comment dumps.

Streams a Politosphere ``comments_YYYY-MM.bz2`` dump line by line, applies the
scope filters from ``docs/decisions.md`` while reading, and writes the kept
comments to Parquet. Nothing is fully decompressed to disk or held in memory.

Outputs, per month, under ``data/interim/comments/``:
    comments_YYYY-MM.parquet                 kept comments
    stats/comments_YYYY-MM.stats.json        line counts per filter
    dropped/comments_YYYY-MM.dropped.parquet ids dropped by the author/language filters

Usage:
    python src/dataset/parse_comments.py 2016-06
    python src/dataset/parse_comments.py 2016-01 2016-02 ... 2016-12 --workers 6
"""

from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import orjson

    _loads = orjson.loads
except ImportError:  # pragma: no cover - orjson is only a speed-up
    _loads = json.loads

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "comments"

NON_US_REGIONS = frozenset({"ca", "eu", "me", "uk", "world"})
DELETED_AUTHOR = "[deleted]"
KEPT_LANGUAGES = frozenset({"en", "un"})  # "un" = undetermined, mostly short comments
REPLY_PREFIX = "t1_"
BATCH_SIZE = 200_000

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("parent_id", pa.string()),
    ("link_id", pa.string()),
    ("author", pa.string()),
    ("subreddit", pa.dictionary(pa.int32(), pa.string())),
    ("created_utc", pa.timestamp("s", tz="UTC")),
    ("body", pa.string()),
    ("score", pa.int32()),
    ("controversiality", pa.int8()),
    ("is_reply", pa.bool_()),
])


@dataclass
class ParseStats:
    """Line counts for one parsed dump; ``dropped`` is keyed by the first failing filter."""

    source: str
    output: str = ""
    lines: int = 0
    kept: int = 0
    kept_replies: int = 0
    dropped: Counter = field(default_factory=Counter)
    seconds: float = 0.0
    output_bytes: int = 0


def load_us_subreddits(metadata_path: str | Path) -> frozenset[str]:
    """Return subreddits from ``subreddits_metadata.json`` not tagged with a non-US region."""
    keep = set()
    with open(metadata_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if row["region"] not in NON_US_REGIONS:
                    keep.add(row["subreddit"])
    return frozenset(keep)


_FLAGGED = re.compile(r'"(?:bot|automoderator)":\s*1\b')


def load_excluded_authors(users_metadata_path: str | Path) -> frozenset[str]:
    """Return pseudonymised authors flagged ``bot`` or ``automoderator``, plus ``[deleted]``."""
    excluded = {DELETED_AUTHOR}
    with open(users_metadata_path, encoding="utf-8") as fh:
        for line in fh:
            if _FLAGGED.search(line):  # cheap pre-check; most lines are all zeros
                row = json.loads(line)
                if row.get("bot") or row.get("automoderator"):
                    excluded.add(row["author"])
    return frozenset(excluded)


def iter_filtered_comments(
    raw_path: str | Path,
    *,
    start: datetime,
    end: datetime,
    subreddits: frozenset[str],
    excluded_authors: frozenset[str],
    stats: ParseStats,
    dropped_sink: list[tuple[str, str, str]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield comments from a ``.bz2`` dump that pass the scope filters.

    Kept: ``start <= created_utc < end``, subreddit in ``subreddits``, author not
    excluded, ``language`` English or undetermined. Top-level comments are kept (as potential
    parents) with ``is_reply = False``; only ``t1_`` parents are replies.

    If ``dropped_sink`` is given, ``(id, author, reason)`` is appended for every comment
    dropped by the author or language filter, so reply edges with a missing parent can be
    explained later. Bot and AutoModerator authors are reported as ``author_bot`` and
    ``[deleted]`` authors as ``author_deleted``.
    """
    lo, hi = int(start.timestamp()), int(end.timestamp())
    drop = stats.dropped
    with bz2.open(raw_path, "rb") as fh:
        for line in fh:
            stats.lines += 1
            try:
                c = _loads(line)
                ts = int(c["created_utc"])
            except (ValueError, KeyError, TypeError):
                drop["malformed"] += 1
                continue
            if not lo <= ts < hi:
                drop["date"] += 1
            elif c["subreddit"] not in subreddits:
                drop["subreddit"] += 1
            elif c["author"] in excluded_authors:
                drop["author"] += 1
                if dropped_sink is not None:
                    reason = "author_deleted" if c["author"] == DELETED_AUTHOR else "author_bot"
                    dropped_sink.append((c["id"], c["author"], reason))
            elif c.get("language") not in KEPT_LANGUAGES:
                drop["language"] += 1
                if dropped_sink is not None:
                    dropped_sink.append((c["id"], c["author"], "language"))
            else:
                is_reply = c["parent_id"].startswith(REPLY_PREFIX)
                stats.kept += 1
                stats.kept_replies += is_reply
                yield {
                    "id": c["id"],
                    "parent_id": c["parent_id"],
                    "link_id": c["link_id"],
                    "author": c["author"],
                    "subreddit": c["subreddit"],
                    "created_utc": ts,
                    "body": c["body"],
                    "score": c["score"],
                    "controversiality": c["controversiality"],
                    "is_reply": is_reply,
                }


def parse_raw_comments(
    raw_path: str | Path,
    out_path: str | Path,
    *,
    start: datetime,
    end: datetime,
    subreddits: frozenset[str],
    excluded_authors: frozenset[str],
    dropped_path: str | Path | None = None,
    batch_size: int = BATCH_SIZE,
) -> ParseStats:
    """Stream ``raw_path``, filter, and write the kept comments to Parquet at ``out_path``.

    If ``dropped_path`` is given, also write ``(id, author, reason)`` for comments dropped by
    the author or language filters there.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    stats = ParseStats(source=str(raw_path), output=str(out_path))
    t0 = time.perf_counter()
    dropped: list[tuple[str, str, str]] | None = [] if dropped_path else None

    batch: list[dict[str, Any]] = []
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:
        for row in iter_filtered_comments(raw_path, start=start, end=end, subreddits=subreddits,
                                          excluded_authors=excluded_authors, stats=stats,
                                          dropped_sink=dropped):
            batch.append(row)
            if len(batch) >= batch_size:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
    tmp.replace(out_path)
    if dropped_path:
        Path(dropped_path).parent.mkdir(parents=True, exist_ok=True)
        ids, authors, reasons = zip(*dropped) if dropped else ((), (), ())
        pq.write_table(pa.table({"id": list(ids), "author": list(authors),
                                 "reason": pa.array(reasons, pa.string()).dictionary_encode()}),
                       dropped_path, compression="zstd")

    stats.seconds = time.perf_counter() - t0
    stats.output_bytes = out_path.stat().st_size
    return stats


def month_bounds(month: str) -> tuple[datetime, datetime]:
    """Return the UTC ``[start, end)`` bounds of a ``YYYY-MM`` month."""
    y, m = map(int, month.split("-"))
    start = datetime(y, m, 1, tzinfo=timezone.utc)
    end = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=timezone.utc)
    return start, end


def _utc_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _parse_month(month: str, raw_dir: Path, out_dir: Path, start: datetime | None, end: datetime | None,
                 subreddits: frozenset[str], excluded: frozenset[str]) -> ParseStats:
    m_start, m_end = month_bounds(month)
    stats = parse_raw_comments(
        raw_dir / f"comments_{month}.bz2",
        out_dir / f"comments_{month}.parquet",
        start=start or m_start, end=end or m_end,
        subreddits=subreddits, excluded_authors=excluded,
        dropped_path=out_dir / "dropped" / f"comments_{month}.dropped.parquet",
    )
    record = asdict(stats)
    record["dropped"] = dict(stats.dropped)
    stats_path = out_dir / "stats" / f"comments_{month}.stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(record, indent=2))
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("months", nargs="+", metavar="YYYY-MM")
    ap.add_argument("--start", type=_utc_date, help="inclusive UTC date; default: each month's start")
    ap.add_argument("--end", type=_utc_date, help="exclusive UTC date; default: each month's end")
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    ap.add_argument("--out-dir", type=Path, default=INTERIM_DIR)
    ap.add_argument("--workers", type=int, default=1, help="months parsed in parallel (one process each)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    subreddits = load_us_subreddits(args.raw_dir / "subreddits_metadata.json")
    excluded = load_excluded_authors(args.raw_dir / "users_metadata.json")
    print(f"{len(subreddits)} US subreddits, {len(excluded)} excluded authors "
          f"(loaded in {time.perf_counter() - t0:.1f}s)", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_parse_month, m, args.raw_dir, args.out_dir, args.start, args.end,
                               subreddits, excluded): m for m in args.months}
        for fut in as_completed(futures):
            stats = fut.result()
            print(f"{futures[fut]}: {stats.lines:,} lines -> {stats.kept:,} kept ({stats.kept_replies:,} replies) "
                  f"in {stats.seconds:.0f}s, {stats.output_bytes / 1e6:.0f} MB; dropped {dict(stats.dropped)}",
                  flush=True)
    print(f"total wall time {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

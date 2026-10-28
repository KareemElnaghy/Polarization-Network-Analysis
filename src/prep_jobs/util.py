"""Helpers shared by the framework and the jobs: atomic writes, memory, weeks, boundary-week folding."""

from __future__ import annotations

import contextlib
import json
import os
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.dataset.windowing import assign_windows


# ------------------------------------------------------------------ files
@contextlib.contextmanager
def atomic_path(final: Path) -> Iterator[Path]:
    """Yield a temporary sibling of ``final``; rename it over ``final`` if the block succeeds."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        yield tmp
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_parquet(table: pa.Table, final: Path, **kwargs) -> int:
    """Write ``table`` atomically (zstd) and return the file size."""
    compression = kwargs.pop("compression", "zstd")
    with atomic_path(final) as tmp:
        pq.write_table(table, tmp, compression=compression, **kwargs)
    return Path(final).stat().st_size


def write_json(obj, final: Path) -> None:
    with atomic_path(final) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def dir_size(path: Path) -> int:
    path = Path(path)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


# ------------------------------------------------------------------ memory
def peak_memory_gb() -> float:
    """Peak resident memory of this process so far, in GB."""
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or getattr(info, "peak_pagefile", None)
        return (peak or info.rss) / 1e9
    except Exception:  # pragma: no cover
        return 0.0


def rss_gb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1e9


# ------------------------------------------------------------------ weeks
def week_label(week_start) -> str:
    """``YYYY-MM-DD`` for a week start given as datetime, date or Timestamp."""
    if isinstance(week_start, datetime):
        return week_start.date().isoformat()
    if isinstance(week_start, date):
        return week_start.isoformat()
    return str(week_start)[:10]


def with_week_start(table: pa.Table, timestamp_column: str = "created_utc") -> pa.Table:
    """Add ``week_start`` with the repo's windowing function (Monday 00:00 UTC); no month column."""
    return assign_windows(table, timestamp_column).drop_columns(["month_start"])


def month_start(month: str) -> datetime:
    y, m = map(int, month.split("-"))
    return datetime(y, m, 1, tzinfo=timezone.utc)


# ------------------------------------------------------------------ boundary-week folding
def sum_by_keys(table: pa.Table, keys: list[str], sums: list[str]) -> pa.Table:
    """Group ``table`` on ``keys``, sum ``sums`` and keep the original column types and order."""
    agg = table.group_by(keys).aggregate([(c, "sum") for c in sums])
    agg = agg.rename_columns([c[:-4] if c.endswith("_sum") and c[:-4] in sums else c for c in agg.column_names])
    agg = agg.select(keys + sums)
    for c in sums:
        want = table.schema.field(c).type
        if agg.schema.field(c).type != want:
            agg = agg.set_column(agg.schema.get_field_index(c), c, pc.cast(agg[c], want, safe=False))
    return agg


def _plain(t: pa.Table) -> pa.Table:
    """Dictionary columns as plain strings (group_by keys), for a small table only."""
    for i, f in enumerate(t.schema):
        if pa.types.is_dictionary(f.type):
            t = t.set_column(i, f.name, pc.cast(t[f.name], f.type.value_type))
    return t


def fold_boundary_weeks(
    months: list[str],
    load: Callable[[str], pa.Table],
    save: Callable[[str, pa.Table], None],
    keys: list[str],
    sums: list[str],
    week_column: str = "week_start",
) -> dict[str, int]:
    """Fold weeks that span two monthly tables and save every month once.

    ``load(month)`` returns that month's table with ``keys`` and ``sums`` columns, where each week's
    rows hold only that month's days. Rows of month M whose ``week_start`` is before M's first day belong
    to a week that began in M-1: they are summed into M-1's rows for the same keys and removed from M.
    The first month keeps its leading partial week. Only the boundary weeks are regrouped (as plain
    strings), so dictionary-encoded columns of the rest stay compact. ``save`` is called once per month;
    to be crash-safe it should write somewhere temporary and let the caller swap the results in.
    Returns the rows saved per month.
    """
    written: dict[str, int] = {}
    carry: pa.Table | None = None  # rows of the later month that belong to the month being processed
    for month in reversed(months):
        cur = load(month)
        if month != months[0]:
            first_day = pa.scalar(month_start(month), cur.schema.field(week_column).type)
            moved = pc.less(cur[week_column], first_day)
            outgoing = cur.filter(moved)
            own = cur.filter(pc.invert(moved))
        else:
            outgoing, own = None, cur
        if carry is not None and carry.num_rows:
            touched = pc.unique(carry[week_column])
            is_b = pc.is_in(own[week_column], touched)
            rest = own.filter(pc.invert(is_b))
            both = pa.concat_tables([_plain(own.filter(is_b)), _plain(carry.select(own.column_names))])
            merged_b = sum_by_keys(both, keys, sums)
            for i, f in enumerate(rest.schema):  # back to the stored column types
                if pa.types.is_dictionary(f.type):
                    merged_b = merged_b.set_column(i, f.name, pc.dictionary_encode(merged_b[f.name]))
                elif merged_b.schema.field(i).type != f.type:
                    merged_b = merged_b.set_column(i, f.name, pc.cast(merged_b[f.name], f.type))
            merged = pa.concat_tables([rest, merged_b]).unify_dictionaries()
            del both, merged_b, rest
        else:
            merged = own
        save(month, merged)
        written[month] = merged.num_rows
        del cur, own, merged
        carry = outgoing
    return written

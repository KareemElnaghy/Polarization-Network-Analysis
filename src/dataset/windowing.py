"""Construct time-windowed graphs from interaction edges.

Every edge gets both a weekly and a monthly window (``week_start``, ``month_start``),
so the two granularities can be compared on the same edges. Weeks start Monday
00:00 UTC; months are calendar months in UTC. Windows are half-open
``[start, next start)``, matching :class:`src.common.types.Window`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterator, Literal

import networkx as nx
import pyarrow as pa
import pyarrow.compute as pc

from src.common.types import GraphWindow, Window

WindowKind = Literal["week", "month"]
WINDOW_COLUMNS: dict[str, str] = {"week": "week_start", "month": "month_start"}


def assign_windows(edges: pa.Table, timestamp_column: str = "created_utc") -> pa.Table:
    """Add ``week_start`` (Monday 00:00 UTC) and ``month_start`` columns to ``edges``."""
    ts = edges[timestamp_column]
    week = pc.floor_temporal(ts, unit="week", week_starts_monday=True)
    month = pc.floor_temporal(ts, unit="month")
    return edges.append_column("week_start", week).append_column("month_start", month)


def window_for(kind: WindowKind, start: datetime) -> Window:
    """Return the half-open window of ``kind`` that begins at ``start``."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if kind == "week":
        return Window(start, start + timedelta(days=7))
    end = datetime(start.year + (start.month == 12), start.month % 12 + 1, 1, tzinfo=start.tzinfo)
    return Window(start, end)


def is_complete(window: Window, data_start: datetime, data_end: datetime) -> bool:
    """True if ``window`` lies entirely within the data's ``[data_start, data_end)`` range."""
    return data_start <= window.start and window.end <= data_end


def window_graph(edges: pa.Table, min_activity_threshold: int = 1) -> nx.DiGraph:
    """Weighted ``source -> target`` graph from one window's edges.

    Edge weight is the number of replies. Nodes with fewer than ``min_activity_threshold``
    replies sent plus received are removed, along with their edges (a single pass, not
    iterated to a fixed point).
    """
    pairs = edges.group_by(["source", "target"]).aggregate([("source", "count")])
    src, dst, w = (pairs[c].to_pylist() for c in ("source", "target", "source_count"))
    if min_activity_threshold > 1:
        activity: dict[str, int] = {}
        for u, v, n in zip(src, dst, w):
            activity[u] = activity.get(u, 0) + n
            activity[v] = activity.get(v, 0) + n
        keep = {node for node, a in activity.items() if a >= min_activity_threshold}
        kept = [(u, v, n) for u, v, n in zip(src, dst, w) if u in keep and v in keep]
        src, dst, w = zip(*kept) if kept else ((), (), ())
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(zip(src, dst, w))
    return graph


def iter_time_windowed_graphs(
    edges: pa.Table,
    kind: WindowKind,
    min_activity_threshold: int = 1,
) -> Iterator[GraphWindow]:
    """Yield one :class:`GraphWindow` per window of ``kind``, in time order.

    ``edges`` needs ``source``, ``target`` and the window column added by
    :func:`assign_windows`. ``metadata["edges"]`` is the window's reply count before
    thresholding.
    """
    column = WINDOW_COLUMNS[kind]
    for start in sorted(pc.unique(edges[column]).to_pylist()):
        rows = edges.filter(pc.equal(edges[column], pa.scalar(start, edges.schema.field(column).type)))
        yield GraphWindow(window_for(kind, start), window_graph(rows, min_activity_threshold),
                          {"kind": kind, "edges": rows.num_rows, "min_activity_threshold": min_activity_threshold})


def build_time_windowed_graphs(
    edges: pa.Table,
    kind: WindowKind,
    min_activity_threshold: int = 1,
) -> list[GraphWindow]:
    """Return graphs grouped into weekly or monthly windows after activity thresholding."""
    return list(iter_time_windowed_graphs(edges, kind, min_activity_threshold))

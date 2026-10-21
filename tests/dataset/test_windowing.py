from datetime import datetime, timezone

import pyarrow as pa

from src.dataset.windowing import (
    assign_windows,
    build_time_windowed_graphs,
    is_complete,
    window_for,
    window_graph,
)

UTC = timezone.utc


def _edges(rows):
    ts = [int(datetime(*d, tzinfo=UTC).timestamp()) for _, _, d in rows]
    return pa.table({
        "source": [s for s, _, _ in rows],
        "target": [t for _, t, _ in rows],
        "created_utc": pa.array(ts, pa.timestamp("s", tz="UTC")),
    })


def test_assign_windows_uses_monday_weeks_and_calendar_months():
    edges = assign_windows(_edges([
        ("a", "b", (2016, 1, 1, 12)),       # Friday: week starts Mon 2015-12-28
        ("a", "b", (2016, 2, 29, 0)),       # Monday, leap day
        ("a", "b", (2016, 3, 6, 23, 59)),   # Sunday: same week as 2016-02-29
    ])).to_pylist()
    assert [r["week_start"] for r in edges] == [datetime(2015, 12, 28, tzinfo=UTC),
                                                datetime(2016, 2, 29, tzinfo=UTC),
                                                datetime(2016, 2, 29, tzinfo=UTC)]
    assert [r["month_start"] for r in edges] == [datetime(2016, 1, 1, tzinfo=UTC),
                                                 datetime(2016, 2, 1, tzinfo=UTC),
                                                 datetime(2016, 3, 1, tzinfo=UTC)]


def test_window_bounds_and_completeness():
    data = (datetime(2016, 1, 1, tzinfo=UTC), datetime(2017, 1, 1, tzinfo=UTC))
    first_week = window_for("week", datetime(2015, 12, 28, tzinfo=UTC))
    december = window_for("month", datetime(2016, 12, 1, tzinfo=UTC))
    assert first_week.end == datetime(2016, 1, 4, tzinfo=UTC)
    assert december.end == datetime(2017, 1, 1, tzinfo=UTC)
    assert not is_complete(first_week, *data)
    assert is_complete(december, *data)


def test_window_graph_weights_and_threshold():
    edges = _edges([("a", "b", (2016, 6, 1)), ("a", "b", (2016, 6, 2)), ("c", "a", (2016, 6, 3))])
    graph = window_graph(edges)
    assert graph["a"]["b"]["weight"] == 2 and graph.number_of_nodes() == 3
    # activity: a=3, b=2, c=1 -> threshold 2 drops c and its edge
    assert sorted(window_graph(edges, min_activity_threshold=2).edges) == [("a", "b")]
    assert window_graph(edges, min_activity_threshold=10).number_of_nodes() == 0


def test_build_time_windowed_graphs_splits_by_week():
    edges = assign_windows(_edges([("a", "b", (2016, 6, 6)), ("b", "c", (2016, 6, 13)),
                                   ("c", "a", (2016, 6, 14))]))
    windows = build_time_windowed_graphs(edges, "week")
    assert [w.window.start.day for w in windows] == [6, 13]
    assert [w.graph.number_of_edges() for w in windows] == [1, 2]
    assert [w.metadata["edges"] for w in windows] == [1, 2]

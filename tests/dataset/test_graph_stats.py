import networkx as nx
import pytest

from src.dataset.graph_stats import compute_graph_stats


def test_compute_graph_stats_directed_weighted():
    g = nx.DiGraph()
    g.add_weighted_edges_from([("a", "b", 3), ("b", "a", 1), ("c", "a", 1), ("d", "e", 1)])
    s = compute_graph_stats(g)
    assert (s["nodes"], s["edges"], s["interactions"]) == (5, 4, 6)
    assert s["density"] == pytest.approx(4 / 20)
    assert s["reciprocity"] == pytest.approx(0.5)
    assert s["in_degree_max"] == 2 and s["out_degree_max"] == 1
    assert (s["lcc_nodes"], s["lcc_fraction"]) == (3, pytest.approx(0.6))


def test_compute_graph_stats_empty_graph():
    s = compute_graph_stats(nx.DiGraph())
    assert (s["nodes"], s["edges"], s["density"], s["lcc_fraction"]) == (0, 0, 0.0, 0.0)


def test_subreddit_counts_per_window():
    from datetime import date, datetime, timezone

    import pyarrow as pa

    from src.dataset.graph_stats import subreddit_counts
    from src.dataset.windowing import assign_windows

    ts = [datetime(2016, 6, d, tzinfo=timezone.utc) for d in (6, 7, 8, 13)]
    edges = assign_windows(pa.table({
        "source": ["a", "b", "c", "d"], "target": ["b", "c", "a", "a"],
        "subreddit": ["politics", "NeutralPolitics", "politics", "The_Donald"],
        "created_utc": pa.array(ts, pa.timestamp("s", tz="UTC")),
    }))
    out = subreddit_counts(edges, "week", ("politics", "NeutralPolitics", "Ask_Politics")).set_index("window_start")
    first, second = out.loc[date(2016, 6, 6)], out.loc[date(2016, 6, 13)]
    assert (first["politics_replies"], first["NeutralPolitics_replies"], first["Ask_Politics_replies"]) == (2, 1, 0)
    assert first["politics_share"] == pytest.approx(2 / 3)
    assert (second["politics_replies"], second["politics_share"]) == (0, 0.0)

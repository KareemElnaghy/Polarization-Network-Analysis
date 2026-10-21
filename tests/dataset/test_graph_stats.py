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

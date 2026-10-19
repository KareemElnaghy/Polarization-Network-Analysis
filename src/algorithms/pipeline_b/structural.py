"""Structural scores for signed graphs."""

from typing import Any


def compute_frustration_score(signed_graph: Any) -> float:
    """Compute a frustration score for a signed graph.

    This is the extension point for a future Eigen-Sign implementation.
    """
    raise NotImplementedError("TODO: compute_frustration_score")

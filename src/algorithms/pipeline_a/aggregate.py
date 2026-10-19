"""Aggregation for interaction-level pipeline scores."""

from typing import Any, Iterable


def aggregate_window_score(scored_interactions: Iterable[Any]) -> float:
    """Aggregate interaction scores heuristically into a 0-10 window score."""
    raise NotImplementedError("TODO: aggregate_window_score")

"""LLM-based stance and affect scoring."""

from typing import Any, Iterable


def score_interactions(interactions: Iterable[Any], model: str) -> list[dict[str, Any]]:
    """Call an LLM per interaction and return stance, affect, and agreement scores."""
    raise NotImplementedError("TODO: score_interactions")

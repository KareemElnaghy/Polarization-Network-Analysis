"""Null models for signed graph evaluation."""

from typing import Any


def randomized_score_distribution(
    graph: Any,
    score_function: Any,
    n_samples: int,
    random_seed: int | None = None,
) -> list[float]:
    """Randomize signs or edges while preserving degrees and return score samples."""
    raise NotImplementedError("TODO: randomized_score_distribution")

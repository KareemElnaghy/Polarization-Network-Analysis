"""Ground-truth proxy construction from subreddit party labels."""

from typing import Any, Iterable, Mapping


def build_ground_truth_polarization(
    windowed_edges: Iterable[Any],
    subreddit_party_tags: Mapping[str, str],
) -> list[Any]:
    """Map subreddit party tags to a per-window polarization proxy."""
    raise NotImplementedError("TODO: build_ground_truth_polarization")

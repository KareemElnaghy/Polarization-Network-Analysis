"""Construct time-windowed graphs from interaction edges."""

from datetime import timedelta
from typing import Any, Iterable


def build_time_windowed_graphs(
    edges: Iterable[dict[str, Any]],
    window_size: timedelta,
    min_activity_threshold: int,
) -> list[Any]:
    """Return graphs grouped into time windows after activity thresholding."""
    raise NotImplementedError("TODO: build_time_windowed_graphs")

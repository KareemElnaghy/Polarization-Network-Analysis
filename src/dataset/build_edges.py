"""Build interaction edges from parsed Reddit comments."""

from typing import Any, Iterable


def build_reply_edges(comments: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert parsed comments into ``parent_id -> reply`` edge records."""
    raise NotImplementedError("TODO: build_reply_edges")

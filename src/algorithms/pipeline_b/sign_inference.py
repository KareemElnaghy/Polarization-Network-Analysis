"""LLM-based signed edge inference."""

from typing import Any


def infer_edge_sign(interaction: Any, model: str) -> dict[str, Any]:
    """Call an LLM and return a signed edge label with confidence."""
    raise NotImplementedError("TODO: infer_edge_sign")

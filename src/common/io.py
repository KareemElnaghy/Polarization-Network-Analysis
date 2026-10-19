"""Shared persistence helpers for graphs and tabular data."""

from pathlib import Path
from typing import Any


def load_graph(path: str | Path) -> Any:
    """Load a graph from ``path`` using the project's chosen serialization format."""
    raise NotImplementedError("TODO: load_graph")


def save_graph(graph: Any, path: str | Path) -> None:
    """Save ``graph`` to ``path`` using the project's chosen serialization format."""
    raise NotImplementedError("TODO: save_graph")


def load_dataframe(path: str | Path) -> Any:
    """Load a tabular dataset from ``path`` and return a pandas DataFrame."""
    raise NotImplementedError("TODO: load_dataframe")


def save_dataframe(dataframe: Any, path: str | Path) -> None:
    """Save a pandas DataFrame to ``path``."""
    raise NotImplementedError("TODO: save_dataframe")

"""Parsing utilities for raw Reddit comment dumps."""

from pathlib import Path
from typing import Any


def parse_raw_comments(raw_path: str | Path) -> list[dict[str, Any]]:
    """Parse a raw Reddit dump into structured comment records.

    The input is a dump path; the output contains normalized records with stable
    identifiers, author, subreddit, parent, timestamp, and text fields.
    """
    raise NotImplementedError("TODO: parse_raw_comments")

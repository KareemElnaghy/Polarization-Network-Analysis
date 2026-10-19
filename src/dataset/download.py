"""Utilities for selecting and fetching Reddit Politosphere data."""

from pathlib import Path
from typing import Iterable


def fetch_politosphere_months(
    months: Iterable[str],
    output_dir: str | Path,
    source_url: str | None = None,
) -> list[Path]:
    """Fetch selected ``YYYY-MM`` Politosphere dumps into ``output_dir``.

    Inputs are month identifiers, an output directory, and an optional source URL
    template or endpoint. The output is a list of downloaded raw dump paths.
    """
    raise NotImplementedError("TODO: fetch_politosphere_months")

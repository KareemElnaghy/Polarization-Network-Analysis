"""Configuration for the ground-truth runs: config/gt_runs.yaml and config/labels.yaml.

Jobs run from a snapshot of ``src/`` and ``config/`` (see scripts/launch_job.py), so these are read from
the snapshot's ``config`` folder: the exact rule used by a run is the one copied at launch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.prep_jobs.config import SOURCE_ROOT, project_root

CONFIG_DIR = SOURCE_ROOT / "config"


def _load(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def gt_cfg() -> dict[str, Any]:
    return _load("gt_runs.yaml")


def labels_cfg() -> dict[str, Any]:
    return _load("labels.yaml")


def gt_path(g: dict[str, Any], key: str) -> Path:
    return project_root() / g["paths"][key]


def label_sets(g: dict[str, Any], stage: str) -> list[str]:
    """Label sets that run in ``stage`` (gt1, gt2_full, gt2_samples, ...): main, gold where allowed, plus extras."""
    sets = ["main"]
    if stage in g["gold_in"]:
        sets.append("gold")
    return sets + [s for s in g["label_sets_extra"] if s not in sets]


def venues(g: dict[str, Any]) -> dict[str, list[str]]:
    """``venue name -> subreddits``: shared, one per special subreddit, plus the optional extra venues."""
    out = {k: list(v) for k, v in g["venues"].items()}
    for name in g["special_venues"] + g["extra_venues"]:
        out[name] = [name]
    return out

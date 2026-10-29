"""Configuration and paths for the prep jobs.

Jobs run from a snapshot of ``src/`` and the config under ``runs/<job>/snapshot/``; data paths
still point at the real project, given by the ``PREP_PROJECT_ROOT`` environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

SOURCE_ROOT = Path(__file__).resolve().parents[2]  # repo root, or the snapshot when run from one


def project_root() -> Path:
    return Path(os.environ.get("PREP_PROJECT_ROOT", SOURCE_ROOT)).resolve()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else SOURCE_ROOT / "config" / "prep_jobs.yaml"
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def data_path(cfg: dict[str, Any], key: str) -> Path:
    """Absolute path of ``cfg['paths'][key]``."""
    return project_root() / cfg["paths"][key]


def runs_dir(cfg: dict[str, Any]) -> Path:
    return data_path(cfg, "runs")


def job_dir(cfg: dict[str, Any], job: str) -> Path:
    return runs_dir(cfg) / job


def job_cfg(cfg: dict[str, Any], job: str) -> dict[str, Any]:
    return cfg["jobs"][job]

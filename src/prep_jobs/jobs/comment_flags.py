"""Job comment_flags: ids of distinguished (moderator/admin) comments, which the Parquet lacks.

The parsed comment Parquet files kept only a subset of the raw fields; ``distinguished`` was not among
them. This scans the raw dumps for it (byte-level pre-filter, so most lines are never JSON-parsed) and
writes ``data/interim/comment_flags/distinguished_YYYY-MM.parquet`` (id, distinguished). It is a
prerequisite of the other jobs and does not depend on any labelling decision.
"""

from __future__ import annotations

import bz2

import pyarrow as pa

try:
    import orjson
    _loads = orjson.loads
except ImportError:  # pragma: no cover
    import json
    _loads = json.loads

from src.prep_jobs.config import data_path
from src.prep_jobs.data import flags_file
from src.prep_jobs.util import write_parquet

NAME = "comment_flags"
DEADLINE_CUTS: list[str] = []
NOT_DISTINGUISHED = b'"distinguished":null'


def phases(cfg):
    raw = data_path(cfg, "raw")
    units = [{"id": m, "weight": (raw / f"comments_{m}.bz2").stat().st_size} for m in cfg["months"]]
    return [{"name": "scan", "fn": "src.prep_jobs.jobs.comment_flags:scan_month", "units": units}]


def scan_month(ctx, unit):
    month = unit["id"]
    ids, kinds, lines = [], [], 0
    with bz2.open(data_path(ctx.cfg, "raw") / f"comments_{month}.bz2", "rb") as fh:
        for line in fh:
            lines += 1
            if NOT_DISTINGUISHED in line:
                continue
            rec = _loads(line)
            if rec.get("distinguished"):
                ids.append(rec["id"])
                kinds.append(rec["distinguished"])
            if lines % 200_000 == 0:
                ctx.report(rows=lines)
    size = write_parquet(pa.table({"id": ids, "distinguished": kinds}), flags_file(ctx.cfg, month))
    ctx.log(f"{month}: {lines:,} lines, {len(ids):,} distinguished")
    return {"rows": lines, "bytes": size, "distinguished": len(ids)}

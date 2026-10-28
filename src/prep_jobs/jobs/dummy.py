"""Dummy job for testing the framework: sleeps in a few units and writes a file per unit."""

from __future__ import annotations

import time

from src.prep_jobs.config import job_cfg, project_root
from src.prep_jobs.util import write_json

NAME = "dummy"


def phases(cfg):
    j = job_cfg(cfg, NAME)
    units = [{"id": f"u{i}", "weight": 1000 * (i + 1), "seconds": j["seconds"] / j["units"]} for i in range(j["units"])]
    return [{"name": "sleep", "fn": "src.prep_jobs.jobs.dummy:unit", "units": units, "workers": None},
            {"name": "finish", "fn": "src.prep_jobs.jobs.dummy:finish",
             "units": [{"id": "all", "weight": 100}], "workers": 1, "needs_all_prior": True}]


def unit(ctx, unit):
    j = job_cfg(ctx.cfg, NAME)
    if j.get("flaky_unit") == unit["id"] and ctx.attempt < j.get("flaky_attempts", 2):
        raise RuntimeError("planned failure for the retry test")
    n = int(unit["seconds"])
    for i in range(n):
        time.sleep(1)
        ctx.report(rows=(i + 1) * 10, fraction=(i + 1) / n)
    out = project_root() / "runs" / "dummy" / "out" / f"{unit['id']}.json"
    write_json({"unit": unit["id"], "attempt": ctx.attempt}, out)
    return {"rows": n * 10, "bytes": out.stat().st_size}


def finish(ctx, unit):
    return {"rows": 0, "bytes": 0}

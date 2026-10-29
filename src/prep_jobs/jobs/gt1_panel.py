"""Job R4, gt1_panel: is the mid-2016 fall in GT#1 excess behaviour (same users sort differently) or composition?

Same cell as the main GT#1 series (main labels, shared venue, full week, heavy posters excluded). Phases:
``rows`` saves the labelled (thread, user) rows of each week; ``panels`` counts the weeks each user is active and the
before-split side ratio and r/politics share; ``variants`` computes, per week, the excess of every variant with thread-bootstrap
replicates; ``finalise`` makes the before/after tables, event tests, plots and ``results/gt1_panel/summary.md``.
Existing outputs are never overwritten.
"""

from __future__ import annotations

import gc
import zlib

import numpy as np
import pandas as pd

from src.gt_runs import gt1_panel as P
from src.gt_runs import gt1_robust as R
from src.gt_runs.config import gt_cfg, gt_path, labels_cfg, venues
from src.gt_runs.labels import heavy_set, load_side_map
from src.prep_jobs.config import project_root, runs_dir
from src.prep_jobs.jobs.clean_edges_and_sampling import weeks_of
from src.prep_jobs.jobs.gt1_robustness import week_rows
from src.prep_jobs.util import read_json, write_json

NAME = "gt1_panel"
DEADLINE_CUTS: list[str] = []
VARIANTS = ["main", "panel_a", "panel_b", "balanced", "pol_balanced", "strict", "reweight"]


def phases(cfg):
    mod = "src.prep_jobs.jobs.gt1_panel"
    ws = weeks_of(cfg)
    return [
        {"name": "rows", "fn": f"{mod}:rows_unit", "units": [{"id": w, "weight": 20} for w in ws]},
        {"name": "panels", "fn": f"{mod}:panels_unit", "units": [{"id": "panels", "weight": 40}], "workers": 1, "needs_all_prior": True},
        {"name": "variants", "fn": f"{mod}:variant_unit", "units": [{"id": w, "weight": 100} for w in ws], "needs_all_prior": True},
        {"name": "finalise", "fn": f"{mod}:finalise", "units": [{"id": "finalise", "weight": 50}], "workers": 1, "needs_all_prior": True},
    ]


def work(cfg):
    d = runs_dir(cfg) / NAME / "work"
    d.mkdir(parents=True, exist_ok=True)
    return d


def rows_unit(ctx, unit):
    cfg, g, lab, week = ctx.cfg, gt_cfg(), labels_cfg(), unit["id"]
    root = gt_path(g, "labels")
    heavy = heavy_set(root, week) if lab["exclude_heavy_posters"] else None
    rows = week_rows(cfg, week, venues(g)["shared"], load_side_map(root, "shared", "main"), heavy)
    rows.to_parquet(work(cfg) / f"rows_{week}.parquet")
    return {"rows": len(rows), "bytes": 0}


def panels_unit(ctx, unit):
    cfg, p = ctx.cfg, gt_cfg()["gt1_panel"]
    weeks = weeks_of(cfg)
    before = [w for w in weeks if w < p["split_week"]]
    present, tables = {}, {}
    for i, w in enumerate(weeks):
        rows = pd.read_parquet(work(cfg) / f"rows_{w}.parquet")
        present[w] = set(rows["user"].unique())
        if w in before:
            t = P.user_table(rows)
            tables[w] = {"ratio": float((t["side"] == 1).sum() / (t["side"] == 0).sum()), "pol": float((t["home"] == "politics").mean())}
        ctx.report(fraction=(i + 1) / len(weeks))
    users = sorted(set().union(*present.values()))
    tab = pd.DataFrame(0, index=users, columns=["n_weeks", "n_before", "n_after"], dtype=int)
    for w, u in present.items():
        tab.loc[list(u), "n_weeks"] += 1
        tab.loc[list(u), "n_before" if w in before else "n_after"] += 1
    tab.index.name = "user"
    tab.to_parquet(work(cfg) / "panels.parquet")
    a = tab.index[tab["n_weeks"] >= p["min_weeks_a"]]
    b = tab.index[(tab["n_before"] >= p["min_each_half"]) & (tab["n_after"] >= p["min_each_half"])]
    tgt = {"rep_dem_ratio": float(np.mean([v["ratio"] for v in tables.values()])),
           "politics_share": float(np.mean([v["pol"] for v in tables.values()])),
           "n_users": len(tab), "panel_a_users": len(a), "panel_b_users": len(b), "weeks_before": len(before),
           "weeks_after": len(weeks) - len(before)}
    write_json(tgt, work(cfg) / "targets.json")
    ctx.log(f"targets {tgt}")
    return {"rows": len(tab), "bytes": 0}


def _clean(v):
    if isinstance(v, (float, np.floating)):
        return None if np.isnan(v) else float(v)
    return v


def variant_unit(ctx, unit):
    cfg, g, lab, week = ctx.cfg, gt_cfg(), labels_cfg(), unit["id"]
    p = g["gt1_panel"]
    min_users = lab["min_users_per_side"]
    tab = pd.read_parquet(work(cfg) / "panels.parquet")
    tgt = read_json(work(cfg) / "targets.json")
    pan_a = tab.index[tab["n_weeks"] >= p["min_weeks_a"]]
    pan_b = tab.index[(tab["n_before"] >= p["min_each_half"]) & (tab["n_after"] >= p["min_each_half"])]
    rows = pd.read_parquet(work(cfg) / f"rows_{week}.parquet")
    is_after = week >= p["split_week"]
    politics = rows["sub"] == "politics"

    def pick(name):
        """(fixed rows, function drawing the random part or None)."""
        if name == "main":
            return rows, None
        if name == "panel_a":
            return P.keep_users(rows, pan_a), None
        if name == "panel_b":
            return P.keep_users(rows, pan_b), None
        if name == "balanced":
            return rows, P.balanced
        if name == "pol_balanced":
            return rows[politics], P.balanced
        if name == "strict":
            return P.keep_users(rows[politics], pan_b), P.balanced
        if name == "reweight":
            return rows, lambda r, rng: P.reweighted(r, rng, tgt["rep_dem_ratio"], tgt["politics_share"])
        raise KeyError(name)

    out: dict = {"week_start": week}
    for vi, name in enumerate(VARIANTS):
        if name == "reweight" and not is_after:
            continue
        rng = np.random.default_rng(g["seed"] + zlib.crc32(week.encode()) + 1000 * vi)
        fixed, draw = pick(name)
        if draw is None:
            cell = R.Cell(fixed)
            e = cell.excess(rng, p["n_perm"], min_users)
            boot = P.boot_values(cell, rng, p["n_boot"], p["boot_perm"]) if cell.ok(min_users) else np.full(p["n_boot"], np.nan)
            nd, nr, nt = cell.n_dem, cell.n_rep, len(cell.threads)
        else:
            ex, nds, nrs, nts = [], [], [], []
            for _ in range(p["n_repeats"]):
                c = R.Cell(draw(fixed, rng))
                ex.append(c.excess(rng, p["n_perm_sample"], min_users)["excess"])
                nds.append(c.n_dem); nrs.append(c.n_rep); nts.append(len(c.threads))
            e = {"excess": float(np.nanmean(ex)) if not np.all(np.isnan(ex)) else np.nan}
            boot = np.full(p["n_boot"], np.nan)
            for b in range(p["n_boot"]):
                c = R.Cell(draw(fixed, rng))
                if c.ok(min_users):
                    boot[b] = P.boot_values(c, rng, 1, p["boot_perm"])[0]
            nd, nr, nt = float(np.mean(nds)), float(np.mean(nrs)), float(np.mean(nts))
        out[name] = {"excess": _clean(e["excess"]), "n_dem": nd, "n_rep": nr, "n_threads": nt,
                     "boot": [_clean(v) for v in boot]}
        ctx.report(fraction=(vi + 1) / len(VARIANTS))
        gc.collect()
    write_json(out, work(cfg) / f"variants_{week}.json")
    return {"rows": 1, "bytes": 0}


def finalise(ctx, unit):
    from src.prep_jobs.jobs import gt1_panel_report as rep

    cfg, g = ctx.cfg, gt_cfg()
    weeks = weeks_of(cfg)
    files = {w: read_json(work(cfg) / f"variants_{w}.json") for w in weeks}
    out = project_root() / "results" / "gt1_panel"
    out.mkdir(parents=True, exist_ok=True)
    tgt = read_json(work(cfg) / "targets.json")
    rep.make_report(weeks, files, tgt, g, out, ctx.log)
    return {"rows": len(weeks), "bytes": 0}

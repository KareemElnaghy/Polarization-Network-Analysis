"""Overnight queue of Pipeline-B re-runs for P1.6 (A5 noise-injection curve) and P1.7 (A6 thinning and windows). Label-free (reads the slim export).
Each job = a set of (group, edges) -> B's graph rules unchanged (build_graph, giant, normalised_frustration with R sign-shuffle nulls). Output per job:
<out>/<job>.csv (one row per group) and <out>/<job>.camps.parquet. Resumable (finished jobs are skipped), stops when <out>/STOP exists or the wall budget would be exceeded.
usage: python scripts/final/b_jobs.py --hours 6.5 [--workers 14] [--R 20] [--only noise,thin,win4,month,win2] [--list]"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, ".")
from src.algorithms.pipeline_b import sign_inference as si
from src.algorithms.pipeline_b import structural as st

INP = Path("exports/analysis_inputs")


def guo(E):
    return (E.p_agree - E.p_disagree) * (1 - E.p_neutral)


def groups_weekly(E):
    return [(wk, g) for wk, g in E.groupby("week")]


def groups_rolling(E, k, weeks):
    by = {wk: g for wk, g in E.groupby("week")}
    return [(weeks[i], pd.concat([by[w] for w in weeks[i - k + 1: i + 1] if w in by])) for i in range(k - 1, len(weeks))]


def groups_month(E, weeks):
    m = pd.to_datetime(E.week).dt.to_period("M").astype(str)
    return [(k, g) for k, g in E.groupby(m)]


def thin(E, frac, seed):
    rng = np.random.default_rng(seed)
    keep = set()
    for wk, g in E.groupby("week"):
        th = np.sort(g.thread.unique())
        keep |= set(rng.choice(th, size=int(round(frac * len(th))), replace=False))
    return E[E.thread.isin(keep)]


def make_jobs(E, weeks):
    J = []  # (name, family, builder)
    def noise(p, s): return (f"noise_p{p:g}_s{s}", "noise", lambda: groups_weekly(E.assign(w=si.noise_injection(guo(E), p, s))))
    def thinj(f, s): return (f"thin_f{f:g}_s{s}", "thin", lambda: groups_weekly(thin(E, f, s).pipe(lambda d: d.assign(w=guo(d)))))
    J.append(noise(0, 0))
    for p in (0.1, 0.4, 0.2, 0.05): J.append(noise(p, 0))
    for f in (0.5, 0.25): J.append(thinj(f, 0))
    J.append(("win4_s0", "win4", lambda: groups_rolling(E.assign(w=guo(E)), 4, weeks)))
    J.append(("month_s0", "month", lambda: groups_month(E.assign(w=guo(E)), weeks)))
    for s in range(1, 5):
        for p in (0.1, 0.4, 0.2, 0.05): J.append(noise(p, s))
        for f in (0.5, 0.25): J.append(thinj(f, s))
    for s in range(5, 10):
        for p in (0.1, 0.4, 0.2, 0.05): J.append(noise(p, s))
    J.append(("win2_s0", "win2", lambda: groups_rolling(E.assign(w=guo(E)), 2, weeks)))
    return J


def run_job(groups, bc, R, ex):
    rows, camps = [], []
    for label, g in groups:
        row = {"group": label, "n_edges": len(g)}
        full = st.build_graph(g, bc["pair_aggregation"])
        row["n_pairs"] = len(full["w"])
        if len(full["w"]) >= bc["min_edges"]:
            gi = st.giant(full)
            row["n_giant"] = len(gi["nodes"])
            if len(gi["nodes"]) >= bc["min_giant_nodes"]:
                r = st.normalised_frustration(gi, R, bc["restarts"], bc["seed"], 0, ex)
                side = r.pop("side")
                row |= {"P_B": r["P_B"], "F_obs": r["F_obs"], "frac_negative": float((gi["w"] < 0).mean()), "mean_degree_giant": 2 * len(gi["w"]) / len(gi["nodes"])}
                camps.append(pd.DataFrame({"group": label, "user": gi["nodes"], "side": side.astype(int)}))
        rows.append(row)
    return pd.DataFrame(rows), (pd.concat(camps) if camps else pd.DataFrame(columns=["group", "user", "side"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6.5)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--out", default="runs/final_b_jobs")
    ap.add_argument("--only", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bc = yaml.safe_load(open(INP / "pipeline_b.yaml"))
    E = pd.read_parquet(INP / "edges_5pct.parquet")
    weeks = sorted(E.week.unique())
    jobs = make_jobs(E, weeks)
    if a.only:
        jobs = [j for j in jobs if j[1] in a.only.split(",")]
    if a.list:
        print("\n".join(j[0] for j in jobs))
        return
    ex = ProcessPoolExecutor(a.workers)
    t0, last = time.time(), 0.0
    for name, fam, build in jobs:
        if (out / f"{name}.csv").exists():
            continue
        if (out / "STOP").exists():
            print("STOP file found", flush=True)
            break
        if time.time() - t0 + last > a.hours * 3600:
            print(f"budget reached ({(time.time() - t0) / 3600:.2f} h); next job {name} not started", flush=True)
            break
        ts = time.time()
        rows, camps = run_job(build(), bc, a.R, ex)
        camps.to_parquet(out / f"{name}.camps.parquet")
        rows.to_csv(out / f"{name}.csv", index=False)
        last = time.time() - ts
        print(f"{name} done {last:.0f}s (total {(time.time() - t0) / 3600:.2f} h) P_B mean {rows.P_B.mean():+.4f}", flush=True)


if __name__ == "__main__":
    main()

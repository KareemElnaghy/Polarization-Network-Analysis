"""P1.2 / SPEC_ANALYSES A2: split-half series. For split seed s, each thread goes to half H1 or H2 (seeded coin per thread). For each half and week
compute P_A (mean edge score) and P_B (Guo weight, giant component, normalised frustration, R sign-shuffle nulls, same graph rules as structural.py).
Label-free, runs from the slim export (scripts/final/export_analysis_inputs.py). Resumable per split; shardable with --splits 0-9 / 10-19; STOP file honoured.
usage: python scripts/final/noise_ceiling.py --inputs exports/analysis_inputs --splits 0-19 --R 20 --out runs/noise_ceiling"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, ".")
from src.algorithms.pipeline_b import structural as st


def split_halves(threads: np.ndarray, s: int) -> dict:
    u = np.unique(threads)
    bit = np.random.default_rng(s).integers(0, 2, len(u))
    return dict(zip(u, bit))


def half_week(g: pd.DataFrame, bc: dict, R: int, ex) -> dict:
    row = {"n_edges": len(g), "P_A": float(g.score.mean()) if len(g) else np.nan, "P_B": np.nan}
    full = st.build_graph(g, bc["pair_aggregation"])
    row["n_pairs"] = len(full["w"])
    if len(full["w"]) < bc["min_edges"]:
        return row
    gi = st.giant(full)
    row["n_giant"] = len(gi["nodes"])
    if len(gi["nodes"]) < bc["min_giant_nodes"]:
        return row
    r = st.normalised_frustration(gi, R, bc["restarts"], bc["seed"], 0, ex)
    row["P_B"] = r["P_B"]
    return row


def parse(rng: str) -> list[int]:
    a, b = rng.split("-") if "-" in rng else (rng, rng)
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="exports/analysis_inputs")
    ap.add_argument("--splits", default="0-19")
    ap.add_argument("--R", type=int, default=20)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="runs/noise_ceiling")
    ap.add_argument("--weeks", default=None, help="restrict to weeks (plumbing checks)")
    a = ap.parse_args()
    inp, out = Path(a.inputs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    bc = yaml.safe_load(open(inp / "pipeline_b.yaml"))
    E = pd.read_parquet(inp / "edges_5pct.parquet")
    E["w"] = (E.p_agree - E.p_disagree) * (1 - E.p_neutral)  # = pipeline_c.weights.guo_weight, Pipeline B continuous_llm
    weeks = sorted(E.week.unique()) if not a.weeks else a.weeks.split(",")
    nw = a.workers or bc.get("workers", 1)
    ex = ProcessPoolExecutor(nw) if nw > 1 else None
    t0 = time.time()
    for s in parse(a.splits):
        f = out / f"split_{s:02d}_R{a.R}.csv"
        if f.exists():
            print(f"split {s} already done", flush=True)
            continue
        if (out / "STOP").exists():
            print("STOP file found", flush=True)
            break
        half = E.thread.map(split_halves(E.thread.to_numpy(), s))
        rows = []
        for h in (0, 1):
            Eh = E[half == h]
            for wk in weeks:
                rows.append({"split": s, "half": h + 1, "week": wk, **half_week(Eh[Eh.week == wk], bc, a.R, ex)})
        pd.DataFrame(rows).to_csv(f, index=False)
        print(f"split {s} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

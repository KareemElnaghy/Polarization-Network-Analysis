"""P2.3 (SPEC_ANALYSES A8): Pipelines A, B (continuous, binary) and C1 at one density from the frozen encoder store (D-E6a, deviation-labelled).
Unit of work = (density, pipeline, week); each unit writes <out>/d<density>/<pipe>/<week>.csv (+ .camps.parquet for graph pipelines), so runs are
resumable and shardable by week across machines. A STOP file in <out> ends the run after the current unit. No GT is read here.
R: 50 sign-shuffle nulls for continuous B and C1 at full density, 20 otherwise (A8.2). Rewiring null R=5 at full density for continuous B only (A8.3).
usage: python scripts/final/density_ladder.py --density 0.2 [--pipes A,B,B_bin,C1] [--weeks 1-27 | --weeks odd] [--workers 10]"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, ".")
from src.algorithms.encoder import outputs as eo
from src.algorithms.llm.data import load_full, load_shared, load_users
from src.algorithms.llm.schema import pair_hash
from src.algorithms.llm.store import Store
from src.algorithms.pipeline_a import aggregate as pa
from src.algorithms.pipeline_b import sign_inference as si
from src.algorithms.pipeline_b import structural as st
from src.algorithms.pipeline_c import weights as wc

BACKBONE = "microsoft/deberta-v3-large"  # D-E6a frozen encoder
ODD_WEEKS_NOTE = "fallback: the 27 odd-numbered ISO weeks of 2016 (D-E6 §D)"


def resolve_weeks(spec, all_weeks):
    if spec is None:
        return all_weeks
    if spec == "odd":
        return [w for w in all_weeks if pd.Timestamp(w).isocalendar()[1] % 2 == 1]  # 26 odd ISO weeks of 2016 + 2015-W53 (week of 2015-12-28) = 27
    if "-" in spec and not spec.startswith("20"):
        lo, hi = map(int, spec.split("-"))
        return all_weeks[lo - 1: hi]
    return spec.split(",")


def load_edges(density, weeks, enc, mc):
    if density >= 1.0:
        e = load_full(weeks, users=True)
    else:
        e = load_shared(density)
        e["week"] = e.week.astype(str)
        e = e[e.week.isin(weeks)].merge(load_users(density), on="comment_id")
    e["pair_hash"] = [pair_hash(p, b, mc) for p, b in zip(e.parent, e.body)]
    n0 = len(e)
    e = e.drop(columns=["parent"]).merge(enc, on="pair_hash", how="inner")
    return e, n0


def graph_unit(g, bc, R, Rr, ex):
    row = {"n_edges_in": len(g)}
    full = st.build_graph(g, bc["pair_aggregation"])
    row |= {"n_pairs": len(full["w"]), "n_nodes_full": len(full["nodes"])}
    if len(full["w"]) < bc["min_edges"]:
        return row | {"nan_reason": f"fewer than {bc['min_edges']} user pairs"}, None
    gi = st.giant(full)
    if len(gi["nodes"]) < bc["min_giant_nodes"]:
        return row | {"nan_reason": f"giant component under {bc['min_giant_nodes']} nodes"}, None
    r = st.normalised_frustration(gi, R, bc["restarts"], bc["seed"], Rr, ex)
    side = r.pop("side")
    row |= {"n_nodes_giant": len(gi["nodes"]), "n_pairs_giant": len(gi["w"]), "mean_degree_giant": 2 * len(gi["w"]) / len(gi["nodes"]),
            "frac_negative": float((gi["w"] < 0).mean()), **r, "R": R, "nan_reason": ""}
    return row, pd.DataFrame({"user": gi["nodes"], "side": side.astype(int)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--density", type=float, required=True, choices=[0.01, 0.05, 0.2, 1.0])
    ap.add_argument("--pipes", default="A,B,B_bin,C1")
    ap.add_argument("--weeks", default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--store", default="data/llm/encoder_outputs.sqlite")
    ap.add_argument("--out", default="runs/final_ladder")
    a = ap.parse_args()
    bc = yaml.safe_load(open("configs/llm/pipeline_b.yaml"))
    mc = yaml.safe_load(open("configs/llm/llm.yaml"))["max_chars"]
    all_weeks = sorted(p.name.split("=", 1)[1] for p in Path("data/processed/edges_clean").glob("week=*"))
    weeks = resolve_weeks(a.weeks, all_weeks)
    out = Path(a.out) / f"d{a.density:g}"
    pipes = a.pipes.split(",")
    for p in pipes:
        (out / p).mkdir(parents=True, exist_ok=True)
    todo = [w for w in weeks if any(not (out / p / f"{w}.csv").exists() for p in pipes)]
    if not todo:
        print("nothing to do", flush=True)
        print("DONE", flush=True)
        return
    ctx = eo.ctx_for(BACKBONE)
    enc = Store(a.store).table(ctx).drop(columns=["model", "quant", "prompt_version", "temperature", "seed"], errors="ignore")
    t0 = time.time()
    E, n0 = load_edges(a.density, todo, enc, mc)
    del enc
    print(f"density {a.density:g}: {len(E)} of {n0} edges have encoder rows ({len(todo)} weeks to do), {time.time() - t0:.0f}s", flush=True)
    rules = pa.load_rules()
    ex = ProcessPoolExecutor(a.workers)
    full = a.density >= 1.0
    for wk in todo:
        if (Path(a.out) / "STOP").exists():
            print("STOP file found", flush=True)
            break
        g = E[E.week == wk]
        for p in pipes:
            f = out / p / f"{wk}.csv"
            if f.exists():
                continue
            ts = time.time()
            if p == "A":
                row = pa.aggregate_weekly(pa.score_edges(g, rules), rules).iloc[0].to_dict()
                row["n_missing_encoder"] = 0
                camps = None
            else:
                if p == "B":
                    w, R, Rr = si.continuous_llm(g), (50 if full else 20), (5 if full else 0)
                elif p == "B_bin":
                    w, R, Rr = si.binary_llm(g, bc["neutral_policy"]), 20, 0
                elif p == "C1":
                    w, R, Rr = wc.c1(g), (50 if full else 20), 0
                d = g.assign(w=w).dropna(subset=["w"])
                row, camps = graph_unit(d, bc, R, Rr, ex)
            row = {"week": wk, "density": a.density, "pipeline": p, **row, "seconds": time.time() - ts,
                   "label": "deviation: encoder failed G-ENC check 1 (D-E6a)"}
            if camps is not None:
                camps.assign(week=wk).to_parquet(out / p / f"{wk}.camps.parquet")
            pd.DataFrame([row]).to_csv(f, index=False)
            print(f"d{a.density:g} {p} {wk} P={row.get('P_B', row.get('P_A'))} n={row.get('n_nodes_giant', row.get('n'))} {row['seconds']:.0f}s "
                  f"(total {(time.time() - t0) / 3600:.2f} h)", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()

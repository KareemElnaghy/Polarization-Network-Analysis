"""Cost of the frustration heuristic + nulls on a synthetic graph the size of the 5% / 20% weekly graphs, and the optimality
gap against the exact ILP on small graphs. usage: python scripts/llm/bench_frustration.py"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from src.algorithms.pipeline_b import structural as st
from src.algorithms.pipeline_b.frustration import csr_from_edges, exact_frustration, frustration

rows = []
for n, deg in [(2_000, 5), (10_000, 5), (40_000, 6), (150_000, 6)]:
    rng = np.random.default_rng(0)
    m = n * deg // 2
    u, v = rng.integers(0, n, m), rng.integers(0, n, m)
    k = u != v
    u, v = u[k], v[k]
    camp = rng.integers(0, 2, n)
    w = np.where(camp[u] == camp[v], 1.0, -1.0) * rng.uniform(0.2, 1, len(u))
    flip = rng.random(len(u)) < 0.15
    w[flip] *= -1
    g = {"nodes": np.arange(n), "u": u, "v": v, "w": w, "count": np.ones(len(u))}
    frustration(csr_from_edges(3, [0], [1], [1.0]), 1)  # warm JIT
    t0 = time.time(); F, _ = frustration(csr_from_edges(n, u, v, w), restarts=10); t_one = time.time() - t0
    t0 = time.time(); st.rewire(g, rng); t_rw = time.time() - t0
    rows.append({"nodes": n, "edges": len(u), "sec_per_frustration_10_restarts": t_one, "sec_per_rewire": t_rw, "F": F})
    print(rows[-1], flush=True)

gaps = []
for seed in range(16):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 80))
    m = int(n * 2.5)
    u, v = rng.integers(0, n, m), rng.integers(0, n, m)
    u, v = np.unique(np.c_[np.minimum(u, v), np.maximum(u, v)][u != v], axis=0).T  # one weight per pair, as build_graph gives
    w = rng.normal(size=len(u))
    Fh, _ = frustration(csr_from_edges(n, u, v, w), restarts=10, seed=seed)
    ex = exact_frustration(n, u, v, w, time_limit=120, detail=True)
    gaps.append({"nodes": n, "edges": len(u), "F_heur": Fh, "F_exact": ex["F"], "proved_optimal": ex["optimal"],
                 "rel_gap_vs_best_known": (Fh - min(Fh, ex["F"])) / min(Fh, ex["F"]), "ilp_bound": ex["bound"]})
import pandas as pd
d = pd.DataFrame(gaps)
print(d.round(4).to_string())
print("proved optimal: %d/%d; heuristic == best known in %.0f%%; mean rel gap vs best known %.4f; max %.4f" % (
    d.proved_optimal.sum(), len(d), 100 * (d.rel_gap_vs_best_known < 1e-9).mean(), d.rel_gap_vs_best_known.mean(), d.rel_gap_vs_best_known.max()))
pd.DataFrame(rows).to_csv("results/llm/bench/frustration_cost.csv", index=False)
d.to_csv("results/llm/bench/frustration_gap.csv", index=False)

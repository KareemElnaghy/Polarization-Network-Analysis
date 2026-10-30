"""Diagnostic: is the P_B decline with graph size real, or a heuristic failure? Compare the heuristic's F with the F of the planted camps
(an upper bound on the optimum) and with a 10x longer anneal. usage: python scripts/llm/synth_diag_scale.py"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.algorithms.pipeline_b import structural as st
from src.algorithms.pipeline_b.frustration import csr_from_edges, frustration
from src.algorithms.synthetic.generator import generate

POOL = np.load("results/llm/synthetic/degree_pool.npy")
rows = []
for n in (500, 2000, 4000, 10000, 40000):
    for lam in (0.5, 0.8):
        for seed in range(3):
            full = generate(n, 3.2, lam, POOL, seed=seed)
            camp = full["camp"]
            g = st.giant({k: v for k, v in full.items() if k != "camp"})
            s_pl = np.where(camp[g["nodes"]] == 1, 1.0, -1.0)
            A = np.abs(g["w"]).sum()
            F_pl = 0.5 * (A - (g["w"] * s_pl[g["u"]] * s_pl[g["v"]]).sum())
            W = csr_from_edges(len(g["nodes"]), g["u"], g["v"], g["w"])
            F_h = frustration(W, 10, seed)[0]
            sweeps_default = int(min(5000, max(100, 2_000_000 // len(g["nodes"]))))
            F_long = frustration(W, 10, seed, sa_sweeps=5000)[0]  # convergence reference
            Fn = np.mean([frustration(csr_from_edges(len(g["nodes"]), g["u"], g["v"], np.random.default_rng(100 + r).permutation(g["w"])), 10, r)[0] for r in range(10)])
            rows.append({"n": n, "lam": lam, "seed": seed, "giant": len(g["nodes"]), "F_null": Fn, "F_planted": F_pl, "F_heur": F_h, "F_long": F_long,
                         "PB_heur": 1 - F_h / Fn, "PB_long": 1 - F_long / Fn, "PB_planted": 1 - F_pl / Fn})
    print(n, flush=True)
d = pd.DataFrame(rows).groupby(["lam", "n"])[["giant", "PB_heur", "PB_long", "PB_planted", "F_heur", "F_long", "F_planted"]].mean().round(3)
print(d.to_string())
d.to_csv("results/llm/synthetic/diag_scale.csv")

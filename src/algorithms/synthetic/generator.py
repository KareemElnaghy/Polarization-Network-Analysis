"""Parameterized synthetic signed graph generation."""

from typing import Any


def generate_signed_graph(
    n_nodes: int,
    density: float,
    n_factions: int,
    noise_level: float,
) -> Any:
    """Generate a signed graph from node count, density, factions, and noise."""
    raise NotImplementedError("TODO: generate_signed_graph")


import numpy as np


def generate(n: int, mean_degree: float, lam: float, pool, rep_share: float = 0.5, homophily: float = 0.5,
             intermediate_share: float = 0.0, discordant_share: float = 0.0, flip: float = 0.0, seed: int = 0) -> dict:
    """Degree-corrected signed block model (see docs/synthetic_spec.md). Returns the graph dict used by pipeline_b.structural
    plus `camp` (1/0, -1 for no camp). Pairs are unique, no self-loops."""
    rng = np.random.default_rng(seed)
    theta = np.maximum(rng.choice(np.asarray(pool), n), 1).astype(float)
    camp = (rng.random(n) < rep_share).astype(int)
    camp[rng.random(n) < intermediate_share] = -1
    m = int(1.15 * n * mean_degree / 2)  # a little extra: duplicate pairs are dropped below
    u = rng.choice(n, m, p=theta / theta.sum())
    same = rng.random(m) < homophily
    v = rng.choice(n, m, p=theta / theta.sum())  # default: any node (used for no-camp endpoints)
    for c in (0, 1):
        for want_same, tgt in ((True, c), (False, 1 - c)):
            sel = np.flatnonzero((camp[u] == c) & (same == want_same))
            pool_c = np.flatnonzero(camp == tgt)
            if len(sel) and len(pool_c):
                p = theta[pool_c] / theta[pool_c].sum()
                v[sel] = rng.choice(pool_c, len(sel), p=p)
    k = u != v
    pairs = np.unique(np.c_[np.minimum(u, v), np.maximum(u, v)][k], axis=0)
    u, v = pairs[:, 0], pairs[:, 1]
    cu, cv = camp[u], camp[v]
    nocamp = (cu < 0) | (cv < 0)
    withincamp = (cu == cv) & ~nocamp
    aligned = np.where(cu == cv, 1.0, -1.0)
    rand = rng.choice([-1.0, 1.0], len(u))
    sign = np.where((rng.random(len(u)) < lam) & ~nocamp, aligned, rand)
    sign = np.where(withincamp & (rng.random(len(u)) < discordant_share), -1.0, sign)
    sign = np.where(rng.random(len(u)) < flip, -sign, sign)
    w = sign * rng.uniform(0.3, 1.0, len(u))
    return {"nodes": np.arange(n), "u": u, "v": v, "w": w, "count": np.ones(len(u)), "camp": camp}

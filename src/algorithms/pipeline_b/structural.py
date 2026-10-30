"""Pipeline B structure: weekly signed user graph, giant component, normalised frustration, null models."""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .frustration import csr_from_edges, frustration


def build_graph(df: pd.DataFrame, agg: str = "mean") -> dict:
    """df: source, target, w (NaN weights are dropped). Undirected; a user pair's weight is the `agg` of its interaction
    weights (TODO(verify) against Guo et al.); `count` keeps the number of interactions. Self-loops dropped."""
    d = df.dropna(subset=["w"])
    d = d[d.source != d.target]
    a = np.where(d.source.values < d.target.values, d.source.values, d.target.values)
    b = np.where(d.source.values < d.target.values, d.target.values, d.source.values)
    pairs = pd.DataFrame({"a": a, "b": b, "w": d.w.values}).groupby(["a", "b"]).w.agg([agg, "size"]).reset_index()
    pairs.columns = ["a", "b", "w", "count"]
    pairs = pairs[pairs.w != 0]
    nodes, inv = np.unique(np.r_[pairs.a.values, pairs.b.values], return_inverse=True)
    m = len(pairs)
    return {"nodes": nodes, "u": inv[:m], "v": inv[m:], "w": pairs.w.values.astype(float), "count": pairs["count"].values}


def giant(g: dict) -> dict:
    """Restrict to the largest connected component (by edges present)."""
    n = len(g["nodes"])
    if n == 0:
        return g
    ncomp, lab = connected_components(csr_from_edges(n, g["u"], g["v"], np.ones(len(g["w"]))), directed=False)
    big = np.argmax(np.bincount(lab))
    keep = lab == big
    e = keep[g["u"]]
    remap = -np.ones(n, dtype=int)
    remap[keep] = np.arange(keep.sum())
    return {"nodes": g["nodes"][keep], "u": remap[g["u"][e]], "v": remap[g["v"][e]], "w": g["w"][e], "count": g["count"][e]}


def _F(g: dict, restarts: int, seed: int):
    return frustration(csr_from_edges(len(g["nodes"]), g["u"], g["v"], g["w"]), restarts, seed)


def sign_shuffle(g: dict, rng) -> dict:
    """Null 1: same topology, edge weights permuted among edges (weight multiset preserved)."""
    return {**g, "w": rng.permutation(g["w"])}


def rewire(g: dict, rng, swaps_per_edge: int = 10) -> dict:
    """Null 2: signed-degree-preserving rewiring. Double edge swaps inside the positive and the negative edge sets separately
    keep every node's positive and negative degree; weights travel with their edge slot. Rejects self-loops and duplicates."""
    u, v = g["u"].copy(), g["v"].copy()
    seen = set(zip(np.minimum(u, v).tolist(), np.maximum(u, v).tolist()))
    for sign_mask in (g["w"] > 0, g["w"] < 0):
        idx = np.flatnonzero(sign_mask)
        if len(idx) < 2:
            continue
        for _ in range(swaps_per_edge * len(idx)):
            i, j = rng.choice(idx, 2, replace=False)
            a, b, c, d = u[i], v[i], u[j], v[j]
            if rng.random() < 0.5:
                c, d = d, c
            if a == d or c == b or (min(a, d), max(a, d)) in seen or (min(c, b), max(c, b)) in seen:
                continue
            seen -= {(min(a, b), max(a, b)), (min(u[j], v[j]), max(u[j], v[j]))}
            seen |= {(min(a, d), max(a, d)), (min(c, b), max(c, b))}
            u[i], v[i], u[j], v[j] = a, d, c, b
    return {**g, "u": u, "v": v}


def _task(a):
    """One null draw (top-level so it can run in a process pool). kind: shuffle (weights given) or rewire (seeded in the worker)."""
    kind, g, restarts, seed = a
    if kind == "rewire":
        g = rewire(g, np.random.default_rng(seed))
    return _F(g, restarts, seed)[0]


def _map(tasks, executor):
    return list(executor.map(_task, tasks)) if executor is not None and len(tasks) > 1 else [_task(t) for t in tasks]


def normalised_frustration(g: dict, R: int, restarts: int, seed: int, R_rewire: int = 0, executor=None) -> dict:
    """P_B = 1 - F_obs / F_null with F_null the mean frustration over R sign-shuffled copies, same heuristic.
    TODO(verify) the exact normalisation in Guo et al. P_B = 0 when F_null = 0 (no frustration possible, e.g. all-positive)."""
    rng = np.random.default_rng(seed)
    F_obs, s = _F(g, restarts, seed)
    Fn = np.array(_map([("shuffle", sign_shuffle(g, rng), restarts, seed + 1 + r) for r in range(R)], executor))
    out = {"F_obs": F_obs, "F_null_mean": Fn.mean() if R else np.nan, "F_null_sd": Fn.std() if R else np.nan,
           "A": float(np.abs(g["w"]).sum()), "P_B": (1 - F_obs / Fn.mean()) if R and Fn.mean() > 0 else 0.0}
    if R_rewire:
        Fr = np.array(_map([("rewire", g, restarts, seed + 1000 + r) for r in range(R_rewire)], executor))
        out |= {"F_rewire_mean": Fr.mean(), "P_B_rewire": (1 - F_obs / Fr.mean()) if Fr.mean() > 0 else 0.0}
    return out | {"side": s}

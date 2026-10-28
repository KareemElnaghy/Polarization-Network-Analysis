"""Ground truth #1: thread dissimilarity index with a shuffled-label null.

Rows are ``(thread, user, label)``. Each thread counts a user once ("distinct labelled users"). With ``d_t``
and ``r_t`` the dem and rep users in thread ``t`` and ``D``, ``R`` their totals over all threads, the
index is ``0.5 * sum_t |d_t / D - r_t / R|``: 0 when both camps spread over threads in the same
proportions, 1 when they never share a thread. Finite samples give a positive index even with no
segregation, so the null shuffles labels across users (same group sizes) and ``excess = observed - null_mean``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

DEM_LABELS = {"dem", "left", 0, False}
REP_LABELS = {"rep", "right", 1, True}


def _to_frame(rows) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows[["thread", "user", "label"]].copy()
    return pd.DataFrame(list(rows), columns=["thread", "user", "label"])


def _side(label) -> float:
    if label in REP_LABELS:
        return 1.0
    if label in DEM_LABELS:
        return 0.0
    return np.nan


def index_from_counts(d: np.ndarray, r: np.ndarray) -> float:
    return 0.5 * float(np.abs(d / d.sum() - r / r.sum()).sum())


def dissimilarity_index(rows: Iterable | pd.DataFrame, n_perm: int = 200, seed: int = 0, min_users: int = 50) -> dict:
    """Index, shuffled-label null and excess for ``(thread, user, label)`` rows.

    Labels are ``dem``/``left``/0 or ``rep``/``right``/1; other labels are ignored. A user with two
    different labels keeps the first. Returns NaN values and a ``reason`` if either side has fewer than
    ``min_users`` distinct users.
    """
    df = _to_frame(rows)
    df["user"], df["thread"] = df["user"].astype(str), df["thread"].astype(str)
    df["side"] = df["label"].map(_side)
    df = df.dropna(subset=["side"]).drop_duplicates(["thread", "user"])
    user_side = df.drop_duplicates("user").set_index("user")["side"]
    df = df[df["user"].map(user_side) == df["side"]]
    users, u_idx = np.unique(df["user"].to_numpy(), return_inverse=True)
    threads, t_idx = np.unique(df["thread"].to_numpy(), return_inverse=True)
    side = user_side.reindex(users).to_numpy()
    n_rep, n_dem = int(side.sum()), int((1 - side).sum())
    out = {"observed": np.nan, "null_mean": np.nan, "null_sd": np.nan, "excess": np.nan,
           "n_dem": n_dem, "n_rep": n_rep, "n_threads": len(threads), "reason": None}
    if min(n_rep, n_dem) < min_users:
        out["reason"] = f"fewer than {min_users} labelled users on one side (dem {n_dem}, rep {n_rep})"
        return out
    inc = sparse.csr_matrix((np.ones(len(u_idx)), (u_idx, t_idx)), shape=(len(users), len(threads)))
    r_t = inc.T @ side
    d_t = inc.T @ (1 - side)
    out["observed"] = index_from_counts(d_t, r_t)
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for b in range(n_perm):
        s = rng.permutation(side)
        null[b] = index_from_counts(inc.T @ (1 - s), inc.T @ s)
    out["null_mean"], out["null_sd"] = float(null.mean()), float(null.std(ddof=1)) if n_perm > 1 else float("nan")
    out["excess"] = out["observed"] - out["null_mean"]
    return out

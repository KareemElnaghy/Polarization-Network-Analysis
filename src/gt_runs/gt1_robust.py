"""Robustness checks for ground truth #1 (thread dissimilarity index); one week at a time.

All functions work on ``rows``: a DataFrame ``thread, user, side (0 dem / 1 rep), sub, n_comments``, one row per
(thread, user). ``excess = observed - mean(shuffled-label null)`` exactly as in ``src.ground_truth.dissimilarity``;
the variants only change what is resampled, how labels are shuffled, or which rows go in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

NAN = {"observed": np.nan, "null_mean": np.nan, "null_sd": np.nan, "excess": np.nan}


class Cell:
    """Incidence of users in threads plus the user sides."""

    def __init__(self, rows: pd.DataFrame) -> None:
        self.users, u = np.unique(rows["user"].to_numpy(), return_inverse=True)
        self.threads, t = np.unique(rows["thread"].to_numpy(), return_inverse=True)
        first = np.zeros(len(self.users), dtype=np.int64)
        first[u[::-1]] = np.arange(len(u))[::-1]  # first row of each user
        self.side = rows["side"].to_numpy()[first].astype(float)
        self.u, self.t = u, t
        self.n_dem, self.n_rep = int((self.side == 0).sum()), int((self.side == 1).sum())
        self.incT = sparse.csr_matrix((np.ones(len(u)), (t, u)), shape=(len(self.threads), len(self.users)))

    def ok(self, min_users: int) -> bool:
        return min(self.n_dem, self.n_rep) >= min_users

    def index(self, side: np.ndarray, w: np.ndarray | None = None) -> float:
        d, r = self.incT @ (1 - side), self.incT @ side
        if w is not None:
            d, r = d * w, r * w
        if d.sum() == 0 or r.sum() == 0:
            return np.nan
        return 0.5 * float(np.abs(d / d.sum() - r / r.sum()).sum())

    def excess(self, rng, n_perm: int, min_users: int, strata: np.ndarray | None = None, w: np.ndarray | None = None) -> dict:
        """Observed, null mean/sd and excess. ``strata`` (per user): labels are shuffled only within a stratum."""
        if not self.ok(min_users):
            return dict(NAN)
        obs = self.index(self.side, w)
        if strata is None:
            null = [self.index(rng.permutation(self.side), w) for _ in range(n_perm)]
        else:
            order0 = np.argsort(strata, kind="stable")
            null = []
            for _ in range(n_perm):
                order = np.lexsort((rng.random(len(strata)), strata))
                s = np.empty_like(self.side)
                s[order] = self.side[order0]
                null.append(self.index(s, w))
        nm = float(np.nanmean(null))
        return {"observed": obs, "null_mean": nm, "null_sd": float(np.nanstd(null, ddof=1)) if n_perm > 1 else np.nan,
                "excess": obs - nm}


def home_strata(rows: pd.DataFrame, thread_stratum: pd.Series, cell: Cell) -> np.ndarray:
    """Per user (in ``cell.users`` order): the stratum of the thread where the user commented most."""
    r = rows.assign(stratum=rows["thread"].map(thread_stratum)).sort_values(["user", "n_comments"], ascending=[True, False])
    top = r.drop_duplicates("user").set_index("user")["stratum"]
    codes, _ = pd.factorize(top.reindex(cell.users))
    return codes


def thread_strata(rows: pd.DataFrame, thread_day: pd.Series) -> pd.Series:
    """``subreddit x day x thread-size bin`` code per thread; size = labelled users, binned by log2."""
    size = rows.groupby("thread")["user"].nunique()
    sub = rows.drop_duplicates("thread").set_index("thread")["sub"]
    day = thread_day.reindex(size.index).fillna(-1).astype(int)
    key = sub.reindex(size.index).astype(str) + "|" + day.astype(str) + "|" + np.floor(np.log2(size)).astype(int).astype(str)
    return pd.Series(pd.factorize(key)[0], index=size.index)


def bootstrap(cell: Cell, rng, n_boot: int, n_perm: int, min_users: int) -> dict:
    """Thread-resampled excess: multinomial thread weights; the null mean uses ``n_perm`` shuffles per resample."""
    if not cell.ok(min_users):
        return {"boot_lo": np.nan, "boot_hi": np.nan, "boot_sd": np.nan}
    nt = len(cell.threads)
    vals = []
    for _ in range(n_boot):
        w = rng.multinomial(nt, np.full(nt, 1 / nt)).astype(float)
        obs = cell.index(cell.side, w)
        nm = np.nanmean([cell.index(rng.permutation(cell.side), w) for _ in range(n_perm)])
        vals.append(obs - nm)
    v = np.array(vals)
    return {"boot_lo": float(np.nanpercentile(v, 2.5)), "boot_hi": float(np.nanpercentile(v, 97.5)), "boot_sd": float(np.nanstd(v, ddof=1))}


def subset(rows: pd.DataFrame, keep: np.ndarray) -> pd.DataFrame:
    return rows[keep]


def flip_sides(rows: pd.DataFrame, rng, frac: float) -> pd.DataFrame:
    """Flip the side of a random ``frac`` of the users."""
    users = rows["user"].unique()
    flip = set(rng.choice(users, size=int(round(frac * len(users))), replace=False))
    side = rows["side"].to_numpy().copy()
    m = rows["user"].isin(flip).to_numpy()
    side[m] = 1 - side[m]
    return rows.assign(side=side)


def event_test(series: pd.Series, event_weeks: list[str], n_perm: int = 10000, seed: int = 0) -> dict:
    """Mean in event weeks vs other weeks and a permutation test that shuffles which weeks are the event weeks."""
    s = series.dropna()
    ev = s.index.isin(event_weeks)
    diff = s[ev].mean() - s[~ev].mean()
    rng = np.random.default_rng(seed)
    vals = s.to_numpy()
    k = int(ev.sum())
    perm = np.empty(n_perm)
    for i in range(n_perm):
        m = np.zeros(len(vals), dtype=bool)
        m[rng.choice(len(vals), k, replace=False)] = True
        perm[i] = vals[m].mean() - vals[~m].mean()
    return {"n_event_weeks": k, "event_mean": float(s[ev].mean()), "other_mean": float(s[~ev].mean()), "diff": float(diff),
            "p_one_sided": float((perm >= diff).mean()), "p_two_sided": float((np.abs(perm) >= abs(diff)).mean())}

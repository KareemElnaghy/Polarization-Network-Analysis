"""Fixed-panel and composition controls for ground truth #1.

``rows`` is the frame used by ``gt1_robust``: ``thread, user, side (0 dem / 1 rep), sub, n_comments``, one row per
(thread, user). Every function returns a subset of the rows (all rows of the kept users) so that the excess is computed
with the same ``gt1_robust.Cell`` and shuffled-label null as the main series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.gt_runs.gt1_robust import Cell


def user_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per user: side, home subreddit (most commented), comments."""
    r = rows.sort_values(["user", "n_comments"], ascending=[True, False])
    t = r.drop_duplicates("user").set_index("user")[["side", "sub"]].rename(columns={"sub": "home"})
    t["comments"] = rows.groupby("user")["n_comments"].sum()
    return t


def keep_users(rows: pd.DataFrame, users) -> pd.DataFrame:
    return rows[rows["user"].isin(users)]


def balanced(rows: pd.DataFrame, rng) -> pd.DataFrame:
    """Subsample users of the larger side down to the size of the smaller side."""
    ut = rows.drop_duplicates("user").set_index("user")["side"]
    d, r = ut.index[ut == 0].to_numpy(), ut.index[ut == 1].to_numpy()
    k = min(len(d), len(r))
    if len(d) > k:
        d = rng.choice(d, k, replace=False)
    if len(r) > k:
        r = rng.choice(r, k, replace=False)
    return rows[rows["user"].isin(np.concatenate([d, r]))]


def reweighted(rows: pd.DataFrame, rng, rep_dem_ratio: float, politics_share: float, home_sub: str = "politics") -> pd.DataFrame:
    """Subsample users so that rep/dem = ``rep_dem_ratio`` and the share with home subreddit ``home_sub`` = ``politics_share``.

    Users are drawn without replacement inside the four side x home cells; the sample is as large as the scarcest cell
    allows for the target mix.
    """
    ut = user_table(rows)
    ps = rep_dem_ratio / (1.0 + rep_dem_ratio)   # republican share of users
    shares = {(s, h): (ps if s == 1 else 1 - ps) * (politics_share if h else 1 - politics_share) for s in (0, 1) for h in (0, 1)}
    cells = {(s, h): ut.index[(ut["side"] == s) & ((ut["home"] == home_sub) == bool(h))].to_numpy() for s in (0, 1) for h in (0, 1)}
    n = min(len(cells[c]) / shares[c] for c in cells if shares[c] > 0)
    keep = []
    for c, sh in shares.items():
        k = int(round(n * sh))
        if k > 0:
            keep.append(rng.choice(cells[c], min(k, len(cells[c])), replace=False))
    return rows[rows["user"].isin(np.concatenate(keep))]


def boot_values(cell: Cell, rng, n_boot: int, n_perm: int) -> np.ndarray:
    """Thread-resampled excess, one value per resample (NaN when the cell is empty)."""
    nt = len(cell.threads)
    out = np.empty(n_boot)
    for b in range(n_boot):
        w = rng.multinomial(nt, np.full(nt, 1 / nt)).astype(float)
        obs = cell.index(cell.side, w)
        out[b] = obs - np.nanmean([cell.index(rng.permutation(cell.side), w) for _ in range(n_perm)])
    return out


def mean_ci(values: np.ndarray) -> tuple[float, float]:
    v = values[~np.isnan(values)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)

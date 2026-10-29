"""Ground truth #2: the leave-out estimator of partisanship (Gentzkow, Shapiro and Taddy, 2019).

Input: a users x phrases count matrix ``C`` and labels (1 = rep, 0 = dem). With ``m_i`` the row sum, for each
side P the phrase frequencies are ``rho_P_j = sum_{i in P} c_ij / sum_{i in P} m_i``. For user ``i`` both are
recomputed without ``i``; ``q_{-i,j} = rho_R_{-i,j} / (rho_R_{-i,j} + rho_D_{-i,j})`` and
``qhat_i = sum_j (c_ij / m_i) q_{-i,j}``. Then ``pi = 0.5 * mean_{i in R} qhat_i + 0.5 * mean_{i in D} (1 - qhat_i)``.
0.5 means the camps use phrases alike; 1 means they share none.

Everything runs over the non-zero entries only, so the cost is proportional to ``nnz``. A phrase used by
no one else gets ``q = 0.5`` (no information). The bootstrap draws multinomial user weights instead of copying
rows, and leaves a user out with all of its copies. The null shuffles labels across users (same group sizes).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _pi(rows, cols, vals, m, y, w, n_cols, active) -> tuple[float, np.ndarray]:
    """``pi`` and ``qhat`` (per user) for 0/1 labels ``y`` and non-negative user weights ``w``."""
    wy = w * y
    wd = w * (1 - y)
    c_r = np.bincount(cols, weights=vals * wy[rows], minlength=n_cols)
    c_d = np.bincount(cols, weights=vals * wd[rows], minlength=n_cols)
    m_r, m_d = float((m * wy).sum()), float((m * wd).sum())
    cr, cd, mi, wi, yi = c_r[cols], c_d[cols], m[rows], w[rows], y[rows]
    # remove the user's own (weighted) counts from its own side
    num_r = np.where(yi == 1, cr - wi * vals, cr)
    den_r = np.where(yi == 1, m_r - wi * mi, m_r)
    num_d = np.where(yi == 1, cd, cd - wi * vals)
    den_d = np.where(yi == 1, m_d, m_d - wi * mi)
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_r, rho_d = num_r / den_r, num_d / den_d
        q = rho_r / (rho_r + rho_d)
    q = np.where(np.isfinite(q), q, 0.5)
    qhat = np.bincount(rows, weights=vals / mi[rows] * q, minlength=len(m))
    wr, wdm = (w * y)[active], (w * (1 - y))[active]
    pi = 0.5 * (qhat[active] * wr).sum() / wr.sum() + 0.5 * ((1 - qhat[active]) * wdm).sum() / wdm.sum()
    return float(pi), qhat


def leave_out_estimator(counts, labels, n_perm: int = 200, n_boot: int = 200, seed: int = 0, min_users: int = 50,
                        m=None) -> dict:
    """Leave-out ``pi`` with a shuffled-label null and bootstrap bands.

    ``counts``: scipy sparse users x phrases; ``labels``: array of 1 (rep) / 0 (dem); ``m``: optional row totals
    (default: row sums). Users with ``m_i == 0`` are dropped. Returns ``pi``, ``null_mean``, ``null_sd``,
    ``boot_lo``/``boot_hi`` (2.5 and 97.5 percentiles), ``boot_sd``, group sizes and ``reason`` (set, with NaNs, if
    a side has fewer than ``min_users`` users).
    """
    coo = sparse.coo_matrix(counts)
    m_all = np.asarray(coo.tocsr().sum(axis=1)).ravel() if m is None else np.asarray(m, dtype=float)
    y_all = np.asarray(labels, dtype=float)
    keep = m_all > 0
    remap = np.cumsum(keep) - 1
    sel = keep[coo.row]
    rows, cols, vals = remap[coo.row[sel]], coo.col[sel], coo.data[sel].astype(float)
    m, y = m_all[keep], y_all[keep]
    n_users, n_cols = len(m), coo.shape[1]
    n_rep, n_dem = int(y.sum()), int((1 - y).sum())
    out = {"pi": np.nan, "null_mean": np.nan, "null_sd": np.nan, "boot_lo": np.nan, "boot_hi": np.nan, "boot_sd": np.nan,
           "n_rep": n_rep, "n_dem": n_dem, "reason": None}
    if min(n_rep, n_dem) < min_users:
        out["reason"] = f"fewer than {min_users} users on one side (dem {n_dem}, rep {n_rep})"
        return out
    everyone = np.ones(n_users, dtype=bool)
    ones = np.ones(n_users)
    out["pi"], _ = _pi(rows, cols, vals, m, y, ones, n_cols, everyone)
    rng = np.random.default_rng(seed)
    null = np.array([_pi(rows, cols, vals, m, rng.permutation(y), ones, n_cols, everyone)[0] for _ in range(n_perm)])
    if n_perm:
        out["null_mean"], out["null_sd"] = float(null.mean()), float(null.std(ddof=1)) if n_perm > 1 else float("nan")
    if n_boot:
        boots = []
        for _ in range(n_boot):
            w = rng.multinomial(n_users, np.full(n_users, 1 / n_users)).astype(float)
            active = w > 0
            if w[active & (y == 1)].sum() == 0 or w[active & (y == 0)].sum() == 0:
                continue
            boots.append(_pi(rows, cols, vals, m, y, w, n_cols, active)[0])
        boots = np.array(boots)
        out["boot_lo"], out["boot_hi"] = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        out["boot_sd"] = float(boots.std(ddof=1))
    return out


# Name used in docs/gt_runs_spec.md
leave_out_polarization = leave_out_estimator

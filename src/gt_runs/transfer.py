"""Ground truth #2, transfer design: learn how each camp talks in the supporter subs, predict side in neutral subs.

Users are rows of a users x phrases count matrix. Three ways to weight users in training (``WEIGHTINGS``):
``cap100`` / ``cap300`` cap each user's phrases per week (a user-week contributes at most 100 / 300 phrases) before the
weeks are summed, so the user's influence is the capped volume; ``equal`` converts each user's counts to frequencies (every
user counts once). A cap on the yearly total would make all three identical, because almost every user exceeds it. Two instruments read a user's neutral-sub phrases with models frozen from training:

* ``phrase``: leaning ``q_j`` = share of phrase j's (weighted, side-normalised) use by Republican supporters; user
  estimate = frequency-weighted mean of ``q`` over the user's phrases; weekly ``pi = 0.5 mean_R(est) + 0.5 mean_D(1 - est)``
  (0.5 when the estimate carries no side information);
* ``logit``: L2 logistic regression on the weighted rows; user score = P(Republican); weekly value = AUC.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

WEIGHTINGS = {"cap100": 100.0, "cap300": 300.0, "equal": None}
SMOOTH = 0.5        # pseudo-count per side, in units where each side has total mass SIDE_MASS
SIDE_MASS = 1e6


def frequencies(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """Rows scaled to sum to 1 (empty rows stay empty)."""
    X = sparse.csr_matrix(X, dtype=np.float32)
    tot = np.asarray(X.sum(axis=1)).ravel()
    return sparse.diags((1.0 / np.maximum(tot, 1.0)).astype(np.float32)) @ X


def user_weights(X: sparse.csr_matrix, weighting: str) -> np.ndarray:
    """Influence of each training user: capped phrase volume (X already holds per-week capped counts) or 1 for equal."""
    tot = np.asarray(X.sum(axis=1)).ravel().astype(float)
    return np.ones(len(tot)) if weighting == "equal" else tot / tot.mean()


def profile_matrix(X: sparse.csr_matrix, weighting: str) -> sparse.csr_matrix:
    """Rows whose column sums give each side's weighted phrase profile: capped counts, or frequencies for equal."""
    return frequencies(X) if weighting == "equal" else sparse.csr_matrix(X, dtype=np.float32)


FEATURE_SCALE = 100.0   # logistic features: frequencies x 100, so every user row sums to 100 whatever the weighting


def leanings(Xw: sparse.csr_matrix, y: np.ndarray) -> np.ndarray:
    """``q_j`` per phrase from weighted rows ``Xw`` and sides ``y`` (1 rep, 0 dem)."""
    cr = np.asarray(Xw[y == 1].sum(axis=0)).ravel().astype(float)
    cd = np.asarray(Xw[y == 0].sum(axis=0)).ravel().astype(float)
    cr, cd = cr * SIDE_MASS / cr.sum(), cd * SIDE_MASS / cd.sum()
    return (cr + SMOOTH) / (cr + cd + 2 * SMOOTH)


def pi_value(est: np.ndarray, y: np.ndarray) -> float:
    return 0.5 * float(est[y == 1].mean()) + 0.5 * float((1 - est[y == 0]).mean())


def auc_value(score: np.ndarray, y: np.ndarray) -> float:
    r = rankdata(score)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def weekly_stat(values: np.ndarray, y: np.ndarray, kind: str, rng, n_perm: int, n_boot: int, min_users: int) -> dict:
    """Weekly pi (``kind='pi'``) or AUC (``'auc'``) with a shuffled-label null and bootstrap bands over users."""
    f = pi_value if kind == "pi" else auc_value
    out = {"value": np.nan, "null_mean": np.nan, "null_sd": np.nan, "boot_lo": np.nan, "boot_hi": np.nan,
           "n_dem": int((y == 0).sum()), "n_rep": int((y == 1).sum())}
    if min(out["n_dem"], out["n_rep"]) < min_users:
        return out
    out["value"] = f(values, y)
    null = np.array([f(values, rng.permutation(y)) for _ in range(n_perm)])
    out["null_mean"], out["null_sd"] = float(null.mean()), float(null.std(ddof=1))
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        boots.append(f(values[i], y[i]))
    if boots:
        out["boot_lo"], out["boot_hi"] = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    return out


def fit_logit(Xw: sparse.csr_matrix, y: np.ndarray, sw: np.ndarray, rng, grid=(0.3, 3.0, 30.0), cv_users: int = 20000, folds: int = 3):
    """L2 logistic regression with user weights sw; C picked by ``folds``-fold CV (log-loss) on a subsample of the training users."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold

    n = Xw.shape[0]
    sub = rng.choice(n, min(n, cv_users), replace=False)
    Xs, ys, ws = Xw[sub], y[sub], sw[sub]
    best, best_loss = grid[0], np.inf
    for c in grid:
        losses = []
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(Xs, ys):
            m = LogisticRegression(C=c, solver="lbfgs", max_iter=150, tol=1e-3, class_weight="balanced").fit(Xs[tr], ys[tr], sample_weight=ws[tr])
            losses.append(log_loss(ys[te], m.predict_proba(Xs[te])[:, 1]))
        if np.mean(losses) < best_loss:
            best, best_loss = c, float(np.mean(losses))
    m = LogisticRegression(C=best, solver="lbfgs", max_iter=300, tol=1e-3, class_weight="balanced").fit(Xw, y, sample_weight=sw)
    return m.coef_.ravel().astype(np.float32), float(m.intercept_[0]), best


def accuracy(score: np.ndarray, y: np.ndarray, threshold: float) -> float:
    return float(((score > threshold) == (y == 1)).mean())

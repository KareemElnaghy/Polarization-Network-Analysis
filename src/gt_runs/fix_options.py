"""Candidate fixes for the ground-truth #2 estimator (exploration only; the main T2/T3 outputs are not touched).

Everything works on cached ``(author, phrase, count)`` triplets of the shared venue (labelled users, heavy posters
and contraction-filter phrases already removed). Variants:

* ``equal``: every user counts once (party profile = mean of user frequencies) instead of in proportion to volume;
* ``phrase_users``: keep only phrases used by at least this many distinct users in the cell;
* ``cap``: scale each user's counts down so nobody contributes more than ``cap`` phrases;
* ``window``: pool the trailing N weeks.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from src.ground_truth.leave_out_estimator import leave_out_estimator


def build_matrix(a, p, c, side_vec, n_phrases, min_phrases, equal=False, phrase_users=0, cap=None):
    """Users x phrases matrix (and labels, row totals for the estimator) after the variant's filters."""
    c = c.astype(float)
    if cap:
        totals = np.bincount(a, weights=c, minlength=len(side_vec))
        c = c * np.minimum(1.0, cap / np.maximum(totals[a], 1.0))
    users, inv = np.unique(a, return_inverse=True)
    C = sparse.csr_matrix((c, (inv, p)), shape=(len(users), n_phrases))
    if phrase_users:
        used = np.diff(C.tocsc().indptr)
        C = C[:, np.flatnonzero(used >= phrase_users)]
    tot = np.asarray(C.sum(axis=1)).ravel()
    ok = tot >= min_phrases
    C, users, tot = C[ok], users[ok], tot[ok]
    m = None
    if equal:
        C = sparse.diags(1.0 / tot) @ C
        m = np.ones(len(users))
    return C, side_vec[users].astype(float), m


def fix_cell(a, p, c, side_vec, n_phrases, min_phrases, n_perm, n_boot, seed, min_users, **variant) -> dict:
    C, y, m = build_matrix(a, p, c, side_vec, n_phrases, min_phrases, **variant)
    r = leave_out_estimator(C, y, n_perm=n_perm, n_boot=n_boot, seed=seed, min_users=min_users, m=m)
    return {k: r[k] for k in ("n_dem", "n_rep", "pi", "null_mean", "null_sd", "boot_lo", "boot_hi")}


def top_share(a, c, frac=0.01) -> float:
    """Share of all phrases held by the top ``frac`` of users."""
    tot = np.bincount(a, weights=c.astype(float))
    tot = np.sort(tot[tot > 0])[::-1]
    k = max(1, int(round(frac * len(tot))))
    return float(tot[:k].sum() / tot.sum())


def classifier_auc(a, p, c, side_vec, n_phrases, min_phrases, phrase_users=20, folds=5, n_null=3, seed=0) -> dict:
    """Held-out-user AUC of a logistic regression on user phrase frequencies, with a shuffled-label null."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    C, y, m = build_matrix(a, p, c, side_vec, n_phrases, min_phrases, equal=True, phrase_users=phrase_users)
    C = C.tocsr()

    def auc(labels):
        score = np.zeros(len(labels))
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(C, labels):
            clf = LogisticRegression(C=10.0, solver="liblinear", class_weight="balanced")
            clf.fit(C[tr], labels[tr])
            score[te] = clf.decision_function(C[te])
        return roc_auc_score(labels, score)

    rng = np.random.default_rng(seed)
    real = auc(y)
    null = np.array([auc(rng.permutation(y)) for _ in range(n_null)])
    return {"auc": float(real), "auc_null_mean": float(null.mean()), "auc_null_sd": float(null.std(ddof=1)),
            "n_dem": int((y == 0).sum()), "n_rep": int((y == 1).sum())}

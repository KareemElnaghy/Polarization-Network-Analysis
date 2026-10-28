import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score

from src.gt_runs import transfer as T


def toy(n=400, v=60, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    p = np.full(v, 1.0 / v)
    p_rep, p_dem = p.copy(), p.copy()
    p_rep[:10] *= 3
    p_dem[10:20] *= 3
    rows = [rng.multinomial(60, (p_rep if s else p_dem) / (p_rep if s else p_dem).sum()) for s in y]
    return sparse.csr_matrix(np.array(rows, dtype=np.float32)), y


def test_auc_matches_sklearn():
    rng = np.random.default_rng(1)
    s, y = rng.normal(size=200), rng.integers(0, 2, 200)
    assert abs(T.auc_value(s, y) - roc_auc_score(y, s)) < 1e-9


def test_leanings_and_pi_follow_the_signal():
    X, y = toy()
    q = T.leanings(T.profile_matrix(X, "equal"), y)
    assert q[:10].mean() > 0.5 > q[10:20].mean()
    est = T.frequencies(X) @ q
    assert T.pi_value(est, y) > 0.55
    rng = np.random.default_rng(0)
    r = T.weekly_stat(est, y, "pi", rng, 50, 30, 10)
    assert r["value"] > r["null_mean"] + 5 * r["null_sd"] and abs(r["null_mean"] - 0.5) < 0.02


def test_logit_with_user_weights_separates_and_equal_has_unit_weights():
    X, y = toy()
    F = T.frequencies(X) * T.FEATURE_SCALE
    sw = T.user_weights(X, "cap100")
    assert abs(sw.mean() - 1) < 1e-9 and (T.user_weights(X, "equal") == 1).all()
    coef, b, C = T.fit_logit(F, y, sw, np.random.default_rng(0), cv_users=200)
    assert T.auc_value(F @ coef + b, y) > 0.8

import numpy as np

from src.algorithms.analysis import forecast as fc


def test_perfectly_predictable_change_is_learned():
    rng = np.random.default_rng(0)
    T = 53
    p = rng.normal(size=T).cumsum()
    dp = np.r_[np.nan, np.diff(p)]
    gt = np.zeros(T)
    for s in range(2, T):                     # dGT_s = 2 * dP_{s-1}
        gt[s] = gt[s - 1] + 2 * dp[s - 1]
    f = fc.rolling_forecasts(gt, {"A": p}, first_target=24)
    assert len(f) == 30 and f.t.iloc[0] == 24 and f.t.iloc[-1] == 53
    assert np.abs(f.actual - f.A).max() < 1e-8
    dm = fc.diebold_mariano((f.actual - f.A).to_numpy(), (f.actual - f.persistence).to_numpy())
    assert dm["mean_loss_diff"] < 0


def test_no_lookahead():
    rng = np.random.default_rng(1)
    gt, p = rng.normal(size=53).cumsum(), rng.normal(size=53).cumsum()
    base = fc.rolling_forecasts(gt, {"A": p}, 24)
    gt2 = gt.copy()
    gt2[-1] += 100.0                          # changing the last week must not change earlier forecasts
    alt = fc.rolling_forecasts(gt2, {"A": p}, 24)
    assert np.allclose(base.A[:-1], alt.A[:-1])


def test_dm_identical_errors_is_nan_or_null():
    e = np.random.default_rng(2).normal(size=30)
    assert np.isnan(fc.diebold_mariano(e, e)["p"])

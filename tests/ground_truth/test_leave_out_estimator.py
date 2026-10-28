import time

import numpy as np
import pytest
from scipy import sparse

from src.ground_truth.leave_out_estimator import leave_out_estimator


def synth(n_users=600, n_phrases=300, mixing=0.0, seed=0, tokens=80):
    """Both camps draw phrases from a common distribution; with mixing m < 1 each camp also has its own half."""
    rng = np.random.default_rng(seed)
    y = np.array([i % 2 for i in range(n_users)])
    common = np.ones(n_phrases) / n_phrases
    own_r = np.r_[np.zeros(n_phrases // 2), np.ones(n_phrases - n_phrases // 2)]
    own_d = 1 - own_r
    rows = []
    for i in range(n_users):
        own = (own_r if y[i] else own_d)
        p = (1 - mixing) * common + mixing * own / own.sum()
        rows.append(rng.multinomial(tokens, p))
    return sparse.csr_matrix(np.array(rows)), y


def test_identical_distributions_give_half():
    C, y = synth(mixing=0.0)
    r = leave_out_estimator(C, y, n_perm=30, n_boot=30, min_users=50)
    assert r["pi"] == pytest.approx(0.5, abs=0.02)
    assert r["boot_lo"] <= r["pi"] <= r["boot_hi"]


def test_pi_rises_monotonically_with_separation():
    pis = [leave_out_estimator(synth(mixing=m)[0], synth(mixing=m)[1], n_perm=0, n_boot=0)["pi"]
           for m in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(a < b for a, b in zip(pis, pis[1:]))
    assert pis[-1] > 0.9


def test_shuffled_label_null_is_half_even_for_separated_data():
    C, y = synth(mixing=1.0)
    r = leave_out_estimator(C, y, n_perm=60, n_boot=0)
    assert r["pi"] > 0.9
    assert r["null_mean"] == pytest.approx(0.5, abs=0.02)


def test_few_users_gives_nan_with_reason():
    C, y = synth(n_users=40)
    r = leave_out_estimator(C, y, min_users=50)
    assert np.isnan(r["pi"]) and "fewer than 50" in r["reason"]


def test_matches_a_slow_reference_implementation():
    C, y = synth(n_users=60, n_phrases=40, mixing=0.4, tokens=30)
    D = C.toarray().astype(float)
    m = D.sum(1)
    qhat = np.zeros(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        rho_r = D[keep & (y == 1)].sum(0) / m[keep & (y == 1)].sum()
        rho_d = D[keep & (y == 0)].sum(0) / m[keep & (y == 0)].sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.where(rho_r + rho_d > 0, rho_r / (rho_r + rho_d), 0.5)
        qhat[i] = (D[i] / m[i] * q).sum()
    expected = 0.5 * qhat[y == 1].mean() + 0.5 * (1 - qhat[y == 0]).mean()
    assert leave_out_estimator(C, y, n_perm=0, n_boot=0, min_users=10)["pi"] == pytest.approx(expected, abs=1e-9)


def test_speed_50k_users_100k_phrases():
    rng = np.random.default_rng(1)
    n_users, n_phrases, per_user = 50_000, 100_000, 100
    rows = np.repeat(np.arange(n_users), per_user)
    cols = np.minimum(rng.zipf(1.3, size=rows.size) - 1, n_phrases - 1)
    C = sparse.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n_users, n_phrases)).tocsr()
    y = rng.integers(0, 2, n_users)
    t0 = time.perf_counter()
    r = leave_out_estimator(C, y, n_perm=10, n_boot=10)
    elapsed = time.perf_counter() - t0
    assert C.nnz > 2_000_000
    assert elapsed < 300
    assert r["pi"] == pytest.approx(0.5, abs=0.02)

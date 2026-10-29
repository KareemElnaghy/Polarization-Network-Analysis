import numpy as np
import pytest

from src.ground_truth.dissimilarity import dissimilarity_index


def make_rows(n_users, n_threads, mix, seed=0, per_user=6):
    """mix=0: dem users only in the first half of threads, rep in the second; mix=1: both anywhere."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        label = "dem" if u % 2 == 0 else "rep"
        half = n_threads // 2
        for _ in range(per_user):
            home = rng.integers(0, half) + (0 if label == "dem" else half)
            anywhere = rng.integers(0, n_threads)
            rows.append((f"t{anywhere if rng.random() < mix else home}", f"u{u}", label))
    return rows


def test_separated_camps_give_index_near_one():
    r = dissimilarity_index(make_rows(400, 40, mix=0.0), n_perm=50, min_users=50)
    assert r["observed"] == pytest.approx(1.0, abs=1e-9)
    assert r["excess"] > 0.5


def test_fully_mixed_camps_sit_at_the_null():
    r = dissimilarity_index(make_rows(400, 40, mix=1.0), n_perm=100, min_users=50)
    assert abs(r["excess"]) < 0.05
    assert r["null_mean"] == pytest.approx(r["observed"], abs=0.05)


def test_excess_rises_with_segregation():
    ex = [dissimilarity_index(make_rows(400, 40, mix=m), n_perm=50)["excess"] for m in (1.0, 0.5, 0.0)]
    assert ex[0] < ex[1] < ex[2]


def test_counts_distinct_users_not_comments():
    rows = [("a", "u1", "dem")] * 10 + [("b", "u2", "rep")] + [("a", f"d{i}", "dem") for i in range(60)] + \
           [("b", f"r{i}", "rep") for i in range(60)]
    assert dissimilarity_index(rows, n_perm=5, min_users=50)["observed"] == pytest.approx(1.0)


def test_too_few_users_gives_nan_with_reason():
    r = dissimilarity_index([("a", "u1", "dem"), ("b", "u2", "rep")], min_users=50)
    assert np.isnan(r["observed"]) and "fewer than 50" in r["reason"]

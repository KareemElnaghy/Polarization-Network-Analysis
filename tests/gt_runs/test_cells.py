import numpy as np
import pandas as pd
import pytest

from src.gt_runs.cells import gt1_cell, gt2_cell


def thread_users(n_per_side=80, separated=True, seed=0):
    rng = np.random.default_rng(seed)
    rows, side = [], {}
    for i in range(2 * n_per_side):
        u, s = f"u{i}", ("dem" if i % 2 == 0 else "rep")
        side[u] = s
        for _ in range(4):
            home = rng.integers(0, 10) + (0 if s == "dem" else 10)
            t = home if separated else rng.integers(0, 20)
            rows.append(("politics", f"t{t}", u))
    # comments in a subreddit outside the venue must be ignored
    rows += [("other", f"o{i}", f"u{i}") for i in range(20)]
    return pd.DataFrame(rows, columns=["subreddit", "link_id", "author"]), side


def test_gt1_separated_week_has_index_one_and_counts_threads():
    tu, side = thread_users(separated=True)
    r = gt1_cell(tu, ["politics"], side, None, None, n_perm=30, seed=0, min_users=50)
    assert r["observed"] == pytest.approx(1.0) and r["excess"] > 0.4 and not r["low_n"]
    assert (r["n_dem"], r["n_rep"], r["n_threads"], r["n_threads_both"]) == (80, 80, 20, 0)


def test_gt1_mixed_week_is_near_its_null():
    tu, side = thread_users(separated=False)
    r = gt1_cell(tu, ["politics"], side, None, None, n_perm=60, seed=0, min_users=50)
    assert abs(r["excess"]) < 0.05 and r["n_threads_both"] > 15


def test_gt1_heavy_posters_and_samples_restrict_the_cell():
    tu, side = thread_users()
    heavy = {f"u{i}" for i in range(0, 40)}                      # 20 dem and 20 rep users
    r = gt1_cell(tu, ["politics"], side, heavy, None, n_perm=5, seed=0, min_users=10)
    assert (r["n_dem"], r["n_rep"]) == (60, 60)
    r = gt1_cell(tu, ["politics"], side, None, {"t0", "t1", "t10", "t11"}, n_perm=5, seed=0, min_users=1)
    assert r["n_threads"] == 4


def test_gt1_low_n_cell_is_nan_and_flagged_but_still_returned():
    tu, side = thread_users(n_per_side=10)
    r = gt1_cell(tu, ["politics"], side, None, None, n_perm=5, seed=0, min_users=50)
    assert np.isnan(r["observed"]) and np.isnan(r["excess"]) and r["low_n"]
    assert r["n_dem"] == 10 and r["n_rep"] == 10


def phrase_triplets(n_per_side=120, n_phrases=40, overlap=0.0, seed=0, tokens=60):
    rng = np.random.default_rng(seed)
    a, p, c, side = [], [], [], np.zeros(2 * n_per_side, dtype=np.int8)
    for i in range(2 * n_per_side):
        s = i % 2
        side[i] = s
        own = np.arange(n_phrases // 2) + (n_phrases // 2) * s
        probs = np.full(n_phrases, overlap / n_phrases)
        probs[own] += (1 - overlap) / len(own)
        draws = rng.multinomial(tokens, probs / probs.sum())
        for ph in np.flatnonzero(draws):
            a.append(i), p.append(ph), c.append(draws[ph])
    return np.array(a), np.array(p), np.array(c), side


def test_gt2_separated_vocabularies_give_high_pi_and_null_near_half():
    a, p, c, side = phrase_triplets(overlap=0.0)
    r = gt2_cell(a, p, c, side, None, None, 40, 20, n_perm=30, n_boot=10, seed=0, min_users=50)
    assert r["pi"] > 0.9 and r["null_mean"] == pytest.approx(0.5, abs=0.03) and not r["low_n"]
    assert r["boot_lo"] <= r["pi"] <= r["boot_hi"]


def test_gt2_phrase_filter_and_heavy_posters_change_the_cell():
    a, p, c, side = phrase_triplets(overlap=0.0)
    keep = np.ones(40, dtype=bool)
    keep[:10] = False                                             # drop dem-only phrases
    r_all = gt2_cell(a, p, c, side, None, None, 40, 1, 0, 0, 0, 50)
    r_f = gt2_cell(a, p, c, side, keep, None, 40, 1, 0, 0, 0, 50)
    assert r_f["pi"] != r_all["pi"]
    heavy = np.zeros(len(side), dtype=bool)
    heavy[:20] = True
    r_h = gt2_cell(a, p, c, side, None, heavy, 40, 1, 0, 0, 0, 50)
    assert r_h["n_dem"] + r_h["n_rep"] == len(side) - 20


def test_gt2_users_below_min_phrases_are_dropped_and_low_n_is_nan():
    a, p, c, side = phrase_triplets(n_per_side=30)
    r = gt2_cell(a, p, c, side, None, None, 40, 20, n_perm=5, n_boot=0, seed=0, min_users=50)
    assert np.isnan(r["pi"]) and r["low_n"] and r["n_dem"] == 30 and r["n_rep"] == 30
    r = gt2_cell(a, p, c, side, None, None, 40, 10_000, n_perm=5, n_boot=0, seed=0, min_users=1)
    assert r["low_n"] and r["n_dem"] == 0

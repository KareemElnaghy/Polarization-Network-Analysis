import numpy as np
import pandas as pd

from src.gt_runs import gt1_robust as R
from src.ground_truth.dissimilarity import dissimilarity_index


def rows(n=120, seed=0, segregated=True):
    rng = np.random.default_rng(seed)
    recs = []
    for u in range(n):
        side = u % 2
        for t in rng.choice(np.arange(10) + (side * 10 if segregated else 0), 3, replace=False):
            recs.append((f"t{t}", f"u{u}", side, "politics", 1))
    return pd.DataFrame(recs, columns=["thread", "user", "side", "sub", "n_comments"])


def test_matches_reference_index():
    df = rows()
    c = R.Cell(df)
    ref = dissimilarity_index(df.assign(label=df["side"])[["thread", "user", "label"]], n_perm=1, min_users=10)
    assert abs(c.index(c.side) - ref["observed"]) < 1e-9


def test_segregated_has_excess_and_flipping_shrinks_it():
    rng = np.random.default_rng(1)
    df = rows()
    e = R.Cell(df).excess(rng, 50, 10)["excess"]
    f = R.Cell(R.flip_sides(df, rng, 0.2)).excess(rng, 50, 10)["excess"]
    assert e > 0.3 and f < e
    assert abs(R.Cell(rows(segregated=False)).excess(rng, 50, 10)["excess"]) < 0.1


def test_stratified_shuffle_keeps_labels_inside_strata():
    df = rows()
    c = R.Cell(df)
    strata = c.side.astype(int)  # strata equal the sides
    r = c.excess(np.random.default_rng(2), 20, 10, strata=strata)
    assert abs(r["excess"]) < 1e-9  # shuffling within strata that equal the sides changes nothing


def test_event_test_detects_a_shift():
    s = pd.Series(np.r_[np.zeros(40), np.ones(5) * 3], index=[f"w{i}" for i in range(45)]) + np.random.default_rng(0).normal(0, .1, 45)
    r = R.event_test(s, [f"w{i}" for i in range(40, 45)], n_perm=2000)
    assert r["diff"] > 2.5 and r["p_one_sided"] < 0.01

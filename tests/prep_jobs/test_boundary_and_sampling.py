from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from src.common.text_filters import privacy_overwritten_mask, removed_mask, unusable_text_mask
from src.prep_jobs.jobs.clean_edges_and_sampling import graph_metrics, keep_k, sample_threads, thread_seed
from src.prep_jobs.util import fold_boundary_weeks

TS = pa.timestamp("s", tz="UTC")


def week(d):
    return datetime(2016, *d, tzinfo=timezone.utc)


def monthly(rows):
    return pa.table({"week_start": pa.array([week(r[0]) for r in rows], TS), "author": [r[1] for r in rows],
                     "n": pa.array([r[2] for r in rows], pa.int64())})


def as_dict(t):
    return {(t["week_start"][i].as_py().date().isoformat(), t["author"][i].as_py()): t["n"][i].as_py()
            for i in range(t.num_rows)}


def test_boundary_week_rows_are_summed_into_the_earlier_month():
    # week 2016-01-25 has days in Jan (Jan file) and Feb (Feb file); the other weeks are wholly in one month
    jan = monthly([((1, 25), "a", 3), ((1, 25), "b", 1), ((1, 4), "a", 5)])
    feb = monthly([((1, 25), "a", 2), ((1, 25), "c", 4), ((2, 1), "a", 7), ((2, 29), "a", 1)])
    mar = monthly([((2, 29), "a", 6), ((3, 7), "a", 9)])
    store, saved, loads = {"2016-01": jan, "2016-02": feb, "2016-03": mar}, {}, []
    fold_boundary_weeks(list(store), lambda m: (loads.append(m), store[m])[1], lambda m, t: saved.__setitem__(m, t),
                        ["week_start", "author"], ["n"])
    assert sorted(loads) == sorted(store)  # each month is read once
    got = {m: as_dict(t) for m, t in saved.items()}
    assert got["2016-01"] == {("2016-01-25", "a"): 5, ("2016-01-25", "b"): 1, ("2016-01-04", "a"): 5,
                              ("2016-01-25", "c"): 4}
    assert got["2016-02"] == {("2016-02-01", "a"): 7, ("2016-02-29", "a"): 7}
    assert got["2016-03"] == {("2016-03-07", "a"): 9}
    assert sum(sum(v.values()) for v in got.values()) == sum(sum(t["n"].to_pylist()) for t in store.values())


def test_text_filters():
    overwrite = "This comment has been overwritten by an open source script to protect this user's privacy."
    t = pa.array(["[removed]", "[deleted]", "fine", overwrite, None])
    assert removed_mask(t).to_pylist() == [True, True, False, False, False]
    assert privacy_overwritten_mask(t).to_pylist() == [False, False, False, True, False]
    assert unusable_text_mask(t).to_pylist() == [True, True, False, True, False]


def test_keep_k_is_a_single_pass():
    s = np.array(["a", "a", "b", "c", "d"])
    t = np.array(["b", "b", "a", "a", "e"])
    # activity: a=4, b=3, c=1, d=1, e=1; k=3 keeps a and b, so the edges touching c, d or e go
    assert keep_k(s, t, 3).tolist() == [True, True, True, False, False]


def test_graph_metrics_counts_distinct_pairs_and_components():
    m = graph_metrics(np.array(["a", "a", "b", "x"]), np.array(["b", "b", "c", "y"]))
    assert (m["nodes"], m["pairs"]) == (5, 3)
    assert m["lcc_share"] == pytest.approx(3 / 5)
    assert m["mean_degree"] == pytest.approx(2 * 3 / 5)


def test_random_rounding_makes_sample_share_unbiased():
    rng = np.random.default_rng(0)
    threads = pd.DataFrame({"subreddit": rng.choice(["politics", "other_sub"], 400),
                            "link_id": [f"t{i}" for i in range(400)], "n": rng.integers(1, 60, 400)})
    edges = [1, 2, 4, 8, 16, 32]
    shares = [sample_threads(threads, {"politics"}, edges, 0.05, thread_seed(1, 0.05, f"w{i}")).mean() for i in range(200)]
    assert np.mean(shares) == pytest.approx(0.05, abs=0.004)
    a = sample_threads(threads, {"politics"}, edges, 0.2, thread_seed(1, 0.2, "w"))
    b = sample_threads(threads, {"politics"}, edges, 0.2, thread_seed(1, 0.2, "w"))
    assert (a == b).all()

import numpy as np
import pandas as pd

from src.algorithms.analysis import alignment as al


def test_cluster_boot_diff_recovers_known_shift():
    rng = np.random.default_rng(0)
    n = 4000
    thread = rng.integers(0, 200, n)
    cross = rng.integers(0, 2, n)
    x = rng.normal(size=n) - 0.5 * cross
    r = al.cluster_boot_diff(x, cross, thread, boot=300)
    assert r["ci_lo"] < -0.5 < r["ci_hi"] + 0.15 and r["ci_hi"] < 0
    assert r["n_cross"] + r["n_same"] == n


def test_cluster_boot_diff_ignores_nan():
    x = np.array([1.0, np.nan, 3.0, 5.0])
    r = al.cluster_boot_diff(x, np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]), boot=50)
    assert r["n_cross"] == 1 and r["n_same"] == 2 and abs(r["delta"] - (1.0 - 4.0)) < 1e-12


def test_weekly_alignment_excess_positive_when_negatives_are_cross():
    rng = np.random.default_rng(1)
    n = 600
    lab = rng.choice(["dem", "rep"], size=400)
    s, t = rng.integers(0, 400, n), rng.integers(0, 400, n)
    keep = s != t
    s, t = s[keep], t[keep]
    cross = lab[s] != lab[t]
    df = pd.DataFrame({"week": "w", "source": s.astype(str), "target": t.astype(str), "lab_replier": lab[s], "lab_parent": lab[t],
                       "w": np.where(cross, -1.0, 1.0)})
    # users appear with one label each in this construction
    out = al.weekly_antagonism_alignment(df, "w", n_perm=50).iloc[0]
    assert out.share_neg_cross == 1.0 and out.alignment_excess > 0.3
    shuffled = df.assign(w=rng.choice([-1.0, 1.0], len(df)))
    assert abs(al.weekly_antagonism_alignment(shuffled, "w", n_perm=50).iloc[0].alignment_excess) < 0.1

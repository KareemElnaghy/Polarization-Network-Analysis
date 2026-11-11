from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.algorithms.eval import metrics as M

RUN_A = Path("runs/llm/20261107T121812_pipeA_5pct_full")
RUN_B = Path("runs/llm/20261107T121822_pipeB_5pct_full")


def test_holm_known_values():
    h = M.holm({"a": 0.01, "b": 0.04, "c": 0.03}).set_index("test")
    assert np.allclose(h.p_holm, [0.03, 0.06, 0.06])  # 3*.01, max(.03, 2*.03), max(.06, .04)
    assert h.loc["a", "significant_at_alpha"] and not h.loc["b", "significant_at_alpha"]


def test_partial_removes_confounder():
    rng = np.random.default_rng(0)
    z = rng.normal(size=200)
    x, y = z + 0.1 * rng.normal(size=200), z + 0.1 * rng.normal(size=200)
    assert M.rho(x, y) > 0.9 and abs(M.partial(x, y, z)) < 0.3


def test_block_boot_reproducible():
    x = np.random.default_rng(1).normal(size=40)
    a = M.block_boot(lambda u: np.mean(u), [x], boot=200, seed=0)
    b = M.block_boot(lambda u: np.mean(u), [x], boot=200, seed=0)
    assert a[1:3] == b[1:3]


@pytest.mark.skipif(not (RUN_A.exists() and RUN_B.exists()), reason="run folders not present")
def test_reproduces_committed_full5pct_correlations():
    ref = pd.read_csv("results/llm/rq1_5pct/rq1_5pct_full_correlations.csv")
    gt = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start")
    gt.index = gt.index.astype(str)
    A = pd.read_csv(RUN_A / "weekly_A.csv").set_index("week").reindex(sorted(gt.index))
    B = pd.read_csv(RUN_B / "weekly_B.csv")
    B = B[B.sign_source == "continuous_llm"].set_index("week").reindex(sorted(gt.index))
    for truth in ("gt1_excess", "gt2"):
        for name, s in (("P_A", A.P_A), ("P_B", B.P_B)):
            r = ref[(ref.truth == truth) & (ref.series == name)].iloc[0]
            assert abs(M.rho(s, gt[truth]) - r.rho) < 1e-9 and abs(M.circ_p(s, gt[truth]) - r.perm_p) < 1e-9

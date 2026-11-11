import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from src.algorithms.pipeline_b import sign_inference as si
from src.algorithms.pipeline_c import weights as wc


def _grid():
    rows = [dict(agreement=a, emotional_charge=e, hostility=h, personal_attack=p)
            for a, e, h, p in itertools.product(["agree", "neutral", "disagree"], ["low", "medium", "high"], ["none", "mild", "strong"], [False, True])]
    df = pd.DataFrame(rows)
    pa = df.agreement.map({"agree": [.8, .1, .1], "neutral": [.1, .8, .1], "disagree": [.1, .1, .8]})
    df[["p_agree", "p_neutral", "p_disagree"]] = np.array(pa.tolist())
    return df


def test_c1_total_over_all_54_combinations_and_multiplier_range():
    df = _grid()
    assert len(df) == 54
    m = wc.intensity_multiplier(df)
    assert m.min() == 1.0 and m.max() == 3.5 and m.notna().all()
    assert wc.c1(df).notna().all()


def test_c1_equals_b_when_intensities_zero_and_sign_matches_b():
    df = _grid()
    zero = df[(df.hostility == "none") & (df.emotional_charge == "low") & (~df.personal_attack)]
    assert np.allclose(wc.c1(zero), si.continuous_llm(zero))
    assert (np.sign(wc.c1(df)) == np.sign(si.continuous_llm(df))).all()


def test_c1_binary_drops_neutral():
    df = _grid()
    w = wc.c1_binary(df)
    assert w[df.agreement == "neutral"].isna().all() and (w[df.agreement == "agree"] > 0).all() and (w[df.agreement == "disagree"] < 0).all()


def test_pipeline_c_does_not_read_labels():
    for f in Path("src/algorithms/pipeline_c").glob("*.py"):
        t = f.read_text()
        assert "algorithms.eval" not in t and "processed/labels" not in t and "camp_recovery" not in t, f

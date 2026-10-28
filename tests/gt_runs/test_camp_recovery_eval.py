import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("camp_recovery_eval", Path(__file__).resolve().parents[2] / "scripts" / "camp_recovery_eval.py")
cre = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cre)

N = 400
AUTHORS = [f"u{i}" for i in range(N)]
TRUTH = ["dem" if i % 2 == 0 else "rep" for i in range(N)]
LABELS = pd.DataFrame({"author": AUTHORS, "side_main": TRUTH, "side_gold": [t if i < 100 else None for i, t in enumerate(TRUTH)]})


def pred(sides):
    return pd.DataFrame({"author": AUTHORS, "side": sides})


def row(df, name):
    return df[df["label_set"] == name].iloc[0]


def test_perfect_prediction():
    out = cre.evaluate(pred(TRUTH), LABELS)
    r = row(out, "main")
    assert r["accuracy"] == 1.0 and r["ari"] == pytest.approx(1.0) and not r["swapped"]
    assert (r["tp_dem"], r["fn_dem"], r["tp_rep"], r["fn_rep"]) == (200, 0, 200, 0)
    assert row(out, "gold")["users_compared"] == 100


def test_swapped_sides_are_scored_as_perfect_and_flagged():
    flipped = ["rep" if t == "dem" else "dem" for t in TRUTH]
    r = row(cre.evaluate(pred(flipped), LABELS), "main")
    assert r["accuracy"] == 1.0 and r["swapped"] and r["ari"] == pytest.approx(1.0)


def test_arbitrary_group_names_are_accepted():
    r = row(cre.evaluate(pred(["A" if t == "dem" else "B" for t in TRUTH]), LABELS), "main")
    assert r["accuracy"] == 1.0


def test_random_prediction_is_near_chance():
    rng = np.random.default_rng(0)
    r = row(cre.evaluate(pred(list(rng.choice(["dem", "rep"], N))), LABELS), "main")
    assert 0.4 < r["accuracy"] < 0.65 and abs(r["ari"]) < 0.05


def test_no_overlap_and_wrong_group_count_do_not_crash():
    other = pd.DataFrame({"author": ["zz1", "zz2"], "side": ["dem", "rep"]})
    assert np.isnan(row(cre.evaluate(other, LABELS), "main")["accuracy"])
    three = pred(["a", "b", "c"] * (N // 3) + ["a"])
    assert "expected 2" in row(cre.evaluate(three, LABELS), "main")["note"]

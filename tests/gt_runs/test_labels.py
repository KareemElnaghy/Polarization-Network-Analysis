import numpy as np
import pandas as pd
import pytest

from src.gt_runs.labels import AuthorIndex, gold_sides, posting_sides

PARTY = {"The_Donald": "rep", "Republican": "rep", "hillaryclinton": "dem", "democrats": "dem"}
RULE = {"count_version": "score_ge_1", "min_comments": 10, "share": 0.75}


def counts(rows):
    return pd.DataFrame(rows, columns=["author", "subreddit", "n_ge1"])


def test_sides_follow_the_share_rule_and_minimum():
    c = counts([("r", "The_Donald", 9), ("r", "democrats", 1),            # 0.9 rep, 10 comments -> rep
                ("d", "hillaryclinton", 8), ("d", "The_Donald", 2),       # 0.2 rep -> dem
                ("m", "The_Donald", 5), ("m", "hillaryclinton", 5),       # mixed
                ("s", "The_Donald", 9)])                                  # only 9 comments
    s = posting_sides(c, PARTY, RULE).set_index("author")
    assert s["side"].to_dict() == {"r": "rep", "d": "dem"}
    assert s.loc["r", "share"] == pytest.approx(0.9) and s.loc["r", "n_comments"] == 10


def test_leave_the_venue_out_subtracts_the_venue_counts():
    c = counts([("u", "The_Donald", 40), ("u", "hillaryclinton", 12)])
    assert posting_sides(c, PARTY, RULE).set_index("author").loc["u", "side"] == "rep"
    out = posting_sides(c, PARTY, RULE, venue_subs=["The_Donald"])
    assert out.set_index("author").loc["u", "side"] == "dem"           # only the Clinton comments remain
    small = posting_sides(counts([("v", "The_Donald", 30), ("v", "democrats", 3)]), PARTY, RULE, venue_subs=["The_Donald"])
    assert small.empty                                                  # 3 comments left: unlabelled


def test_unknown_count_version_is_an_error():
    with pytest.raises(KeyError):
        posting_sides(counts([("a", "The_Donald", 20)]), PARTY, {**RULE, "count_version": "nope"})


def test_gold_excludes_third_party_and_conflicting_votes():
    d = pd.DataFrame([("t", "vote", "right"), ("t", "vote", "right"), ("c", "vote", "left"), ("s", "vote", "left"),
                      ("j", "vote", "third"), ("x", "vote", "left"), ("x", "vote", "right"), ("y", "vote", "left"),
                      ("y", "vote", "third"), ("i", "identity", "right")], columns=["author", "kind", "label"])
    g = gold_sides(d).set_index("author")["side"].to_dict()
    assert g == {"t": "rep", "c": "dem", "s": "dem"}


def test_author_index_vectors():
    idx = AuthorIndex(["a", "b", "c"])
    assert idx.side_vector({"a": "rep", "c": "dem", "zz": "rep"}).tolist() == [1, -1, 0]
    assert idx.flag_vector({"b"}).tolist() == [False, True, False]

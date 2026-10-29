from collections import Counter

import numpy as np

from src.prep_jobs.config import job_cfg, load_config
from src.prep_jobs.jobs import phrase_counts as pcj

JC = job_cfg(load_config(), "phrase_counts")
TEXTS = [
    "i think that the republican party is going to win the election in november . http www example com 123 does n't matter",
    "the the of and",
    "bernie sanders supporters bernie sanders",
    "",
    "one",
]


def expected_keys(text, tok):
    out = Counter()
    for phrase, n in pcj.python_phrases(text, JC, tok).items():
        a, b = phrase.split(" ")
        out[pcj.pair_key(tok.code(a), tok.code(b), JC["key_bits"])] += n
    return out


def test_vectorised_phrases_match_a_plain_python_recount():
    tok = pcj.Tokenizer(JC)
    keys, cid, _, _ = pcj.bigram_keys(TEXTS, tok, JC["key_bits"])
    for i, text in enumerate(TEXTS):
        assert Counter(keys[cid == i].tolist()) == expected_keys(text, tok), text


def test_stopword_pairs_urls_numbers_and_short_tokens_are_dropped():
    tok = pcj.Tokenizer(JC)
    phrases = pcj.python_phrases("the of and 123 http www com i think trump wins", JC, tok)
    # dropped tokens do not break adjacency: "and" and "think" become neighbours, and only one is a stopword
    assert set(phrases) == {"and think", "think trump", "trump wins"}


def test_phrases_do_not_cross_comments():
    tok = pcj.Tokenizer(JC)
    keys, cid, _, _ = pcj.bigram_keys(["alpha beta", "gamma delta"], tok, JC["key_bits"])
    assert sorted(cid.tolist()) == [0, 1]


def test_sum_duplicates():
    k, c = pcj.sum_duplicates(np.array([3, 1, 3, 2], dtype=np.uint64), np.array([1, 5, 2, 7], dtype=np.uint32))
    assert k.tolist() == [1, 2, 3] and c.tolist() == [5, 7, 3]

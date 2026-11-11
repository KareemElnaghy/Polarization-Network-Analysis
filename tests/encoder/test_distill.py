import hashlib

from src.algorithms.encoder.distill import split_of


def test_split_is_deterministic_and_roughly_90_5_5():
    ts = [f"t3_{i}" for i in range(20000)]
    s = [split_of(t) for t in ts]
    assert s == [split_of(t) for t in ts]
    assert abs(s.count("test") / len(s) - 0.05) < 0.01 and abs(s.count("val") / len(s) - 0.05) < 0.01
    k = int(hashlib.sha1(b"abc").hexdigest(), 16) % 20
    assert split_of("abc") == ("test" if k == 0 else "val" if k == 1 else "train")

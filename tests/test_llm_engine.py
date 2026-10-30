import pandas as pd
import pytest

from src.algorithms.llm import backends, extract, schema
from src.algorithms.llm.backends import MockBackend, normalise, probs_from_logprobs
from src.algorithms.llm.store import Store

CFG = {"model": "m", "temperature": 0.0, "seed": 0, "prompt_version": "v1", "prob_method": "logprobs",
       "max_chars": {"parent": 20, "body": 20}, "parallel": 2, "max_retries": 2}


def edges(n=6, dup=True):
    rows = [{"week": f"w{i % 2}", "parent": "p" + str(i), "body": "b" + str(i)} for i in range(n)]
    return pd.DataFrame(rows + (rows[:2] if dup else []))


def test_dedup_and_cache(tmp_path):
    s, b = Store(str(tmp_path / "s.db")), MockBackend()
    c = extract.run(edges(), b, s, CFG)
    assert c["unique"] == 6 and b.calls == 6 and c["done"] == 6
    extract.run(edges(), b, s, CFG)
    assert b.calls == 6  # all cache hits


def test_retry_on_invalid_json(tmp_path):
    s, b = Store(str(tmp_path / "s.db")), MockBackend(fail_first=2)
    c = extract.run(edges(2, False), b, s, CFG)
    assert c["done"] == 2 and c["failed"] == 0 and (s.table(extract.context(CFG, b)).retries == 2).all()
    c = extract.run(edges(2, False).assign(body=["x1", "x2"]), MockBackend(fail_first=9), s, CFG)
    assert c["failed"] == 2


def test_resume_after_kill(tmp_path):
    s, b = Store(str(tmp_path / "s.db")), MockBackend()
    n = []
    extract.run(edges(), b, s, {**CFG, "parallel": 1}, stop=lambda: len(n) >= 3 or n.append(1) or False)
    done = len(s.have(extract.context(CFG, b)))
    assert 0 < done < 6
    b2 = MockBackend()
    extract.run(edges(), b2, s, CFG)
    assert b2.calls == 6 - done


def test_cache_key_sensitivity(tmp_path):
    s, b = Store(str(tmp_path / "s.db")), MockBackend()
    extract.run(edges(), b, s, CFG)
    for k, v in [("model", "m2"), ("prompt_version", "v2"), ("temperature", 0.5), ("seed", 1)]:
        before = b.calls
        extract.run(edges(), b, s, {**CFG, k: v})
        assert b.calls == before + 6, k


def test_truncation_changes_identity():
    mc = {"parent": 3, "body": 3}
    assert schema.pair_hash("abcdef", "x", mc) == schema.pair_hash("abcxyz", "x", mc)  # same visible text -> one call


def test_normalise():
    assert normalise({"agree": 1, "disagree": 3}) == {"agree": 0.25, "neutral": 0.0, "disagree": 0.75}
    assert normalise({}) is None


def test_logprobs_extraction():
    toks = [{"token": t, "logprob": 0.0} for t in ['{"', 'agreement', '":', ' "']]
    toks.append({"token": "dis", "logprob": -0.1, "top_logprobs": [
        {"token": "dis", "logprob": -0.1}, {"token": "agree", "logprob": -2.3}, {"token": "neutral", "logprob": -4.6},
        {"token": "The", "logprob": -1.0}]})
    p = probs_from_logprobs(toks)
    assert abs(sum(p.values()) - 1) < 1e-9 and p["disagree"] > 0.8


def test_validate():
    ok = {"agreement": "agree", "emotional_charge": "low", "hostility": "none", "personal_attack": False}
    assert backends.validate(ok) == ok
    with pytest.raises(backends.InvalidOutput):
        backends.validate({**ok, "hostility": "x"})


def test_prompt_has_no_forbidden_fields():
    msg = schema.user_message("parent text", "reply text", CFG["max_chars"]) + schema.SYSTEM_PROMPT
    low = msg.lower()
    for w in ("subreddit", "username", "r/", "u/", "democrat label", "ground truth"):
        assert w not in low
    assert set(schema.SCHEMA["properties"]) == {"agreement", "emotional_charge", "hostility", "personal_attack"}


def test_export_manifest(tmp_path):
    s = Store(str(tmp_path / "s.db"))
    extract.run(edges(), MockBackend(), s, CFG)
    f = s.export_parquet(str(tmp_path / "shards"))
    assert len(pd.read_parquet(f)) == 6 and (tmp_path / "shards" / "MANIFEST.tsv").exists()


def test_run_folder_and_index(tmp_path):
    from src.algorithms.llm import provenance as pv
    d = pv.new_run("t", {"seed": 1, "model": "m"}, [], root=str(tmp_path / "runs"))
    (d / "out.csv").write_text("a\n1\n")
    pv.finish_run(d, "done", {"x": 1}, "test", index_dir=str(tmp_path / "res"))
    m = __import__("json").loads((d / "manifest.json").read_text())
    assert m["status"] == "done" and "out.csv" in m["outputs"] and m["git_sha"]
    assert d.name in (tmp_path / "res" / "INDEX.md").read_text()

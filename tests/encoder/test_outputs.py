"""Encoder rows written in the store schema must parse through Pipeline A's rule table and Pipeline B's sign functions (SPEC_ENCODER section 4)."""
import numpy as np
import pandas as pd

from src.algorithms.encoder import model as em
from src.algorithms.encoder import outputs as eo
from src.algorithms.llm.store import Store
from src.algorithms.pipeline_a import aggregate as pa
from src.algorithms.pipeline_b import sign_inference as si


def _pred(n, seed=0):
    r = np.random.default_rng(seed)
    sm = lambda x: np.exp(x) / np.exp(x).sum(1, keepdims=True)
    return {"agree": sm(r.normal(size=(n, 3))).astype(np.float32), "emo": sm(r.normal(size=(n, 3))), "host": sm(r.normal(size=(n, 3))),
            "attack": r.random(n).astype(np.float32)}


def test_rows_roundtrip_and_parse(tmp_path):
    n = 40
    pred = _pred(n)
    store = Store(str(tmp_path / "enc.sqlite"))
    ctx = eo.ctx_for("microsoft/deberta-v3-base")
    assert ctx["model"] == "enc-deberta-v3-base-distill-v1" and ctx["quant"] == "fp16" and ctx["prompt_version"] == "enc-v1"
    assert eo.write(store, [f"h{i}" for i in range(n)], pred, ctx) == n
    t = store.table(ctx)
    assert len(t) == n and set(t.agreement) <= {"agree", "neutral", "disagree"}
    np.testing.assert_allclose(t.p_agree.to_numpy(), pred["agree"][:, 0], rtol=1e-6)
    assert (t.agreement.to_numpy() == np.array(eo.AGREE)[pred["agree"].argmax(1)]).all()
    rules = pa.load_rules()
    s = pa.score_edges(t.assign(week="2016-01-04"), rules)
    assert s.score.between(0, 10).all()
    assert pa.aggregate_weekly(s, rules).n.iloc[0] == n
    w = si.continuous_llm(t)
    assert np.isfinite(w).all() and (w.abs() <= 1).all()
    b = si.binary_llm(t)
    assert set(b.dropna().unique()) <= {-1.0, 1.0}


def test_length_batches_cover_all_once():
    lens = np.random.default_rng(0).integers(5, 400, 1000)
    bs = em.length_batches(lens, 32, 0)
    allidx = np.concatenate(bs)
    assert sorted(allidx.tolist()) == list(range(1000))
    assert max(len(b) for b in bs) <= 32

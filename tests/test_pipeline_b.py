from pathlib import Path

import numpy as np
import pandas as pd

from src.algorithms.pipeline_b import sign_inference as si
from src.algorithms.pipeline_b import structural as st
from src.algorithms.pipeline_b.frustration import csr_from_edges, exact_frustration, frustration


def two_camps(n=30, p=0.3, seed=0):
    """Random graph where positive edges are inside camps and negative across: perfectly balanced."""
    rng = np.random.default_rng(seed)
    camp = rng.integers(0, 2, n)
    iu = np.triu_indices(n, 1)
    keep = rng.random(len(iu[0])) < p
    u, v = iu[0][keep], iu[1][keep]
    return n, u, v, np.where(camp[u] == camp[v], 1.0, -1.0), camp


def graph(n, u, v, w):
    return {"nodes": np.arange(n), "u": u, "v": v, "w": w, "count": np.ones(len(w))}


def test_balanced_graph_zero_frustration_and_recovers_camps():
    n, u, v, w, camp = two_camps()
    F, s = frustration(csr_from_edges(n, u, v, w))
    assert F == 0
    assert (np.sign(s) == np.where(camp == camp[0], 1, -1) * np.sign(s[0])).all()  # camps recovered up to side swap
    out = st.normalised_frustration(graph(n, u, v, w), R=10, restarts=5, seed=0)
    assert out["F_obs"] == 0 and out["P_B"] == 1.0


def test_all_positive_not_polarized():
    n, u, v, w, _ = two_camps()
    out = st.normalised_frustration(graph(n, u, v, np.ones_like(w)), R=5, restarts=3, seed=0)
    assert out["F_obs"] == 0 and out["P_B"] == 0.0


def test_random_signs_have_low_P_B():
    n, u, v, w, _ = two_camps(60, 0.2)
    rng = np.random.default_rng(1)
    out = st.normalised_frustration(graph(n, u, v, rng.choice([-1.0, 1.0], len(w))), R=20, restarts=5, seed=0)
    assert abs(out["P_B"]) < 0.15


def test_heuristic_not_below_exact_and_close():
    gaps = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n, p = 14, 0.35
        iu = np.triu_indices(n, 1)
        keep = rng.random(len(iu[0])) < p
        u, v = iu[0][keep], iu[1][keep]
        w = rng.normal(size=len(u))
        Fh, _ = frustration(csr_from_edges(n, u, v, w), restarts=10, seed=seed)
        Fe = exact_frustration(n, u, v, w)
        assert Fh >= Fe - 1e-9
        gaps.append(Fh - Fe)
    assert np.mean(np.array(gaps) <= 1e-9) >= 0.75  # optimal on most tiny graphs


def test_sign_shuffle_keeps_topology_and_rewire_keeps_signed_degrees():
    n, u, v, w, _ = two_camps(40, 0.2)
    g = graph(n, u, v, w * np.random.default_rng(2).uniform(0.5, 1, len(w)))
    rng = np.random.default_rng(0)
    sh = st.sign_shuffle(g, rng)
    assert (sh["u"] == g["u"]).all() and (sh["v"] == g["v"]).all() and np.allclose(np.sort(sh["w"]), np.sort(g["w"]))
    rw = st.rewire(g, rng)

    def degs(x):
        d = np.zeros((n, 2))
        for a, b, ww in zip(x["u"], x["v"], x["w"]):
            d[a, ww < 0] += 1; d[b, ww < 0] += 1
        return d
    assert (degs(rw) == degs(g)).all() and (rw["u"] != rw["v"]).all()
    assert len({(min(a, b), max(a, b)) for a, b in zip(rw["u"], rw["v"])}) == len(rw["u"])
    assert (rw["u"] != g["u"]).any()


def test_build_graph_mean_and_giant():
    df = pd.DataFrame({"source": ["a", "b", "a", "c", "x", "y", "z"], "target": ["b", "a", "c", "c", "y", "z", "x"],
                       "w": [1.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0]})  # a-b mean .5 (2 interactions); c->c self loop dropped
    g = st.build_graph(df)
    assert len(g["nodes"]) == 6 and len(g["w"]) == 5 and sorted(g["count"])[-1] == 2
    assert len(st.giant(g)["nodes"]) == 3  # {x,y,z} triangle ties with {a,b,c} path; either has 3 nodes


def test_sign_sources():
    df = pd.DataFrame({"p_agree": [.8, .1, .1], "p_neutral": [.1, .8, .1], "p_disagree": [.1, .1, .8], "body": ["great", "ok", "terrible awful"]})
    assert np.allclose(si.continuous_llm(df), [(.7) * .9, 0.0, -.7 * .9])
    b = si.binary_llm(df)
    assert b[0] == 1 and np.isnan(b[1]) and b[2] == -1
    assert (si.unsigned(df) == 1).all()
    assert si.sentiment(df)[0] > 0 > si.sentiment(df)[2]
    flipped = si.noise_injection(pd.Series(np.ones(2000)), 0.2, seed=0)
    assert abs((flipped < 0).mean() - 0.2) < 0.03


def test_pipelines_do_not_read_labels():
    for pkg in ["pipeline_a", "pipeline_b", "llm"]:
        for f in Path("src/algorithms", pkg).glob("*.py"):
            t = f.read_text()
            assert "camp_recovery" not in t and "processed/labels" not in t and "algorithms.eval" not in t and "..eval" not in t, f


def test_camp_recovery_perfect_and_random():
    from src.algorithms.eval import camp_recovery as cr
    rng = np.random.default_rng(0)
    users = [f"u{i}" for i in range(200)]
    y = pd.Series(rng.integers(0, 2, 200), index=users)
    perfect = pd.DataFrame({"week": "w", "sign_source": "s", "user": users, "side": np.where(y.values == 1, 1, -1) * -1})  # swapped sides
    rand = perfect.assign(side=rng.choice([-1, 1], 200), week="r")
    out = cr.evaluate(pd.concat([perfect, rand]), y, n_null=50).set_index("week")
    assert out.loc["w", "balanced_acc"] == 1.0 and out.loc["r", "balanced_acc"] < out.loc["r", "null_p95"] + 0.05

"""Pipeline A: heuristic edge score from the shared LLM attributes, then weekly aggregation."""
import itertools

import pandas as pd
import yaml

from ..llm.schema import SCHEMA


def load_rules(path: str = "configs/llm/pipeline_a_rules.yaml") -> dict:
    return yaml.safe_load(open(path))


def edge_score(f: dict, rules: dict) -> int:
    """f: {agreement, emotional_charge, hostility, personal_attack}. Total over all attribute combinations."""
    s = (rules["agreement"][f["agreement"]] + rules["emotional_charge"][f["emotional_charge"]]
         + rules["hostility"][f["hostility"]] + rules["personal_attack"][bool(f["personal_attack"])])
    return int(min(10, max(0, s)))


def all_combinations() -> list[dict]:
    props = SCHEMA["properties"]
    vals = [props[k]["enum"] if "enum" in props[k] else [False, True] for k in props]
    return [dict(zip(props, v)) for v in itertools.product(*vals)]


def score_edges(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """df has the four attribute columns; adds score, score_hostility_only (0-10) and is_disagree."""
    out = df.copy()
    out["score"] = [edge_score(r, rules) for r in df[list(SCHEMA["properties"])].to_dict("records")]
    out["score_hostility_only"] = df.hostility.map(rules["hostility_only"])
    out["is_disagree"] = (df.agreement == "disagree").astype(float)
    return out


def aggregate_weekly(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """df has week + the columns from score_edges. P_A = mean score; also median, share >= threshold, n, variants."""
    g = df.groupby("week")
    w = pd.DataFrame({"P_A": g.score.mean(), "median": g.score.median(),
                      "share_high": g.score.apply(lambda s: (s >= rules["high_threshold"]).mean()),
                      "n": g.size(), "A_disagree": g.is_disagree.mean(), "A_hostility": g.score_hostility_only.mean()})
    w["low_n"] = w.n < rules["min_edges_per_week"]
    return w.reset_index()

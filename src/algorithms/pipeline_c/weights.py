"""Pipeline C edge weights (SPEC_PIPELINE_C). Pure functions of the per-edge LLM/encoder outputs: no labels, GT or eval imports (firewall).
C1: w = w_B * m with m = 1 + h + a + 0.5 e. Coefficients are fixed by the spec and never tuned."""
import numpy as np
import pandas as pd

H = {"none": 0.0, "mild": 0.5, "strong": 1.0}
E = {"low": 0.0, "medium": 0.5, "high": 1.0}


def guo_weight(df: pd.DataFrame) -> pd.Series:
    """Pipeline B's continuous weight, w_B = (p_agree - p_disagree)(1 - p_neutral)."""
    return (df.p_agree - df.p_disagree) * (1 - df.p_neutral)


def intensity_multiplier(df: pd.DataFrame) -> pd.Series:
    """m = 1 + h + a + 0.5 e in [1, 3.5]; h from hostility, a = personal_attack, e from emotional_charge."""
    h = df.hostility.map(H)
    e = df.emotional_charge.map(E)
    a = df.personal_attack.astype(float)
    m = 1.0 + h + a + 0.5 * e
    if m.isna().any():
        raise ValueError("unmapped attribute value(s) in hostility / emotional_charge / personal_attack")
    return m


def c1(df: pd.DataFrame) -> pd.Series:
    """Intensity-weighted Guo weight (headline Pipeline C variant)."""
    return guo_weight(df) * intensity_multiplier(df)


def c1_binary(df: pd.DataFrame) -> pd.Series:
    """Robustness variant C1-bin: sign(argmax agreement) * m, neutral argmax dropped (NaN)."""
    am = df[["p_agree", "p_neutral", "p_disagree"]].to_numpy().argmax(1)
    sign = pd.Series(np.select([am == 0, am == 2], [1.0, -1.0], np.nan), index=df.index)
    return sign * intensity_multiplier(df)


VARIANTS = {"C1": c1, "C1bin": c1_binary, "_Bparity": guo_weight}  # _Bparity: plumbing check only (must reproduce Pipeline B)

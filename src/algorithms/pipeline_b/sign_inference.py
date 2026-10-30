"""Sign sources: edge weight/sign from the cached extraction (no new LLM calls) or from lexicon sentiment. NaN = edge dropped."""
import numpy as np
import pandas as pd


def continuous_llm(df: pd.DataFrame) -> pd.Series:
    """Guo weight w = (p_agree - p_disagree) * (1 - p_neutral)."""
    return (df.p_agree - df.p_disagree) * (1 - df.p_neutral)


def binary_llm(df: pd.DataFrame, neutral: str = "drop") -> pd.Series:
    """argmax label -> +1 / -1; neutral dropped (default) or mapped to +1."""
    am = df[["p_agree", "p_neutral", "p_disagree"]].values.argmax(1)
    w = pd.Series(np.select([am == 0, am == 2], [1.0, -1.0], np.nan), index=df.index)
    return w.fillna(1.0) if neutral == "plus" else w


def sentiment(df: pd.DataFrame) -> pd.Series:
    """VADER compound score of the reply text (no LLM, no parent context)."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    return pd.Series([an.polarity_scores(t)["compound"] for t in df.body], index=df.index)


def unsigned(df: pd.DataFrame) -> pd.Series:
    """Degenerate control: every edge +1 (never frustrated, so P_B is 0 by construction)."""
    return pd.Series(1.0, index=df.index)


def noise_injection(w: pd.Series, p: float, seed: int = 0) -> pd.Series:
    """Flip the sign of a random share p of edges (for sign-quality experiments)."""
    flip = np.random.default_rng(seed).random(len(w)) < p
    return w.where(~flip, -w)


SOURCES = {"continuous_llm": continuous_llm, "binary_llm": binary_llm, "sentiment": sentiment, "unsigned": unsigned}

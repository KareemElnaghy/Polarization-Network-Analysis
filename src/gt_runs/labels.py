"""Who counts as dem or rep: posting-based sides from Job G's counts, and self-declared votes (gold).

Sides are ``dem`` / ``rep`` (earlier outputs say left / right). Mixed and unlabelled users get no side.
Leave-the-venue-out: for a venue the user's side is computed after subtracting the venue's own comments
from their counts, so measuring inside a tagged subreddit never uses that subreddit to build the label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.prep_jobs.data import label_counts
from src.prep_jobs.jobs.label_versions import VERSION_COLS

SIDE_OF = {"left": "dem", "right": "rep"}


def count_column(count_version: str) -> str:
    if count_version not in VERSION_COLS:
        raise KeyError(f"count version {count_version!r} was not written by label_versions: {list(VERSION_COLS)}")
    return VERSION_COLS[count_version]


def posting_sides(counts: pd.DataFrame, party: dict[str, str], rule: dict, venue_subs: list[str] | tuple = ()) -> pd.DataFrame:
    """Sides from per-user-per-sub ``counts`` (author, subreddit, n_*...), leaving ``venue_subs`` out.

    Returns ``author, side, share, n_comments`` for users who are right (rep share >= ``share``) or left
    (<= 1 - share) on at least ``min_comments`` comments outside the venue; ``share`` is the rep share.
    """
    col = count_column(rule["count_version"])
    df = counts[~counts["subreddit"].isin(list(venue_subs))] if len(venue_subs) else counts
    df = df.assign(party=df["subreddit"].map(party))
    p = df.pivot_table(index="author", columns="party", values=col, aggfunc="sum", fill_value=0)
    p = p.reindex(columns=["rep", "dem"], fill_value=0)
    lab = label_counts(p["rep"], p["dem"], rule["min_comments"], rule["share"])
    lab = lab[lab["label"].isin(SIDE_OF)]
    return pd.DataFrame({"author": lab.index.astype(str), "side": lab["label"].map(SIDE_OF).to_numpy(),
                         "share": lab["share"].to_numpy(), "n_comments": lab["n"].to_numpy()})


def gold_sides(declarations: pd.DataFrame) -> pd.DataFrame:
    """Sides from self-declared votes: Trump means rep, Clinton or Sanders means dem.

    ``declarations`` has ``author, kind, label`` (label right = Trump, left = Clinton/Sanders, third = Johnson/Stein).
    A user whose vote declarations include a third-party vote or more than one distinct label is excluded.
    """
    v = declarations[declarations["kind"] == "vote"]
    labels = v.groupby("author")["label"].agg(lambda s: frozenset(s))
    n = v.groupby("author").size()
    single = labels[labels.map(lambda s: s in (frozenset({"left"}), frozenset({"right"})))]
    side = single.map(lambda s: SIDE_OF[next(iter(s))])
    return pd.DataFrame({"author": single.index.astype(str), "side": side.to_numpy(), "share": np.nan,
                         "n_comments": n.reindex(single.index).to_numpy()})


def label_path(root: Path, venue: str, label_set: str) -> Path:
    return Path(root) / f"venue={venue}" / f"set={label_set}.parquet"


def load_side_map(root: Path, venue: str, label_set: str) -> dict[str, str]:
    t = pd.read_parquet(label_path(root, venue, label_set), columns=["author", "side"])
    return dict(zip(t["author"], t["side"]))


def heavy_set(root: Path, week: str) -> set[str]:
    f = Path(root) / "heavy_posters.parquet"
    t = pd.read_parquet(f, filters=[("week_start", "==", week)], columns=["author"])
    return set(t["author"])


class AuthorIndex:
    """Stable integer ids for authors (the sorted ``user_summary`` authors that Job A used)."""

    def __init__(self, authors: list[str]) -> None:
        self.authors = authors
        self.index = {a: i for i, a in enumerate(authors)}

    def side_vector(self, side_map: dict[str, str]) -> np.ndarray:
        """int8 array over all authors: -1 no side, 0 dem, 1 rep."""
        v = np.full(len(self.authors), -1, dtype=np.int8)
        for a, s in side_map.items():
            i = self.index.get(a)
            if i is not None:
                v[i] = 1 if s == "rep" else 0
        return v

    def flag_vector(self, names: set[str]) -> np.ndarray:
        v = np.zeros(len(self.authors), dtype=bool)
        for a in names:
            i = self.index.get(a)
            if i is not None:
                v[i] = True
        return v

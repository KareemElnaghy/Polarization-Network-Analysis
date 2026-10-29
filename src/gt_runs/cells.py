"""Per-week inputs and the two ground-truth cell computations shared by the GT jobs.

A *cell* is (week, venue, label set, sample version). GT#1 uses ``thread_user`` rows; GT#2 uses phrase
counts as ``(author_idx, phrase_id, count)`` triplets. Cells with fewer than ``min_users_per_side`` labelled users
on a side come back as NaN with ``low_n`` True (the row is kept).
"""

from __future__ import annotations

import gc
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy import sparse

from src.ground_truth.dissimilarity import dissimilarity_index
from src.ground_truth.leave_out_estimator import leave_out_estimator
from src.prep_jobs.config import data_path


# ------------------------------------------------------------------ inputs
def read_thread_user(cfg: dict, week: str) -> pd.DataFrame:
    """The week's ``thread_user`` rows: subreddit, link_id, author."""
    f = data_path(cfg, "thread_user") / f"week={week}" / "part.parquet"
    return pq.read_table(f, columns=["subreddit", "link_id", "author"]).to_pandas()


def sampled_links(cfg: dict, week: str, rates: list[float]) -> dict[float, set[str]]:
    t = pq.read_table(data_path(cfg, "edges_sampled") / "sampled_threads.parquet",
                      filters=[("week_start", "==", week)], columns=["link_id", "rate"]).to_pandas()
    return {r: set(t.loc[np.isclose(t["rate"], r), "link_id"]) for r in rates}


def phrase_month(week: str, cfg: dict) -> str:
    """Phrase-count month file holding ``week`` (the first partial week sits in January)."""
    m = week[:7]
    return m if m in cfg["months"] else cfg["months"][0]


def read_phrase_week(cfg: dict, week: str) -> pa.Table:
    """The week's phrase counts (week_start, subreddit, author, phrase_id, count); strings stay dictionary-encoded."""
    f = data_path(cfg, "phrase_counts") / f"month={phrase_month(week, cfg)}" / "counts.parquet"
    typ = pq.read_schema(f).field("week_start").type
    start = datetime.fromisoformat(week).replace(tzinfo=timezone.utc)
    return pq.read_table(f, filters=pc.field("week_start") == pa.scalar(start, typ))


def keep_phrase_mask(cfg: dict, exclude_tokens: list[str]) -> tuple[np.ndarray, int]:
    """Boolean array over phrase ids: False where either word is a contraction fragment; and how many dropped."""
    vocab = pq.read_table(data_path(cfg, "phrase_counts") / "vocab.parquet", columns=["phrase_id", "phrase"]).to_pandas()
    bad = set(exclude_tokens)
    drop = vocab["phrase"].fillna("").map(lambda p: any(w in bad for w in p.split(" ")))
    keep = np.ones(int(vocab["phrase_id"].max()) + 1, dtype=bool)
    keep[vocab.loc[drop, "phrase_id"].to_numpy()] = False
    return keep, int(drop.sum())


class PhraseWeek:
    """One week's phrase counts decoded once: global author ids, phrase ids, counts and subreddit codes."""

    def __init__(self, table: pa.Table, author_index) -> None:
        df = table.select(["subreddit", "author", "phrase_id", "count"]).to_pandas()
        self.sub_categories = df["subreddit"].cat.categories
        self.sub_codes = df["subreddit"].cat.codes.to_numpy()
        cats = df["author"].cat.categories
        to_idx = np.fromiter((author_index.index.get(a, -1) for a in cats), dtype=np.int64, count=len(cats))
        self.author = to_idx[df["author"].cat.codes.to_numpy()]
        self.phrase = df["phrase_id"].to_numpy()
        self.count = df["count"].to_numpy()
        del df

    def rows(self, subs: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(author_idx, phrase_id, count)`` for rows in ``subs`` by authors known to the author index."""
        ok = self.sub_categories.isin(subs)[self.sub_codes] & (self.author >= 0)
        return self.author[ok], self.phrase[ok], self.count[ok]


# ------------------------------------------------------------------ GT#1
def gt1_cell(tu: pd.DataFrame, subs: list[str], side_map: dict[str, str], heavy: set[str] | None,
             links: set[str] | None, n_perm: int, seed: int, min_users: int) -> dict:
    """Dissimilarity index for one cell. ``links`` restricts to sampled threads (None = full week)."""
    df = tu[tu["subreddit"].isin(subs)]
    if links is not None:
        df = df[df["link_id"].isin(links)]
    label = df["author"].map(side_map)
    df = df.assign(label=label)
    df = df[df["label"].notna()]
    if heavy:
        df = df[~df["author"].isin(heavy)]
    rows = pd.DataFrame({"thread": df["link_id"], "user": df["author"], "label": df["label"]})
    r = dissimilarity_index(rows, n_perm=n_perm, seed=seed, min_users=min_users)
    if len(rows):
        both = rows.groupby("thread")["label"].nunique()
        n_both = int((both == 2).sum())
    else:
        n_both = 0
    return {"n_dem": r["n_dem"], "n_rep": r["n_rep"], "n_threads": r["n_threads"], "n_threads_both": n_both,
            "observed": r["observed"], "null_mean": r["null_mean"], "null_sd": r["null_sd"], "excess": r["excess"],
            "low_n": bool(np.isnan(r["observed"]))}


# ------------------------------------------------------------------ GT#2
def gt2_cell(author_idx: np.ndarray, phrase_id: np.ndarray, count: np.ndarray, side_vec: np.ndarray,
             keep_phrase: np.ndarray | None, heavy_vec: np.ndarray | None, n_phrases: int, min_phrases: int,
             n_perm: int, n_boot: int, seed: int, min_users: int, cap: float | None = None) -> dict:
    """Leave-out estimator for one cell from ``(author, phrase, count)`` triplets.

    ``cap`` (variant only, not used by the main runs): scale each user's counts down so that nobody contributes
    more than ``cap`` in-vocabulary phrases; very active users otherwise dominate the pooled frequencies.
    """
    sel = side_vec[author_idx] >= 0
    if keep_phrase is not None:
        sel &= keep_phrase[phrase_id]
    if heavy_vec is not None:
        sel &= ~heavy_vec[author_idx]
    a, p, c = author_idx[sel], phrase_id[sel], count[sel].astype(float)
    if cap:
        totals = np.bincount(a, weights=c, minlength=len(side_vec))
        c = c * np.minimum(1.0, cap / np.maximum(totals[a], 1.0))
    users, inv = np.unique(a, return_inverse=True)
    C = sparse.csr_matrix((c, (inv, p)), shape=(len(users), n_phrases))
    enough = np.asarray(C.sum(axis=1)).ravel() >= min_phrases
    C, users = C[enough], users[enough]
    y = side_vec[users].astype(float)
    r = leave_out_estimator(C, y, n_perm=n_perm, n_boot=n_boot, seed=seed, min_users=min_users)
    out = {"n_dem": r["n_dem"], "n_rep": r["n_rep"], "pi": r["pi"], "null_mean": r["null_mean"], "null_sd": r["null_sd"],
           "boot_lo": r["boot_lo"], "boot_hi": r["boot_hi"], "low_n": bool(np.isnan(r["pi"]))}
    del C, a, p, c, users
    gc.collect()
    return out

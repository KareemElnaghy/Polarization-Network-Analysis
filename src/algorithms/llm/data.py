"""Load sampled shared-venue edges as LLM input (cleaned text only; no users, subreddits or labels leave this function)."""
import pandas as pd
import pyarrow.dataset as ds

SHARED = ["politics", "PoliticalDiscussion"]  # config/gt_runs.yaml venues.shared


def load_shared(rate: float, root: str = "data/processed/edges_sampled", keep_ids: bool = True) -> pd.DataFrame:
    cols = ["comment_id", "link_id", "week", "body_clean", "parent_body_clean", "reply_removed", "parent_removed", "subreddit"]
    t = ds.dataset(f"{root}/rate={rate}", format="parquet", partitioning="hive").to_table(columns=cols).to_pandas()
    t = t[t.subreddit.isin(SHARED) & ~t.reply_removed & ~t.parent_removed]
    t = t.rename(columns={"body_clean": "body", "parent_body_clean": "parent"})
    t = t[(t.body.str.strip() != "") & (t.parent.str.strip() != "")]
    return t[["comment_id", "link_id", "week", "parent", "body"]].reset_index(drop=True)


def load_full(weeks=None, root: str = "data/processed/edges_clean", users: bool = False) -> pd.DataFrame:
    """P2.1: the full cleaned shared venue with the same filters as load_shared (drop removed/empty). weeks: list of week strings or None.
    Same output columns as load_shared (plus source/target if users=True); truncation is applied by the caller via schema.clip."""
    import pyarrow.compute as pc
    cols = ["comment_id", "link_id", "week", "body_clean", "parent_body_clean"] + (["source", "target"] if users else [])
    f = pc.field("subreddit").isin(SHARED) & ~pc.field("reply_removed") & ~pc.field("parent_removed")
    if weeks is not None:
        f = f & pc.field("week").isin([str(w) for w in weeks])
    t = ds.dataset(root, format="parquet", partitioning="hive").to_table(columns=cols, filter=f).to_pandas()
    t = t.rename(columns={"body_clean": "body", "parent_body_clean": "parent"})
    t["week"] = t.week.astype(str)
    t = t[(t.body.str.strip() != "") & (t.parent.str.strip() != "")]
    return t.reset_index(drop=True)


def load_users(rate: float, root: str = "data/processed/edges_sampled") -> pd.DataFrame:
    """comment_id -> (source, target) pseudonymised user ids. Used by Pipeline B only; never passed to the LLM."""
    t = ds.dataset(f"{root}/rate={rate}", format="parquet", partitioning="hive").to_table(columns=["comment_id", "source", "target"])
    return t.to_pandas()

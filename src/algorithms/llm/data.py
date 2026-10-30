"""Load sampled shared-venue edges as LLM input (cleaned text only; no users, subreddits or labels leave this function)."""
import pandas as pd
import pyarrow.dataset as ds

SHARED = ["politics", "PoliticalDiscussion"]  # config/gt_runs.yaml venues.shared


def load_shared(rate: float, root: str = "data/processed/edges_sampled", keep_ids: bool = True) -> pd.DataFrame:
    cols = ["comment_id", "week", "body_clean", "parent_body_clean", "reply_removed", "parent_removed", "subreddit"]
    t = ds.dataset(f"{root}/rate={rate}", format="parquet", partitioning="hive").to_table(columns=cols).to_pandas()
    t = t[t.subreddit.isin(SHARED) & ~t.reply_removed & ~t.parent_removed]
    t = t.rename(columns={"body_clean": "body", "parent_body_clean": "parent"})
    t = t[(t.body.str.strip() != "") & (t.parent.str.strip() != "")]
    return t[["comment_id", "week", "parent", "body"]].reset_index(drop=True)

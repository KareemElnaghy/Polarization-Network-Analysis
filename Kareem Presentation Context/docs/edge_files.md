# Edge files: column guide

Reply edges from the Reddit Politosphere, 2016. Each row is one reply to a comment, directed from the person replying to the author of the comment they replied to. Edges are unsigned; signs come later from the LLM pipelines.

**Where:** `data/interim/edges/edges_YYYY-MM.parquet`, one file per month of the reply's timestamp. There are 12 files, 28,099,021 rows and 6.4 GB in total. Read only the columns you need:

```python
import pandas as pd
edges = pd.read_parquet("data/interim/edges/edges_2016-06.parquet",
                        columns=["source", "target", "created_utc", "week_start", "subreddit"])
```

| Column | Type | Meaning |
|---|---|---|
| `comment_id` | string | Reddit ID of the reply, without the `t1_` prefix. Unique across all files. |
| `parent_comment_id` | string | ID of the comment being replied to. It can be in an earlier month's file. |
| `link_id` | string | The thread (post) both comments belong to, with its `t3_` prefix. |
| `source` | string | Pseudonymised author of the reply. This is where the edge starts. |
| `target` | string | Pseudonymised author of the parent comment. This is where the edge ends. Never equal to `source`. |
| `created_utc` | timestamp (UTC) | When the reply was posted. |
| `subreddit` | string | Subreddit of the reply (the parent is in the same thread). |
| `body` | string | Reply text: raw Reddit markdown with HTML entities still escaped (`&gt;`, `&amp;`). |
| `parent_body` | string | Parent comment text, in the same format. 1,456 are empty strings; none are null. |
| `reply_removed` | bool | `body` is exactly `[removed]` or `[deleted]`. |
| `parent_removed` | bool | `parent_body` is exactly `[removed]` or `[deleted]`. |
| `body_hash` | uint64 | 64-bit hash of `body`, for grouping identical texts cheaply. |
| `week_start` | timestamp (UTC) | Start of the reply's weekly window: Monday 00:00 UTC. |
| `month_start` | timestamp (UTC) | Start of the reply's monthly window: the 1st at 00:00 UTC. |
| `author_week_replies` | int | Replies `source` sent in that week, across all subreddits. |
| `body_repeat_count` | int | Edges in 2016 whose `body` is identical to this one. 1 means the text is unique. |

## What is already excluded

Self-replies, and replies whose parent is missing, are excluded. A parent is missing if it was written by a deleted account, a bot or AutoModerator, was not in English, or was posted before 2016. Upstream, the comments themselves are limited to the 522 US political subreddits and to English or undetermined language.

## What is not applied yet

- **No activity threshold.** Every user is present. The threshold is still to be chosen.
- **No flag filters.** `reply_removed`, `parent_removed`, `author_week_replies` and `body_repeat_count` are there for filtering, but nothing has been filtered on them.
- **No aggregation.** A user who replies to the same person several times has several rows. To get weighted graphs, group by `source, target` within a window, or use `src.dataset.windowing.window_graph`, which returns a `networkx.DiGraph` with `weight` = reply count.

The first week (2015-12-28, containing only Jan 1–3) and the last week (2016-12-26) are partial. Per-window statistics are in `results/graph_stats/`.

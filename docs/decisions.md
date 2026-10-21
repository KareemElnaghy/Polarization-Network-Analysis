# Ground-truth mapping rule

Placeholder: define how subreddit party tags map to the ground-truth proxy.

# Null-model definition

Placeholder: define the degree-preserving sign and edge randomization procedure.

# Scope cuts

Decided 2026-10-21.

- **Time range:** all of 2016 (`comments_2016-01` … `comments_2016-12`). The midway scope was June–December 2016; processing proved fast enough (one month parses in about 1.5 minutes), so it was expanded on 2026-10-21.
- **Subreddits:** all US political subreddits in the Politosphere record. Operationally: every subreddit in `subreddits_metadata.json` except those whose `region` is a non-US region (`ca`, `eu`, `me`, `uk`, `world`). Subreddits with an empty `region` are kept, since only 18 of 605 carry an explicit `us` tag.
- **Windows:** monthly windows are the primary analysis; weekly windows are a sensitivity check.
- **Comment filters (applied while streaming in `src/dataset/parse_comments.py`):**
  - `created_utc` within the scope's date range;
  - subreddit in the US set above;
  - author not `[deleted]`, and not flagged `bot` or `automoderator` in `users_metadata.json` (author names are pseudonymised, so bots cannot be matched by name);
  - `language` is `en` or `un` (the dataset's own language tag). `un` = undetermined, about 14% of comments, mostly short ones, and is kept; only comments tagged as another language are dropped;
  - edges are comment-to-comment replies only (`parent_id` starts with `t1_`). Top-level comments are kept in the interim Parquet, marked `is_reply = False`, because they are the parents of first-level replies; replies to posts (`t3_`) never become edges.

# Reply graph construction

Decided 2026-10-21. Code: `src/dataset/build_edges.py`, `windowing.py`, `graph_stats.py`.

- **Edge:** one reply to a comment, directed from the replier to the parent comment's author. Parents are looked up across all parsed months, so a reply can point to an older comment.
- **Dropped edges (counted by reason in `data/interim/edges/stats/edges_summary.json`):** self-replies, and replies whose parent is not among the kept comments (parent author `[deleted]`, a bot, non-English, or posted before 2016).
- **Flagged, not dropped (columns on each edge):** `reply_removed` / `parent_removed` (text is `[removed]`/`[deleted]`), `author_week_replies` (replies the source sent that week), `body_repeat_count` (edges with identical reply text). Cutoffs are still to be chosen.
- **Windows:** every edge carries both `week_start` (Monday 00:00 UTC) and `month_start`. The first (2015-12-28) and last (2016-12-26) weeks are partial and marked `complete = False` in the stats.
- **Window graph:** `networkx.DiGraph`, edge weight = number of replies. The minimum-activity threshold (replies sent plus received in the window, single pass) is not chosen yet; the per-threshold sweep is in `results/graph_stats/*_threshold_sweep.csv`.

# Open questions

Placeholder: record unresolved methodological and implementation questions.

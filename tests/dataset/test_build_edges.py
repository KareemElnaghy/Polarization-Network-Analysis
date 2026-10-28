from datetime import datetime, timezone

import pyarrow as pa

from src.dataset.build_edges import add_repeat_flags, build_reply_edges, count_repeats
from src.dataset.windowing import assign_windows

TS = pa.timestamp("s", tz="UTC")


def _comments(rows):
    cols = ["id", "parent_id", "link_id", "author", "subreddit", "created_utc", "body", "is_reply"]
    table = pa.table({c: [r[i] for r in rows] for i, c in enumerate(cols)})
    return table.set_column(5, "created_utc", table["created_utc"].cast(TS))


def _ts(day):
    return int(datetime(2016, 6, day, tzinfo=timezone.utc).timestamp())


def test_build_reply_edges_links_replier_to_parent_author_and_counts_losses():
    comments = _comments([
        ("top", "t3_post", "t3_post", "alice", "politics", _ts(1), "root", False),
        ("r1", "t1_top", "t3_post", "bob", "politics", _ts(2), "agree", True),
        ("r2", "t1_r1", "t3_post", "alice", "politics", _ts(3), "[removed]", True),
        ("self", "t1_r2", "t3_post", "alice", "politics", _ts(4), "me again", True),
        ("orphan_bot", "t1_botc", "t3_post", "bob", "politics", _ts(5), "hi bot", True),
        ("orphan_old", "t1_old", "t3_post", "bob", "politics", _ts(6), "hi old", True),
        ("to_post", "t3_post", "t3_post", "carol", "politics", _ts(7), "top level", False),
    ])
    parents = comments.select(["id", "author", "body"])
    dropped = pa.table({"id": ["botc"], "reason": ["author_bot"]})

    edges, losses = build_reply_edges(comments, parents, dropped)

    rows = edges.to_pylist()
    assert [(r["comment_id"], r["source"], r["target"]) for r in rows] == [("r1", "bob", "alice"),
                                                                           ("r2", "alice", "bob")]
    assert rows[0]["parent_body"] == "root" and rows[1]["body"] == "[removed]"
    assert [r["reply_removed"] for r in rows] == [False, True]
    assert [r["parent_removed"] for r in rows] == [False, False]
    assert dict(losses) == {"replies": 5, "edges": 2, "self_reply": 1,
                            "parent_missing:author_bot": 1, "parent_missing:parent_not_in_data": 1}


def test_repeat_flags_count_author_weeks_and_identical_texts(tmp_path):
    comments = _comments([
        ("p", "t3_x", "t3_x", "alice", "politics", _ts(6), "root", False),   # Monday 2016-06-06
        ("a", "t1_p", "t3_x", "bob", "politics", _ts(6), "same", True),
        ("b", "t1_p", "t3_x", "bob", "politics", _ts(12), "same", True),     # Sunday, same week
        ("c", "t1_p", "t3_x", "bob", "politics", _ts(13), "other", True),    # next Monday
    ])
    edges, _ = build_reply_edges(comments, comments.select(["id", "author", "body"]))
    edges = assign_windows(edges)
    path = tmp_path / "edges.parquet"
    pa.parquet.write_table(edges, path)

    flagged = add_repeat_flags(edges, *count_repeats([path])).to_pylist()

    assert [r["comment_id"] for r in flagged] == ["a", "b", "c"]
    assert [r["author_week_replies"] for r in flagged] == [2, 2, 1]
    assert [r["body_repeat_count"] for r in flagged] == [2, 2, 1]

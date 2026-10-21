import bz2
import json
from datetime import datetime, timezone

import pyarrow.parquet as pq

from src.dataset.parse_comments import (
    load_excluded_authors,
    load_us_subreddits,
    month_bounds,
    parse_raw_comments,
)

JUNE = int(datetime(2016, 6, 15, tzinfo=timezone.utc).timestamp())


def _comment(cid, **overrides):
    c = {"id": cid, "parent_id": "t1_parent", "link_id": "t3_post", "author": "alice",
         "subreddit": "politics", "created_utc": JUNE, "body": "text", "score": 1,
         "controversiality": 0, "language": "en"}
    c.update(overrides)
    return c


def _write_jsonl(path, rows, compress=False):
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    path.write_bytes(bz2.compress(data) if compress else data)


def test_parse_raw_comments_applies_scope_filters(tmp_path):
    _write_jsonl(tmp_path / "subs.json", [
        {"subreddit": "politics", "region": ""},
        {"subreddit": "The_Donald_CA", "region": "us"},
        {"subreddit": "ukpolitics", "region": "uk"},
    ])
    _write_jsonl(tmp_path / "users.json", [
        {"author": "alice", "bot": 0, "automoderator": 0},
        {"author": "botty", "bot": 1, "automoderator": 0},
        {"author": "mod", "bot": 0, "automoderator": 1},
    ])
    _write_jsonl(tmp_path / "comments.bz2", [
        _comment("keep_reply"),
        _comment("keep_top", parent_id="t3_post"),
        _comment("early", created_utc=JUNE - 30 * 86400),
        _comment("uk", subreddit="ukpolitics"),
        _comment("bot", author="botty"),
        _comment("automod", author="mod"),
        _comment("deleted", author="[deleted]"),
        _comment("undetermined", language="un"),
        _comment("dutch", language="nl"),
    ], compress=True)

    subs = load_us_subreddits(tmp_path / "subs.json")
    assert subs == {"politics", "The_Donald_CA"}
    excluded = load_excluded_authors(tmp_path / "users.json")
    assert excluded == {"botty", "mod", "[deleted]"}

    start, end = month_bounds("2016-06")
    out = tmp_path / "out.parquet"
    stats = parse_raw_comments(tmp_path / "comments.bz2", out, start=start, end=end,
                               subreddits=subs, excluded_authors=excluded)

    table = pq.read_table(out).to_pydict()
    assert table["id"] == ["keep_reply", "keep_top", "undetermined"]
    assert table["is_reply"] == [True, False, True]
    assert (stats.lines, stats.kept, stats.kept_replies) == (9, 3, 2)
    assert stats.dropped == {"date": 1, "subreddit": 1, "author": 3, "language": 1}


def test_month_bounds_wraps_december():
    start, end = month_bounds("2016-12")
    assert (start.month, end.year, end.month) == (12, 2017, 1)

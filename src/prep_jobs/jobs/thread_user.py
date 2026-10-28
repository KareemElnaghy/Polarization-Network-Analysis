"""Job B, thread_user: per (week, subreddit, thread, author) comment counts for ground truth #1.

Columns: n_comments, n_replies (reply to a comment), n_accepted (score >= 1), score_sum. Removed/deleted,
privacy-script and distinguished comments are excluded. Monthly units write per-week staging files; a
finalise step sums the weeks that span two months and writes ``week=YYYY-MM-DD/part.parquet``.
"""

from __future__ import annotations

import shutil
from collections import Counter

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.prep_jobs.config import data_path, runs_dir
from src.prep_jobs.data import comment_file, iter_comments, month_rows
from src.prep_jobs.util import dir_size, sum_by_keys, week_label, with_week_start, write_parquet

NAME = "thread_user"
DEADLINE_CUTS: list[str] = []
KEYS = ["week_start", "subreddit", "link_id", "author"]
SUMS = ["n_comments", "n_replies", "n_accepted", "score_sum"]
COLUMNS = ["created_utc", "subreddit", "link_id", "author", "score", "is_reply"]


def out_dir(cfg):
    return data_path(cfg, "thread_user")


def phases(cfg):
    mod = "src.prep_jobs.jobs.thread_user"
    months = [{"id": m, "weight": month_rows(cfg, m)} for m in cfg["months"]]
    total = sum(u["weight"] for u in months)
    fin = [{"id": "finalise", "weight": total * 0.05}]
    return [
        {"name": "months", "fn": f"{mod}:month_unit", "units": months},
        {"name": "finalise", "fn": f"{mod}:finalise", "units": fin, "workers": 1, "needs_all_prior": True},
        {"name": "validate", "fn": f"{mod}:validate", "units": [{"id": "validate", "weight": total * 0.02}],
         "workers": 1, "needs_all_prior": True},
    ]


def month_unit(ctx, unit):
    cfg, month = ctx.cfg, unit["id"]
    stats: Counter = Counter()
    parts: list[pa.Table] = []

    def fold(parts):
        t = pa.concat_tables(parts)
        return [sum_by_keys(t, KEYS, SUMS)]

    for t in iter_comments(cfg, month, COLUMNS, ctx.batch_rows, stats):
        t = with_week_start(t)
        t = t.set_column(t.schema.get_field_index("subreddit"), "subreddit", pc.cast(t["subreddit"], pa.string()))
        score = pc.cast(t["score"], pa.int64())
        t = pa.table({
            "week_start": t["week_start"], "subreddit": t["subreddit"], "link_id": t["link_id"], "author": t["author"],
            "n_comments": pa.array([1] * t.num_rows, pa.int64()),
            "n_replies": pc.cast(t["is_reply"], pa.int64()),
            "n_accepted": pc.cast(pc.greater_equal(score, 1), pa.int64()),
            "score_sum": score})
        parts.append(sum_by_keys(t, KEYS, SUMS))
        if len(parts) >= 8 or ctx.over_limit():
            parts = fold(parts)
            if ctx.over_limit():
                ctx.shrink()
        ctx.report(rows=stats["rows"], fraction=min(0.99, stats["rows"] / unit["weight"]))
    table = fold(parts)[0] if parts else None
    nbytes = 0
    stage = out_dir(cfg) / "_staging" / f"month={month}"
    shutil.rmtree(stage, ignore_errors=True)
    if table is not None:
        weeks = pc.unique(table["week_start"])
        for w in weeks:
            sub = table.filter(pc.equal(table["week_start"], w))
            nbytes += write_parquet(sub, stage / f"week={week_label(w.as_py())}.parquet")
    (stage / "stats.json").write_text(__import__("json").dumps({"rows": stats["rows"], "kept": stats["kept"],
                                                                 **{k: v for k, v in stats.items()}}))
    ctx.log(f"{month}: {dict(stats)}")
    return {"rows": stats["rows"], "bytes": nbytes, "kept": stats["kept"]}


def finalise(ctx, unit):
    cfg = ctx.cfg
    root = out_dir(cfg)
    stage = root / "_staging"
    by_week: dict[str, list] = {}
    for f in sorted(stage.glob("month=*/week=*.parquet")):
        by_week.setdefault(f.stem.removeprefix("week="), []).append(f)
    rows = nbytes = 0
    for week, files in sorted(by_week.items()):
        t = pa.concat_tables(pq.read_table(f) for f in files)
        if len(files) > 1:
            t = sum_by_keys(t, KEYS, SUMS)
        nbytes += write_parquet(t, root / f"week={week}" / "part.parquet")
        rows += t.num_rows
    ctx.log(f"{len(by_week)} weeks, {rows:,} rows, {nbytes / 1e6:.0f} MB")
    return {"rows": rows, "bytes": nbytes, "weeks": len(by_week)}


def validate(ctx, unit):
    """Sum of n_comments must equal the in-scope comments after exclusions, recounted independently."""
    cfg = ctx.cfg
    root = out_dir(cfg)
    total = 0
    for f in sorted(root.glob("week=*/part.parquet")):
        total += pc.sum(pq.read_table(f, columns=["n_comments"])["n_comments"]).as_py()
    stats_files = sorted((root / "_staging").glob("month=*/stats.json"))
    import json
    from_units = sum(json.loads(f.read_text())["kept"] for f in stats_files)
    # independent recount: a fresh pass with a different code path (dataset filter expression)
    import pyarrow.dataset as ds
    import pyarrow.compute as pc2
    from src.common.text_filters import PRIVACY_OVERWRITE_REGEX, REMOVED_TEXTS
    from src.prep_jobs.data import distinguished_ids
    recount = 0
    for m in cfg["months"]:
        dist = distinguished_ids(cfg, m)
        expr = (~ds.field("body").isin(REMOVED_TEXTS)) & (~ds.field("id").isin(dist))
        for batch in ds.dataset(comment_file(cfg, m)).to_batches(columns=["body"], filter=expr):
            recount += batch.num_rows - (pc2.sum(pc2.match_substring_regex(batch["body"], PRIVACY_OVERWRITE_REGEX)).as_py() or 0)
    text = (f"# thread_user validation\n\nsum of n_comments in the output : {total:,}\n"
            f"in-scope comments after exclusions (units' count): {from_units:,}\n"
            f"in-scope comments after exclusions (independent recount): {recount:,}\n\n"
            f"**{'PASS' if total == from_units == recount else 'FAIL'}**\n")
    (runs_dir(cfg) / NAME / "VALIDATION.md").write_text(text, encoding="utf-8")
    print(text)
    if not total == from_units == recount:
        raise RuntimeError("thread_user validation failed")
    shutil.rmtree(root / "_staging", ignore_errors=True)
    return {"rows": total, "bytes": 0}

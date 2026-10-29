"""Job C, clean_edges_and_sampling: the cleaned weekly edge set and the thread samples for the LLM pipelines.

Step 1 (per month): drop edges whose reply or parent text is removed/deleted or a privacy-script
overwrite; add ``body_clean`` / ``parent_body_clean`` (html.unescape, originals kept) and a
``distinguished`` flag for the reply comment (the edge is kept).
Step 2 (per week, 53 units), on the whole week:
  heavy posters: an author who sent more than ``heavy_poster_threshold`` replies that week (counted on the
      raw edges before step 1, across all subreddits) is excluded with the edges sent to them;
  k = ``min_activity``, once: a user is kept if sent + received >= k on the cleaned week; an edge is kept if
      both ends are kept;
  write the full cleaned week, then whole-thread samples at each rate (strata: top subreddits x thread size),
      without re-applying k.
"""

from __future__ import annotations

import hashlib
import html
import json
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.common.text_filters import privacy_overwritten_mask, removed_mask
from src.prep_jobs.config import data_path, job_cfg, runs_dir
from src.prep_jobs.data import distinguished_ids
from src.prep_jobs.util import atomic_path, read_json, week_label, write_json, write_parquet

NAME = "clean_edges_and_sampling"
DEADLINE_CUTS = ["sampling.rates: drop 0.20 first"]


def edge_file(cfg, month):
    return data_path(cfg, "edges") / f"edges_{month}.parquet"


def clean_root(cfg):
    return data_path(cfg, "edges_clean")


def weeks_of(cfg) -> list[str]:
    start = datetime.fromisoformat(cfg["first_week"])
    out = []
    while start.year <= cfg["year"] and (start.year == cfg["year"] or start.year == cfg["year"] - 1):
        out.append(start.date().isoformat())
        start += timedelta(days=7)
    return out


def phases(cfg):
    mod = "src.prep_jobs.jobs.clean_edges_and_sampling"
    rows = {m: pq.ParquetFile(edge_file(cfg, m)).metadata.num_rows for m in cfg["months"]}
    total = sum(rows.values())
    sweep = pd.read_csv(data_path(cfg, "graph_stats") / "weekly_stats.csv")
    wk = dict(zip(sweep["window_start"].astype(str), sweep["replies"]))
    weeks = [{"id": w, "weight": float(wk.get(w, total / 53))} for w in weeks_of(cfg)]
    return [
        {"name": "setup", "fn": f"{mod}:setup", "units": [{"id": "top_subreddits", "weight": total * 0.02}], "workers": 1},
        {"name": "clean_months", "fn": f"{mod}:clean_month", "units": [{"id": m, "weight": rows[m]} for m in cfg["months"]],
         "needs_all_prior": True},
        {"name": "weeks", "fn": f"{mod}:week_unit", "units": weeks, "needs_all_prior": True},
        {"name": "finalise", "fn": f"{mod}:finalise", "units": [{"id": "finalise", "weight": total * 0.02}],
         "workers": 1, "needs_all_prior": True},
    ]


# ------------------------------------------------------------------ setup
def setup(ctx, unit):
    cfg, jc = ctx.cfg, ctx.jcfg
    counts: Counter = Counter()
    for m in cfg["months"]:
        t = pq.read_table(edge_file(cfg, m), columns=["subreddit"])
        for v in pc.value_counts(pc.cast(t["subreddit"], pa.string())).to_pylist():
            counts[v["values"]] += v["counts"]
        ctx.report(fraction=0.5)
    top = [s for s, _ in counts.most_common(jc["top_subreddits"])]
    write_json({"top_subreddits": top, "edge_counts": dict(counts.most_common(60))}, runs_dir(cfg) / NAME / "top_subreddits.json")
    return {"rows": sum(counts.values()), "bytes": 0}


# ------------------------------------------------------------------ step 1
def unescape(arr: pa.ChunkedArray | pa.Array) -> pa.Array:
    vals = arr.to_pylist()
    return pa.array([html.unescape(s) if s and "&" in s else s for s in vals], pa.string())


def clean_month(ctx, unit):
    cfg, month = ctx.cfg, unit["id"]
    flt = cfg["text_filters"]
    dist = distinguished_ids(cfg, month)
    pf = pq.ParquetFile(edge_file(cfg, month))
    stage = clean_root(cfg) / "_staging"
    tmp = stage / f"month={month}.tmp"
    final = stage / f"month={month}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    writers: dict[str, pq.ParquetWriter] = {}
    stats: dict[str, Counter] = {}
    rows = kept_total = nbytes = 0
    out_schema = None
    for batch in pf.iter_batches(batch_size=ctx.batch_rows):
        t = pa.Table.from_batches([batch])
        rows += t.num_rows
        rr = pc.or_(t["reply_removed"], removed_mask(t["body"]))
        pr = pc.or_(t["parent_removed"], removed_mask(t["parent_body"]))
        rp = privacy_overwritten_mask(t["body"], flt["privacy_overwrite_regex"])
        pp = privacy_overwritten_mask(t["parent_body"], flt["privacy_overwrite_regex"])
        drop = pc.or_(pc.or_(rr, pr), pc.or_(rp, pp))
        wk = t["week_start"]
        for w in pc.unique(wk):
            m = pc.equal(wk, w)
            c = stats.setdefault(week_label(w.as_py()), Counter())
            c["raw"] += pc.sum(pc.cast(m, pa.int64())).as_py()
            for name, mask in (("removed_reply", rr), ("removed_parent", pr), ("privacy_reply", rp),
                               ("privacy_parent", pp), ("text_rules_union", drop)):
                c[name] += pc.sum(pc.cast(pc.and_(m, mask), pa.int64())).as_py() or 0
        t = t.filter(pc.invert(drop))
        if t.num_rows == 0:
            continue
        t = t.append_column("body_clean", unescape(t["body"])).append_column("parent_body_clean", unescape(t["parent_body"]))
        t = t.append_column("distinguished", pc.is_in(t["comment_id"], dist))
        kept_total += t.num_rows
        wk = t["week_start"]
        for w in pc.unique(wk):
            label = week_label(w.as_py())
            if label not in writers:
                writers[label] = pq.ParquetWriter(tmp / f"week={label}.parquet", t.schema, compression="zstd")
            writers[label].write_table(t.filter(pc.equal(wk, w)))
        ctx.report(rows=rows, fraction=min(0.99, rows / unit["weight"]))
    for w in writers.values():
        w.close()
    (tmp / "stats.json").write_text(json.dumps({k: dict(v) for k, v in stats.items()}))
    nbytes = sum(f.stat().st_size for f in tmp.glob("*.parquet"))
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)
    ctx.log(f"{month}: {rows:,} edges -> {kept_total:,} after text rules")
    return {"rows": rows, "bytes": nbytes, "kept": kept_total}


# ------------------------------------------------------------------ step 2
def keep_k(src: np.ndarray, dst: np.ndarray, k: int) -> np.ndarray:
    """One pass: keep users with sent + received >= k; keep an edge if both ends are kept."""
    codes, uniques = pd.factorize(np.concatenate([src, dst]))
    n = len(src)
    activity = np.bincount(codes, minlength=len(uniques))
    ok = activity >= k
    return ok[codes[:n]] & ok[codes[n:]]


def graph_metrics(src, dst) -> dict:
    if len(src) == 0:
        return {"nodes": 0, "pairs": 0, "mean_degree": 0.0, "lcc_share": 0.0}
    pairs = pd.DataFrame({"s": src, "t": dst}).drop_duplicates()
    codes, uniques = pd.factorize(np.concatenate([pairs["s"].to_numpy(), pairs["t"].to_numpy()]))
    m, n = len(pairs), len(uniques)
    mat = csr_matrix((np.ones(m, dtype=np.int8), (codes[:m], codes[m:])), shape=(n, n))
    _, labels = connected_components(mat, directed=True, connection="weak")
    return {"nodes": n, "pairs": m, "mean_degree": 2 * m / n, "lcc_share": float(np.bincount(labels).max() / n)}


def thread_seed(seed: int, rate: float, week: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{rate}|{week}".encode()).digest()[:8], "little")


def sample_threads(threads: pd.DataFrame, top: set[str], bucket_edges: list[int], rate: float, seed: int) -> np.ndarray:
    """Boolean mask over ``threads`` (columns subreddit, link_id, n): stratified sample with random rounding."""
    group = np.where(threads["subreddit"].isin(top), threads["subreddit"], "other")
    bucket = np.searchsorted(np.asarray(bucket_edges), threads["n"].to_numpy(), side="right") - 1
    rng = np.random.default_rng(seed)
    chosen = np.zeros(len(threads), dtype=bool)
    strata = pd.DataFrame({"g": group, "b": bucket}).groupby(["g", "b"]).indices
    for _, idx in sorted(strata.items()):
        x = rate * len(idx)
        k = int(np.floor(x)) + int(rng.random() < x - np.floor(x))
        if k:
            chosen[rng.choice(idx, size=min(k, len(idx)), replace=False)] = True
    return chosen


def read_raw_week(cfg, week: str) -> pa.Table:
    d = datetime.fromisoformat(week)
    months = sorted({d.strftime("%Y-%m"), (d + timedelta(days=6)).strftime("%Y-%m")} & set(cfg["months"]))
    parts = []
    for m in months:
        f = edge_file(cfg, m)
        typ = pq.read_schema(f).field("week_start").type
        flt = pc.field("week_start") == pa.scalar(d.replace(tzinfo=timezone.utc), typ)
        parts.append(pq.read_table(f, columns=["source", "target", "author_week_replies"], filters=flt))
    return pa.concat_tables(parts)


def week_unit(ctx, unit):
    cfg, jc, week = ctx.cfg, ctx.jcfg, unit["id"]
    top = set(read_json(runs_dir(cfg) / NAME / "top_subreddits.json")["top_subreddits"])
    # -- raw week: heavy posters, and the k-only numbers that must match the threshold sweep
    raw = read_raw_week(cfg, week)
    rs, rt = raw["source"].to_numpy(zero_copy_only=False), raw["target"].to_numpy(zero_copy_only=False)
    sent = pd.Series(rs).value_counts()
    heavy = set(sent.index[sent > jc["heavy_poster_threshold"]])
    awr = raw["author_week_replies"].to_numpy()
    recomputed = sent.reindex(rs).to_numpy()
    mismatch = int((awr != recomputed).sum())
    heavy_mask_raw = np.isin(rs, list(heavy)) | np.isin(rt, list(heavy))
    k_raw = keep_k(rs, rt, jc["min_activity"])
    kraw_pairs = pd.DataFrame({"s": rs[k_raw], "t": rt[k_raw]}).drop_duplicates()
    kraw_nodes = len(set(rs[k_raw]) | set(rt[k_raw]))
    row = {"week_start": week, "raw_edges": len(rs), "heavy_authors": len(heavy), "drop_heavy_alone": int(heavy_mask_raw.sum()),
           "drop_k_alone": int((~k_raw).sum()), "author_week_replies_mismatch_rows": mismatch,
           "k_only_nodes": kraw_nodes, "k_only_pairs": len(kraw_pairs), "k_only_replies": int(k_raw.sum())}
    del raw, rs, rt, awr, recomputed
    ctx.report(fraction=0.2)

    # -- step-1 output for the week (one or two monthly staging files)
    files = sorted((clean_root(cfg) / "_staging").glob(f"month=*/week={week}.parquet"))
    stats = Counter()
    for sf in (clean_root(cfg) / "_staging").glob("month=*/stats.json"):
        stats.update(json.loads(sf.read_text()).get(week, {}))
    row.update({f"drop_{k}_alone": stats.get(k, 0) for k in ("removed_reply", "removed_parent", "privacy_reply",
                                                              "privacy_parent", "text_rules_union")})
    light = pa.concat_tables(pq.read_table(f, columns=["source", "target", "link_id", "subreddit"]) for f in files)
    s, t = light["source"].to_numpy(zero_copy_only=False), light["target"].to_numpy(zero_copy_only=False)
    n_text = len(s)
    not_heavy = ~(np.isin(s, list(heavy)) | np.isin(t, list(heavy)))
    k_ok = keep_k(s[not_heavy], t[not_heavy], jc["min_activity"])
    keep = np.zeros(n_text, dtype=bool)
    keep[np.flatnonzero(not_heavy)[k_ok]] = True
    row.update({"after_text_rules": n_text, "after_heavy": int(not_heavy.sum()), "after_k": int(keep.sum()),
                "seq_drop_text_rules": row["raw_edges"] - n_text, "seq_drop_heavy": n_text - int(not_heavy.sum()),
                "seq_drop_k": int(not_heavy.sum()) - int(keep.sum())})
    light = light.filter(pa.array(keep))
    s, t = s[keep], t[keep]
    full = graph_metrics(s, t)
    row.update({"final_edges": len(s), "final_nodes": full["nodes"], "final_pairs": full["pairs"],
                "final_mean_degree": full["mean_degree"], "final_lcc_share": full["lcc_share"]})
    ctx.report(fraction=0.4)

    # -- write the cleaned week, streaming
    out = clean_root(cfg) / f"week={week}" / "part.parquet"
    offset, nbytes = 0, 0
    with atomic_path(out) as tmp:
        writer = None
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=ctx.batch_rows):
                seg = keep[offset: offset + batch.num_rows]
                offset += batch.num_rows
                tbl = pa.Table.from_batches([batch]).filter(pa.array(seg))
                if tbl.num_rows:
                    writer = writer or pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
                    writer.write_table(tbl)
        if writer:
            writer.close()
    nbytes += out.stat().st_size if out.exists() else 0
    row["distinguished_edges"] = int(pc.sum(pc.cast(pq.read_table(out, columns=["distinguished"])["distinguished"], pa.int64())).as_py() or 0)
    ctx.report(fraction=0.6)

    # -- thread samples
    threads = pd.DataFrame({"subreddit": light["subreddit"].to_numpy(zero_copy_only=False).astype(str),
                            "link_id": light["link_id"].to_numpy(zero_copy_only=False)})
    sizes = threads.groupby(["subreddit", "link_id"]).size().rename("n").reset_index()
    link = light["link_id"].to_numpy(zero_copy_only=False)
    sample_rows, thread_parts = [], []
    cleaned = pq.ParquetFile(out)
    for rate in jc["sampling"]["rates"]:
        chosen = sample_threads(sizes, top, jc["size_buckets"], rate, thread_seed(cfg["seed"], rate, week))
        picked = sizes[chosen]
        pick_ids = set(picked["link_id"])
        m = np.fromiter((x in pick_ids for x in link), dtype=bool, count=len(link))
        sm = graph_metrics(s[m], t[m])
        sample_rows.append({"week_start": week, "rate": rate, "threads_sampled": int(chosen.sum()), "threads_total": len(sizes),
                            "edges_sampled": int(m.sum()), "edges_total": len(link),
                            "realised_edge_share": float(m.sum() / max(1, len(link))), "nodes": sm["nodes"], "pairs": sm["pairs"],
                            "mean_degree": sm["mean_degree"], "lcc_share": sm["lcc_share"],
                            "full_nodes": full["nodes"], "full_pairs": full["pairs"], "full_mean_degree": full["mean_degree"],
                            "full_lcc_share": full["lcc_share"]})
        sout = data_path(cfg, "edges_sampled") / f"rate={rate}" / f"week={week}" / "part.parquet"
        off, writer = 0, None
        with atomic_path(sout) as tmp:
            for batch in cleaned.iter_batches(batch_size=ctx.batch_rows):
                seg = m[off: off + batch.num_rows]
                off += batch.num_rows
                tbl = pa.Table.from_batches([batch]).filter(pa.array(seg))
                if tbl.num_rows:
                    writer = writer or pq.ParquetWriter(tmp, tbl.schema, compression="zstd")
                    writer.write_table(tbl)
            if writer:
                writer.close()
            else:  # empty sample: still write a valid file
                pq.write_table(cleaned.schema_arrow.empty_table(), tmp)
        nbytes += sout.stat().st_size
        thread_parts.append(pa.table({"week_start": [week] * len(picked), "subreddit": picked["subreddit"].tolist(),
                                      "link_id": picked["link_id"].tolist(), "rate": [rate] * len(picked)}))
        ctx.report(fraction=0.6 + 0.1 * len(thread_parts))
    write_parquet(pa.concat_tables(thread_parts), data_path(cfg, "edges_sampled") / "_threads" / f"{week}.parquet")
    write_json({"filter": row, "sampling": sample_rows}, runs_dir(cfg) / NAME / "week_reports" / f"{week}.json")
    return {"rows": row["raw_edges"], "bytes": nbytes, "final_edges": row["final_edges"]}


# ------------------------------------------------------------------ finalise and validation
def finalise(ctx, unit):
    cfg, jc = ctx.cfg, ctx.jcfg
    reports = [read_json(f) for f in sorted((runs_dir(cfg) / NAME / "week_reports").glob("*.json"))]
    out = data_path(cfg, "results") / "edge_filters"
    out.mkdir(parents=True, exist_ok=True)
    filt = pd.DataFrame([r["filter"] for r in reports])
    samp = pd.DataFrame([x for r in reports for x in r["sampling"]])
    filt.to_csv(out / "filter_report.csv", index=False)
    samp.to_csv(out / "sampling_report.csv", index=False)
    parts = sorted((data_path(cfg, "edges_sampled") / "_threads").glob("*.parquet"))
    write_parquet(pa.concat_tables(pq.read_table(f) for f in parts), data_path(cfg, "edges_sampled") / "sampled_threads.parquet")

    sweep = pd.read_csv(data_path(cfg, "graph_stats") / "weekly_threshold_sweep.csv")
    sweep = sweep[sweep["threshold"] == jc["min_activity"]].assign(week_start=lambda d: d["window_start"].astype(str))
    v = filt.merge(sweep, on="week_start")
    tol = jc["validation_tolerance"]
    for mine, theirs in (("k_only_nodes", "nodes"), ("k_only_pairs", "edges"), ("k_only_replies", "replies")):
        v[f"{mine}_rel_diff"] = (v[mine] - v[theirs]) / v[theirs]
    bad = v[(v[[c for c in v.columns if c.endswith("_rel_diff")]].abs() > tol).any(axis=1)]
    complete = samp[~samp["week_start"].isin([weeks_of(cfg)[0], weeks_of(cfg)[-1]])]
    off = complete[(complete["realised_edge_share"] - complete["rate"]).abs() > jc["realised_share_tolerance"]]
    lines = ["# clean_edges_and_sampling validation", "",
             f"## k = {jc['min_activity']} only (no new filters) vs weekly_threshold_sweep.csv, tolerance {tol:.1%}",
             f"weeks compared: {len(v)}; weeks differing: {len(bad)}"]
    for r in bad.itertuples():
        lines.append(f"- {r.week_start}: nodes {r.k_only_nodes} vs {r.nodes}, edges {r.k_only_pairs} vs {r.edges}, "
                     f"replies {r.k_only_replies} vs {r.replies}")
    lines += ["", f"## Realised edge share vs rate (complete weeks), tolerance {jc['realised_share_tolerance']:.0%} points",
              f"(week, rate) combinations: {len(complete)}; outside tolerance: {len(off)}"]
    for r in off.itertuples():
        lines.append(f"- {r.week_start} rate {r.rate}: realised {r.realised_edge_share:.3f}")
    lines += ["", "## author_week_replies vs recount from raw edges",
              f"rows differing across all weeks: {int(filt['author_week_replies_mismatch_rows'].sum()):,} "
              f"of {int(filt['raw_edges'].sum()):,}"]
    (runs_dir(cfg) / NAME / "VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    shutil.rmtree(clean_root(cfg) / "_staging", ignore_errors=True)
    shutil.rmtree(data_path(cfg, "edges_sampled") / "_threads", ignore_errors=True)
    ctx.log(f"finalised; {len(bad)} weeks differ from the sweep; {len(off)} sample shares outside tolerance")
    return {"rows": len(filt), "bytes": 0}

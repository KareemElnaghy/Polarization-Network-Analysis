"""Job A, phrase_counts: two-word-phrase counts per (week, subreddit, author) for ground truth #2.

Reads the raw dumps (``body_cleaned`` and ``distinguished`` are not in the Parquet) with the same
scope filters as ``parse_comments`` (US subreddit, language en/un, author kept), then excludes
removed/deleted, privacy-script and distinguished comments.

Pipeline (all labelling-free):
  pass1       per month: unique (phrase, author) pairs with counts, written in hash buckets
  vocab_merge per bucket: distinct authors and total count per phrase, keep >= min_authors_per_phrase
  vocab_select top ``vocab_size`` phrases by distinct authors -> phrase ids
  pass2       per month: counts per (week_start, subreddit, author, phrase_id) for eligible authors, plus totals
  finalise    sum boundary weeks that span two monthly files; write vocab.parquet with phrase text
  validate    independent recount of 1,000 comments, top phrases, totals == sum of counts

Phrases are identified by a 44-bit hash while counting (collisions between two rare phrases are
possible, about a few hundred among ~10^8 phrases, and are ignored); text is recovered in pass2.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import random
import re
import shutil
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    import orjson
    _loads = orjson.loads
except ImportError:  # pragma: no cover
    _loads = json.loads

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.dataset.parse_comments import KEPT_LANGUAGES, load_us_subreddits, month_bounds
from src.prep_jobs.config import data_path, job_cfg, runs_dir
from src.prep_jobs.data import party_map
from src.prep_jobs.util import atomic_path, fold_boundary_weeks, read_json, week_label, with_week_start, write_json, write_parquet

NAME = "phrase_counts"
DEADLINE_CUTS = ["vocab_size: 100000 -> 50000", "min_author_comments: 5 -> 10",
                 "subreddits: all -> fallback_subreddits"]

PAIR_MIX_1 = np.uint64(0x9E3779B97F4A7C15)
PAIR_MIX_2 = np.uint64(0xD6E8FEB86659FD93)
SHIFT = np.uint64(32)
WEEK_BITS, SUB_BITS, PHRASE_BITS = 6, 10, 17


def work_dir(cfg):
    return data_path(cfg, "phrase_counts") / "_work"


# ------------------------------------------------------------------ scope, authors, tokens
def scope_subreddits(cfg, jc) -> list[str]:
    us = load_us_subreddits(data_path(cfg, "raw") / "subreddits_metadata.json")
    want = jc["subreddits"]
    if want == "all":
        return sorted(us)
    if want == "fallback":
        want = jc["fallback_subreddits"]
    subs = set()
    for s in want:
        subs.update(party_map(cfg) if s == "@party_tagged" else [s])
    return sorted(subs & us)


def load_authors(cfg):
    t = pq.read_table(data_path(cfg, "user_profiles") / "user_summary.parquet", columns=["author", "n_comments"])
    t = t.sort_by("author")
    return t["author"].to_pylist(), t["n_comments"].to_numpy()


class Tokenizer:
    """Maps a token to an int64 code: 0 if unusable, else (60-bit hash << 1) | is_stopword."""

    def __init__(self, jc):
        self.regex = re.compile(jc["token_regex"])
        self.min_len = jc["min_token_length"]
        self.drop = frozenset(jc["url_fragment_tokens"])
        self.cache: dict[str, int] = {}

    def _compute(self, t: str) -> int:
        if len(t) < self.min_len or t in self.drop or not self.regex.match(t):
            return 0
        h = int.from_bytes(hashlib.blake2b(t.encode(), digest_size=8).digest(), "little") & ((1 << 60) - 1)
        return ((h | (1 << 60)) << 1) | (t in ENGLISH_STOP_WORDS)

    def code(self, t: str) -> int:
        c = self.cache.get(t)
        if c is None:
            c = self._compute(t)
            if len(self.cache) > 6_000_000:
                self.cache.clear()
            self.cache[t] = c
        return c


def mix_pair(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    x = h1 * PAIR_MIX_1 + h2
    x ^= x >> SHIFT
    x *= PAIR_MIX_2
    x ^= x >> SHIFT
    return x


def pair_key(c1: int, c2: int, key_bits: int) -> int:
    """Hash key of a phrase from the codes of its two tokens (scalar twin of ``bigram_keys``)."""
    x = mix_pair(np.array([c1 >> 1], dtype=np.uint64), np.array([c2 >> 1], dtype=np.uint64))
    return int(x[0] & np.uint64((1 << key_bits) - 1))


def bigram_keys(texts: list[str], tok: Tokenizer, key_bits: int, want_tokens: bool = False):
    """Phrase keys for a batch of comments.

    Tokens are kept if alphabetic (apostrophes allowed), long enough and not a URL fragment; pairs are
    adjacent kept tokens in one comment; a pair is dropped if both tokens are stopwords.
    Returns ``(keys, comment_index, flat_position, flat_tokens)`` for the valid pairs.
    """
    code, flat, flat_tok = tok.code, [], []
    lens = np.empty(len(texts), dtype=np.int64)
    for i, text in enumerate(texts):
        start = len(flat)
        for t in text.split():
            c = code(t)
            if c:
                flat.append(c)
                if want_tokens:
                    flat_tok.append(t)
        lens[i] = len(flat) - start
    if len(flat) < 2:
        e = np.empty(0, dtype=np.uint64)
        return e, np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), flat_tok
    arr = np.array(flat, dtype=np.int64)
    cid = np.repeat(np.arange(len(texts)), lens)
    stop = (arr & 1).astype(bool)
    ok = (cid[:-1] == cid[1:]) & ~(stop[:-1] & stop[1:])
    pos = np.flatnonzero(ok)
    x = mix_pair((arr[pos] >> 1).astype(np.uint64), (arr[pos + 1] >> 1).astype(np.uint64))
    return x & np.uint64((1 << key_bits) - 1), cid[pos], pos, flat_tok


def sum_duplicates(keys: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if keys.size == 0:
        return keys, counts
    order = np.argsort(keys, kind="stable")
    keys, counts = keys[order], counts[order]
    first = np.flatnonzero(np.concatenate(([True], keys[1:] != keys[:-1])))
    return keys[first], np.add.reduceat(counts, first)


# ------------------------------------------------------------------ raw stream
class RawComments:
    """Comments of one month that pass the scope filters and the shared exclusions."""

    def __init__(self, cfg, jc, month, subs: list[str], authors: list[str], window: tuple[int, int] | None = None,
                 with_link: bool = False):
        """``window`` = (start, end) epoch seconds to keep instead of the whole month; ``with_link`` also yields ``link_id``."""
        self.cfg, self.month = cfg, month
        self.window, self.with_link = window, with_link
        self.subs = {s: i for i, s in enumerate(subs)}
        self.author_index = {a: i for i, a in enumerate(authors)}
        self.privacy = re.compile(cfg["text_filters"]["privacy_overwrite_regex"])
        self.removed = frozenset(cfg["text_filters"]["removed_texts"])
        self.drop_distinguished = cfg["text_filters"]["exclude_distinguished"]
        self.stats: Counter = Counter()

    def __iter__(self):
        lo, hi = self.window or tuple(int(d.timestamp()) for d in month_bounds(self.month))
        st = self.stats
        with bz2.open(data_path(self.cfg, "raw") / f"comments_{self.month}.bz2", "rb") as fh:
            for line in fh:
                st["lines"] += 1
                try:
                    c = _loads(line)
                    ts = int(c["created_utc"])
                except (ValueError, KeyError, TypeError):
                    continue
                if not lo <= ts < hi:
                    continue
                sub = self.subs.get(c["subreddit"])
                if sub is None:
                    continue
                a = self.author_index.get(c["author"])
                if a is None or c.get("language") not in KEPT_LANGUAGES:
                    continue
                st["in_scope"] += 1
                body = c["body"]
                if body in self.removed or self.privacy.match(body):
                    st["removed_or_privacy"] += 1
                    continue
                if self.drop_distinguished and c.get("distinguished"):
                    st["distinguished"] += 1
                    continue
                st["counted"] += 1
                if self.with_link:
                    yield a, sub, ts, c["body_cleaned"], c["link_id"]
                else:
                    yield a, sub, ts, c["body_cleaned"]


def weight_total(cfg, jc) -> int:
    return sum((data_path(cfg, "raw") / f"comments_{m}.bz2").stat().st_size for m in cfg["months"])


def phases(cfg):
    jc = job_cfg(cfg, NAME)
    raw = data_path(cfg, "raw")
    size = {m: (raw / f"comments_{m}.bz2").stat().st_size for m in cfg["months"]}
    total = sum(size.values())
    months = [{"id": m, "weight": size[m]} for m in cfg["months"]]
    buckets = [{"id": f"b{b:02d}", "bucket": b, "weight": total * 0.02 / jc["buckets"]} for b in range(jc["buckets"])]
    one = lambda name: [{"id": name, "weight": total * 0.01}]  # noqa: E731
    mod = "src.prep_jobs.jobs.phrase_counts"
    return [
        {"name": "pass1", "fn": f"{mod}:pass1_month", "units": months},
        {"name": "vocab_merge", "fn": f"{mod}:merge_bucket", "units": buckets, "needs_all_prior": True},
        {"name": "vocab_select", "fn": f"{mod}:vocab_select", "units": one("select"), "workers": 1, "needs_all_prior": True},
        {"name": "pass2", "fn": f"{mod}:pass2_month", "units": [dict(u) for u in months], "needs_all_prior": True},
        {"name": "finalise", "fn": f"{mod}:finalise", "units": one("finalise"), "workers": 1, "needs_all_prior": True},
        {"name": "validate", "fn": f"{mod}:validate", "units": one("validate"), "workers": 1, "needs_all_prior": True},
    ]


# ------------------------------------------------------------------ pass 1
def pass1_month(ctx, unit):
    cfg, jc, month = ctx.cfg, ctx.jcfg, unit["id"]
    subs = scope_subreddits(cfg, jc)
    authors, _ = load_authors(cfg)
    assert len(authors) < 1 << jc["author_bits"], "author ids do not fit in author_bits"
    nb, kb, ab = jc["buckets"], jc["key_bits"], jc["author_bits"]
    assert nb & (nb - 1) == 0
    tok = Tokenizer(jc)
    raw = RawComments(cfg, jc, month, subs, authors)
    final = work_dir(cfg) / "pass1" / month
    tmp = work_dir(cfg) / "pass1" / f"{month}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(final, ignore_errors=True)
    tmp.mkdir(parents=True)
    keys_l, cnt_l = [], []
    chunk, n_chunks, in_chunk, rows_out = 0, 0, 0, 0

    def flush():
        nonlocal keys_l, cnt_l, n_chunks, rows_out
        if not keys_l:
            return
        k, c = sum_duplicates(np.concatenate(keys_l), np.concatenate(cnt_l))
        bucket = ((k >> np.uint64(ab)) & np.uint64(nb - 1)).astype(np.int64)
        order = np.argsort(bucket, kind="stable")
        k, c, bucket = k[order], c[order], bucket[order]
        edges = np.searchsorted(bucket, np.arange(nb + 1))
        for b in range(nb):
            if edges[b + 1] > edges[b]:
                d = tmp / f"b{b:02d}"
                d.mkdir(exist_ok=True)
                np.savez(d / f"c{n_chunks:03d}.npz", k=k[edges[b]:edges[b + 1]], c=c[edges[b]:edges[b + 1]])
        rows_out += k.size
        n_chunks += 1
        keys_l, cnt_l = [], []

    texts, auth = [], []
    n = 0
    for a, _sub, _ts, text in raw:
        texts.append(text)
        auth.append(a)
        if len(texts) >= jc["batch_comments"]:
            n += len(texts)
            keys, cid, _, _ = bigram_keys(texts, tok, kb)
            comb = (keys << np.uint64(ab)) | np.asarray(auth, dtype=np.uint64)[cid]
            u, c = np.unique(comb, return_counts=True)
            keys_l.append(u)
            cnt_l.append(c.astype(np.uint32))
            in_chunk += len(texts)
            texts, auth = [], []
            if in_chunk >= jc["chunk_comments"] or ctx.over_limit():
                if ctx.over_limit():
                    ctx.shrink()
                flush()
                in_chunk = 0
            ctx.report(rows=raw.stats["lines"], fraction=min(0.99, raw.stats["lines"] / 3.3e6 / 1.0))
    if texts:
        keys, cid, _, _ = bigram_keys(texts, tok, kb)
        comb = (keys << np.uint64(ab)) | np.asarray(auth, dtype=np.uint64)[cid]
        u, c = np.unique(comb, return_counts=True)
        keys_l.append(u)
        cnt_l.append(c.astype(np.uint32))
    flush()
    tmp.rename(final)
    nbytes = sum(f.stat().st_size for f in final.rglob("*.npz"))
    ctx.log(f"{month}: {raw.stats['lines']:,} lines, {raw.stats['counted']:,} comments counted, "
            f"{rows_out:,} pair rows in {n_chunks} chunks; {dict(raw.stats)}")
    write_json(dict(raw.stats), work_dir(cfg) / "pass1_stats" / f"{month}.json")
    return {"rows": raw.stats["lines"], "bytes": nbytes, "counted": raw.stats["counted"]}


# ------------------------------------------------------------------ vocabulary
def merge_bucket(ctx, unit):
    cfg, jc = ctx.cfg, ctx.jcfg
    ab = jc["author_bits"]
    b = unit["bucket"]
    files = sorted((work_dir(cfg) / "pass1").glob(f"*/b{b:02d}/*.npz"))
    ks, cs = [], []
    for f in files:
        with np.load(f) as z:
            ks.append(z["k"])
            cs.append(z["c"])
    if ks:
        k, c = sum_duplicates(np.concatenate(ks), np.concatenate(cs).astype(np.uint64))
        key = k >> np.uint64(ab)
        first = np.flatnonzero(np.concatenate(([True], key[1:] != key[:-1])))
        phrase = key[first]
        n_users = np.diff(np.append(first, key.size)).astype(np.uint32)
        total = np.add.reduceat(c, first)
        keep = n_users >= jc["min_authors_per_phrase"]
        out = {"key": phrase[keep], "n_users": n_users[keep], "total": total[keep]}
        distinct = int(phrase.size)
    else:
        out = {"key": np.empty(0, np.uint64), "n_users": np.empty(0, np.uint32), "total": np.empty(0, np.uint64)}
        distinct = 0
    with atomic_path(work_dir(cfg) / "vocab" / f"b{b:02d}.npz") as tmp:
        with open(tmp, "wb") as fh:
            np.savez(fh, **out)
    return {"rows": distinct, "bytes": int(out["key"].size * 20), "candidates": int(out["key"].size)}


def vocab_select(ctx, unit):
    cfg, jc = ctx.cfg, ctx.jcfg
    parts = [np.load(f) for f in sorted((work_dir(cfg) / "vocab").glob("b*.npz"))]
    key = np.concatenate([p["key"] for p in parts])
    n_users = np.concatenate([p["n_users"] for p in parts])
    total = np.concatenate([p["total"] for p in parts])
    assert jc["vocab_size"] < 1 << PHRASE_BITS
    order = np.lexsort((key, -total.astype(np.int64), -n_users.astype(np.int64)))[: jc["vocab_size"]]
    key, n_users, total = key[order], n_users[order], total[order]
    ids = np.arange(key.size, dtype=np.int32)
    by_key = np.argsort(key)
    np.save(work_dir(cfg) / "vocab_keys.npy", key[by_key])
    np.save(work_dir(cfg) / "vocab_ids.npy", ids[by_key])
    table = pa.table({"phrase_id": ids, "key": key, "n_users": n_users.astype(np.int32),
                      "total_count": total.astype(np.int64)})
    size = write_parquet(table, data_path(cfg, "phrase_counts") / "vocab_hashes.parquet")
    candidates = sum(int(p["key"].size) for p in parts)
    ctx.log(f"{candidates:,} phrases with >= {jc['min_authors_per_phrase']} authors; kept {key.size:,}; "
            f"min authors kept {int(n_users.min()) if n_users.size else 0}")
    return {"rows": int(key.size), "bytes": size, "candidates": candidates}


# ------------------------------------------------------------------ pass 2
def week_index(ts_list, first_week_epoch: int) -> np.ndarray:
    """Week number since ``first_week`` for epoch seconds, using the repo's windowing function."""
    t = pa.table({"created_utc": pa.array(ts_list, pa.timestamp("s", tz="UTC"))})
    ws = with_week_start(t)["week_start"].cast(pa.timestamp("s", tz="UTC")).cast(pa.int64()).to_numpy()
    return (ws - first_week_epoch) // (7 * 86400)


def pass2_month(ctx, unit):
    cfg, jc, month = ctx.cfg, ctx.jcfg, unit["id"]
    subs = scope_subreddits(cfg, jc)
    authors, n_comments = load_authors(cfg)
    eligible = n_comments >= jc["min_author_comments"]
    kb, ab = jc["key_bits"], jc["author_bits"]
    vkeys = np.load(work_dir(cfg) / "vocab_keys.npy")
    vids = np.load(work_dir(cfg) / "vocab_ids.npy")
    n_vocab = int(vids.size)
    first_week = datetime.fromisoformat(cfg["first_week"]).replace(tzinfo=timezone.utc)
    fw = int(first_week.timestamp())
    tok = Tokenizer(jc)
    raw = RawComments(cfg, jc, month, subs, authors)
    seen = np.zeros(n_vocab, dtype=bool)
    strings: dict[int, str] = {}
    pk_l, pc_l, tk_l, tn_l, tm_l = [], [], [], [], []   # packed counts; totals (key, n_comments, m_vocab)

    def consolidate():
        nonlocal pk_l, pc_l, tk_l, tn_l, tm_l
        if pk_l:
            k, c = sum_duplicates(np.concatenate(pk_l), np.concatenate(pc_l))
            pk_l, pc_l = [k], [c]
        if tk_l:
            k = np.concatenate(tk_l)
            n = np.concatenate(tn_l)
            m = np.concatenate(tm_l)
            order = np.argsort(k, kind="stable")
            k, n, m = k[order], n[order], m[order]
            first = np.flatnonzero(np.concatenate(([True], k[1:] != k[:-1])))
            tk_l, tn_l, tm_l = [k[first]], [np.add.reduceat(n, first)], [np.add.reduceat(m, first)]

    buf = []
    n_rows = 0

    def process(buf):
        aidx = np.fromiter((b[0] for b in buf), dtype=np.int64, count=len(buf))
        sidx = np.fromiter((b[1] for b in buf), dtype=np.int64, count=len(buf))
        wk = week_index([b[2] for b in buf], fw)
        texts = [b[3] for b in buf]
        keys, cid, pos, flat_tok = bigram_keys(texts, tok, kb, want_tokens=True)
        i = np.searchsorted(vkeys, keys)
        i[i == n_vocab] = 0
        hit = vkeys[i] == keys
        pid = vids[i[hit]]
        # phrase text, once per vocabulary phrase (any author)
        new = ~seen[pid]
        if new.any():
            u, first = np.unique(pid[new], return_index=True)
            hp = pos[hit][new][first]
            for p, q in zip(u.tolist(), hp.tolist()):
                strings[p] = flat_tok[q] + " " + flat_tok[q + 1]
            seen[u] = True
        el = eligible[aidx]
        # totals: every eligible comment counts in n_comments; in-vocabulary pairs add to m_vocab
        k3 = (wk.astype(np.uint64) << np.uint64(SUB_BITS + ab)) | (sidx.astype(np.uint64) << np.uint64(ab)) | aidx.astype(np.uint64)
        c_el = cid[hit]
        sel = el[c_el]
        packed = (k3[c_el[sel]] << np.uint64(PHRASE_BITS)) | pid[sel].astype(np.uint64)
        u, c = np.unique(packed, return_counts=True)
        pk_l.append(u)
        pc_l.append(c.astype(np.uint32))
        tk_l.append(k3[el])
        tn_l.append(np.ones(int(el.sum()), dtype=np.uint32))
        tm_l.append(np.zeros(int(el.sum()), dtype=np.uint32))
        # m_vocab per key3
        mk, mc = np.unique(packed >> np.uint64(PHRASE_BITS), return_counts=True)
        tk_l.append(mk)
        tn_l.append(np.zeros(mk.size, dtype=np.uint32))
        tm_l.append(mc.astype(np.uint32))

    for rec in raw:
        buf.append(rec)
        if len(buf) >= jc["batch_comments"]:
            process(buf)
            buf = []
            if sum(a.size for a in pk_l) > 20_000_000 or ctx.over_limit():
                if ctx.over_limit():
                    ctx.shrink()
                consolidate()
            ctx.report(rows=raw.stats["lines"], fraction=min(0.99, raw.stats["lines"] / 3.3e6))
    if buf:
        process(buf)
    consolidate()

    packed, count = (pk_l[0], pc_l[0]) if pk_l else (np.empty(0, np.uint64), np.empty(0, np.uint32))
    pid = (packed & np.uint64((1 << PHRASE_BITS) - 1)).astype(np.int32)
    k3 = packed >> np.uint64(PHRASE_BITS)
    counts_tbl = pa.table({
        "week_start": week_ts(k3 >> np.uint64(SUB_BITS + ab), fw),
        "subreddit": dict_col((k3 >> np.uint64(ab)) & np.uint64((1 << SUB_BITS) - 1), subs),
        "author": dict_col(k3 & np.uint64((1 << ab) - 1), authors),
        "phrase_id": pid, "count": count.astype(np.int32)})
    tk, tn, tm = (tk_l[0], tn_l[0], tm_l[0]) if tk_l else (np.empty(0, np.uint64),) * 3
    totals_tbl = pa.table({
        "week_start": week_ts(tk >> np.uint64(SUB_BITS + ab), fw),
        "subreddit": dict_col((tk >> np.uint64(ab)) & np.uint64((1 << SUB_BITS) - 1), subs),
        "author": dict_col(tk & np.uint64((1 << ab) - 1), authors),
        "m_vocab": tm.astype(np.int32), "n_comments": tn.astype(np.int32)})
    out = data_path(cfg, "phrase_counts") / f"month={month}"
    nbytes = write_parquet(counts_tbl, out / "counts.parquet") + write_parquet(totals_tbl, out / "totals.parquet")
    write_json({str(k): v for k, v in strings.items()}, work_dir(cfg) / "pass2_strings" / f"{month}.json")
    write_json(dict(raw.stats), work_dir(cfg) / "pass2_stats" / f"{month}.json")
    ctx.log(f"{month}: {counts_tbl.num_rows:,} count rows, {totals_tbl.num_rows:,} total rows; {dict(raw.stats)}")
    return {"rows": raw.stats["lines"], "bytes": nbytes, "count_rows": counts_tbl.num_rows}


def week_ts(week_idx: np.ndarray, fw: int) -> pa.Array:
    return pa.array((week_idx.astype(np.int64) * 7 * 86400 + fw), pa.int64()).cast(pa.timestamp("s", tz="UTC"))


def dict_col(codes: np.ndarray, names: list[str]) -> pa.DictionaryArray:
    uniq, inv = np.unique(codes.astype(np.int64), return_inverse=True)
    return pa.DictionaryArray.from_arrays(pa.array(inv.astype(np.int32)), pa.array([names[i] for i in uniq.tolist()]))


def after_unit(cfg, phase_name, unit, ck):
    """After the first pass-2 month, project the output size and stop if it would not fit on disk."""
    if phase_name != "pass2":
        return None
    jc = job_cfg(cfg, NAME)
    marker = runs_dir(cfg) / NAME / "disk_check.json"
    if marker.exists():
        return None
    total_w = weight_total(cfg, jc)
    projected = ck["bytes"] / unit["weight"] * total_w
    free = shutil.disk_usage(data_path(cfg, "phrase_counts").parent).free
    write_json({"unit": unit["id"], "unit_bytes": ck["bytes"], "projected_total_bytes": projected,
                "free_bytes": free, "limit_fraction": jc["disk_fraction_limit"]}, marker)
    if projected > jc["disk_fraction_limit"] * free:
        marker.unlink()  # re-check after a resume with a smaller scope
        return (f"projected phrase_counts output {projected / 1e9:.0f} GB exceeds "
                f"{100 * jc['disk_fraction_limit']:.0f}% of free disk ({free / 1e9:.0f} GB). Set jobs.phrase_counts.subreddits "
                f"to 'fallback' or cut vocab_size / raise min_author_comments in config/prep_jobs.yaml, then relaunch with --resume")
    return None


# ------------------------------------------------------------------ finalise
def _load_month(cfg, kind):
    def load(month):
        return pq.read_table(data_path(cfg, "phrase_counts") / f"month={month}" / f"{kind}.parquet")
    return load


def _save_month(cfg, kind):
    """Write the folded month next to the original; ``finalise`` swaps them in once everything succeeded."""
    def save(month, table):
        write_parquet(table, data_path(cfg, "phrase_counts") / f"month={month}" / f"{kind}.folded.parquet")
    return save


def finalise(ctx, unit):
    cfg = ctx.cfg
    months = cfg["months"]
    rows_counts = fold_boundary_weeks(months, _load_month(cfg, "counts"), _save_month(cfg, "counts"),
                                      ["week_start", "subreddit", "author", "phrase_id"], ["count"])
    rows_totals = fold_boundary_weeks(months, _load_month(cfg, "totals"), _save_month(cfg, "totals"),
                                      ["week_start", "subreddit", "author"], ["m_vocab", "n_comments"])
    for m in months:  # swap in the folded files; before this point the originals are untouched
        for kind in ("counts", "totals"):
            d = data_path(cfg, "phrase_counts") / f"month={m}"
            (d / f"{kind}.folded.parquet").replace(d / f"{kind}.parquet")
    # vocabulary text
    strings: dict[int, str] = {}
    for f in sorted((work_dir(cfg) / "pass2_strings").glob("*.json")):
        for k, v in read_json(f, {}).items():
            strings.setdefault(int(k), v)
    vh = pq.read_table(data_path(cfg, "phrase_counts") / "vocab_hashes.parquet")
    ids = vh["phrase_id"].to_pylist()
    missing = [i for i in ids if i not in strings]
    vocab = pa.table({"phrase_id": vh["phrase_id"], "phrase": pa.array([strings.get(i) for i in ids], pa.string()),
                      "n_users": vh["n_users"], "total_count": vh["total_count"]})
    write_parquet(vocab, data_path(cfg, "phrase_counts") / "vocab.parquet")
    write_json({"counts_rows": rows_counts, "totals_rows": rows_totals, "vocab_missing_text": len(missing)},
               runs_dir(cfg) / NAME / "finalise.json")
    ctx.log(f"finalised; {len(missing)} vocabulary phrases without text")
    return {"rows": sum(rows_counts.values()), "bytes": 0, "missing_text": len(missing)}


# ------------------------------------------------------------------ validation
def python_phrases(text: str, jc, tok: Tokenizer) -> Counter:
    """Independent recount: plain Python, phrase strings (no hashing)."""
    drop = frozenset(jc["url_fragment_tokens"])
    rx = re.compile(jc["token_regex"])
    toks = [t for t in text.split() if len(t) >= jc["min_token_length"] and t not in drop and rx.match(t)]
    return Counter(f"{a} {b}" for a, b in zip(toks, toks[1:])
                   if not (a in ENGLISH_STOP_WORDS and b in ENGLISH_STOP_WORDS))


def validate(ctx, unit):
    cfg, jc = ctx.cfg, ctx.jcfg
    out = ["# phrase_counts validation", f"generated {datetime.now():%Y-%m-%d %H:%M}", ""]
    ok = True
    # 1. independent recount of random comments
    subs = scope_subreddits(cfg, jc)
    authors, _ = load_authors(cfg)
    month = cfg["months"][len(cfg["months"]) // 2]
    raw = RawComments(cfg, jc, month, subs, authors)
    rng = random.Random(cfg["seed"])
    sample, seen = [], 0
    for rec in raw:
        seen += 1
        if len(sample) < jc["validation_sample_comments"]:
            sample.append(rec[3])
        else:
            j = rng.randrange(seen)
            if j < jc["validation_sample_comments"]:
                sample[j] = rec[3]
        if seen >= 300_000:
            break
    tok = Tokenizer(jc)
    keys, cid, _, _ = bigram_keys(sample, tok, jc["key_bits"])
    mismatched = 0
    for i, text in enumerate(sample):
        expect = Counter()
        for phrase, n in python_phrases(text, jc, tok).items():
            a, b = phrase.split(" ")
            expect[pair_key(tok.code(a), tok.code(b), jc["key_bits"])] += n
        got = Counter(keys[cid == i].tolist())
        mismatched += expect != got
    out += [f"## 1. Independent recount ({len(sample)} random comments from {month}, first {seen:,} in scope)",
            f"comments whose phrase multiset differs from the plain-Python recount: **{mismatched}**", ""]
    ok &= mismatched == 0
    # 2. top phrases
    vocab = pq.read_table(data_path(cfg, "phrase_counts") / "vocab.parquet").to_pandas()
    top = vocab.sort_values("total_count", ascending=False).head(jc["validation_top_phrases"])
    out += [f"## 2. The {jc['validation_top_phrases']} most frequent vocabulary phrases", "",
            "| phrase | authors | total |", "|---|---|---|"]
    out += [f"| {r.phrase} | {r.n_users:,} | {r.total_count:,} |" for r in top.itertuples()]
    out += ["", f"vocabulary: {len(vocab):,} phrases, {int(vocab.phrase.isna().sum())} without text", ""]
    # 3. totals equal sum of counts
    out += ["## 3. Totals equal the sum of counts", "", "| month | count rows | sum(count) | sum(m_vocab) | keys differing |", "|---|---|---|---|---|"]
    for m in cfg["months"]:
        base = data_path(cfg, "phrase_counts") / f"month={m}"
        c = pq.read_table(base / "counts.parquet")
        t = pq.read_table(base / "totals.parquet")
        key = ["week_start", "subreddit", "author"]
        c = c.set_column(1, "subreddit", pc.cast(c["subreddit"], pa.string())).set_column(2, "author", pc.cast(c["author"], pa.string()))
        t = t.set_column(1, "subreddit", pc.cast(t["subreddit"], pa.string())).set_column(2, "author", pc.cast(t["author"], pa.string()))
        cs = c.group_by(key).aggregate([("count", "sum")])
        j = t.select(key + ["m_vocab"]).join(cs, key, join_type="left outer")
        diff = pc.sum(pc.cast(pc.not_equal(j["m_vocab"], pc.fill_null(j["count_sum"], 0)), pa.int64())).as_py() or 0
        out.append(f"| {m} | {c.num_rows:,} | {pc.sum(c['count']).as_py():,} | {pc.sum(t['m_vocab']).as_py():,} | {diff} |")
        ok &= diff == 0 and pc.sum(c["count"]).as_py() == pc.sum(t["m_vocab"]).as_py()
    out += ["", f"**Result: {'PASS' if ok else 'FAIL'}**"]
    (runs_dir(cfg) / NAME / "VALIDATION.md").write_text("\n".join(out), encoding="utf-8")
    if ok:
        shutil.rmtree(work_dir(cfg) / "pass1", ignore_errors=True)
    if not ok:
        raise RuntimeError("validation failed; see VALIDATION.md")
    return {"rows": len(sample), "bytes": 0}

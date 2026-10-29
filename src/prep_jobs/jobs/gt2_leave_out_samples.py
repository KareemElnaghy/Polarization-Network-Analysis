"""Job T3, gt2_leave_out_samples: the leave-out estimator inside the thread samples (ground truth #2, sample versions).

``phrase_counts`` is aggregated per (week, subreddit, author) and cannot be cut to sampled threads, so each
week's unit re-reads that week's comments from the raw dumps with Job A's reader (``RawComments``, same scope
filters and exclusions) and Job A's counting function (``bigram_keys`` with the same ``vocab.parquet`` hashes),
keeping comments in sampled threads (``sampled_threads.parquet``) by labelled users in the venue subreddits.
Counts are kept in memory only, per rate (0.01, then 0.05, then 0.20), the estimator is computed at once and
only result rows are written. Same contraction filter as T2. Label set: main.
"""

from __future__ import annotations

import gc
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pyarrow as pa

from src.gt_runs.cells import gt2_cell, keep_phrase_mask, sampled_links
from src.gt_runs.config import gt_cfg, gt_path, label_sets, labels_cfg, venues
from src.gt_runs.labels import AuthorIndex, heavy_set, load_side_map
from src.prep_jobs.config import data_path, job_cfg, runs_dir
from src.prep_jobs.jobs.clean_edges_and_sampling import weeks_of
from src.prep_jobs.jobs.phrase_counts import (PHRASE_BITS, RawComments, Tokenizer, bigram_keys, load_authors, sum_duplicates,
                                              work_dir)
from src.prep_jobs.util import read_json, write_json, write_parquet

NAME = "gt2_leave_out_samples"
DEADLINE_CUTS = ["drop the 0.20 sample (sample_rates in config/gt_runs.yaml)", "n_perm 100 -> 50"]
COLUMNS = ["week_start", "venue", "label_set", "sample", "n_dem", "n_rep", "pi", "null_mean", "null_sd", "boot_lo",
           "boot_hi", "low_n"]


def phases(cfg):
    mod = "src.prep_jobs.jobs.gt2_leave_out_samples"
    return [
        {"name": "weeks", "fn": f"{mod}:week_unit", "units": [{"id": w, "weight": 100} for w in weeks_of(cfg)]},
        {"name": "finalise", "fn": f"{mod}:finalise", "units": [{"id": "finalise", "weight": 50}], "workers": 1,
         "needs_all_prior": True},
    ]


def months_for_week(cfg, week: str) -> list[str]:
    d = datetime.fromisoformat(week)
    return sorted({d.strftime("%Y-%m"), (d + timedelta(days=6)).strftime("%Y-%m")} & set(cfg["months"]))


class RateCounts:
    """In-memory counts per rate keyed by packed (subreddit, author, phrase); merged now and then."""

    def __init__(self, rates, author_bits):
        self.rates, self.ab = rates, author_bits
        self.keys = {r: [] for r in rates}
        self.cnts = {r: [] for r in rates}

    def add(self, rate, packed):
        if packed.size:
            u, c = np.unique(packed, return_counts=True)
            self.keys[rate].append(u)
            self.cnts[rate].append(c.astype(np.int64))

    def consolidate(self, rate):
        if len(self.keys[rate]) > 1:
            k, c = sum_duplicates(np.concatenate(self.keys[rate]), np.concatenate(self.cnts[rate]))
            self.keys[rate], self.cnts[rate] = [k], [c]

    def size(self):
        return sum(a.size for r in self.rates for a in self.keys[r])

    def take(self, rate):
        self.consolidate(rate)
        k = self.keys[rate][0] if self.keys[rate] else np.empty(0, np.uint64)
        c = self.cnts[rate][0] if self.cnts[rate] else np.empty(0, np.int64)
        self.keys[rate], self.cnts[rate] = [], []
        return k, c


def week_unit(ctx, unit):
    cfg, g, lab, week = ctx.cfg, gt_cfg(), labels_cfg(), unit["id"]
    jc = job_cfg(cfg, "phrase_counts")
    kb, ab = jc["key_bits"], jc["author_bits"]
    rates = sorted(g["sample_rates"])
    vmap = venues(g)
    sub_list = sorted({s for subs in vmap.values() for s in subs})
    authors, _ = load_authors(cfg)
    aidx = AuthorIndex(authors)
    root = gt_path(g, "labels")
    sets = [s for s in label_sets(g, "gt2_samples") if s == "main"]
    side = {(v, s): aidx.side_vector(load_side_map(root, v, s)) for v in vmap for s in sets}
    labelled = np.zeros(len(authors), dtype=bool)
    for sv in side.values():
        labelled |= sv >= 0
    heavy = heavy_set(root, week) if lab["exclude_heavy_posters"] else None
    heavy_vec = aidx.flag_vector(heavy) if heavy else None
    keep, _ = keep_phrase_mask(cfg, g["gt2"]["exclude_tokens"])

    links = sampled_links(cfg, week, rates)
    link_bits: dict[str, int] = {}
    for i, r in enumerate(rates):
        for l in links[r]:
            link_bits[l] = link_bits.get(l, 0) | (1 << i)
    vkeys, vids = np.load(work_dir(cfg) / "vocab_keys.npy"), np.load(work_dir(cfg) / "vocab_ids.npy")
    tok = Tokenizer(jc)
    counts = RateCounts(rates, ab)
    lo = int(datetime.fromisoformat(week).replace(tzinfo=timezone.utc).timestamp())
    sub_code = {s: i for i, s in enumerate(sub_list)}

    def process(buf):
        a = np.fromiter((b[0] for b in buf), dtype=np.int64, count=len(buf))
        sc = np.fromiter((b[1] for b in buf), dtype=np.int64, count=len(buf))
        bits = np.fromiter((b[2] for b in buf), dtype=np.int64, count=len(buf))
        keys, cid, _, _ = bigram_keys([b[3] for b in buf], tok, kb)
        i = np.searchsorted(vkeys, keys)
        i[i == len(vkeys)] = 0
        hit = vkeys[i] == keys
        pid = vids[i[hit]]
        ok = keep[pid]
        c_idx = cid[hit][ok]
        pid = pid[ok]
        packed = (sc[c_idx].astype(np.uint64) << np.uint64(ab + PHRASE_BITS)) | (a[c_idx].astype(np.uint64) << np.uint64(PHRASE_BITS)) \
            | pid.astype(np.uint64)
        for j, r in enumerate(rates):
            counts.add(r, packed[(bits[c_idx] >> j) & 1 == 1])

    buf, n_seen = [], 0
    for month in months_for_week(cfg, week):
        raw = RawComments(cfg, jc, month, sub_list, authors, window=(lo, lo + 7 * 86400), with_link=True)
        for a, sub, _ts, text, link in raw:
            bits = link_bits.get(link)
            if not bits or not labelled[a]:
                continue
            buf.append((a, sub, bits, text))
            if len(buf) >= g["samples"]["batch_comments"]:
                process(buf)
                n_seen += len(buf)
                buf = []
                if counts.size() > 15_000_000 or ctx.over_limit():
                    for r in rates:
                        counts.consolidate(r)
                ctx.report(rows=raw.stats["lines"], fraction=0.5)
    if buf:
        process(buf)
        n_seen += len(buf)
    ctx.log(f"{week}: {n_seen:,} comments in sampled threads by labelled users")

    rows = []
    p = g["gt2"]
    for r in rates:
        k, c = counts.take(r)
        sub_c = (k >> np.uint64(ab + PHRASE_BITS)).astype(np.int64)
        a_i = ((k >> np.uint64(PHRASE_BITS)) & np.uint64((1 << ab) - 1)).astype(np.int64)
        ph = (k & np.uint64((1 << PHRASE_BITS) - 1)).astype(np.int64)
        for venue, subs in vmap.items():
            sel = np.isin(sub_c, [sub_code[s] for s in subs])
            for s in sets:
                boot = p["n_boot"] if (venue == "shared" and s == "main" and g["samples"]["bootstrap_shared"]) else 0
                cell = gt2_cell(a_i[sel], ph[sel], c[sel], side[(venue, s)], None, heavy_vec, len(keep), p["min_phrases"],
                                p["n_perm"], boot, g["seed"], lab["min_users_per_side"])
                rows.append({"week_start": week, "venue": venue, "label_set": s, "sample": str(r), **cell})
        del k, c, sub_c, a_i, ph
        gc.collect()
    write_json(rows, runs_dir(cfg) / NAME / "work" / f"{week}.json")
    return {"rows": n_seen, "bytes": 0}


def finalise(ctx, unit):
    cfg, g = ctx.cfg, gt_cfg()
    rows = [r for f in sorted((runs_dir(cfg) / NAME / "work").glob("*.json")) for r in read_json(f)]
    df = pd.DataFrame(rows)[COLUMNS]
    size = write_parquet(pa.Table.from_pandas(df, preserve_index=False), gt_path(g, "results") / "gt2_weekly_samples.parquet")
    ctx.log(f"{len(df)} cells; low_n {int(df['low_n'].sum())}")
    return {"rows": len(df), "bytes": size}

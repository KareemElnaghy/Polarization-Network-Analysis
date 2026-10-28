"""Explore fixes for ground truth #2 (see docs/decisions.md). Writes only to results/gt2_fix_options/.

  python scripts/gt2_fix_options.py cache-full [workers]
  python scripts/gt2_fix_options.py cache-samples [workers]
  python scripts/gt2_fix_options.py run STAGE VARIANTS WEEKS SOURCES [workers]   # e.g. run s1 base,A,B six full
  python scripts/gt2_fix_options.py auc [workers]
  python scripts/gt2_fix_options.py concentration
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.gt_runs.cells import PhraseWeek, keep_phrase_mask, read_phrase_week
from src.gt_runs.config import gt_cfg, gt_path, labels_cfg, venues
from src.gt_runs.fix_options import classifier_auc, fix_cell, top_share
from src.gt_runs.labels import AuthorIndex, heavy_set, load_side_map
from src.prep_jobs.config import load_config
from src.prep_jobs.jobs.clean_edges_and_sampling import weeks_of

OUT = Path(__file__).resolve().parents[1] / "results" / "gt2_fix_options"
CACHE = OUT / "cache"
SIX = ["2016-01-25", "2016-03-07", "2016-07-18", "2016-04-18", "2016-08-08", "2016-11-07"]  # primary/convention/quiet/election
VARIANTS = {
    "base": {},
    "A_equal": {"equal": True},
    "B_phr20": {"phrase_users": 20},
    "C300": {"cap": 300},
    "C1000": {"cap": 1000},
    "D_win4": {"window": 4},
    "AB": {"equal": True, "phrase_users": 20},
    "ABD": {"equal": True, "phrase_users": 20, "window": 4},
    "ABC300": {"equal": True, "phrase_users": 20, "cap": 300},
    "BD": {"phrase_users": 20, "window": 4},
    "BC300": {"phrase_users": 20, "cap": 300},
    "AD": {"equal": True, "window": 4},
}
_S = {}


def _state():
    if not _S:
        cfg, g, lab = load_config(), gt_cfg(), labels_cfg()
        authors = __import__("src.prep_jobs.jobs.phrase_counts", fromlist=["x"]).load_authors(cfg)[0]
        aidx = AuthorIndex(authors)
        root = gt_path(g, "labels")
        keep, _ = keep_phrase_mask(cfg, g["gt2"]["exclude_tokens"])
        _S.update(cfg=cfg, g=g, lab=lab, authors=authors, aidx=aidx, root=root, keep=keep,
                  side={s: aidx.side_vector(load_side_map(root, "shared", s)) for s in ("main", "gold")})
    return _S


def _save(path, a, p, c):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, a=a.astype(np.int32), p=p.astype(np.int32), c=c.astype(np.int32))


def _cached(path: Path) -> bool:
    return path.exists()


# ------------------------------------------------------------------ caches
def cache_full(week):
    f = CACHE / "full" / f"{week}.npz"
    if _cached(f):
        return week
    s = _state()
    subs = venues(s["g"])["shared"]
    pw = PhraseWeek(read_phrase_week(s["cfg"], week), s["aidx"])
    a, p, c = pw.rows(subs)
    ok = (s["side"]["main"][a] >= 0) | (s["side"]["gold"][a] >= 0)
    ok &= s["keep"][p]
    heavy = heavy_set(s["root"], week)
    ok &= ~s["aidx"].flag_vector(heavy)[a]
    _save(f, a[ok], p[ok], c[ok])
    return week


def cache_samples(week):
    """One raw scan of the week: shared-venue triplets for every sample rate (same filters as job T3)."""
    from datetime import datetime, timezone

    from src.gt_runs.cells import sampled_links
    from src.prep_jobs.config import job_cfg
    from src.prep_jobs.jobs.gt2_leave_out_samples import RateCounts, months_for_week
    from src.prep_jobs.jobs.phrase_counts import PHRASE_BITS, RawComments, Tokenizer, bigram_keys, work_dir

    s = _state()
    cfg, g = s["cfg"], s["g"]
    rates = sorted(g["sample_rates"])
    if all(_cached(CACHE / f"r{r}" / f"{week}.npz") for r in rates):
        return week
    jc = job_cfg(cfg, "phrase_counts")
    kb, ab = jc["key_bits"], jc["author_bits"]
    subs = venues(g)["shared"]
    authors, aidx, keep = s["authors"], s["aidx"], s["keep"]
    labelled = (s["side"]["main"] >= 0) | (s["side"]["gold"] >= 0)
    heavy_vec = aidx.flag_vector(heavy_set(s["root"], week))
    links = sampled_links(cfg, week, rates)
    link_bits: dict[str, int] = {}
    for i, r in enumerate(rates):
        for l in links[r]:
            link_bits[l] = link_bits.get(l, 0) | (1 << i)
    vkeys, vids = np.load(work_dir(cfg) / "vocab_keys.npy"), np.load(work_dir(cfg) / "vocab_ids.npy")
    tok = Tokenizer(jc)
    counts = RateCounts(rates, ab)
    lo = int(datetime.fromisoformat(week).replace(tzinfo=timezone.utc).timestamp())

    def process(buf):
        a = np.fromiter((b[0] for b in buf), dtype=np.int64, count=len(buf))
        bits = np.fromiter((b[1] for b in buf), dtype=np.int64, count=len(buf))
        keys, cid, _, _ = bigram_keys([b[2] for b in buf], tok, kb)
        i = np.searchsorted(vkeys, keys)
        i[i == len(vkeys)] = 0
        hit = vkeys[i] == keys
        pid = vids[i[hit]]
        ok = keep[pid]
        c_idx, pid = cid[hit][ok], pid[ok]
        packed = (a[c_idx].astype(np.uint64) << np.uint64(PHRASE_BITS)) | pid.astype(np.uint64)
        for j, r in enumerate(rates):
            counts.add(r, packed[(bits[c_idx] >> j) & 1 == 1])

    buf = []
    for month in months_for_week(cfg, week):
        raw = RawComments(cfg, jc, month, subs, authors, window=(lo, lo + 7 * 86400), with_link=True)
        for a, _sub, _ts, text, link in raw:
            bits = link_bits.get(link)
            if not bits or not labelled[a] or heavy_vec[a]:
                continue
            buf.append((a, bits, text))
            if len(buf) >= g["samples"]["batch_comments"]:
                process(buf)
                buf = []
                if counts.size() > 15_000_000:
                    for r in rates:
                        counts.consolidate(r)
    if buf:
        process(buf)
    for r in rates:
        k, c = counts.take(r)
        _save(CACHE / f"r{r}" / f"{week}.npz", ((k >> np.uint64(PHRASE_BITS)) & np.uint64((1 << ab) - 1)).astype(np.int64),
              (k & np.uint64((1 << PHRASE_BITS) - 1)).astype(np.int64), c)
    return week


# ------------------------------------------------------------------ runs
def load_window(source, weeks, week, n):
    i = weeks.index(week)
    parts = [np.load(CACHE / source / f"{w}.npz") for w in weeks[max(0, i - n + 1): i + 1]]
    return (np.concatenate([x["a"] for x in parts]).astype(np.int64), np.concatenate([x["p"] for x in parts]).astype(np.int64),
            np.concatenate([x["c"] for x in parts]).astype(np.int64))


def run_task(task):
    stage, variant, label_set, source, week = task
    s = _state()
    params = dict(VARIANTS[variant])
    n = params.pop("window", 1)
    weeks = weeks_of(s["cfg"])
    a, p, c = load_window(source, weeks, week, n)
    side = s["side"][label_set]
    sel = side[a] >= 0
    a, p, c = a[sel], p[sel], c[sel]
    g = s["g"]
    boot = g["gt2"]["n_boot"] if label_set == "main" else 0
    t = time.time()
    r = fix_cell(a, p, c, side, len(s["keep"]), g["gt2"]["min_phrases"], g["gt2"]["n_perm"], boot, g["seed"],
                 s["lab"]["min_users_per_side"], **params)
    return {"stage": stage, "variant": variant, "label_set": label_set, "source": source, "week_start": week,
            "seconds": round(time.time() - t, 1), "top1_share": top_share(a, c), **{k: (None if v != v else v) for k, v in r.items()}}


def cmd_run(stage, variants, weeks_arg, sources, workers=6):
    cfg = load_config()
    allw = weeks_of(cfg)
    weeks = {"six": SIX, "twelve": SIX + ["2016-02-08", "2016-05-02", "2016-06-27", "2016-09-19", "2016-10-17", "2016-12-05"],
             "extra6": ["2016-02-08", "2016-05-02", "2016-06-27", "2016-09-19", "2016-10-17", "2016-12-05"],
             "all": allw}[weeks_arg]
    out = OUT / f"{stage}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["variant"], r["label_set"], r["source"], r["week_start"]))
    tasks = [(stage, v, ls, src, w) for src in sources.split(",") for v in variants.split(",") for ls in ("main", "gold")
             for w in sorted(weeks) if (v, ls, src, w) not in done]
    print(f"{len(tasks)} tasks, {workers} workers", flush=True)
    t0 = time.time()
    with Pool(workers) as pool, open(out, "a") as fh:
        for i, r in enumerate(pool.imap_unordered(run_task, tasks), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if i % 10 == 0:
                print(f"{i}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)


def auc_task(week):
    s = _state()
    a, p, c = load_window("full", weeks_of(s["cfg"]), week, 1)
    side = s["side"]["main"]
    sel = side[a] >= 0
    t = time.time()
    r = classifier_auc(a[sel], p[sel], c[sel], side, len(s["keep"]), s["g"]["gt2"]["min_phrases"])
    return {"week_start": week, "seconds": round(time.time() - t, 1), **r}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd, args = sys.argv[1], sys.argv[2:]
    cfg = load_config()
    if cmd in ("cache-full", "cache-samples"):
        w = int(args[0]) if args else 3
        fn = cache_full if cmd == "cache-full" else cache_samples
        with Pool(w) as pool:
            for i, wk in enumerate(pool.imap_unordered(fn, weeks_of(cfg)), 1):
                print(i, wk, flush=True)
    elif cmd == "run":
        cmd_run(*args[:4], workers=int(args[4]) if len(args) > 4 else 6)
    elif cmd == "auc":
        with Pool(int(args[0]) if args else 3) as pool:
            rows = pool.map(auc_task, SIX)
        pd.DataFrame(rows).to_csv(OUT / "auc_stage1.csv", index=False)
        print(pd.DataFrame(rows))
    elif cmd == "concentration":
        rows = []
        for w in weeks_of(cfg):
            x = np.load(CACHE / "full" / f"{w}.npz")
            s = _state()
            side = s["side"]["main"]
            sel = side[x["a"]] >= 0
            rows.append({"week_start": w, "top1_share": top_share(x["a"][sel].astype(np.int64), x["c"][sel])})
        pd.DataFrame(rows).to_csv(OUT / "concentration.csv", index=False)


if __name__ == "__main__":
    main()

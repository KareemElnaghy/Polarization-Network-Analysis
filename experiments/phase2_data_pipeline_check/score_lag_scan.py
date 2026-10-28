"""Scan raw comment dumps for score, created_utc and retrieved_on in party-tagged subs and r/politics.

Writes one small Parquet per month (default: scratch dir given by --out). Read-only on data/raw.

Usage:
    python experiments/phase2_data_pipeline_check/score_lag_scan.py --out <dir> --workers 6
"""

from __future__ import annotations

import argparse
import bz2
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    import orjson
    _loads = orjson.loads
except ImportError:
    _loads = json.loads

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def target_subreddits() -> frozenset[str]:
    subs = {"politics"}
    for line in (RAW / "subreddits_metadata.json").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["party"]:
                subs.add(row["subreddit"])
    return frozenset(subs)


def scan(args: tuple[str, str]) -> tuple[str, int]:
    month, out = args
    subs = target_subreddits()
    cols: dict[str, list] = {k: [] for k in ("subreddit", "created_utc", "retrieved_on", "score")}
    with bz2.open(RAW / f"comments_{month}.bz2", "rb") as fh:
        for line in fh:
            if b'"subreddit"' not in line:
                continue
            c = _loads(line)
            if c["subreddit"] in subs and c["author"] != "[deleted]":
                cols["subreddit"].append(c["subreddit"])
                cols["created_utc"].append(int(c["created_utc"]))
                cols["retrieved_on"].append(int(c.get("retrieved_on") or 0))
                cols["score"].append(int(c["score"]))
    pq.write_table(pa.table(cols), Path(out) / f"scan_{month}.parquet")
    return month, len(cols["score"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    months = [f"2016-{m:02d}" for m in range(1, 13)]
    with ProcessPoolExecutor(a.workers) as pool:
        for month, n in pool.map(scan, [(m, a.out) for m in months]):
            print(month, n, flush=True)

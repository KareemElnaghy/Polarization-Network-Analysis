"""Shared extraction pass: dedup -> skip cached -> call backend (threaded, retried) -> write store incrementally."""
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from .backends import InvalidOutput
from .schema import pair_hash, was_truncated


def context(cfg: dict, backend) -> dict:
    info = backend.info()
    return {"model": cfg["model"], "quant": info["quantization"], "prompt_version": cfg["prompt_version"],
            "temperature": cfg["temperature"], "seed": cfg["seed"], "prob_method": cfg["prob_method"], **info}


def interleave(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle within week, then round-robin across weeks so a stopped run still covers the whole year."""
    df = df.sample(frac=1, random_state=seed)
    return df.assign(_r=df.groupby("week").cumcount()).sort_values("_r", kind="stable").drop(columns="_r")


def run(edges: pd.DataFrame, backend, store, cfg: dict, stop=lambda: False, progress=None) -> dict:
    """edges needs columns: week, parent (text), body (text). Returns counters. Safe to kill and rerun."""
    ctx, mc = context(cfg, backend), cfg["max_chars"]
    edges = edges.assign(h=[pair_hash(p, b, mc) for p, b in zip(edges.parent, edges.body)])
    trunc = float(np.mean([was_truncated(p, b, mc) for p, b in zip(edges.parent, edges.body)]))
    todo = interleave(edges.drop_duplicates("h"), cfg["seed"])
    todo = todo[~todo.h.isin(store.have(ctx))]
    c = {"edges": len(edges), "unique": int(edges.h.nunique()), "to_run": len(todo), "done": 0, "failed": 0,
         "truncation_rate": trunc}
    lock = threading.Lock()

    def one(row):
        if stop():
            return
        for attempt in range(cfg["max_retries"] + 1):
            try:
                res = backend.extract(row.parent, row.body)
                break
            except InvalidOutput:
                continue
        else:
            with lock:
                c["failed"] += 1
            return
        with lock:  # sqlite connection is shared; serialise writes
            store.put(row.h, ctx, res, attempt)
            c["done"] += 1
            if progress:
                progress(c)

    with ThreadPoolExecutor(cfg["parallel"]) as ex:
        list(ex.map(one, todo.itertuples()))
    return c

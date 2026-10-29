"""Weekly-snapshot completeness check and thread-sampling check (read-only on the edge files).

Part 1: every Monday-start week from 2015-12-28 to 2016-12-26 is present, none is empty,
        and nodes/edges are reported per week after the activity threshold (k=3).
Part 2: for r/politics and r/PoliticalDiscussion only, thread-size distribution per week,
        then one typical week sampled to 20k/50k/100k edges by whole threads (stratified by
        thread size) and by random edges, compared with the full week.

Usage:
    python experiments/phase2_data_pipeline_check/weekly_and_sampling_checks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
EDGES_DIR = ROOT / "data" / "interim" / "edges"
OUT = ROOT / "results" / "sampling_check"
K = 3
SUBREDDITS = ["politics", "PoliticalDiscussion"]
SIZES = (20_000, 50_000, 100_000)
N_SEEDS = 10
SEED = 42
EXPECTED_WEEKS = pd.date_range("2015-12-28", "2016-12-26", freq="7D")


# ---------------------------------------------------------------- part 1
def weekly_k3() -> pd.DataFrame:
    """Per week: replies, nodes/edges at k=1 and k=3 (single pass, as in window_graph), days covered."""
    pair_parts, day_parts = [], []
    for f in sorted(EDGES_DIR.glob("edges_*.parquet")):
        t = pq.read_table(f, columns=["source", "target", "week_start", "created_utc"])
        pair_parts.append(t.group_by(["week_start", "source", "target"]).aggregate([("source", "count")]))
        day = pc.floor_temporal(t["created_utc"], unit="day")
        day_parts.append(pa.table({"week_start": t["week_start"], "day": day})
                         .group_by(["week_start", "day"]).aggregate([]))
        print(f"  read {f.name}", flush=True)
    pairs = pa.concat_tables(pair_parts).group_by(["week_start", "source", "target"]).aggregate(
        [("source_count", "sum")]).to_pandas().rename(columns={"source_count_sum": "w"})
    days = pa.concat_tables(day_parts).group_by(["week_start", "day"]).aggregate([]).to_pandas()
    days = days.groupby("week_start").size().rename("days_with_edges")

    act = pd.concat([pairs[["week_start", "source", "w"]].set_axis(["week_start", "node", "w"], axis=1),
                     pairs[["week_start", "target", "w"]].set_axis(["week_start", "node", "w"], axis=1)])
    act = act.groupby(["week_start", "node"])["w"].sum()
    keep = act[act >= K].reset_index()[["week_start", "node"]]
    kept = pairs.merge(keep.rename(columns={"node": "source"}), on=["week_start", "source"])
    kept = kept.merge(keep.rename(columns={"node": "target"}), on=["week_start", "target"])
    nodes = pd.concat([kept[["week_start", "source"]].set_axis(["week_start", "n"], axis=1),
                       kept[["week_start", "target"]].set_axis(["week_start", "n"], axis=1)]).drop_duplicates()
    out = pd.DataFrame({
        "replies_k1": pairs.groupby("week_start")["w"].sum(),
        "edges_k1": pairs.groupby("week_start").size(),
        "nodes_k1": act.groupby("week_start").size(),
        "replies_k3": kept.groupby("week_start")["w"].sum(),
        "edges_k3": kept.groupby("week_start").size(),
        "nodes_k3": nodes.groupby("week_start").size(),
    }).join(days)
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out


def check_weeks(wk: pd.DataFrame) -> dict:
    present = set(wk.index)
    missing = [d.date().isoformat() for d in EXPECTED_WEEKS if d not in present]
    unexpected = [d.date().isoformat() for d in sorted(present - set(EXPECTED_WEEKS))]
    not_monday = [d.date().isoformat() for d in wk.index if d.weekday() != 0]
    reindexed = wk.reindex(EXPECTED_WEEKS)
    zero_k3 = [d.date().isoformat() for d, v in reindexed["edges_k3"].fillna(0).items() if v == 0]
    short = {d.date().isoformat(): int(v) for d, v in reindexed["days_with_edges"].items() if v < 7}
    return {"expected_weeks": len(EXPECTED_WEEKS), "present_weeks": len(present & set(EXPECTED_WEEKS)),
            "missing": missing, "unexpected": unexpected, "not_monday": not_monday,
            "zero_edges_at_k3": zero_k3, "weeks_with_fewer_than_7_days_of_edges": short}


# ---------------------------------------------------------------- part 2
def load_two_subreddits() -> pd.DataFrame:
    parts = []
    for f in sorted(EDGES_DIR.glob("edges_*.parquet")):
        t = pq.read_table(f, columns=["link_id", "source", "target", "week_start", "subreddit"],
                          filters=[("subreddit", "in", SUBREDDITS)])
        parts.append(t.group_by(["week_start", "link_id", "source", "target"]).aggregate([("subreddit", "count")]))
        print(f"  read {f.name}: {t.num_rows:,} edges", flush=True)
    df = pa.concat_tables(parts).to_pandas().rename(columns={"subreddit_count": "n"})
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.tz_localize(None)
    return df


def thread_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Thread size = reply edges a thread has in a week (a thread can span several weeks)."""
    return df.groupby(["week_start", "link_id"])["n"].sum().rename("size").reset_index()


def thread_summary(sizes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wk, g in sizes.groupby("week_start"):
        s = np.sort(g["size"].to_numpy())[::-1]
        top = max(1, int(np.ceil(0.01 * len(s))))
        rows.append({"week_start": wk.date(), "edges": int(s.sum()), "threads": len(s), "top1pct_threads": top,
                     "top1pct_edge_share": s[:top].sum() / s.sum(), "median_size": float(np.median(s)),
                     "p90_size": float(np.percentile(s, 90)), "p99_size": float(np.percentile(s, 99)),
                     "max_size": int(s[0]), "largest_thread_share": s[0] / s.sum(),
                     "threads_ge_100_edges": int((s >= 100).sum()),
                     "share_edges_in_threads_ge_100": s[s >= 100].sum() / s.sum()})
    return pd.DataFrame(rows)


def graph_from(df_edges: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_edges_from(zip(df_edges["source"], df_edges["target"]))
    return g


def graph_metrics(g: nx.DiGraph, ref_deg: np.ndarray | None = None) -> dict:
    n, m = g.number_of_nodes(), g.number_of_edges()
    deg = np.fromiter((d for _, d in g.degree()), int, n)
    lcc = max((len(c) for c in nx.weakly_connected_components(g)), default=0)
    out = {"nodes": n, "edges": m, "density": nx.density(g), "deg_mean": deg.mean(),
           "deg_median": float(np.median(deg)), "deg_p90": float(np.percentile(deg, 90)),
           "deg_p99": float(np.percentile(deg, 99)), "deg_max": int(deg.max()),
           "frac_deg1": float((deg == 1).mean()), "lcc_nodes": lcc, "lcc_fraction": lcc / n}
    if ref_deg is not None:  # two-sample KS distance between degree distributions
        grid = np.union1d(deg, ref_deg)
        a = np.searchsorted(np.sort(deg), grid, side="right") / len(deg)
        b = np.searchsorted(np.sort(ref_deg), grid, side="right") / len(ref_deg)
        out["deg_ks_vs_full"] = float(np.abs(a - b).max())
    return out


def sample_threads(week: pd.DataFrame, sizes: pd.Series, target: int, rng: np.random.Generator) -> pd.DataFrame:
    """Whole-thread sample, stratified by log2 thread size, edge budget per stratum proportional to its edge share.

    Threads are added in random order within a stratum while they fit the stratum's budget (a thread
    that would overshoot is skipped); any remaining deficit is filled from the unused threads that fit.
    Overshoot is never allowed, so the result is at most ``target`` edges.
    """
    stratum = np.floor(np.log2(sizes.to_numpy())).astype(int)
    total = sizes.sum()
    chosen, used = [], 0
    unused_all = []
    for s in np.unique(stratum):
        idx = np.flatnonzero(stratum == s)
        budget = target * sizes.iloc[idx].sum() / total
        rng.shuffle(idx)
        got = 0
        for i in idx:
            sz = sizes.iloc[i]
            if got + sz <= budget:
                chosen.append(i); got += sz
            else:
                unused_all.append(i)
        used += got
    rng.shuffle(unused_all)
    for i in unused_all:
        sz = sizes.iloc[i]
        if used + sz <= target:
            chosen.append(i); used += sz
    return week[week["link_id"].isin(set(sizes.index[chosen]))]


def sampling_comparison(week_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sizes = week_df.groupby("link_id")["n"].sum()
    edges = week_df.loc[week_df.index.repeat(week_df["n"]), ["link_id", "source", "target"]].reset_index(drop=True)
    full = graph_from(edges)
    ref_deg = np.fromiter((d for _, d in full.degree()), int, full.number_of_nodes())
    rows = [{"method": "full", "target": len(edges), "seed": 0, "reply_edges": len(edges), "threads": len(sizes),
             **graph_metrics(full, ref_deg)}]
    for target in SIZES:
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(SEED + seed)
            st = sample_threads(edges, sizes, target, rng)
            rows.append({"method": "thread_stratified", "target": target, "seed": seed, "reply_edges": len(st),
                         "threads": st["link_id"].nunique(), **graph_metrics(graph_from(st), ref_deg)})
            re = edges.iloc[rng.choice(len(edges), size=min(target, len(edges)), replace=False)]
            rows.append({"method": "random_edge", "target": target, "seed": seed, "reply_edges": len(re),
                         "threads": re["link_id"].nunique(), **graph_metrics(graph_from(re), ref_deg)})
        print(f"  sampled {target:,}", flush=True)
    return pd.DataFrame(rows), {"reply_edges": len(edges), "threads": len(sizes)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("part 1: weekly snapshots", flush=True)
    wk = weekly_k3()
    wk.reset_index(names="week_start").to_csv(OUT / "weekly_k3.csv", index=False)
    report = check_weeks(wk)
    (OUT / "weekly_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    print("part 2: threads", flush=True)
    df = load_two_subreddits()
    sizes = thread_sizes(df)
    summ = thread_summary(sizes)
    summ.to_csv(OUT / "thread_size_by_week.csv", index=False)
    sizes.to_csv(OUT / "thread_sizes_by_week.csv.gz", index=False)

    complete = summ[(summ["week_start"] > pd.Timestamp("2015-12-28").date()) &
                    (summ["week_start"] < pd.Timestamp("2016-12-26").date())]
    typical = complete.iloc[(complete["edges"] - complete["edges"].median()).abs().argsort().iloc[0]]
    week = pd.Timestamp(typical["week_start"])
    print(f"typical week: {week.date()} ({int(typical['edges']):,} edges)", flush=True)
    cmp_, meta = sampling_comparison(df[df["week_start"] == week].drop(columns="week_start"))
    cmp_.to_csv(OUT / "sampling_runs.csv", index=False)
    cols = ["reply_edges", "threads", "nodes", "edges", "density", "deg_mean", "deg_median", "deg_p90", "deg_p99",
            "deg_max", "frac_deg1", "lcc_fraction", "deg_ks_vs_full"]
    agg = cmp_.groupby(["method", "target"])[cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.reset_index().to_csv(OUT / "sampling_summary.csv", index=False)
    (OUT / "typical_week.json").write_text(json.dumps({"week_start": str(week.date()), **meta}, indent=2))
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())

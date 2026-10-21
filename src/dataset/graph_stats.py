"""Graph summaries used to calibrate synthetic graphs.

``compute_graph_stats`` works on any networkx graph, so real windows and synthetic
graphs are measured identically. The CLI computes it for every weekly and monthly
window of the reply graph and writes tables and plots to ``results/graph_stats/``.

Usage:
    python src/dataset/graph_stats.py
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

if __package__ in (None, ""):  # run as a script: make ``src`` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.dataset.windowing import WINDOW_COLUMNS, is_complete, iter_time_windowed_graphs  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EDGES_DIR = PROJECT_ROOT / "data" / "interim" / "edges"
RESULTS_DIR = PROJECT_ROOT / "results" / "graph_stats"
THRESHOLDS = (1, 2, 3, 5, 10, 20)
# Subreddits whose reply counts and shares are reported per window.
FOCUS_SUBREDDITS = ("politics", "PoliticalDiscussion", "NeutralPolitics", "Ask_Politics")


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {f"{prefix}_{k}": 0.0 for k in ("mean", "median", "p90", "p99", "max")}
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
        f"{prefix}_max": float(values.max()),
    }


def compute_graph_stats(graph: Any) -> dict[str, Any]:
    """Compute graph size, density, and degree-distribution summary statistics.

    Degrees count distinct neighbours (unweighted). ``interactions`` sums the ``weight``
    attribute (1 per edge if absent). The largest connected component is weakly connected
    for directed graphs.
    """
    n, m = graph.number_of_nodes(), graph.number_of_edges()
    stats: dict[str, Any] = {
        "nodes": n,
        "edges": m,
        "interactions": int(graph.size(weight="weight")) if m else 0,
        "density": nx.density(graph) if n > 1 else 0.0,
    }
    if graph.is_directed():
        stats["reciprocity"] = nx.reciprocity(graph) if m else 0.0
        stats.update(_summary(np.fromiter((d for _, d in graph.in_degree()), int, n), "in_degree"))
        stats.update(_summary(np.fromiter((d for _, d in graph.out_degree()), int, n), "out_degree"))
        components = nx.weakly_connected_components(graph)
    else:
        components = nx.connected_components(graph)
    stats.update(_summary(np.fromiter((d for _, d in graph.degree()), int, n), "degree"))
    lcc = max((len(c) for c in components), default=0)
    stats["lcc_nodes"] = lcc
    stats["lcc_fraction"] = lcc / n if n else 0.0
    return stats


def degree_counts(graph: nx.DiGraph) -> pd.DataFrame:
    """Long table of ``direction, degree, count`` (how many nodes have each degree)."""
    frames = []
    for direction, view in (("in", graph.in_degree()), ("out", graph.out_degree()), ("total", graph.degree())):
        counts = pd.Series([d for _, d in view]).value_counts().sort_index()
        frames.append(pd.DataFrame({"direction": direction, "degree": counts.index, "count": counts.values}))
    return pd.concat(frames, ignore_index=True)


def window_stats(edges: pa.Table, kind: str, data_start: datetime, data_end: datetime,
                 min_activity_threshold: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-window graph stats plus long-format degree counts for each window."""
    rows, degrees = [], []
    for gw in iter_time_windowed_graphs(edges, kind, min_activity_threshold):
        row = {"window_start": gw.window.start.date(), "complete": is_complete(gw.window, data_start, data_end),
               "replies": gw.metadata["edges"]}
        row.update(compute_graph_stats(gw.graph))
        rows.append(row)
        deg = degree_counts(gw.graph)
        deg.insert(0, "window_start", gw.window.start.date())
        degrees.append(deg)
        print(f"  {kind} {gw.window.start.date()}: {row['nodes']:,} nodes, {row['edges']:,} edges", flush=True)
    stats = pd.DataFrame(rows).merge(subreddit_counts(edges, kind), on="window_start", how="left")
    return stats, pd.concat(degrees, ignore_index=True)


def subreddit_counts(edges: pa.Table, kind: str, subreddits=FOCUS_SUBREDDITS) -> pd.DataFrame:
    """Per window: ``<subreddit>_replies`` (edge rows) and ``<subreddit>_share`` of all replies."""
    col = WINDOW_COLUMNS[kind]
    counts = edges.group_by([col, "subreddit"]).aggregate([("source", "count")]).to_pandas()
    wide = counts.pivot_table(index=col, columns="subreddit", values="source_count", aggfunc="sum", fill_value=0)
    total = wide.sum(axis=1)
    out = pd.DataFrame(index=wide.index)
    for sub in subreddits:
        n = wide[sub] if sub in wide.columns else 0
        out[f"{sub}_replies"] = n
        out[f"{sub}_share"] = n / total
    out = out.reset_index().rename(columns={col: "window_start"})
    out["window_start"] = pd.to_datetime(out["window_start"]).dt.date
    return out


def update_subreddit_counts(stats_csv: Path, edges: pa.Table, kind: str) -> pd.DataFrame:
    """Replace the per-subreddit columns of an existing stats CSV without recomputing graphs."""
    stats = pd.read_csv(stats_csv)
    stats = stats.drop(columns=[c for c in stats.columns if c.endswith(("_replies", "_share")) and c != "replies"])
    stats["window_start"] = pd.to_datetime(stats["window_start"]).dt.date
    stats = stats.merge(subreddit_counts(edges, kind), on="window_start", how="left")
    stats.to_csv(stats_csv, index=False)
    return stats


def threshold_sweep(edges: pa.Table, kind: str, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Nodes, distinct edges and replies kept per window at each minimum-activity threshold.

    Mirrors :func:`src.dataset.windowing.window_graph`: a node's activity is replies sent
    plus received in the window; an edge survives if both endpoints do.
    """
    col = WINDOW_COLUMNS[kind]
    pairs = edges.group_by([col, "source", "target"]).aggregate([("source", "count")]).to_pandas()
    pairs = pairs.rename(columns={"source_count": "w"})
    act = pd.concat([pairs[[col, "source", "w"]].rename(columns={"source": "node"}),
                     pairs[[col, "target", "w"]].rename(columns={"target": "node"})])
    act = act.groupby([col, "node"], observed=True)["w"].sum()
    out = []
    for k in thresholds:
        keep = act[act >= k].reset_index()[[col, "node"]]
        kept = pairs.merge(keep.rename(columns={"node": "source"}), on=[col, "source"])
        kept = kept.merge(keep.rename(columns={"node": "target"}), on=[col, "target"])
        nodes = pd.concat([kept[[col, "source"]].set_axis([col, "n"], axis=1),
                           kept[[col, "target"]].set_axis([col, "n"], axis=1)]).drop_duplicates()
        g = kept.groupby(col).agg(edges=("w", "size"), replies=("w", "sum"))
        g["nodes"] = nodes.groupby(col).size()
        g["threshold"] = k
        out.append(g.reset_index())
    res = pd.concat(out, ignore_index=True).set_index([col, "threshold"])
    full = pd.MultiIndex.from_product([sorted(pairs[col].unique()), list(thresholds)], names=[col, "threshold"])
    res = res.reindex(full, fill_value=0).reset_index().rename(columns={col: "window_start"})  # empty -> 0
    res["window_start"] = pd.to_datetime(res["window_start"]).dt.date
    return res[["window_start", "threshold", "nodes", "edges", "replies"]].astype(
        {"nodes": int, "edges": int, "replies": int})


def author_activity(edges: pa.Table) -> pd.DataFrame:
    """Distribution of replies sent per (author, week)."""
    per = edges.group_by(["source", "week_start"]).aggregate([("target", "count")])["target_count"]
    v = per.to_numpy()
    qs = [50, 90, 99, 99.9, 99.99]
    rows = [{"stat": f"p{q}", "value": float(np.percentile(v, q))} for q in qs]
    rows += [{"stat": "max", "value": float(v.max())}, {"stat": "author_weeks", "value": float(v.size)}]
    return pd.DataFrame(rows)


def make_plots(weekly: pd.DataFrame, monthly: pd.DataFrame, monthly_deg: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(ax, col, label, log=False):
        for df, name, style in ((weekly, "weekly", "-o"), (monthly, "monthly", "-s")):
            ax.plot(pd.to_datetime(df["window_start"]), df[col], style, ms=3, label=name)
        ax.set_title(label)
        if log:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    series(axes[0, 0], "nodes", "Nodes (users)")
    series(axes[0, 1], "edges", "Distinct directed edges")
    series(axes[1, 0], "lcc_fraction", "Largest weakly connected component (fraction of nodes)")
    series(axes[1, 1], "politics_share", "Share of replies in r/politics")
    axes[0, 0].legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "windows_over_time.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, direction in zip(axes, ("in", "out")):
        d = monthly_deg[monthly_deg.direction == direction]
        for month, g in d.groupby("window_start"):
            g = g.sort_values("degree")
            ccdf = g["count"][::-1].cumsum()[::-1] / g["count"].sum()
            ax.loglog(g["degree"], ccdf, lw=1, label=str(month)[:7])
        ax.set_title(f"{direction}-degree CCDF, monthly windows")
        ax.set_xlabel("degree (distinct neighbours)")
        ax.set_ylabel("P(degree >= x)")
        ax.grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "degree_ccdf_monthly.png", dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--edges-dir", type=Path, default=EDGES_DIR)
    ap.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--start", default="2016-01-01", help="first day covered by the data (UTC)")
    ap.add_argument("--end", default="2017-01-01", help="day after the last day covered (UTC)")
    ap.add_argument("--subreddit-counts-only", action="store_true",
                    help="only refresh the per-subreddit columns of existing weekly/monthly stats CSVs")
    args = ap.parse_args()
    data_start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    data_end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    t0 = time.perf_counter()
    files = sorted(args.edges_dir.glob("edges_*.parquet"))
    cols = ["source", "target", "subreddit", "week_start", "month_start"]
    edges = pa.concat_tables(pq.read_table(f, columns=cols) for f in files)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{edges.num_rows:,} edges from {len(files)} files", flush=True)
    if args.subreddit_counts_only:
        for kind in ("week", "month"):
            update_subreddit_counts(args.out_dir / f"{kind}ly_stats.csv", edges, kind)
        print(f"updated subreddit columns in {time.perf_counter() - t0:.0f}s")
        return 0

    outputs = {}
    for kind in ("week", "month"):
        stats, degrees = window_stats(edges, kind, data_start, data_end)
        stats.to_csv(args.out_dir / f"{kind}ly_stats.csv", index=False)
        degrees.to_csv(args.out_dir / f"{kind}ly_degree_counts.csv", index=False)
        threshold_sweep(edges, kind).to_csv(args.out_dir / f"{kind}ly_threshold_sweep.csv", index=False)
        outputs[kind] = (stats, degrees)
    author_activity(edges).to_csv(args.out_dir / "author_week_activity.csv", index=False)
    make_plots(outputs["week"][0], outputs["month"][0], outputs["month"][1], args.out_dir)
    print(f"wrote {args.out_dir} in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

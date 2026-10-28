"""Evaluate Pipeline B's camp recovery against the label sets.

    python scripts/camp_recovery_eval.py <pipeline_b_user_sides.csv> [--labels data/processed/camp_recovery/labels.parquet]

The input has (author, side) per user; ``side`` can be dem/rep, left/right, 0/1 or any two group names. A graph
split has no inherent direction, so accuracy is computed with the sides as given and swapped, and the better
one is reported (``swapped`` says which). For every label set (columns ``side_*`` of the labels file) the report
gives users compared, accuracy, adjusted Rand index (direction-free) and confusion counts.
This evaluates Pipeline B only; nothing the pipelines read may import or depend on the labels file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data" / "processed" / "camp_recovery" / "labels.parquet"
SIDE_ALIASES = {"dem": "dem", "left": "dem", "0": "dem", "d": "dem", "rep": "rep", "right": "rep", "1": "rep", "r": "rep"}


def adjusted_rand_index(a: pd.Series, b: pd.Series) -> float:
    table = pd.crosstab(a, b).to_numpy().astype(float)
    comb = lambda x: x * (x - 1) / 2  # noqa: E731
    sum_cells = comb(table).sum()
    sum_rows, sum_cols = comb(table.sum(axis=1)).sum(), comb(table.sum(axis=0)).sum()
    total = comb(table.sum())
    expected = sum_rows * sum_cols / total if total else 0.0
    maximum = (sum_rows + sum_cols) / 2
    return 1.0 if maximum == expected else float((sum_cells - expected) / (maximum - expected))


def evaluate(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per label set. ``pred`` has author, side; ``labels`` has author and side_<set> columns."""
    pred = pred.drop_duplicates("author").copy()
    pred["side"] = pred["side"].astype(str).str.lower().map(lambda s: SIDE_ALIASES.get(s, s))
    groups = sorted(pred["side"].unique())
    rows = []
    for col in [c for c in labels.columns if c.startswith("side_")]:
        m = labels[["author", col]].dropna().merge(pred, on="author")
        row = {"label_set": col.removeprefix("side_"), "users_labelled": int(labels[col].notna().sum()), "users_compared": len(m)}
        if len(m) == 0 or len(groups) != 2:
            rows.append({**row, "accuracy": np.nan, "swapped": False, "ari": np.nan, "tp_dem": 0, "fn_dem": 0, "tp_rep": 0, "fn_rep": 0,
                         "note": "no overlap" if len(m) == 0 else f"expected 2 predicted groups, got {len(groups)}"})
            continue
        as_given = m["side"].map({groups[0]: "dem", groups[1]: "rep"})
        swapped = m["side"].map({groups[0]: "rep", groups[1]: "dem"})
        acc, acc_sw = (as_given == m[col]).mean(), (swapped == m[col]).mean()
        use_swapped = acc_sw > acc
        best = swapped if use_swapped else as_given
        rows.append({**row, "accuracy": float(max(acc, acc_sw)), "swapped": bool(use_swapped),
                     "ari": adjusted_rand_index(m[col], m["side"]),
                     "tp_dem": int(((best == "dem") & (m[col] == "dem")).sum()), "fn_dem": int(((best == "rep") & (m[col] == "dem")).sum()),
                     "tp_rep": int(((best == "rep") & (m[col] == "rep")).sum()), "fn_rep": int(((best == "dem") & (m[col] == "rep")).sum()),
                     "note": ""})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sides_csv", type=Path)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = ap.parse_args()
    pred = pd.read_csv(args.sides_csv, usecols=["author", "side"], dtype=str)
    out = evaluate(pred, pd.read_parquet(args.labels))
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

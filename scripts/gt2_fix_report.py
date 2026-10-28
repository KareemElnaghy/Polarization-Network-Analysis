"""Tables for results/gt2_fix_options/summary.md from the stage jsonl files (stage names given on the command line)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gt2_fix_options"
GT = ROOT / "results" / "ground_truth"


def sp(x, y):
    d = pd.concat([x, y], axis=1).dropna()
    return float(spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0]) if len(d) >= 4 else np.nan


def series(d, variant, label_set, source):
    s = d[(d.variant == variant) & (d.label_set == label_set) & (d.source == source)]
    return s.set_index("week_start")


def table(d: pd.DataFrame, sources: list[str]) -> pd.DataFrame:
    gt1 = pd.read_parquet(GT / "gt1_weekly.parquet")
    gt1 = gt1[(gt1.venue == "shared") & (gt1.label_set == "main") & (gt1["sample"] == "full")].set_index("week_start")["excess"]
    rows = []
    for v in d.variant.unique():
        m, g = series(d, v, "main", "full"), series(d, v, "gold", "full")
        z = (m.pi - m.null_mean) / m.null_sd
        r = {"variant": v, "weeks": len(m), "real": m.pi.mean(), "null_mean": m.null_mean.mean(), "null_sd": m.null_sd.mean(),
             "boot_lo": m.boot_lo.mean(), "boot_hi": m.boot_hi.mean(), "frac_above_null": float((m.pi > m.null_mean).mean()),
             "frac_z>2": float((z > 2).mean()), "dem_users": m.n_dem.mean(), "rep_users": m.n_rep.mean(),
             "rho_gold": sp(m.pi, g.pi), "rho_gt1": sp(m.pi, gt1.reindex(m.index)), "sec_per_cell": m.seconds.mean()}
        for s in sources:
            x = series(d, v, "main", s)
            r[f"rho_full_vs_{s}"] = sp(m.pi, x.pi)
            r[f"real_{s}"] = x.pi.mean()
            r[f"null_{s}"] = x.null_mean.mean()
            r[f"users_{s}"] = (x.n_dem.mean() + x.n_rep.mean()) / 2
        rows.append(r)
    return pd.DataFrame(rows).set_index("variant")


def baseline() -> dict:
    f = pd.read_parquet(GT / "gt2_weekly_full.parquet")
    f = f[(f.venue == "shared") & (f.label_set == "main")].set_index("week_start")["pi"]
    s = pd.read_parquet(GT / "gt2_weekly_samples.parquet")
    s = s[(s.venue == "shared") & (s.label_set == "main")]
    out = {}
    for r, x in s.groupby("sample"):
        x = x.set_index("week_start")["pi"]
        d = pd.concat([f, x], axis=1).dropna()
        out[r] = (float(spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0]), float(pearsonr(d.iloc[:, 0], d.iloc[:, 1])[0]), len(d))
    return out


if __name__ == "__main__":
    stages = sys.argv[1].split(",")
    sources = sys.argv[2].split(",") if len(sys.argv) > 2 else []
    d = pd.concat([pd.read_json(OUT / f"{s}.jsonl", lines=True) for s in stages])
    t = table(d, sources)
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(t.round(3).T.to_string())
    print(baseline())


def md(df: pd.DataFrame, nd: int = 3) -> str:
    df = df.reset_index()
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(out)

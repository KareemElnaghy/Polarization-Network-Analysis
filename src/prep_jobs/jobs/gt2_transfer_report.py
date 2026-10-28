"""Tables, plots and summary.md for the gt2_transfer job."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.gt_runs import transfer as T  # noqa: E402
from src.prep_jobs.config import project_root  # noqa: E402

INSTR = ["phrase", "logit"]


def md(df: pd.DataFrame, nd: int = 3) -> str:
    df = df.reset_index()
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def rho(a: pd.Series, b: pd.Series) -> float:
    d = pd.concat([a, b], axis=1).dropna()
    return float(spearmanr(d.iloc[:, 0], d.iloc[:, 1])[0]) if len(d) >= 5 else np.nan


def series(res, inst, w, src="full", labels="main", col="value"):
    s = res[(res.instrument == inst) & (res.weighting == w) & (res.source == src) & (res.labels == labels)]
    return s.set_index("week_start")[col]


def make_report(res, sanity, users, g, out, log) -> None:
    root = project_root()
    gt1 = pd.read_parquet(root / "results" / "ground_truth" / "gt1_weekly.parquet")
    gt1 = gt1[(gt1.venue == "shared") & (gt1.label_set == "main") & (gt1["sample"] == "full")].set_index("week_start")["excess"]
    a_path = root / "results" / "gt2_fix_options" / "s3.jsonl"
    var_a = None
    if a_path.exists():
        d = pd.read_json(a_path, lines=True)
        var_a = d[(d.variant == "A_equal") & (d.label_set == "main") & (d.source == "full")].set_index("week_start")["pi"]
    old = pd.read_parquet(root / "results" / "ground_truth" / "gt2_weekly_full.parquet")
    old = old[(old.venue == "shared") & (old.label_set == "main")].set_index("week_start")["pi"]

    rows = []
    for inst in INSTR:
        for w in T.WEIGHTINGS:
            m = res[(res.instrument == inst) & (res.weighting == w) & (res.source == "full") & (res.labels == "main")].set_index("week_start")
            gsub = sanity[(sanity.instrument == inst) & (sanity.weighting == w)]
            sm, sg = gsub[gsub.labels == "main"], gsub[gsub.labels == "gold"]
            u = users[(users.side_gold >= 0)]
            sc = u[f"{inst}_{w}"].to_numpy()
            yg = u["side_gold"].to_numpy().astype(int)
            thr = 0.5 if inst == "logit" else np.nan
            if inst == "phrase":
                thr = float(np.mean([np.load(root / "runs" / "gt2_transfer" / "work" / f"model_{k}.npz")[f"meta_{w}"][2] for k in range(g["gt2_transfer"]["n_folds"])]))
            z = (m["value"] - m["null_mean"]) / m["null_sd"]
            r = {"instrument": inst, "weighting": w,
                 "supporter acc (main, held-out)": float(np.average(sm["balanced_accuracy"], weights=sm["n"])),
                 "supporter AUC (main)": float(np.average(sm["auc"], weights=sm["n"])),
                 "supporter acc (gold)": float(np.average(sg["balanced_accuracy"], weights=sg["n"])),
                 "supporter AUC (gold)": float(np.average(sg["auc"], weights=sg["n"])),
                 "neutral AUC (gold, pooled)": T.auc_value(sc, yg),
                 "neutral acc (gold, pooled)": 0.5 * (float((sc[yg == 1] > thr).mean()) + float((sc[yg == 0] <= thr).mean())),
                 "weeks usable": int(m["value"].notna().sum()), "weeks above null+2sd": int((z > 2).sum()),
                 "weeks above band lo": int((m["value"] > m["null_mean"] + 2 * m["null_sd"]).sum()),
                 "mean value": float(m["value"].mean()), "mean null": float(m["null_mean"].mean()), "mean null sd": float(m["null_sd"].mean()),
                 "mean band width": float((m["boot_hi"] - m["boot_lo"]).mean())}
            for src in ("r0.2", "r0.05", "r0.01"):
                r[f"rho full vs {src}"] = rho(m["value"], series(res, inst, w, src))
                r[f"weeks usable {src}"] = int(series(res, inst, w, src).notna().sum())
            r["rho vs variant A"] = rho(m["value"], var_a) if var_a is not None else np.nan
            r["rho vs GT#1"] = rho(m["value"], gt1)
            r["rho vs old GT#2"] = rho(m["value"], old)
            r["rho vs gold-label series"] = rho(m["value"], series(res, inst, w, labels="gold"))
            r["users per side / week (dem, rep)"] = f"{m['n_dem'].mean():.0f}, {m['n_rep'].mean():.0f}"
            r["mean phrases / user"] = float(m["mean_phrases"].mean())
            rows.append(r)
    tab = pd.DataFrame(rows).set_index(["instrument", "weighting"])
    cross = pd.DataFrame({w: {"rho phrase vs logit": rho(series(res, "phrase", w), series(res, "logit", w))} for w in T.WEIGHTINGS}).T

    parts = {"a. Held-out accuracy (balanced) and AUC": ["supporter acc (main, held-out)", "supporter AUC (main)", "supporter acc (gold)", "supporter AUC (gold)",
                                                         "neutral AUC (gold, pooled)", "neutral acc (gold, pooled)"],
             "b. Real vs shuffled null, full week": ["weeks usable", "weeks above null+2sd", "mean value", "mean null", "mean null sd", "mean band width"],
             "c. Full week vs samples (Spearman across weeks)": ["rho full vs r0.2", "rho full vs r0.05", "rho full vs r0.01", "weeks usable r0.2", "weeks usable r0.05", "weeks usable r0.01"],
             "d. Agreement with other measures (Spearman across weeks)": ["rho vs variant A", "rho vs GT#1", "rho vs old GT#2", "rho vs gold-label series"],
             "e. Coverage": ["users per side / week (dem, rep)", "mean phrases / user"]}
    for name, cols in parts.items():
        pass

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for i, inst in enumerate(INSTR):
        for w in T.WEIGHTINGS:
            s = series(res, inst, w)
            ax[i].plot(pd.to_datetime(s.index), s, label=w)
        ax[i].plot(pd.to_datetime(series(res, inst, "equal", col="null_mean").index), series(res, inst, "equal", col="null_mean"), "k:", label="shuffled null")
        ax[i].set_title("phrase instrument: weekly pi" if inst == "phrase" else "logistic instrument: weekly AUC"); ax[i].legend()
    fig.tight_layout(); fig.savefig(out / "series.png", dpi=110); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, inst in enumerate(INSTR):
        for w in T.WEIGHTINGS:
            full, s20 = series(res, inst, w), series(res, inst, w, "r0.2")
            ax[i].scatter(full, s20, s=10, label=w)
        ax[i].set_xlabel("full week"); ax[i].set_ylabel("20% sample"); ax[i].set_title(inst); ax[i].legend()
    fig.tight_layout(); fig.savefig(out / "full_vs_sample.png", dpi=110); plt.close(fig)

    txt = ["# GT#2 by transfer (train in supporter subs, score in neutral subs)", "",
           "Training: comments in the party-tagged supporter subs by main-labelled users (all weeks, heavy posters excluded, same phrase vocabulary and",
           "contraction filter, phrases that occur in the neutral subs). Users are split at random into 5 piles (fixed seed); a user is only ever scored by the",
           "model trained on the other four piles. Neutral subs = r/politics + r/PoliticalDiscussion, main labels, users with at least 20 phrases that week.",
           "Instruments: `phrase` = frozen phrase leanings, weekly pi (null 0.5); `logit` = regularised logistic regression, weekly AUC (null 0.5).",
           "Weightings: `cap100`, `cap300`, `equal`. Null: labels shuffled among the scored users (100 shuffles); bands: 100 bootstrap resamples of users.",
           "Plots: `series.png`, `full_vs_sample.png`. User scores (full week): `user_scores.parquet`. Weekly results with bands and all sources: `weekly_results.parquet`.", ""]
    for name, cols in parts.items():
        txt += [f"## {name}", md(tab[cols]), ""]
    txt += ["## Instruments against each other (Spearman across weeks, same weighting)", md(cross), ""]
    (out / "summary.md").write_text("\n".join(txt), encoding="utf-8")
    tab.to_csv(out / "tables.csv")
    log(f"summary written to {out / 'summary.md'}")

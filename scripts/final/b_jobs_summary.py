"""Summarise runs/final_b_jobs: P1.6 noise-injection curve (mean +- sd over finished seeds) and P1.7 thinning, rolling 4-week / 2-week windows, calendar months.
Window GT = mean of the weekly GT values inside the window (an approximation, stated). Exploratory/secondary. Reads labels (evaluation script).
usage: python scripts/final/b_jobs_summary.py [--dir runs/final_b_jobs]"""
import argparse, glob, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from src.algorithms.eval import camp_recovery as cr, metrics as M
ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="runs/final_b_jobs"); a = ap.parse_args()
d = Path(a.dir)
GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start"); GT.index = GT.index.astype(str)
weeks = sorted(GT.index); TR = ["gt1_excess", "gt2"]; labels = cr.load_labels()

def camp_ba(f):
    c = pd.read_parquet(f).rename(columns={"group": "week"}); c["sign_source"] = "x"
    return float(cr.evaluate(c, labels).balanced_acc.mean()) if len(c) else np.nan

def series(f):
    return pd.read_csv(f).set_index("group").P_B.reindex(weeks)

rows = []
for f in sorted(glob.glob(str(d / "*.csv"))):
    name = Path(f).stem
    m = re.match(r"(noise|thin)_[pf]([\d.]+)_s(\d+)", name)
    if not m: continue
    s = series(f); r = pd.read_csv(f)
    rows.append({"job": name, "family": m[1], "level": float(m[2]), "seed": int(m[3]), "mean_P_B": s.mean(), "weeks_scored": int(s.notna().sum()),
                 **{f"rho_{t}": M.rho(s, GT[t]) for t in TR}, "camp_BA": camp_ba(f.replace(".csv", ".camps.parquet")),
                 "mean_degree_giant": r.mean_degree_giant.mean(), "mean_giant_nodes": r.n_giant.mean()})
J = pd.DataFrame(rows); out = Path("results/llm/final/rq3_rq2_jobs"); out.mkdir(parents=True, exist_ok=True)
if len(J):
    J.to_csv(out / "jobs_by_seed.csv", index=False)
    agg = J.groupby(["family", "level"]).agg(n_seeds=("seed", "nunique"), **{f"{c}_mean": (c, "mean") for c in ("mean_P_B", "rho_gt1_excess", "rho_gt2", "camp_BA", "mean_degree_giant")},
                                            **{f"{c}_sd": (c, "std") for c in ("mean_P_B", "rho_gt1_excess", "rho_gt2", "camp_BA")}).reset_index()
    agg.to_csv(out / "noise_and_thinning_summary.csv", index=False)
    print(agg.round(3).to_string(index=False))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        for fam in ("noise", "thin"):
            g = agg[agg.family == fam]
            if len(g) < 2: continue
            fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
            for k, c in enumerate(("mean_P_B", "rho_gt1_excess", "rho_gt2", "camp_BA")):
                ax[k].errorbar(g.level, g[f"{c}_mean"], yerr=g[f"{c}_sd"].fillna(0), marker="o"); ax[k].set(xlabel="flip share p" if fam == "noise" else "thread fraction kept", ylabel=c)
            plt.tight_layout(); plt.savefig(out / f"{fam}_curve.png", dpi=150); plt.close()
    except Exception as e:
        print("figure skipped:", e)

res = {}
for name, k in (("win4_s0", 4), ("win2_s0", 2), ("month_s0", None)):
    f = d / f"{name}.csv"
    if not f.exists(): continue
    r = pd.read_csv(f).set_index("group")
    if k:
        idx = [weeks[i - k + 1: i + 1] for i in range(k - 1, len(weeks))]
        g = {t: pd.Series([GT[t].loc[w].mean() for w in idx], index=r.index.astype(str)) for t in TR}
        r.index = r.index.astype(str); x = r.P_B
    else:
        mo = pd.to_datetime(pd.Series(weeks)).dt.to_period("M").astype(str)
        g = {t: GT[t].groupby(mo.values).mean().reindex(r.index.astype(str)) for t in TR}; x = r.P_B.set_axis(r.index.astype(str))
    ent = {"n_windows": int(x.notna().sum())}
    for t in TR:
        ent[t] = {"rho": M.rho(x, g[t]), "perm_p": M.circ_p(x, g[t]) if len(x) >= 20 else None}
        if k:
            stat = lambda u, v: M.rho(pd.Series(u), pd.Series(v)); mm = (x.notna() & g[t].notna()).to_numpy()
            pt, lo, hi, _ = M.block_boot(stat, [x.to_numpy()[mm], g[t].to_numpy()[mm]], boot=2000, block=max(4, k))
            ent[t] |= {"ci": [lo, hi]}
    res[name] = ent
(out / "windows_summary.json").write_text(json.dumps(res, indent=1, default=float)); print(json.dumps(res, indent=1, default=float))

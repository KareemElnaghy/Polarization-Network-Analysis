"""P1.8 / RQ4 (D-E6 section G, confirmatory-descriptive): rolling-origin one-step forecasts of weekly change in GT#1 and GT#2.
Models: persistence (dGT=0), A (dGT_t ~ dP_A_{t-1}), B, A+B; OLS refit each origin, targets t = 24..53; MAE; Diebold-Mariano vs persistence.
usage: python scripts/final/rq4_forecast.py RUN_A RUN_B [--out results/llm/final/rq4]"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.algorithms.analysis import forecast as fc
from src.algorithms.llm import provenance

ap = argparse.ArgumentParser()
ap.add_argument("run_a")
ap.add_argument("run_b")
ap.add_argument("--out", default="results/llm/final/rq4")
ap.add_argument("--sign-source", default="continuous_llm")
a = ap.parse_args()
run = provenance.new_run("final_rq4_forecast", {"first_target": 24, "sign_source": a.sign_source},
                         [str(Path(a.run_a) / "weekly_A.csv"), str(Path(a.run_b) / "weekly_B.csv")], sample_rate=0.05, run_a=a.run_a, run_b=a.run_b)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start")
GT.index = GT.index.astype(str)
weeks = sorted(GT.index)
A = pd.read_csv(Path(a.run_a) / "weekly_A.csv").set_index("week").reindex(weeks).P_A.to_numpy()
B = pd.read_csv(Path(a.run_b) / "weekly_B.csv")
B = B[B.sign_source == a.sign_source].set_index("week").reindex(weeks).P_B.to_numpy()
assert np.isfinite(A).all() and np.isfinite(B).all() and len(weeks) == 53

res, preds = [], {}
for truth in ("gt1_excess", "gt2"):
    g = GT.loc[weeks, truth].to_numpy()
    f = fc.rolling_forecasts(g, {"A": A, "B": B}, first_target=24)
    preds[truth] = f
    f.assign(week=[weeks[t - 1] for t in f.t]).to_csv(out / f"forecasts_{truth}.csv", index=False)
    base_err = f.actual - f.persistence
    for m in ["persistence", "A", "B", "A+B"]:
        e = f.actual - f[m]
        r = {"truth": truth, "model": m, "n_forecasts": len(f), "MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean()))}
        if m != "persistence":
            r |= {f"DM_{k}": v for k, v in fc.diebold_mariano(e.to_numpy(), base_err.to_numpy()).items()}
        res.append(r)
tab = pd.DataFrame(res)
tab["rel_MAE_vs_persistence"] = tab.MAE / tab.groupby("truth").MAE.transform("first")
tab.to_csv(out / "rq4_mae_dm.csv", index=False)

best = {}
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
for i, truth in enumerate(("gt1_excess", "gt2")):
    t = tab[(tab.truth == truth) & (tab.model != "persistence")].sort_values("MAE").iloc[0]
    best[truth] = t.model
    f = preds[truth]
    ax[i].scatter(f.actual, f[t.model], s=14, color="tab:blue")
    lim = float(max(np.abs(f.actual).max(), np.nanmax(np.abs(f[t.model]))))
    ax[i].plot([-lim, lim], [-lim, lim], color="grey", lw=1)
    ax[i].set(xlabel=f"actual dGT ({truth})", ylabel=f"predicted ({t.model})", title=f"best model: {t.model}, MAE {t.MAE:.3f} (persistence {tab[(tab.truth == truth) & (tab.model == 'persistence')].MAE.iloc[0]:.3f})")
fig.tight_layout()
fig.savefig(out / "rq4_pred_vs_actual.png", dpi=150)

beats = tab[(tab.model != "persistence") & (tab.MAE < tab.groupby("truth").MAE.transform("first")) & (tab.DM_p < 0.05)]
verdict = ("A model beats persistence at p < 0.05 (two-sided DM): " + ", ".join(f"{r.model}/{r.truth}" for r in beats.itertuples())) if len(beats) else \
    "No model beats persistence at p < 0.05 (two-sided DM, HAC variance). Pre-registered expectation: null; reported as a finding."
summary = {"label": "RQ4 per D-E6 section G; expectation stated in advance: no model beats persistence", "n_forecasts": 30, "targets": "t=24..53 (1-indexed weeks)",
           "verdict": verdict, "best_model_by_MAE": best, "table": tab.round(4).to_dict("records")}
(out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
for fl in out.iterdir():
    (run / fl.name).write_bytes(fl.read_bytes())
provenance.finish_run(run, "done", {"verdict": verdict[:80]}, purpose="P1.8 RQ4 rolling-origin forecasting")
pd.set_option("display.width", 220)
print(tab.round(4).to_string(index=False)); print(verdict)

"""P1.3 / H-AL (confirmatory: continuous_llm delta-w; everything else secondary/diagnostic) on the full-5% edges.
usage: python scripts/final/alignment_test.py RUN_A [--out results/llm/final/alignment]"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from src.algorithms.analysis import alignment as al
from src.algorithms.eval import metrics as M
from src.algorithms.llm import provenance

ap = argparse.ArgumentParser()
ap.add_argument("run_a")
ap.add_argument("--out", default="results/llm/final/alignment")
ap.add_argument("--boot", type=int, default=1000)
a = ap.parse_args()
run = provenance.new_run("final_alignment", {"seed": 0, "boot": a.boot, "n_perm": 200}, [str(Path(a.run_a) / "edge_scores.parquet")],
                         sample_rate=0.05, run_a=a.run_a)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

E = al.build_edge_table(a.run_a)
lb = E[E.lb].reset_index(drop=True)
GT = pd.read_csv("results/ground_truth/ground_truth_weekly.csv").set_index("week_start")
GT.index = GT.index.astype(str)

SRC = {"continuous_llm": "w_continuous_llm", "binary_llm": "w_binary_llm", "sentiment": "w_sentiment"}
OUT = {"p_disagree": "p_disagree", "disagree_argmax": "disagree_argmax", "host_gt_none": "host_gt_none", "attack": "attack"}
rows = []
for name, col in SRC.items():
    rows.append({"measure": f"w[{name}]", "sign_source": name, **al.cluster_boot_diff(lb[col].to_numpy(), lb.cross.to_numpy(), lb.thread.to_numpy(), a.boot)})
for name, col in OUT.items():
    rows.append({"measure": name, "sign_source": "llm_attribute", **al.cluster_boot_diff(lb[col].to_numpy(), lb.cross.to_numpy(), lb.thread.to_numpy(), a.boot)})
eff = pd.DataFrame(rows).assign(excludes_zero=lambda d: (d.ci_lo > 0) | (d.ci_hi < 0))
lo = al.logit_odds(lb)
eff.to_csv(out / "alignment_edges_summary.csv", index=False)

wk, serie = [], []
for name, col in SRC.items():
    w = al.weekly_antagonism_alignment(lb, col).assign(sign_source=name)
    wk.append(w)
    for m in ("antagonism", "alignment_excess"):
        for t in ("gt1_excess", "gt2"):
            s = w.set_index("week")[m].reindex(sorted(GT.index))
            serie.append({"sign_source": name, "series": m, "truth": t, "rho": M.rho(s, GT[t]), "perm_p": M.circ_p(s, GT[t]), "mean": float(s.mean())})
pd.concat(wk).to_csv(out / "alignment_weekly.csv", index=False)
corr = pd.DataFrame(serie)
corr.to_csv(out / "alignment_weekly_gt_correlations_DIAGNOSTIC.csv", index=False)

h = eff[eff.sign_source == "continuous_llm"].iloc[0]
verdict = ("H-AL SUPPORTED: continuous_llm weight is lower on cross-camp than same-camp edges, CI excludes 0" if h.delta < 0 and h.ci_hi < 0 else
           "H-AL NOT SUPPORTED: CI for delta-w (cross - same) includes 0" if h.ci_lo <= 0 <= h.ci_hi else
           "H-AL CONTRADICTED: continuous_llm weight is HIGHER on cross-camp edges, CI excludes 0")
summary = {"label": "H-AL CONFIRMATORY (continuous_llm delta-w); all other rows secondary/diagnostic", "run_a": a.run_a,
           "n_edges": int(len(E)), "n_lb_edges": int(len(lb)), "cross_share": float(lb.cross.mean()), "threads": int(lb.thread.nunique()),
           "verdict": verdict, "primary": h.to_dict(), "logit_disagree_on_cross_week_FE": lo,
           "delta_w_by_sign_source": {r.sign_source: round(float(r.delta), 4) for r in eff[eff.measure.str.startswith("w[")].itertuples()}}
(out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
for f in out.iterdir():
    (run / f.name).write_bytes(f.read_bytes())
provenance.finish_run(run, "done", {"delta_w_continuous": round(float(h.delta), 4), "ci": [round(float(h.ci_lo), 4), round(float(h.ci_hi), 4)]},
                      purpose="P1.3 edge-level alignment (H-AL)")
pd.set_option("display.width", 220)
print(eff.round(4).to_string(index=False)); print(lo); print(corr.round(3).to_string(index=False)); print(verdict)

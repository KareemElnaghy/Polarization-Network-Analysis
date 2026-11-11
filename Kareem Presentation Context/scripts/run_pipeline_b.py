"""Pipeline B from stored extractions (no LLM calls). usage: python scripts/llm/run_pipeline_b.py EXTRACTION_RUN_DIR [--name pipeB]
Weekly undirected signed graph per sign source -> giant component -> normalised frustration P_B = 1 - F_obs/F_null (sign-shuffle null),
plus the rewiring null as a secondary check. Writes weekly_B.csv/parquet, camp_assignments.parquet, plot (png/pdf/csv)."""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from concurrent.futures import ProcessPoolExecutor

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
from src.algorithms.llm import provenance
from src.algorithms.llm.data import load_shared, load_users
from src.algorithms.llm.store import Store
from src.algorithms.pipeline_b import sign_inference as si
from src.algorithms.pipeline_b import structural as st


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("extraction_run")
    ap.add_argument("--name", default="pipeB")
    ap.add_argument("--sources", default=None)
    ap.add_argument("--min-edges", type=int, default=None, help="plumbing checks only")
    ap.add_argument("--min-giant", type=int, default=None)
    a = ap.parse_args()
    bc = yaml.safe_load(open("configs/llm/pipeline_b.yaml"))
    if a.min_edges:
        bc["min_edges"] = a.min_edges
    if a.min_giant:
        bc["min_giant_nodes"] = a.min_giant
    src_man = json.load(open(Path(a.extraction_run) / "manifest.json"))
    cfg, rate = src_man["config"], src_man["sample_rate"]
    sel = pd.read_parquet(Path(a.extraction_run) / "selected_edges.parquet")
    store = Store(cfg["store"])
    quant = store.db.execute("SELECT DISTINCT quant FROM extractions WHERE model=?", (cfg["model"],)).fetchone()[0]
    ctx = {"model": cfg["model"], "quant": quant, "prompt_version": cfg["prompt_version"], "temperature": cfg["temperature"], "seed": cfg["seed"]}
    ext = store.table(ctx)[["pair_hash", "p_agree", "p_neutral", "p_disagree", "agreement"]]
    df = sel.merge(ext, left_on="h", right_on="pair_hash", how="inner")
    df = df.merge(load_users(rate), on="comment_id").merge(load_shared(rate)[["comment_id", "body"]], on="comment_id")
    sources = (a.sources or ",".join(bc["sign_sources"])).split(",")
    print(f"{len(df)} extracted edges of {len(sel)}; sources {sources}")

    run = provenance.new_run(a.name, {**cfg, "pipeline_b": bc}, [str(Path(a.extraction_run) / "selected_edges.parquet")], sample_rate=rate,
                             source_run=src_man["run_id"])
    EX = ProcessPoolExecutor(bc.get("workers", 1)) if bc.get("workers", 1) > 1 else None
    rows, camps = [], []
    t0 = time.time()
    for src in sources:
        w = si.SOURCES[src](df) if src != "binary_llm" else si.binary_llm(df, bc["neutral_policy"])
        d = df.assign(w=w)
        R = bc["R_null"].get(src, bc["R_null"]["default"])
        Rr = bc["R_rewire"].get(src, bc["R_rewire"]["default"])
        for wk, g in d.groupby("week"):
            row = {"week": wk, "sign_source": src, "n_edges_in": len(g)}
            full = st.build_graph(g, bc["pair_aggregation"])
            row |= {"n_pairs": len(full["w"]), "n_nodes_full": len(full["nodes"])}
            if len(full["w"]) < bc["min_edges"]:
                rows.append(row | {"nan_reason": f"fewer than {bc['min_edges']} user pairs"})
                continue
            gi = st.giant(full)
            if len(gi["nodes"]) < bc["min_giant_nodes"]:
                rows.append(row | {"nan_reason": f"giant component under {bc['min_giant_nodes']} nodes"})
                continue
            res = st.normalised_frustration(gi, R, bc["restarts"], bc["seed"], Rr, EX)
            full_res = st.normalised_frustration(full, max(R // 2, 5), bc["restarts"], bc["seed"], 0, EX)
            side = res.pop("side")
            rows.append(row | {"n_nodes_giant": len(gi["nodes"]), "n_pairs_giant": len(gi["w"]), "frac_negative": float((gi["w"] < 0).mean()),
                               **res, "P_B_full_graph": full_res["P_B"], "F_obs_full_graph": full_res["F_obs"], "nan_reason": ""})
            camps.append(pd.DataFrame({"week": wk, "sign_source": src, "user": gi["nodes"], "side": side.astype(int)}))
        print(src, "done", round(time.time() - t0, 1), "s", flush=True)
    weekly = pd.DataFrame(rows)
    weekly.to_parquet(run / "weekly_B.parquet"); weekly.to_csv(run / "weekly_B.csv", index=False)
    (pd.concat(camps) if camps else pd.DataFrame(columns=["week", "sign_source", "user", "side"])).to_parquet(run / "camp_assignments.parquet")
    weekly.to_csv(run / "pipelineB_weekly.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for src, g in weekly.groupby("sign_source"):
        ax.plot(pd.to_datetime(g.week), g.P_B, marker="o", ms=3, label=src)
    ax.axvline(pd.Timestamp("2016-06-27"), ls="--", color="gray", lw=1)
    ax.set_ylabel("P_B = 1 - F_obs / F_null"); ax.legend(frameon=False, ncol=4)
    ax.set_title(f"Pipeline B, weekly, giant component ({cfg['model']}, {len(df)} edges)")
    fig.tight_layout(); fig.savefig(run / "pipelineB_weekly.png", dpi=150); fig.savefig(run / "pipelineB_weekly.pdf")
    head = {"edges": int(len(df)), "weeks_scored": int(weekly.P_B.notna().sum()), "weeks_nan": int(weekly.P_B.isna().sum()),
            **{f"P_B_mean_{s}": float(g.P_B.mean()) for s, g in weekly.groupby("sign_source")}, "seconds": time.time() - t0}
    provenance.finish_run(run, "done", head, "Pipeline B weekly scores")
    print(run, head)


if __name__ == "__main__":
    main()

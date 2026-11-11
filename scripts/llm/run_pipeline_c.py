"""Pipeline C from stored extractions (no LLM calls): B's graph / giant component / normalised frustration / nulls UNCHANGED, with the edge
weight replaced by a pre-specified richer one (src/algorithms/pipeline_c/weights.py). Same inputs, graph rules and seeds as run_pipeline_b.py.
Resumable (per-week checkpoints in the run folder), stops cleanly when <run>/STOP exists, --weeks restricts to a subset (sharding).
usage: python scripts/llm/run_pipeline_c.py EXTRACTION_RUN_DIR [--variant C1|C1bin] [--name pipeC1] [--weeks 2016-01-04,...] [--resume RUN_DIR]"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, ".")
from src.algorithms.llm import provenance
from src.algorithms.llm.data import load_users
from src.algorithms.llm.store import Store
from src.algorithms.pipeline_b import structural as st
from src.algorithms.pipeline_c import weights as wc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extraction_run")
    ap.add_argument("--variant", default="C1", choices=list(wc.VARIANTS))
    ap.add_argument("--name", default=None)
    ap.add_argument("--weeks", default=None, help="comma-separated week_start values (default all)")
    ap.add_argument("--resume", default=None, help="existing pipeC run folder to continue")
    ap.add_argument("--min-edges", type=int, default=None, help="plumbing checks only")
    ap.add_argument("--min-giant", type=int, default=None)
    ap.add_argument("--R", type=int, default=None, help="override null draws (plumbing checks only)")
    a = ap.parse_args()
    bc = yaml.safe_load(open("configs/llm/pipeline_b.yaml"))
    for k, v in (("min_edges", a.min_edges), ("min_giant_nodes", a.min_giant)):
        if v:
            bc[k] = v
    src_man = json.load(open(Path(a.extraction_run) / "manifest.json"))
    cfg, rate = src_man["config"], src_man["sample_rate"]
    sel = pd.read_parquet(Path(a.extraction_run) / "selected_edges.parquet")
    store = Store(cfg["store"])
    quant = store.db.execute("SELECT DISTINCT quant FROM extractions WHERE model=?", (cfg["model"],)).fetchone()[0]
    ctx = {"model": cfg["model"], "quant": quant, "prompt_version": cfg["prompt_version"], "temperature": cfg["temperature"], "seed": cfg["seed"]}
    ext = store.table(ctx)[["pair_hash", "p_agree", "p_neutral", "p_disagree", "agreement", "emotional_charge", "hostility", "personal_attack"]]
    df = sel.merge(ext, left_on="h", right_on="pair_hash", how="inner").merge(load_users(rate), on="comment_id")
    weeks = sorted(df.week.unique()) if not a.weeks else a.weeks.split(",")
    label = a.variant
    print(f"{len(df)} extracted edges of {len(sel)}; variant {label}; {len(weeks)} weeks")

    if a.resume:
        run = Path(a.resume)
    else:
        run = provenance.new_run(a.name or f"pipe{label}", {**cfg, "pipeline_b": bc, "pipeline_c_variant": label},
                                 [str(Path(a.extraction_run) / "selected_edges.parquet")], sample_rate=rate, source_run=src_man["run_id"],
                                 weeks_requested=weeks)
    pw, pc = run / "weekly_C.partial.csv", run / "camps.partial.parquet"
    rows = pd.read_csv(pw).to_dict("records") if pw.exists() else []
    camps = [pd.read_parquet(pc)] if pc.exists() else []
    done = {r["week"] for r in rows}

    R = a.R or bc["R_null"].get("continuous_llm", bc["R_null"]["default"])
    Rr = 0 if a.R else bc["R_rewire"].get("continuous_llm", bc["R_rewire"]["default"])
    EX = ProcessPoolExecutor(bc.get("workers", 1)) if bc.get("workers", 1) > 1 else None
    d = df.assign(w=wc.VARIANTS[label](df))
    t0, stopped = time.time(), False
    for wk in weeks:
        if wk in done:
            continue
        if (run / "STOP").exists():
            stopped = True
            break
        g = d[d.week == wk]
        row = {"week": wk, "sign_source": label, "n_edges_in": len(g)}
        full = st.build_graph(g, bc["pair_aggregation"])
        row |= {"n_pairs": len(full["w"]), "n_nodes_full": len(full["nodes"])}
        if len(full["w"]) < bc["min_edges"]:
            rows.append(row | {"nan_reason": f"fewer than {bc['min_edges']} user pairs"})
        else:
            gi = st.giant(full)
            if len(gi["nodes"]) < bc["min_giant_nodes"]:
                rows.append(row | {"nan_reason": f"giant component under {bc['min_giant_nodes']} nodes"})
            else:
                res = st.normalised_frustration(gi, R, bc["restarts"], bc["seed"], Rr, EX)
                full_res = st.normalised_frustration(full, max(R // 2, 5), bc["restarts"], bc["seed"], 0, EX)
                side = res.pop("side")
                rows.append(row | {"n_nodes_giant": len(gi["nodes"]), "n_pairs_giant": len(gi["w"]), "frac_negative": float((gi["w"] < 0).mean()),
                                   **res, "P_B_full_graph": full_res["P_B"], "F_obs_full_graph": full_res["F_obs"], "nan_reason": ""})
                camps.append(pd.DataFrame({"week": wk, "sign_source": label, "user": gi["nodes"], "side": side.astype(int)}))
        pd.DataFrame(rows).to_csv(pw, index=False)
        if camps:
            pd.concat(camps).to_parquet(pc)
        print(f"{wk} done ({len(rows)}/{len(weeks)}) {time.time() - t0:.0f}s", flush=True)

    weekly = pd.DataFrame(rows)
    status = "partial" if stopped or set(weeks) - set(weekly.week) else "done"
    weekly.to_csv(run / "weekly_C.csv", index=False)
    weekly.to_parquet(run / "weekly_C.parquet")
    (pd.concat(camps) if camps else pd.DataFrame(columns=["week", "sign_source", "user", "side"])).to_parquet(run / "camp_assignments.parquet")
    for f in (pw, pc):
        f.unlink(missing_ok=True)
    head = {"variant": label, "edges": int(len(g) if False else len(d)), "weeks_scored": int(weekly.P_B.notna().sum()), "weeks_nan": int(weekly.P_B.isna().sum()),
            f"P_mean_{label}": float(weekly.P_B.mean()), "seconds": time.time() - t0}
    provenance.finish_run(run, status, head, f"Pipeline {label} weekly scores")
    print(run, status, head)


if __name__ == "__main__":
    main()

"""G1.5 (SPEC_ENCODER sections 5-6): select the encoder on VAL only, then run the G-ENC acceptance checks once on the selected model.
Selection: higher val agreement-argmax accuracy vs the LLM; under 0.5 pp apart -> the faster one (pairs/s on this GPU). Test, pilot and GT are not read before
the choice is fixed and logged. Checks: (1) test agreement acc >= 0.85, (2) test Guo-weight Spearman >= 0.80, (3) pilot 3-class acc vs consensus >= 0.65,
(4) weekly P_A encoder vs LLM on the 5% tier, Spearman >= 0.90, (5) throughput >= 200 pairs/s. Also writes the 5%-tier encoder outputs into
data/llm/encoder_outputs.sqlite (store schema). No GT is read here.
usage: python scripts/final/encoder_gate.py RUN_E1 [RUN_E2 ...]"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from src.algorithms.encoder import model as em
from src.algorithms.encoder import outputs as eo
from src.algorithms.llm import provenance
from src.algorithms.llm.data import load_shared
from src.algorithms.llm.schema import clip, pair_hash
from src.algorithms.llm.store import Store
from src.algorithms.pipeline_a import aggregate as pa

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--store", default="data/llm/encoder_outputs.sqlite")
ap.add_argument("--extraction-run", default="runs/llm/20261102T150142_tier5pct")
ap.add_argument("--run-a", default="runs/llm/20261107T121812_pipeA_5pct_full")
a = ap.parse_args()
from transformers import AutoTokenizer

run = provenance.new_run("encoder_gate", {"candidates": a.runs}, ["data/llm/distill/distill_v1.parquet"], sample_rate=0.05)
log = {}


def load(ck):
    c = torch.load(ck, map_location="cpu", weights_only=False)
    m = em.DistillEncoder(c["backbone"])
    m.load_state_dict(c["state_dict"])
    return m.cuda().eval(), AutoTokenizer.from_pretrained(c["backbone"]), c


def throughput(m, tok, P, R, max_length):
    em.predict(m, tok, P[:2000], R[:2000], max_length)  # warm-up
    torch.cuda.synchronize()
    t = time.time()
    em.predict(m, tok, P, R, max_length, batch=256)
    torch.cuda.synchronize()
    return len(P) / (time.time() - t)


# ---------- 1. selection on val only ----------
D = pd.read_parquet("data/llm/distill/distill_v1.parquet")
cands = []
for r in a.runs:
    s = json.load(open(Path(r) / "summary.json"))
    cands.append({"run": r, "checkpoint": s["checkpoint"], "val_agree_acc": s["best_val_agree_acc"]})
cands.sort(key=lambda c: -c["val_agree_acc"])
tp_sample = D[D.split == "val"].sample(20000, random_state=0)
if len(cands) > 1 and cands[0]["val_agree_acc"] - cands[1]["val_agree_acc"] < 0.005:
    for c in cands[:2]:
        m, tok, ck = load(c["checkpoint"])
        c["pairs_per_s_val20k"] = throughput(m, tok, tp_sample.parent_text.tolist(), tp_sample.reply_text.tolist(), ck["max_length"])
        del m
        torch.cuda.empty_cache()
    sel = max(cands[:2], key=lambda c: c["pairs_per_s_val20k"])
    rule = "val accuracies within 0.5 pp -> faster candidate"
else:
    sel, rule = cands[0], "higher val agreement accuracy"
log["selection"] = {"candidates": cands, "selected": sel["run"], "rule_applied": rule}
(run / "selection.json").write_text(json.dumps(log["selection"], indent=1))
print("SELECTED", json.dumps(log["selection"], indent=1), flush=True)

# ---------- 2. score the whole 5% tier (617,041 pairs incl. pilot) with the selected model ----------
m, tok, ck = load(sel["checkpoint"])
L = ck["max_length"]
sha = hashlib.sha256(Path(sel["checkpoint"]).read_bytes()).hexdigest()
cfg = yaml.safe_load(open("configs/llm/llm.yaml"))
mc = cfg["max_chars"]
e = load_shared(0.05)
e["pair_hash"] = [pair_hash(p, b, mc) for p, b in zip(e.parent, e.body)]
U = e.drop_duplicates("pair_hash")
U = pd.DataFrame({"pair_hash": U.pair_hash, "comment_id": U.comment_id, "parent_text": U.parent.map(lambda s: clip(s, mc["parent"])),
                  "reply_text": U.body.map(lambda s: clip(s, mc["body"]))}).reset_index(drop=True)
t0 = time.time()
pred = em.predict(m, tok, U.parent_text.tolist(), U.reply_text.tolist(), L, batch=256)
score_s = time.time() - t0
store = Store(a.store)
ctx = eo.ctx_for(ck["backbone"])
store.db.execute("DELETE FROM extractions WHERE model=? AND prompt_version=?", (ctx["model"], ctx["prompt_version"]))  # the gate is the only writer of the 5% rows for this model
n_written = eo.write(store, U.pair_hash.tolist(), pred, ctx)
print(f"scored {len(U)} pairs in {score_s:.0f}s ({len(U) / score_s:.0f} pairs/s incl. tokenization); wrote {n_written} rows", flush=True)
P = pd.DataFrame({"pair_hash": U.pair_hash, "e_agree": pred["agree"][:, 0], "e_neutral": pred["agree"][:, 1], "e_disagree": pred["agree"][:, 2],
                  "e_emo": pred["emo"].argmax(1), "e_host": pred["host"].argmax(1), "e_attack": pred["attack"] > 0.5})

# ---------- 3. checks ----------
T = D[D.split == "test"].merge(P, on="pair_hash", validate="1:1")
llm = T[em.AGREE_COLS].to_numpy()
enc = T[["e_agree", "e_neutral", "e_disagree"]].to_numpy()
guo = lambda x: (x[:, 0] - x[:, 2]) * (1 - x[:, 1])
c1 = test_acc = float((llm.argmax(1) == enc.argmax(1)).mean())
c2 = float(spearmanr(guo(llm), guo(enc))[0])
mult_l = 1 + T.hostility.map(em.HOST).map({0: 0, 1: .5, 2: 1}) + T.personal_attack.astype(float) + 0.5 * T.emotional_charge.map(em.EMO).map({0: 0, 1: .5, 2: 1})
mult_e = 1 + T.e_host.map({0: 0, 1: .5, 2: 1}) + T.e_attack.astype(float) + 0.5 * T.e_emo.map({0: 0, 1: .5, 2: 1})
nongating = {"test_emo_acc": float((T.e_emo == T.emotional_charge.map(em.EMO)).mean()), "test_host_acc": float((T.e_host == T.hostility.map(em.HOST)).mean()),
             "test_attack_acc": float((T.e_attack == T.personal_attack.astype(bool)).mean()),
             "test_C1_weight_spearman": float(spearmanr(guo(llm) * mult_l, guo(enc) * mult_e)[0]),
             "test_confusion_agreement_llm_rows_enc_cols": pd.crosstab(llm.argmax(1), enc.argmax(1)).values.tolist(), "n_test": int(len(T))}

key = pd.read_csv("data/llm/pilot/pilot_key_PRIVATE.csv", usecols=["pilot_id", "comment_id"])
lab = pd.read_csv("data/llm/pilot/labels_kareem.csv").merge(key, on="pilot_id")  # D-P3: agreed consensus labels
pp = lab.merge(U[["comment_id", "pair_hash"]], on="comment_id").merge(P, on="pair_hash")
pp["enc_label"] = np.array(eo.AGREE)[pp[["e_agree", "e_neutral", "e_disagree"]].to_numpy().argmax(1)]
c3 = float((pp.enc_label == pp.label).mean())
llm_tab = Store(cfg["store"]).table({"model": cfg["model"], "quant": "Q4_K_M", "prompt_version": cfg["prompt_version"], "temperature": cfg["temperature"], "seed": cfg["seed"]})
pp = pp.merge(llm_tab[["pair_hash", "agreement"]], on="pair_hash", how="left")
nongating["pilot_n"] = int(len(pp))
nongating["pilot_llm_acc_same_items"] = float((pp.agreement == pp.label).mean())

sel_e = pd.read_parquet(Path(a.extraction_run) / "selected_edges.parquet")
rules = pa.load_rules()
enc_tab = store.table(ctx).drop(columns=["model", "quant", "prompt_version", "temperature", "seed"])
EA = pa.aggregate_weekly(pa.score_edges(sel_e.merge(enc_tab, left_on="h", right_on="pair_hash", how="inner"), rules), rules).set_index("week").P_A
LA = pd.read_csv(Path(a.run_a) / "weekly_A.csv").set_index("week").P_A
c4 = float(spearmanr(LA.reindex(EA.index), EA)[0])
nongating["weekly_P_A_pearson"] = float(np.corrcoef(LA.reindex(EA.index), EA)[0, 1])
nongating["n_weeks"] = int(len(EA))

c5 = throughput(m, tok, tp_sample.parent_text.tolist(), tp_sample.reply_text.tolist(), L)
checks = {"1_test_agree_acc": (c1, 0.85), "2_test_guo_spearman": (c2, 0.80), "3_pilot_3class_acc": (c3, 0.65), "4_weekly_PA_spearman": (c4, 0.90),
          "5_pairs_per_s": (c5, 200.0)}
res = {"selected_checkpoint": sel["checkpoint"], "sha256": sha, "backbone": ck["backbone"], "best_step": ck["step"], "store_ctx": ctx,
       "checks": {k: {"value": v, "threshold": t, "pass": bool(v >= t)} for k, (v, t) in checks.items()}, "all_pass": all(v >= t for v, t in checks.values()),
       "non_gating": nongating, "full_tier_scoring_pairs_per_s": len(U) / score_s, "selection": log["selection"]}
(run / "gate.json").write_text(json.dumps(res, indent=1, default=float))
Path("results/llm/final/encoder").mkdir(parents=True, exist_ok=True)
Path("results/llm/final/encoder/gate.json").write_text(json.dumps(res, indent=1, default=float))
pd.DataFrame({"week": EA.index, "P_A_llm": LA.reindex(EA.index).values, "P_A_encoder": EA.values}).to_csv("results/llm/final/encoder/weekly_PA_fidelity.csv", index=False)
provenance.finish_run(run, "done", {k: round(v["value"], 4) for k, v in res["checks"].items()} | {"all_pass": res["all_pass"]}, purpose="G1.5 encoder selection + G-ENC checks")
print(json.dumps(res, indent=1, default=float))

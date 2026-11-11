"""G1.7 / C2 (SPEC_PIPELINE_C): bge-base-en-v1.5 embeddings of the unique parent/reply texts of the 5% tier, a ridge head trained on w_C1 with
5-fold GroupKFold-by-thread out-of-fold predictions (alpha chosen by nested CV on training folds only), a final head for other densities, and the
selection-free quality report (logged before any GT comparison). Writes data/llm/c2/ (git-ignored) and a run folder.
usage: python scripts/final/c2_embed_head.py [--limit N] [--out data/llm/c2]"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from src.algorithms.llm import provenance
from src.algorithms.llm.data import load_shared
from src.algorithms.llm.schema import clip, pair_hash
from src.algorithms.llm.store import Store
from src.algorithms.pipeline_c import embed as em
from src.algorithms.pipeline_c import weights as wc

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=None, help="smoke test on the first N pairs")
ap.add_argument("--out", default="data/llm/c2")
ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
ap.add_argument("--folds", type=int, default=5)
a = ap.parse_args()
import torch
from sklearn.model_selection import GroupKFold

cfg = yaml.safe_load(open("configs/llm/llm.yaml"))
mc = cfg["max_chars"]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
run = provenance.new_run("c2_embed_head" + ("_smoke" if a.limit else ""), {"embedder": a.model, "folds": a.folds, "alphas": list(em.ALPHAS), "seed": 0},
                         ["configs/llm/llm.yaml"], sample_rate=0.05)
t0 = time.time()

# ---- pairs and LLM targets ----
e = load_shared(0.05)
e["pair_hash"] = [pair_hash(p, b, mc) for p, b in zip(e.parent, e.body)]
e = e.drop_duplicates("pair_hash").assign(parent_text=lambda d: d.parent.map(lambda s: clip(s, mc["parent"])), reply_text=lambda d: d.body.map(lambda s: clip(s, mc["body"])))
store = Store(cfg["store"])
ctx = {"model": cfg["model"], "quant": "Q4_K_M", "prompt_version": cfg["prompt_version"], "temperature": cfg["temperature"], "seed": cfg["seed"]}
t = store.table(ctx)
P = e.merge(t[["pair_hash", "p_agree", "p_neutral", "p_disagree", "agreement", "emotional_charge", "hostility", "personal_attack"]], on="pair_hash", validate="1:1")
P = P.rename(columns={"link_id": "thread"})
key = pd.read_csv("data/llm/pilot/pilot_key_PRIVATE.csv", usecols=["pilot_id", "comment_id"])
P["is_pilot"] = P.comment_id.isin(key.comment_id)
P["w_C1"] = wc.c1(P)
if a.limit:
    P = pd.concat([P[P.is_pilot], P[~P.is_pilot].head(a.limit)]).reset_index(drop=True)
print(f"{len(P)} pairs ({int(P.is_pilot.sum())} pilot), {time.time() - t0:.0f}s", flush=True)

# ---- embeddings (cached per unique text hash) ----
texts = pd.unique(pd.concat([P.parent_text, P.reply_text]).to_numpy())
emb_f = out / f"emb_{'smoke' if a.limit else 'full'}.npy"
idx_f = out / f"emb_{'smoke' if a.limit else 'full'}_texthash.parquet"
if emb_f.exists() and idx_f.exists():
    hs = pd.read_parquet(idx_f).text_hash.tolist()
    assert hs == [em.text_hash(x) for x in texts], "embedding cache does not match the current text set"
    U_np = np.load(emb_f)
else:
    U_np = em.embed_texts(list(texts), a.model)
    np.save(emb_f, U_np)
    pd.DataFrame({"text_hash": [em.text_hash(x) for x in texts]}).to_parquet(idx_f)
print(f"{len(texts)} unique texts embedded {U_np.shape} {time.time() - t0:.0f}s", flush=True)
pos = {x: i for i, x in enumerate(texts)}
dev = "cuda"
U = torch.from_numpy(U_np).to(dev)
ir = torch.tensor([pos[x] for x in P.reply_text], device=dev)
ip = torch.tensor([pos[x] for x in P.parent_text], device=dev)
y = torch.tensor(P.w_C1.to_numpy(), dtype=torch.float32, device=dev)

# ---- folds (training pairs only; pilot pairs are never trained on) ----
tr = np.flatnonzero(~P.is_pilot.to_numpy())
fold = np.full(len(P), -1)
for k, (_, te) in enumerate(GroupKFold(a.folds).split(tr, groups=P.thread.to_numpy()[tr])):
    fold[tr[te]] = k
rows_of = {k: torch.tensor(np.flatnonzero(fold == k), device=dev) for k in range(a.folds)}
stats = {k: em.fold_stats(U, ir[r], ip[r], y[r]) for k, r in rows_of.items()}
print(f"fold stats done {time.time() - t0:.0f}s", flush=True)

# ---- nested alpha selection, OOF predictions ----
inner_mse = np.zeros((a.folds, a.folds, len(em.ALPHAS)))
oof = np.full(len(P), np.nan)
alpha_o = {}
for o in range(a.folds):
    for i in range(a.folds):
        if i == o:
            continue
        b, c0 = em.ridge_path(em.add_stats([stats[f] for f in range(a.folds) if f not in (o, i)]))
        pr = em.predict(U, ir[rows_of[i]], ip[rows_of[i]], b, c0)
        inner_mse[o, i] = ((pr - y[rows_of[i]]) ** 2).mean(1).cpu().numpy()
    ao = int(np.argmin(inner_mse[o][[i for i in range(a.folds) if i != o]].mean(0)))
    alpha_o[o] = float(em.ALPHAS[ao])
    b, c0 = em.ridge_path(em.add_stats([stats[f] for f in range(a.folds) if f != o]), [em.ALPHAS[ao]])
    oof[rows_of[o].cpu().numpy()] = em.predict(U, ir[rows_of[o]], ip[rows_of[o]], b, c0)[0].cpu().numpy()
    print(f"outer fold {o}: alpha {alpha_o[o]:.3g} {time.time() - t0:.0f}s", flush=True)

# ---- final head (all non-pilot pairs) for other densities and for the pilot pairs ----
a_fin = int(np.argmin(np.nanmean(np.where(inner_mse == 0, np.nan, inner_mse), axis=(0, 1))))
b, c0 = em.ridge_path(em.add_stats(list(stats.values())), [em.ALPHAS[a_fin]])
pil = torch.tensor(np.flatnonzero(P.is_pilot.to_numpy()), device=dev)
oof[pil.cpu().numpy()] = em.predict(U, ir[pil], ip[pil], b, c0)[0].cpu().numpy()
np.savez(out / f"c2_head{'_smoke' if a.limit else ''}.npz", beta=b[0].cpu().numpy(), intercept=c0.cpu().numpy(), alpha=em.ALPHAS[a_fin])
P["yhat_oof"] = oof
P["w_C2"] = np.clip(oof, -3.5, 3.5)
assert P.w_C2.notna().all()
P[["pair_hash", "yhat_oof", "w_C2", "w_C1", "is_pilot"]].to_parquet(out / f"c2_weights_5pct{'_smoke' if a.limit else ''}.parquet")

# ---- quality report (selection-free; no camp labels or GT) ----
m = ~P.is_pilot
yy, yh = P.w_C1[m].to_numpy(), P.yhat_oof[m].to_numpy()
r2 = 1 - ((yy - yh) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
am = P[["p_agree", "p_neutral", "p_disagree"]].to_numpy().argmax(1)
dec = m.to_numpy() & (am != 1)
sign_agree = float((np.sign(P.yhat_oof[dec]) == np.where(am[dec] == 0, 1, -1)).mean())
lab = pd.read_csv("data/llm/pilot/labels_kareem.csv").merge(key, on="pilot_id")
pp = P.set_index("comment_id").loc[lab.comment_id]
h = lab.label.to_numpy()
ok = np.isin(h, ["agree", "disagree"])
pil_acc_c2 = float((np.sign(pp.yhat_oof.to_numpy()[ok]) == np.where(h[ok] == "agree", 1, -1)).mean())
pam = pp[["p_agree", "p_neutral", "p_disagree"]].to_numpy().argmax(1)
pil_acc_llm = float((np.where(pam[ok] == 0, 1, np.where(pam[ok] == 2, -1, 0)) == np.where(h[ok] == "agree", 1, -1)).mean())
q = {"label": "C2 quality report, selection-free, logged before any GT comparison", "n_pairs": int(len(P)), "n_train_pairs": int(m.sum()), "unique_texts": int(len(texts)),
     "R2_oof_vs_wC1": float(r2), "spearman_oof_vs_wC1": float(spearmanr(yy, yh)[0]), "sign_agreement_with_LLM_argmax_nonneutral": sign_agree,
     "pilot_n_agree_disagree": int(ok.sum()), "pilot_accuracy_C2_sign": pil_acc_c2, "pilot_accuracy_LLM_argmax_same_items": pil_acc_llm,
     "alpha_per_outer_fold": alpha_o, "alpha_final_head": float(em.ALPHAS[a_fin]), "embedder": a.model, "seconds": time.time() - t0}
(out / f"c2_quality{'_smoke' if a.limit else ''}.json").write_text(json.dumps(q, indent=1))
(run / "c2_quality.json").write_text(json.dumps(q, indent=1))
provenance.finish_run(run, "done", {k: q[k] for k in ("R2_oof_vs_wC1", "spearman_oof_vs_wC1", "pilot_accuracy_C2_sign")}, purpose="G1.7 C2 embeddings and ridge head (OOF)")
print(json.dumps(q, indent=1))

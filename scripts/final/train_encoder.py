"""G1.3 / G1.4 (SPEC_ENCODER section 3): train one distilled encoder candidate on data/llm/distill/distill_v1.parquet (train split only),
evaluate on val every --eval-every steps and keep the best checkpoint by val agreement-argmax accuracy against the LLM. Never touches test, pilot or GT.
Checkpoint -> data/llm/encoder/<name>/best.pt (git-ignored); curves + manifest -> runs/llm/<ts>_encoder_<name>/.
usage: python scripts/final/train_encoder.py --name E1 --backbone microsoft/deberta-v3-base --lr 2e-5 [--max-length 384] [--limit N]"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from src.algorithms.encoder import model as em
from src.algorithms.llm import provenance

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--backbone", required=True)
ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--max-length", type=int, default=384)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--accum", type=int, default=1)
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--eval-every", type=int, default=5000)
ap.add_argument("--limit", type=int, default=None, help="smoke test: first N train rows, val capped at 2000")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = "cuda"
dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
ck_dir = Path("data/llm/encoder") / (a.name + ("_smoke" if a.limit else ""))
ck_dir.mkdir(parents=True, exist_ok=True)
run = provenance.new_run(f"encoder_{a.name}" + ("_smoke" if a.limit else ""), vars(a) | {"dtype": str(dt)}, ["data/llm/distill/distill_v1.parquet"], sample_rate=0.05)
t0 = time.time()

D = pd.read_parquet("data/llm/distill/distill_v1.parquet")
tr, va = D[D.split == "train"].reset_index(drop=True), D[D.split == "val"].reset_index(drop=True)
del D
if a.limit:
    tr, va = tr.head(a.limit), va.head(2000)
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
tok = AutoTokenizer.from_pretrained(a.backbone)
ids, tti = em.tokenize(tok, tr.parent_text, tr.reply_text, a.max_length)
lens = np.array([len(x) for x in ids])
Y = {k: torch.from_numpy(v) for k, v in em.targets(tr).items()}
print(f"{len(tr)} train / {len(va)} val pairs tokenized, truncated share {(lens >= a.max_length).mean():.4f}, {time.time() - t0:.0f}s", flush=True)

model = em.DistillEncoder(a.backbone).to(dev)
nd = ["bias", "LayerNorm.weight", "layernorm", "norm.weight"]
groups = [{"params": [p for n, p in model.named_parameters() if not any(x in n for x in nd)], "weight_decay": 0.01},
          {"params": [p for n, p in model.named_parameters() if any(x in n for x in nd)], "weight_decay": 0.0}]
opt = torch.optim.AdamW(groups, lr=a.lr, fused=True)
micro = a.batch // a.accum
steps_per_epoch = math.ceil(len(tr) / a.batch)
total = steps_per_epoch * a.epochs
sch = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
scaler = torch.amp.GradScaler(enabled=(dt == torch.float16))


def evaluate():
    p = em.predict(model, tok, va.parent_text, va.reply_text, a.max_length, 256, dev, dt)
    model.train()
    llm = va[em.AGREE_COLS].to_numpy()
    w_llm = (llm[:, 0] - llm[:, 2]) * (1 - llm[:, 1])
    w_enc = (p["agree"][:, 0] - p["agree"][:, 2]) * (1 - p["agree"][:, 1])
    return {"val_agree_acc": float((p["agree"].argmax(1) == llm.argmax(1)).mean()), "val_guo_spearman": float(spearmanr(w_llm, w_enc)[0]),
            "val_emo_acc": float((p["emo"].argmax(1) == va.emotional_charge.map(em.EMO).to_numpy()).mean()),
            "val_host_acc": float((p["host"].argmax(1) == va.hostility.map(em.HOST).to_numpy()).mean()),
            "val_attack_acc": float(((p["attack"] > 0.5) == va.personal_attack.astype(bool).to_numpy()).mean())}


curve, best, step, run_loss = [], -1.0, 0, []
model.train()
for ep in range(a.epochs):
    for bi, b_idx in enumerate(em.length_batches(lens, a.batch, a.seed + ep)):
        for mi in range(0, len(b_idx), micro):
            m_idx = b_idx[mi: mi + micro]
            b = em.collate([ids[i] for i in m_idx], [tti[i] for i in m_idx] if tti is not None else None, tok.pad_token_id, dev)
            y = {k: v[m_idx].to(dev, non_blocking=True) for k, v in Y.items()}
            with torch.autocast("cuda", dtype=dt):
                out = model(**b)
            loss, parts = em.loss_fn(out, y)
            scaler.scale(loss * len(m_idx) / len(b_idx)).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sch.step()
        step += 1
        run_loss.append(float(loss))
        if step % 200 == 0:
            print(f"step {step}/{total} loss {np.mean(run_loss[-200:]):.4f} lr {sch.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if step % a.eval_every == 0 or step == total:
            ev = evaluate() | {"step": step, "epoch": ep, "train_loss_recent": float(np.mean(run_loss[-500:])), "seconds": time.time() - t0}
            curve.append(ev)
            pd.DataFrame(curve).to_csv(run / "val_curve.csv", index=False)
            torch.cuda.empty_cache()
            if ev["val_agree_acc"] > best:
                best = ev["val_agree_acc"]
                torch.save({"state_dict": model.state_dict(), "backbone": a.backbone, "max_length": a.max_length, "step": step, "val": ev}, ck_dir / "best.pt")
            print("EVAL", json.dumps(ev), flush=True)

sha = hashlib.sha256((ck_dir / "best.pt").read_bytes()).hexdigest()
summary = {"best_val_agree_acc": best, "best_step": max(curve, key=lambda r: r["val_agree_acc"])["step"], "checkpoint": str(ck_dir / "best.pt"), "sha256": sha,
           "truncated_share_train": float((lens >= a.max_length).mean()), "train_pairs": len(tr), "steps": step, "seconds": time.time() - t0}
(run / "summary.json").write_text(json.dumps(summary, indent=1))
provenance.finish_run(run, "done", summary, purpose=f"G1.3/G1.4 encoder candidate {a.name} ({a.backbone})")
print(json.dumps(summary, indent=1))

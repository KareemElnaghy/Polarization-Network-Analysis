"""Encoder predictions -> rows in the extraction-store schema (SPEC_ENCODER section 4), so Pipelines A/B/C read them unchanged.
model = "enc-<backbone>-distill-v1", quant = "fp16", prompt_version = "enc-v1", temperature 0, seed 0."""
import json
import time

import numpy as np

from . import model as em

AGREE = ["agree", "neutral", "disagree"]


def ctx_for(backbone: str) -> dict:
    return {"model": f"enc-{backbone.split('/')[-1]}-distill-v1", "quant": "fp16", "prompt_version": "enc-v1", "temperature": 0.0, "seed": 0,
            "prob_method": "encoder_softmax"}


def fields(pred: dict, i: int) -> dict:
    return {"agreement": AGREE[int(pred["agree"][i].argmax())], "emotional_charge": em.EMO_INV[int(pred["emo"][i].argmax())],
            "hostility": em.HOST_INV[int(pred["host"][i].argmax())], "personal_attack": bool(pred["attack"][i] > 0.5)}


def write(store, hashes, pred: dict, ctx: dict) -> int:
    now = time.time()
    rows = [(h, ctx["model"], ctx["quant"], ctx["prompt_version"], ctx["temperature"], ctx["seed"], json.dumps(fields(pred, i)),
             float(pred["agree"][i, 0]), float(pred["agree"][i, 1]), float(pred["agree"][i, 2]), None, 0, None, ctx["prob_method"], now)
            for i, h in enumerate(hashes)]
    store.db.executemany("INSERT OR IGNORE INTO extractions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    store.db.commit()
    return len(rows)

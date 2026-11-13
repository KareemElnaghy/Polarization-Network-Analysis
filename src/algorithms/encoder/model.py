"""Distilled cross-encoder (SPEC_ENCODER section 3): a (parent, reply) pair -> agreement probabilities (KL to the LLM's soft targets),
emotional_charge / hostility (3-way CE) and personal_attack (BCE). Pure model code: no labels, GT or eval imports (firewall)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMO = {"low": 0, "medium": 1, "high": 2}
HOST = {"none": 0, "mild": 1, "strong": 2}
EMO_INV = {v: k for k, v in EMO.items()}
HOST_INV = {v: k for k, v in HOST.items()}
AGREE_COLS = ["p_agree", "p_neutral", "p_disagree"]  # head order; argmax 0 agree, 1 neutral, 2 disagree


class DistillEncoder(nn.Module):
    def __init__(self, backbone: str, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoModel
        self.backbone_name = backbone
        self.enc = AutoModel.from_pretrained(backbone)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.agree = nn.Linear(h, 3)
        self.emo = nn.Linear(h, 3)
        self.host = nn.Linear(h, 3)
        self.attack = nn.Linear(h, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        x = self.drop(self.enc(**kw).last_hidden_state[:, 0])  # first-token pooling
        return {"agree": self.agree(x), "emo": self.emo(x), "host": self.host(x), "attack": self.attack(x).squeeze(-1)}


def loss_fn(out: dict, y: dict) -> tuple[torch.Tensor, dict]:
    """1.0 * KL(LLM agreement probs || model) + 0.5 * (CE emo + CE host + BCE attack)."""
    la = F.kl_div(F.log_softmax(out["agree"].float(), -1), y["agree"], reduction="batchmean")
    le = F.cross_entropy(out["emo"].float(), y["emo"])
    lh = F.cross_entropy(out["host"].float(), y["host"])
    lk = F.binary_cross_entropy_with_logits(out["attack"].float(), y["attack"])
    return la + 0.5 * (le + lh + lk), {"kl": float(la), "emo": float(le), "host": float(lh), "attack": float(lk)}


def targets(df) -> dict:
    """Training targets from distill rows (numpy)."""
    p = df[AGREE_COLS].to_numpy(np.float32)
    p = np.clip(p, 1e-6, 1.0)
    p = p / p.sum(1, keepdims=True)
    return {"agree": p, "emo": df.emotional_charge.map(EMO).to_numpy(np.int64), "host": df.hostility.map(HOST).to_numpy(np.int64),
            "attack": df.personal_attack.astype(np.float32).to_numpy()}


def length_batches(lengths: np.ndarray, batch: int, seed: int, mega: int = 50) -> list[np.ndarray]:
    """Bucketed batches: shuffle, sort within mega-chunks of batch*mega by length, cut into batches, shuffle the batch order."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(lengths))
    out = []
    for s in range(0, len(idx), batch * mega):
        c = idx[s: s + batch * mega]
        c = c[np.argsort(lengths[c], kind="stable")]
        out += [c[i: i + batch] for i in range(0, len(c), batch)]
    rng.shuffle(out)
    return out


def collate(ids: list[list[int]], tti: list[list[int]] | None, pad_id: int, device) -> dict:
    L = max(len(x) for x in ids)
    a = np.full((len(ids), L), pad_id, dtype=np.int64)
    m = np.zeros((len(ids), L), dtype=np.int64)
    t = np.zeros((len(ids), L), dtype=np.int64) if tti is not None else None
    for i, x in enumerate(ids):
        a[i, : len(x)] = x
        m[i, : len(x)] = 1
        if t is not None:
            t[i, : len(x)] = tti[i]
    b = {"input_ids": torch.from_numpy(a).to(device, non_blocking=True), "attention_mask": torch.from_numpy(m).to(device, non_blocking=True)}
    if t is not None:
        b["token_type_ids"] = torch.from_numpy(t).to(device, non_blocking=True)
    return b


def tokenize(tok, parents, replies, max_length: int) -> tuple[list, list | None]:
    enc = tok(list(parents), list(replies), truncation="longest_first", max_length=max_length)
    return enc["input_ids"], enc.get("token_type_ids")


@torch.inference_mode()
def predict(model, tok, parents, replies, max_length: int = 384, batch: int = 256, device: str = "cuda", dtype=torch.float16,
            max_tokens: int = 32768) -> dict:
    """Probabilities for every head, in input order. Length-sorted batches of up to `batch` pairs and `max_tokens` padded tokens
    (long batches would otherwise spill out of GPU memory on Windows and crawl), autocast."""
    model.eval()
    torch.cuda.empty_cache()
    ids, tti = tokenize(tok, parents, replies, max_length)
    lens = np.array([len(x) for x in ids])
    order = np.argsort(lens, kind="stable")
    res = {k: np.zeros((len(ids), 3), np.float32) for k in ("agree", "emo", "host")}
    res["attack"] = np.zeros(len(ids), np.float32)
    chunks, s = [], 0
    while s < len(order):
        e = s + 1
        while e < len(order) and e - s < batch and (e - s + 1) * lens[order[e]] <= max_tokens:
            e += 1
        chunks.append(order[s:e])
        s = e
    for o in chunks:
        b = collate([ids[i] for i in o], [tti[i] for i in o] if tti is not None else None, tok.pad_token_id, device)
        with torch.autocast("cuda", dtype=dtype):
            out = model(**b)
        for k in ("agree", "emo", "host"):
            res[k][o] = torch.softmax(out[k].float(), -1).cpu().numpy()
        res["attack"][o] = torch.sigmoid(out["attack"].float()).cpu().numpy()
    return res

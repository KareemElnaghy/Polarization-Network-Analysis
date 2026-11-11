"""C2 (SPEC_PIPELINE_C): frozen sentence-encoder embeddings of the parent and reply text and a ridge head on pair features
x = [u_r, u_p, u_r*u_p, |u_r-u_p|]. The ridge is solved from per-fold normal-equation statistics (exact ridge, intercept unpenalised),
so out-of-fold fitting and the inner alpha selection need one pass over the features. No labels, GT or eval imports (firewall)."""
import hashlib

import numpy as np

ALPHAS = np.logspace(-2, 3, 12)  # fixed by the spec


def text_hash(t: str) -> str:
    return hashlib.sha1(t.encode()).hexdigest()


def embed_texts(texts: list[str], model_name: str = "BAAI/bge-base-en-v1.5", batch: int = 256, device: str = "cuda") -> np.ndarray:
    """Normalised float16 embeddings, row i for texts[i]. Sorted by length internally for throughput."""
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(model_name, device=device)
    m.max_seq_length = 512
    m.half()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), m.get_sentence_embedding_dimension()), dtype=np.float16)
    with torch.inference_mode():
        for s in range(0, len(texts), batch * 20):
            idx = order[s: s + batch * 20]
            out[idx] = m.encode([texts[i] for i in idx], batch_size=batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
    return out


def pair_features(U, ir, ip):
    """U: (N, d) torch tensor; ir/ip: index tensors. Returns (len(ir), 4d) float32 features [u_r, u_p, u_r*u_p, |u_r-u_p|]."""
    r, p = U[ir].float(), U[ip].float()
    import torch
    return torch.cat([r, p, r * p, (r - p).abs()], dim=1)


def fold_stats(U, ir, ip, y, chunk: int = 50_000):
    """Sufficient statistics of one row set: n, sum_x, sum_y, XtX, Xty (float64, on the device of U)."""
    import torch
    d4 = 4 * U.shape[1]
    XtX = torch.zeros(d4, d4, dtype=torch.float64, device=U.device)
    Xty = torch.zeros(d4, dtype=torch.float64, device=U.device)
    sx = torch.zeros(d4, dtype=torch.float64, device=U.device)
    sy, n, syy = 0.0, 0, 0.0
    for s in range(0, len(ir), chunk):
        X = pair_features(U, ir[s: s + chunk], ip[s: s + chunk]).double()
        yy = y[s: s + chunk].double()
        XtX += X.T @ X
        Xty += X.T @ yy
        sx += X.sum(0)
        sy += float(yy.sum())
        syy += float((yy * yy).sum())
        n += len(yy)
    return {"n": n, "sx": sx, "sy": sy, "syy": syy, "XtX": XtX, "Xty": Xty}


def add_stats(stats: list[dict]) -> dict:
    out = {k: stats[0][k].clone() if hasattr(stats[0][k], "clone") else stats[0][k] for k in stats[0]}
    for s in stats[1:]:
        for k in out:
            out[k] = out[k] + s[k]
    return out


def ridge_path(st: dict, alphas=ALPHAS):
    """Exact ridge solutions (centred, intercept unpenalised) for every alpha from one eigendecomposition.
    Returns betas (n_alpha, d) and intercepts (n_alpha,) as torch tensors."""
    import torch
    n = st["n"]
    xm, ym = st["sx"] / n, st["sy"] / n
    C = st["XtX"] - n * torch.outer(xm, xm)
    c = st["Xty"] - n * xm * ym
    lam, V = torch.linalg.eigh(C)
    lam = lam.clamp_min(0)
    Vc = V.T @ c
    betas = torch.stack([V @ (Vc / (lam + float(a))) for a in alphas])
    return betas, ym - betas @ xm


def predict(U, ir, ip, betas, icpt, chunk: int = 50_000):
    """Predictions (n_alpha, n_rows) float32 for each alpha."""
    import torch
    outs = []
    B = betas.float().T.contiguous()
    for s in range(0, len(ir), chunk):
        outs.append((pair_features(U, ir[s: s + chunk], ip[s: s + chunk]) @ B + icpt.float()).T)
    return torch.cat(outs, dim=1)

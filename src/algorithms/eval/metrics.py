"""Statistics shared by the final-milestone analyses. Numerically identical to the helpers inside scripts/llm/rq1_compare.py
(which is left untouched; tests/analysis/test_metrics.py reproduces its committed full-5% output)."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


def rho(x, y) -> float:
    x, y = pd.Series(x), pd.Series(y)
    m = x.notna().to_numpy() & y.notna().to_numpy()
    return spearmanr(x[m], y[m])[0] if m.sum() >= 5 else np.nan


def circ_p(x: pd.Series, y: pd.Series) -> float:
    """Circular-shift permutation p over every distinct shift (T-1), two-sided on |rho|."""
    obs, T = rho(x, y), len(x)
    null = [rho(pd.Series(np.roll(x.to_numpy(), k), index=x.index), y) for k in range(1, T)]
    null = np.array([v for v in null if not np.isnan(v)])
    return (1 + (np.abs(null) >= abs(obs)).sum()) / (1 + len(null))


def partial(x, y, z) -> float:
    """Spearman partial correlation of x, y controlling for z (rank residuals)."""
    x, y, z = pd.Series(x), pd.Series(y), pd.Series(z)
    m = (x.notna() & y.notna() & z.notna()).to_numpy()
    if m.sum() < 5:
        return np.nan
    rx, ry, rz = (rankdata(v.to_numpy()[m]) for v in (x, y, z))
    res = lambda u: u - np.polyval(np.polyfit(rz, u, 1), rz)
    return np.corrcoef(res(rx), res(ry))[0, 1]


def block_indices(n: int, block: int, rng) -> np.ndarray:
    nb = int(np.ceil(n / block))
    return np.concatenate([np.arange(s, s + block) for s in rng.integers(0, max(n - block + 1, 1), nb)])[:n]


def block_boot(stat, arrays, boot: int = 2000, block: int = 4, seed: int = 0):
    """Moving-block bootstrap of stat(*resampled arrays). Returns (point, lo, hi, draws). NaN draws are dropped."""
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    n, rng, d = len(arrays[0]), np.random.default_rng(seed), []
    for _ in range(boot):
        idx = block_indices(n, block, rng)
        v = stat(*[a[idx] for a in arrays])
        if not np.isnan(v):
            d.append(v)
    d = np.array(d)
    return stat(*arrays), (np.percentile(d, 2.5) if len(d) else np.nan), (np.percentile(d, 97.5) if len(d) else np.nan), d


def holm(p: dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    """Holm step-down adjusted p values for a family. Returns one row per test, input order preserved."""
    names = list(p)
    order = sorted(names, key=lambda k: p[k])
    m, run, adj = len(names), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[k]))
        adj[k] = run
    return pd.DataFrame({"test": names, "p": [p[k] for k in names], "p_holm": [adj[k] for k in names],
                         "significant_at_alpha": [adj[k] < alpha for k in names]})

"""RQ4 rolling-origin one-step forecasting of weekly change in a ground truth (D-E6 section G). Reads ground truth (analysis/ only)."""
import numpy as np
import pandas as pd
from scipy import stats


def rolling_forecasts(gt: np.ndarray, series: dict[str, np.ndarray], first_target: int = 24) -> pd.DataFrame:
    """Targets t (1-indexed week numbers) = first_target ... T. For target t the model is refit by OLS (with intercept) on all pairs
    (dGT_s, dP_{s-1}) with s <= t-1, then predicts dGT_t from dP_{t-1}. Models: persistence (0), and one per key of `series` plus 'A+B' (all series).
    dX_s = X_s - X_{s-1}. Returns one row per target with actual and each model's prediction."""
    T = len(gt)
    dgt = np.r_[np.nan, np.diff(gt)]                       # index s-1 holds dGT_s (s 1-indexed -> position s-1)
    dX = {k: np.r_[np.nan, np.diff(v)] for k, v in series.items()}
    models = {k: [k] for k in series} | ({"+".join(series): list(series)} if len(series) > 1 else {})
    rows = []
    for t in range(first_target, T + 1):
        pos = t - 1                                          # position of week t
        r = {"t": t, "actual": dgt[pos], "persistence": 0.0}
        for name, ks in models.items():
            tr = np.arange(2, pos)                           # positions s-1 for s=3..t-1: need dGT_s and dP_{s-1}
            X = np.column_stack([np.ones(len(tr))] + [dX[k][tr - 1] for k in ks])
            y = dgt[tr]
            ok = np.isfinite(X).all(1) & np.isfinite(y)
            beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
            x_new = np.r_[1.0, [dX[k][pos - 1] for k in ks]]
            r[name] = float(x_new @ beta) if np.isfinite(x_new).all() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def diebold_mariano(e_model: np.ndarray, e_base: np.ndarray, lag: int | None = None) -> dict:
    """Two-sided DM test on absolute-error loss differentials d = |e_model| - |e_base|, Newey-West (Bartlett) HAC variance,
    Harvey-Leybourne-Newbold small-sample correction (one-step horizon), t(n-1) reference. d < 0 means the model is more accurate."""
    d = np.abs(e_model) - np.abs(e_base)
    d = d[np.isfinite(d)]
    n = len(d)
    lag = int(np.floor(4 * (n / 100) ** (2 / 9))) if lag is None else lag
    dc = d - d.mean()
    g = [np.sum(dc[k:] * dc[: n - k]) / n for k in range(lag + 1)]
    var = (g[0] + 2 * sum((1 - k / (lag + 1)) * g[k] for k in range(1, lag + 1))) / n
    if var <= 0:
        return {"dm": np.nan, "p": np.nan, "mean_loss_diff": float(d.mean()), "n": n, "lag": lag}
    dm = d.mean() / np.sqrt(var) * np.sqrt((n - 1) / n)     # HLN factor for h = 1
    return {"dm": float(dm), "p": float(2 * stats.t.sf(abs(dm), n - 1)), "mean_loss_diff": float(d.mean()), "n": n, "lag": lag}

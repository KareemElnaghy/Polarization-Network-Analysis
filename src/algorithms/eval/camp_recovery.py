"""Camp-recovery evaluation: the ONLY module that reads user camp labels. Does the frustration partition recover the camps?
Balanced accuracy of the partition among labelled users in the graph, taking the better of the two side assignments
(the partition sign is arbitrary), against a shuffled-label null."""
import numpy as np
import pandas as pd


def load_labels(path: str = "data/processed/labels/venue=shared/set=main.parquet") -> pd.Series:
    d = pd.read_parquet(path)
    return d[d.side.isin(["dem", "rep"])].set_index("author").side.eq("rep").astype(int)


def balanced_accuracy(side: np.ndarray, y: np.ndarray) -> float:
    """side in {-1,+1}, y in {0,1}; best of the two orientations."""
    pred = (side > 0).astype(int)
    ba = 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())
    return max(ba, 1 - ba)


def evaluate(camps: pd.DataFrame, labels: pd.Series, n_null: int = 200, seed: int = 0, min_per_side: int = 5) -> pd.DataFrame:
    """camps: columns week, sign_source, user, side (+1/-1). One row per (week, sign_source)."""
    rng = np.random.default_rng(seed)
    rows = []
    for (wk, src), g in camps.groupby(["week", "sign_source"]):
        g = g[g.user.isin(labels.index)]
        y = labels.loc[g.user].values
        n1, n0 = int(y.sum()), int((1 - y).sum())
        if min(n1, n0) < min_per_side:
            rows.append({"week": wk, "sign_source": src, "n_labelled": len(g), "n_rep": n1, "n_dem": n0, "balanced_acc": np.nan,
                         "null_mean": np.nan, "null_p95": np.nan})
            continue
        s = g.side.values
        null = np.array([balanced_accuracy(s, rng.permutation(y)) for _ in range(n_null)])
        rows.append({"week": wk, "sign_source": src, "n_labelled": len(g), "n_rep": n1, "n_dem": n0,
                     "balanced_acc": balanced_accuracy(s, y), "null_mean": null.mean(), "null_p95": np.quantile(null, 0.95)})
    return pd.DataFrame(rows)

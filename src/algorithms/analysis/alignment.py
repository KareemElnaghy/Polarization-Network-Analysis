"""Edge-level alignment diagnostics (H-AL, SPEC_ANALYSES A3). READS CAMP LABELS: lives in analysis/, never imported by a pipeline.
Edges whose two endpoints both carry a main label (dem/rep) are 'LB'. cross = (label_replier != label_parent)."""
import numpy as np
import pandas as pd

from src.algorithms.llm.data import load_shared, load_users

HOST = {"none": 0.0, "mild": 0.5, "strong": 1.0}
LABELS = "data/processed/labels/venue=shared/set=main.parquet"


def load_side_labels(path: str = LABELS) -> pd.Series:
    d = pd.read_parquet(path)
    return d[d["side"].isin(["dem", "rep"])].set_index("author")["side"]


def sign_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guo weight (continuous), binary +-1 (neutral NaN) and the other edge-level outcomes, from cache columns."""
    am = df[["p_agree", "p_neutral", "p_disagree"]].to_numpy().argmax(1)
    return df.assign(w_continuous_llm=(df.p_agree - df.p_disagree) * (1 - df.p_neutral),
                     w_binary_llm=np.select([am == 0, am == 2], [1.0, -1.0], np.nan),
                     disagree_argmax=(am == 2).astype(float),
                     host_gt_none=(df.hostility != "none").astype(float),
                     host_level=df.hostility.map(HOST),
                     attack=df.personal_attack.astype(float))


def build_edge_table(run_a: str, rate: float = 0.05, with_vader: bool = True) -> pd.DataFrame:
    """One row per edge of the Pipeline A run: cache outputs + endpoints + labels (+ VADER of the reply for LB edges only)."""
    e = pd.read_parquet(f"{run_a}/edge_scores.parquet")
    e = e.merge(load_users(rate), on="comment_id", how="left", validate="1:1")
    lab = load_side_labels()
    e["lab_replier"], e["lab_parent"] = e.source.map(lab), e.target.map(lab)
    e["lb"] = e.lab_replier.notna() & e.lab_parent.notna()
    e["cross"] = np.where(e.lb, e.lab_replier != e.lab_parent, np.nan)
    e = sign_columns(e).rename(columns={"link_id": "thread"})
    e["w_sentiment"] = np.nan
    if with_vader:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        an = SentimentIntensityAnalyzer()
        body = load_shared(rate).set_index("comment_id").body
        idx = e.index[e.lb]
        e.loc[idx, "w_sentiment"] = [an.polarity_scores(t)["compound"] for t in body.loc[e.loc[idx, "comment_id"]]]
    return e


def cluster_boot_diff(x: np.ndarray, cross: np.ndarray, thread: np.ndarray, boot: int = 1000, seed: int = 0) -> dict:
    """mean(x | cross) - mean(x | same) with a thread-cluster bootstrap (threads resampled with replacement via multinomial counts)."""
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    x, cross, thread = x[ok], np.asarray(cross)[ok].astype(bool), np.asarray(thread)[ok]
    th, inv = np.unique(thread, return_inverse=True)
    T = len(th)

    def f(m):
        return np.bincount(inv[m], weights=x[m], minlength=T), np.bincount(inv[m], minlength=T).astype(float)

    (sc, nc), (ss, ns) = f(cross), f(~cross)
    point = sc.sum() / nc.sum() - ss.sum() / ns.sum()
    k = np.random.default_rng(seed).multinomial(T, np.full(T, 1 / T), size=boot).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (k @ sc) / (k @ nc) - (k @ ss) / (k @ ns)
    d = d[np.isfinite(d)]
    return {"delta": point, "ci_lo": np.percentile(d, 2.5), "ci_hi": np.percentile(d, 97.5), "mean_cross": sc.sum() / nc.sum(),
            "mean_same": ss.sum() / ns.sum(), "n_cross": int(nc.sum()), "n_same": int(ns.sum()), "n_threads": T}


def logit_odds(d: pd.DataFrame, y: str = "disagree_argmax") -> dict:
    """Logistic regression y ~ cross + week fixed effects, SEs clustered on thread. Returns the odds ratio of `cross`."""
    import statsmodels.api as sm
    X = pd.concat([d[["cross"]].astype(float), pd.get_dummies(d.week, drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    r = sm.GLM(d[y].astype(float), X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.thread)[0]})
    lo, hi = np.exp(r.conf_int().loc["cross"])
    return {"odds_ratio": float(np.exp(r.params["cross"])), "or_lo": float(lo), "or_hi": float(hi), "p": float(r.pvalues["cross"]), "n": int(len(d))}


def weekly_antagonism_alignment(lb: pd.DataFrame, wcol: str, n_perm: int = 200, seed: int = 0) -> pd.DataFrame:
    """Per week among LB edges: antagonism = share with w<0 (edges with undefined sign dropped); alignment = share of negative edges
    that are cross-camp minus the same share under label permutations among that week's labelled users."""
    rng, rows = np.random.default_rng(seed), []
    for wk, g in lb.groupby("week"):
        g = g[g[wcol].notna()]
        neg = (g[wcol] < 0).to_numpy()
        users, ui = np.unique(np.r_[g.source.to_numpy(), g.target.to_numpy()], return_inverse=True)
        s_i, t_i = ui[: len(g)], ui[len(g):]
        lab = np.zeros(len(users), dtype=int)
        lab[s_i], lab[t_i] = (g.lab_replier == "rep").to_numpy().astype(int), (g.lab_parent == "rep").to_numpy().astype(int)
        obs = (lab[s_i] != lab[t_i])[neg].mean() if neg.any() else np.nan
        null = np.array([(p[s_i] != p[t_i])[neg].mean() for p in (rng.permutation(lab) for _ in range(n_perm))]) if neg.any() else np.array([np.nan])
        rows.append({"week": wk, "n_lb_edges": len(g), "antagonism": neg.mean(), "share_neg_cross": obs, "share_neg_cross_null": null.mean(),
                     "alignment_excess": obs - null.mean(), "share_all_cross": float((lab[s_i] != lab[t_i]).mean())})
    return pd.DataFrame(rows)

"""Job R2, gt2_transfer: ground truth #2 by transfer from the supporter subs to the neutral subs.

Phases: ``neutral`` (phrases that occur in the neutral subs, user sides and the 5 user piles), ``extract`` (per week,
supporter-sub phrase counts of labelled users), ``merge`` (one users x phrases matrix), ``fit`` (per held-out pile:
phrase leanings and a logistic model for each of the three user weightings, plus sanity accuracy), ``score`` (per week,
neutral-sub users scored by the model that never saw them, on the full week and the 20 / 5 / 1% samples) and
``finalise`` (tables, plots, ``results/gt2_transfer/summary.md``, ``user_scores.parquet``).

Neutral-sub phrase counts of the full week and the samples come from the shared-venue triplet cache that the
GT#2 fix exploration wrote (``results/gt2_fix_options/cache``: labelled users, heavy posters and contraction-filter
phrases already removed). Existing outputs are not overwritten.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from scipy import sparse

from src.gt_runs import transfer as T
from src.gt_runs.cells import PhraseWeek, keep_phrase_mask, read_phrase_week
from src.gt_runs.config import gt_cfg, gt_path, labels_cfg, venues
from src.gt_runs.labels import AuthorIndex, heavy_set, load_side_map
from src.prep_jobs.config import project_root, runs_dir
from src.prep_jobs.data import party_map
from src.prep_jobs.jobs.clean_edges_and_sampling import weeks_of
from src.prep_jobs.jobs.phrase_counts import load_authors
from src.prep_jobs.util import read_json, write_json, write_parquet

NAME = "gt2_transfer"
DEADLINE_CUTS: list[str] = []
SOURCES = ["full", "r0.2", "r0.05", "r0.01"]
WNAMES = list(T.WEIGHTINGS)


def work(cfg) -> Path:
    d = runs_dir(cfg) / NAME / "work"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir(g) -> Path:
    return project_root() / "results" / "gt2_fix_options" / "cache"


def out_dir() -> Path:
    return project_root() / "results" / "gt2_transfer"


def phases(cfg):
    mod = "src.prep_jobs.jobs.gt2_transfer"
    weeks = weeks_of(cfg)
    k = gt_cfg()["gt2_transfer"]["n_folds"]
    return [
        {"name": "neutral", "fn": f"{mod}:neutral_unit", "units": [{"id": "neutral", "weight": 50}], "workers": 1},
        {"name": "extract", "fn": f"{mod}:extract_unit", "units": [{"id": w, "weight": 100} for w in weeks], "needs_all_prior": True},
        {"name": "merge", "fn": f"{mod}:merge_unit", "units": [{"id": "merge", "weight": 300}], "workers": 1, "needs_all_prior": True},
        {"name": "fit", "fn": f"{mod}:fit_unit", "units": [{"id": str(i), "weight": 600} for i in range(k)],
         "workers": gt_cfg()["gt2_transfer"]["fit_workers"], "needs_all_prior": True},
        {"name": "score", "fn": f"{mod}:score_unit", "units": [{"id": w, "weight": 100} for w in weeks], "needs_all_prior": True},
        {"name": "finalise", "fn": f"{mod}:finalise", "units": [{"id": "finalise", "weight": 100}], "workers": 1, "needs_all_prior": True},
    ]


# ------------------------------------------------------------------ neutral
def neutral_unit(ctx, unit):
    cfg, g = ctx.cfg, gt_cfg()
    p = g["gt2_transfer"]
    cdir = cache_dir(g)
    for s in SOURCES:
        missing = [w for w in weeks_of(cfg) if not (cdir / s / f"{w}.npz").exists()]
        if missing:
            raise FileNotFoundError(f"cache {s} is missing weeks {missing[:3]}...")
    ph = [np.unique(np.load(cdir / "full" / f"{w}.npz")["p"]) for w in weeks_of(cfg)]
    neutral = np.unique(np.concatenate(ph))
    keep, _ = keep_phrase_mask(cfg, g["gt2"]["exclude_tokens"])
    neutral = neutral[keep[neutral]]
    authors, _ = load_authors(cfg)
    aidx = AuthorIndex(authors)
    root = gt_path(g, "labels")
    main = aidx.side_vector(load_side_map(root, "shared", "main"))
    gold = aidx.side_vector(load_side_map(root, "shared", "gold"))
    pile = np.random.default_rng(g["seed"]).integers(0, p["n_folds"], len(authors)).astype(np.int8)
    np.save(work(cfg) / "neutral.npy", neutral)
    np.savez(work(cfg) / "sides.npz", main=main, gold=gold, pile=pile)
    ctx.log(f"{len(neutral):,} phrases in the neutral subs; {int((main >= 0).sum()):,} main-labelled, {int((gold >= 0).sum()):,} gold users")
    return {"rows": len(neutral), "bytes": 0}


def lut_for(cfg, n_vocab):
    neutral = np.load(work(cfg) / "neutral.npy")
    lut = np.full(n_vocab, -1, dtype=np.int32)
    lut[neutral] = np.arange(len(neutral), dtype=np.int32)
    return lut, len(neutral)


# ------------------------------------------------------------------ extract
def extract_unit(ctx, unit):
    cfg, g, lab, week = ctx.cfg, gt_cfg(), labels_cfg(), unit["id"]
    keep, _ = keep_phrase_mask(cfg, g["gt2"]["exclude_tokens"])
    lut, _ = lut_for(cfg, len(keep))
    s = np.load(work(cfg) / "sides.npz")
    labelled = (s["main"] >= 0) | (s["gold"] >= 0)
    aidx = AuthorIndex(load_authors(cfg)[0])
    heavy = heavy_set(gt_path(g, "labels"), week) if lab["exclude_heavy_posters"] else None
    heavy_vec = aidx.flag_vector(heavy) if heavy else None
    supporter = sorted(party_map(cfg))
    pw = PhraseWeek(read_phrase_week(cfg, week), aidx)
    a, p, c = pw.rows(supporter)
    ok = labelled[a] & keep[p] & (lut[p] >= 0)
    if heavy_vec is not None:
        ok &= ~heavy_vec[a]
    a, col, c = a[ok], lut[p[ok]], c[ok]
    M = sparse.coo_matrix((c.astype(np.float32), (a, col)), shape=(len(labelled), int(lut.max()) + 1)).tocsr()
    tot = np.asarray(M.sum(axis=1)).ravel()
    for tag, cap in (("raw", None), ("cap100", 100.0), ("cap300", 300.0)):
        X = M if cap is None else sparse.diags(np.minimum(1.0, cap / np.maximum(tot, 1.0)).astype(np.float32)) @ M
        sparse.save_npz(work(cfg) / f"extract_{tag}_{week}.npz", sparse.csr_matrix(X, dtype=np.float32), compressed=False)
    return {"rows": int(M.nnz), "bytes": 0}


def merge_unit(ctx, unit):
    cfg = ctx.cfg
    weeks = weeks_of(cfg)
    for ti, tag in enumerate(("raw", "cap100", "cap300")):
        M = None
        for i in range(0, len(weeks), 6):
            part = None
            for w in weeks[i:i + 6]:
                x = sparse.load_npz(work(cfg) / f"extract_{tag}_{w}.npz")
                part = x if part is None else part + x
            M = part if M is None else M + part
            ctx.report(fraction=(ti + min(0.9, i / len(weeks))) / 3)
            gc.collect()
        M.sum_duplicates()
        sparse.save_npz(work(cfg) / f"train_{tag}.npz", M, compressed=False)
        for w in weeks:
            (work(cfg) / f"extract_{tag}_{w}.npz").unlink()
        ctx.log(f"training matrix {tag} {M.shape}, {M.nnz:,} non-zeros, {int((np.diff(M.indptr) > 0).sum()):,} users with text")
        del M
    return {"rows": 0, "bytes": 0}


# ------------------------------------------------------------------ fit
def fit_unit(ctx, unit):
    cfg, g = ctx.cfg, gt_cfg()
    p = g["gt2_transfer"]
    k = int(unit["id"])
    s = np.load(work(cfg) / "sides.npz")
    rng = np.random.default_rng(g["seed"] + k)
    out, sanity = {}, []
    for w in WNAMES:
        M = sparse.load_npz(work(cfg) / f"train_{'raw' if w == 'equal' else w}.npz").tocsr()
        raw = sparse.load_npz(work(cfg) / "train_raw.npz").tocsr() if w != "equal" else M
        tot = np.asarray(raw.sum(axis=1)).ravel()   # phrases a user wrote in the supporter subs (uncapped)
        has = tot >= p["min_train_phrases"]
        train = np.flatnonzero(has & (s["main"] >= 0) & (s["pile"] != k))
        held = np.flatnonzero(has & (s["pile"] == k) & ((s["main"] >= 0) | (s["gold"] >= 0)))
        y = s["main"][train].astype(int)
        Xtr = M[train]
        prof = T.profile_matrix(Xtr, w)
        q = T.leanings(prof, y)
        Ftr = T.frequencies(Xtr)
        est_tr = Ftr @ q
        thr = 0.5 * (est_tr[y == 1].mean() + est_tr[y == 0].mean())
        sw = T.user_weights(Xtr, w)
        coef, b, C = T.fit_logit(Ftr * T.FEATURE_SCALE, y, sw, rng, grid=tuple(p["logit_grid"]), cv_users=p["logit_cv_users"])
        out[f"q_{w}"], out[f"coef_{w}"], out[f"meta_{w}"] = q.astype(np.float32), coef, np.array([b, C, thr])
        Fh = T.frequencies(raw[held])
        est_h = Fh @ q
        dec_h = (Fh * T.FEATURE_SCALE) @ coef + b
        for lname, labels in (("main", s["main"][held]), ("gold", s["gold"][held])):
            m = labels >= 0
            for inst, score, t in (("phrase", est_h, thr), ("logit", dec_h, 0.0)):
                yy = labels[m].astype(int)
                sanity.append({"fold": k, "weighting": w, "instrument": inst, "labels": lname, "n": int(m.sum()),
                               "n_rep": int((yy == 1).sum()), "auc": T.auc_value(score[m], yy),
                               "accuracy": T.accuracy(score[m], yy, t),
                               "balanced_accuracy": 0.5 * (float((score[m][yy == 1] > t).mean()) + float((score[m][yy == 0] <= t).mean())),
                               "C": C if inst == "logit" else np.nan, "mean_phrases": float(tot[held][m].mean())})
        del M, raw, Xtr, prof, Ftr
        gc.collect()
        ctx.report(fraction=(WNAMES.index(w) + 1) / len(WNAMES))
    np.savez(work(cfg) / f"model_{k}.npz", **out)
    write_json(sanity, work(cfg) / f"sanity_{k}.json")
    return {"rows": len(sanity), "bytes": 0}


# ------------------------------------------------------------------ score
def week_users(cfg, cdir, source, week, lut, ncol, min_phrases):
    x = np.load(cdir / source / f"{week}.npz")
    a, p, c = x["a"].astype(np.int64), x["p"].astype(np.int64), x["c"].astype(np.float32)
    col = lut[p]
    ok = col >= 0
    a, col, c = a[ok], col[ok], c[ok]
    users, inv = np.unique(a, return_inverse=True)
    X = sparse.csr_matrix((c, (inv, col)), shape=(len(users), ncol))
    tot = np.asarray(X.sum(axis=1)).ravel()
    good = tot >= min_phrases
    return users[good], X[good], tot[good]


def score_unit(ctx, unit):
    cfg, g, week = ctx.cfg, gt_cfg(), unit["id"]
    p = g["gt2_transfer"]
    keep, _ = keep_phrase_mask(cfg, g["gt2"]["exclude_tokens"])
    lut, ncol = lut_for(cfg, len(keep))
    s = np.load(work(cfg) / "sides.npz")
    models = [np.load(work(cfg) / f"model_{k}.npz") for k in range(p["n_folds"])]
    rng = np.random.default_rng(g["seed"] + 1000 + int(week.replace("-", "")))
    min_users = labels_cfg()["min_users_per_side"]
    rows, user_rows = [], None
    for src in SOURCES:
        users, X, tot = week_users(cfg, cache_dir(g), src, week, lut, ncol, g["gt2"]["min_phrases"])
        pile = s["pile"][users]
        F = T.frequencies(X)
        scores = {}
        Fs = F * T.FEATURE_SCALE
        for w in WNAMES:
            est, dec = np.zeros(len(users)), np.zeros(len(users))
            for k, m in enumerate(models):
                sel = np.flatnonzero(pile == k)
                if len(sel):
                    est[sel] = F[sel] @ m[f"q_{w}"]
                    dec[sel] = Fs[sel] @ m[f"coef_{w}"] + m[f"meta_{w}"][0]
            scores[("phrase", w)], scores[("logit", w)] = est, dec
        for lname in ("main", "gold"):
            side = s[lname][users]
            m = side >= 0
            for (inst, w), v in scores.items():
                kind = "pi" if inst == "phrase" else "auc"
                r = T.weekly_stat(v[m], side[m].astype(int), kind, rng, p["n_perm"], p["n_boot"] if lname == "main" else 0, min_users)
                rows.append({"week_start": week, "source": src, "labels": lname, "instrument": inst, "weighting": w,
                             "mean_phrases": float(tot[m].mean()) if m.any() else np.nan, **r})
        if src == "full":
            authors = load_authors(cfg)[0]
            user_rows = pd.DataFrame({"week_start": week, "author": [authors[i] for i in users], "n_phrases": tot,
                                      "side_main": s["main"][users], "side_gold": s["gold"][users], "pile": pile,
                                      **{f"{inst}_{w}": (1 / (1 + np.exp(-v)) if inst == "logit" else v)
                                         for (inst, w), v in scores.items()}})
            del authors
        del X, F, Fs
        gc.collect()
    user_rows.to_parquet(work(cfg) / f"users_{week}.parquet")
    write_json([{k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()} for r in rows],
               work(cfg) / f"rows_{week}.json")
    return {"rows": len(rows), "bytes": 0}


def finalise(ctx, unit):
    from src.prep_jobs.jobs import gt2_transfer_report as rep

    cfg, g = ctx.cfg, gt_cfg()
    out = out_dir()
    out.mkdir(parents=True, exist_ok=True)
    weeks = weeks_of(cfg)
    res = pd.DataFrame([r for w in weeks for r in read_json(work(cfg) / f"rows_{w}.json")])
    sanity = pd.DataFrame([r for k in range(g["gt2_transfer"]["n_folds"]) for r in read_json(work(cfg) / f"sanity_{k}.json")])
    users = pd.concat([pd.read_parquet(work(cfg) / f"users_{w}.parquet") for w in weeks], ignore_index=True)
    size = write_parquet(pa.Table.from_pandas(users, preserve_index=False), out / "user_scores.parquet")
    res.to_parquet(out / "weekly_results.parquet")
    sanity.to_csv(out / "sanity_by_fold.csv", index=False)
    rep.make_report(res, sanity, users, g, out, ctx.log)
    return {"rows": len(res), "bytes": size}

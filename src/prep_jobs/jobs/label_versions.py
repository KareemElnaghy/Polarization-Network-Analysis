"""Job G, label_versions: compare labelling rules with numbers (no rule is chosen here).

Counts each user's comments per supporter-community subreddit under five versions (all, score_ge_1,
score_ge_2, above_sub_median, weighted), then for every version x min_comments x share reports how many
users are labelled, coverage among supporter-sub posters and all users, and agreement with
self-declarations; agreement and flips between versions; and a direct test of whether the score rules remove
the users whose label disagrees with what they say.
"""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.prep_jobs.config import data_path, job_cfg, runs_dir
from src.prep_jobs.data import (declared_lean, iter_comments, label_counts, median_from_histogram, month_rows,
                                party_map, score_histogram, supporter_subs, version_columns)
from src.prep_jobs.util import read_json, sum_by_keys, write_json, write_parquet

NAME = "label_versions"
DEADLINE_CUTS: list[str] = []
VERSION_COLS = {"all": "n_all", "score_ge_1": "n_ge1", "score_ge_2": "n_ge2", "above_sub_median": "n_above_median",
                "weighted": "w_log"}


def work(cfg):
    return runs_dir(cfg) / NAME / "work"


def phases(cfg):
    mod = "src.prep_jobs.jobs.label_versions"
    rows = {m: month_rows(cfg, m) for m in cfg["months"]}
    total = sum(rows.values())
    months = lambda: [{"id": m, "weight": rows[m]} for m in cfg["months"]]  # noqa: E731
    return [
        {"name": "scores", "fn": f"{mod}:scores_month", "units": months()},
        {"name": "medians", "fn": f"{mod}:medians", "units": [{"id": "medians", "weight": total * 0.01}], "workers": 1,
         "needs_all_prior": True},
        {"name": "counts", "fn": f"{mod}:counts_month", "units": months(), "needs_all_prior": True},
        {"name": "finalise", "fn": f"{mod}:finalise", "units": [{"id": "finalise", "weight": total * 0.05}], "workers": 1,
         "needs_all_prior": True},
    ]


def scores_month(ctx, unit):
    subs = supporter_subs(ctx.cfg, ctx.jcfg)
    stats: Counter = Counter()
    hist = score_histogram(ctx.cfg, unit["id"], subs, ctx.batch_rows, stats)
    write_json({s: {str(k): v for k, v in c.items()} for s, c in hist.items()}, work(ctx.cfg) / f"hist_{unit['id']}.json")
    return {"rows": stats["rows"], "bytes": 0, "kept": stats["kept"]}


def medians(ctx, unit):
    total: dict[str, Counter] = {}
    for f in sorted(work(ctx.cfg).glob("hist_*.json")):
        for s, c in read_json(f).items():
            total.setdefault(s, Counter()).update({int(k): v for k, v in c.items()})
    med = {s: median_from_histogram(c) for s, c in total.items()}
    write_json(med, work(ctx.cfg) / "medians.json")
    return {"rows": len(med), "bytes": 0}


def counts_month(ctx, unit):
    cfg, month = ctx.cfg, unit["id"]
    subs = supporter_subs(cfg, ctx.jcfg)
    med = read_json(work(cfg) / "medians.json")
    stats: Counter = Counter()
    parts, keys = [], ["author", "subreddit"]
    sums = list(VERSION_COLS.values())
    for t in iter_comments(cfg, month, ["author", "subreddit", "score"], ctx.batch_rows, stats):
        t = t.set_column(1, "subreddit", pc.cast(t["subreddit"], pa.string()))
        t = t.filter(pc.is_in(t["subreddit"], pa.array(subs)))
        if not t.num_rows:
            continue
        cols = version_columns(t["score"], t["subreddit"], med)
        batch = pa.table({"author": t["author"], "subreddit": t["subreddit"],
                          **{VERSION_COLS[v]: cols[v] for v in VERSION_COLS}})
        parts.append(sum_by_keys(batch, keys, sums))
        if len(parts) >= 8 or ctx.over_limit():
            parts = [sum_by_keys(pa.concat_tables(parts), keys, sums)]
        ctx.report(rows=stats["rows"], fraction=min(0.99, stats["rows"] / unit["weight"]))
    table = sum_by_keys(pa.concat_tables(parts), keys, sums) if parts else None
    size = write_parquet(table, work(cfg) / f"counts_{month}.parquet") if table is not None else 0
    return {"rows": stats["rows"], "bytes": size, "kept": stats["kept"]}


# ------------------------------------------------------------------ analysis
def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else (f"{v:,}" if isinstance(v, (int, np.integer)) else str(v))
                 for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def per_user_party(counts: pd.DataFrame, party: dict[str, str]) -> dict[str, pd.DataFrame]:
    """For each version: DataFrame indexed by author with rep and dem totals."""
    counts = counts.assign(party=counts["subreddit"].map(party))
    out = {}
    for v, col in VERSION_COLS.items():
        p = counts.pivot_table(index="author", columns="party", values=col, aggfunc="sum", fill_value=0)
        out[v] = p.reindex(columns=["rep", "dem"], fill_value=0)
    return out


def finalise(ctx, unit):
    cfg, jc = ctx.cfg, ctx.jcfg
    res = data_path(cfg, "results") / "label_versions"
    res.mkdir(parents=True, exist_ok=True)
    counts = sum_by_keys(pa.concat_tables(pq.read_table(f) for f in sorted(work(cfg).glob("counts_*.parquet"))),
                         ["author", "subreddit"], list(VERSION_COLS.values()))
    write_parquet(counts, data_path(cfg, "label_versions") / "user_counts.parquet")
    df = counts.to_pandas()
    party = party_map(cfg)
    per_version = per_user_party(df, party)
    posters = df.loc[df["n_all"] > 0, "author"].nunique()
    n_all_users = pq.ParquetFile(data_path(cfg, "user_profiles") / "user_summary.parquet").metadata.num_rows
    decl = declared_lean(cfg)
    declared = pd.Series(dict(zip(decl["author"].to_pylist(), decl["declared"].to_pylist()))) if decl is not None else pd.Series(dtype=str)
    clear_decl = declared[declared.isin(["left", "right"])]

    def agreement(labels: pd.Series, other: pd.Series | None = None):
        ref = labels if other is None else other
        common = clear_decl.index.intersection(labels.index[labels.isin(["left", "right"])])
        if other is not None:
            common = common.intersection(other.index[other.isin(["left", "right"])])
        n = len(common)
        agree = int((clear_decl.loc[common] == ref.loc[common]).sum()) if n else 0
        return n, agree, (agree / n if n else float("nan"))

    rows, labels_default = [], {}
    for v in jc["versions"]:
        pv = per_version[v]
        for mc in jc["min_comments_grid"]:
            for sh in jc["share_grid"]:
                lab = label_counts(pv["rep"], pv["dem"], mc, sh)["label"]
                vc = lab.value_counts()
                n_decl, n_agree, rate = agreement(lab)
                labelled = int(vc.get("left", 0) + vc.get("right", 0) + vc.get("mixed", 0))
                rows.append({"version": v, "min_comments": mc, "share": sh, "left": int(vc.get("left", 0)),
                             "right": int(vc.get("right", 0)), "mixed": int(vc.get("mixed", 0)), "labelled": labelled,
                             "share_of_supporter_posters": labelled / posters, "share_of_all_users": labelled / n_all_users,
                             "declarers_clear": n_decl, "agree": n_agree, "agreement_rate": rate})
                if mc == jc["default_min_comments"] and sh == jc["default_share"]:
                    labels_default[v] = lab
    grid = pd.DataFrame(rows)
    grid.to_csv(res / "grid.csv", index=False)

    # agreement and flips between versions (default cell)
    pairs = []
    for a, b in combinations(jc["versions"], 2):
        la, lb = labels_default[a], labels_default[b]
        idx = la.index.union(lb.index)
        la, lb = la.reindex(idx, fill_value="unlabelled"), lb.reindex(idx, fill_value="unlabelled")
        both = (la != "unlabelled") & (lb != "unlabelled")
        flips = both & la.isin(["left", "right"]) & lb.isin(["left", "right"]) & (la != lb)
        pairs.append({"a": a, "b": b, "labelled_in_both": int(both.sum()), "same_label": int((la == lb)[both].sum()),
                      "same_rate": float((la == lb)[both].mean()), "left_right_flips": int(flips.sum()),
                      "only_a": int(((la != "unlabelled") & (lb == "unlabelled")).sum()),
                      "only_b": int(((la == "unlabelled") & (lb != "unlabelled")).sum())})
    pd.DataFrame(pairs).to_csv(res / "version_agreement.csv", index=False)

    # direct test of the score rule: do removed/changed users agree less with their declarations?
    base = labels_default["all"]
    test = []
    for v in jc["versions"]:
        if v == "all":
            continue
        lv = labels_default[v].reindex(base.index, fill_value="unlabelled")
        lab_all = base != "unlabelled"
        groups = {"kept (same label)": lab_all & (lv == base),
                  "dropped (unlabelled in version)": lab_all & (lv == "unlabelled"),
                  "changed (other label)": lab_all & (lv != "unlabelled") & (lv != base)}
        for name, mask in groups.items():
            idx = base.index[mask]
            n, agree, rate = agreement(base.loc[idx])
            row = {"version": v, "group": name, "users": int(mask.sum()), "declarers_clear": n,
                   "agree_with_all_label": agree, "agreement_rate_all_label": rate}
            if name.startswith("changed"):
                n2, a2, r2 = agreement(lv.loc[idx])
                row.update({"agree_with_version_label": a2, "agreement_rate_version_label": r2})
            test.append(row)
    pd.DataFrame(test).to_csv(res / "score_rule_test.csv", index=False)

    # coverage and comment-count distributions at the default cell
    n_all_by_user = per_version["all"].sum(axis=1)
    cov = []
    for v in jc["versions"]:
        lab = labels_default[v]
        isl = lab != "unlabelled"
        for name, mask in (("labelled", isl), ("unlabelled", ~isl)):
            q = n_all_by_user.reindex(lab.index)[mask].quantile([0.25, 0.5, 0.75, 0.9, 0.99])
            cov.append({"version": v, "group": name, "users": int(mask.sum()),
                        **{f"comments_p{int(p * 100)}": float(x) for p, x in q.items()}})
    pd.DataFrame(cov).to_csv(res / "coverage.csv", index=False)

    default = grid[(grid.min_comments == jc["default_min_comments"]) & (grid.share == jc["default_share"])]
    sub_n = len(supporter_subs(cfg, jc))
    md = [f"# Label versions (generated by the label_versions job)", "",
          f"{sub_n} supporter-community subreddits; {posters:,} users posted in them; {n_all_users:,} users overall; "
          f"{len(clear_decl):,} users with a clear left/right self-declaration. Weighted counts use ln(1+max(score,0)), "
          f"so its min_comments threshold is on that scale, not on comments. No rule is chosen here.", "",
          f"## Default cell (min_comments {jc['default_min_comments']}, share {jc['default_share']})", "",
          md_table(default.drop(columns=["min_comments", "share"])), "",
          "## Full grid", "", "See `grid.csv`. Agreement and coverage by cell:", "",
          md_table(grid[["version", "min_comments", "share", "labelled", "share_of_supporter_posters",
                         "declarers_clear", "agreement_rate"]]), "",
          "## Versions against each other (default cell)", "", md_table(pd.DataFrame(pairs)), "",
          "## Do the score rules remove the wrong users?", "",
          "Users labelled by `all` that a version drops or changes, against users it keeps; agreement is with "
          "self-declarations, computed on declarers only (counts shown, they are small).", "",
          md_table(pd.DataFrame(test)), "",
          "## Comment counts of labelled vs unlabelled users (default cell)", "", md_table(pd.DataFrame(cov), "{:.1f}")]
    (res / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return {"rows": len(grid), "bytes": 0}

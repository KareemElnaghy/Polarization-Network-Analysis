"""Job E, within_camp_setup: label users as Sanders-side or Clinton-side for the within-camp test.

Leaning = share of a user's comments in the Sanders subs among comments in Sanders + Clinton subs, counted
under each version (all, score_ge_1, score_ge_2, above_sub_median); classified at the configured
thresholds (not final). The special subs are not used for labelling. Then a light pass counts, per week,
how many labelled users of each side were active in the venue subs and overall.
"""

from __future__ import annotations

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.prep_jobs.config import data_path, job_cfg, runs_dir  # noqa: E402
from src.prep_jobs.data import (iter_comments, median_from_histogram, month_rows, score_histogram,  # noqa: E402
                                version_columns)
from src.prep_jobs.util import read_json, sum_by_keys, week_label, with_week_start, write_json, write_parquet  # noqa: E402

NAME = "within_camp_setup"
DEADLINE_CUTS: list[str] = []


def work(cfg):
    return runs_dir(cfg) / NAME / "work"


def camp_subs(jc):
    return jc["sanders_subs"] + jc["clinton_subs"]


def phases(cfg):
    jc = job_cfg(cfg, NAME)
    mod = "src.prep_jobs.jobs.within_camp_setup"
    rows = {m: month_rows(cfg, m) for m in cfg["months"]}
    total = sum(rows.values())
    months = lambda: [{"id": m, "weight": rows[m]} for m in cfg["months"]]  # noqa: E731
    one = lambda name: [{"id": name, "weight": total * 0.02}]  # noqa: E731
    return [
        {"name": "scores", "fn": f"{mod}:scores_month", "units": months()},
        {"name": "medians", "fn": f"{mod}:medians", "units": one("medians"), "workers": 1, "needs_all_prior": True},
        {"name": "counts", "fn": f"{mod}:counts_month", "units": months(), "needs_all_prior": True},
        {"name": "labels", "fn": f"{mod}:labels", "units": one("labels"), "workers": 1, "needs_all_prior": True},
        {"name": "weekly", "fn": f"{mod}:weekly_month", "units": months(), "needs_all_prior": True},
        {"name": "report", "fn": f"{mod}:report", "units": one("report"), "workers": 1, "needs_all_prior": True},
    ]


def scores_month(ctx, unit):
    stats: Counter = Counter()
    hist = score_histogram(ctx.cfg, unit["id"], camp_subs(ctx.jcfg), ctx.batch_rows, stats)
    write_json({s: {str(k): v for k, v in c.items()} for s, c in hist.items()}, work(ctx.cfg) / f"hist_{unit['id']}.json")
    return {"rows": stats["rows"], "bytes": 0}


def medians(ctx, unit):
    total: dict[str, Counter] = {}
    for f in sorted(work(ctx.cfg).glob("hist_*.json")):
        for s, c in read_json(f).items():
            total.setdefault(s, Counter()).update({int(k): v for k, v in c.items()})
    write_json({s: median_from_histogram(c) for s, c in total.items()}, work(ctx.cfg) / "medians.json")
    return {"rows": len(total), "bytes": 0}


def counts_month(ctx, unit):
    cfg, jc, month = ctx.cfg, ctx.jcfg, unit["id"]
    med = read_json(work(cfg) / "medians.json")
    side_of = {**{s: "sanders" for s in jc["sanders_subs"]}, **{s: "clinton" for s in jc["clinton_subs"]}}
    stats: Counter = Counter()
    versions = jc["count_versions"]
    parts, keys = [], ["author", "side"]
    for t in iter_comments(cfg, month, ["author", "subreddit", "score"], ctx.batch_rows, stats):
        t = t.set_column(1, "subreddit", pc.cast(t["subreddit"], pa.string()))
        t = t.filter(pc.is_in(t["subreddit"], pa.array(list(side_of))))
        if not t.num_rows:
            continue
        cols = version_columns(t["score"], t["subreddit"], med)
        side = pc.index_in(t["subreddit"], pa.array(list(side_of)))
        side_names = pa.array([side_of[s] for s in side_of])
        batch = pa.table({"author": t["author"], "side": pc.take(side_names, side), **{v: cols[v] for v in versions}})
        parts.append(sum_by_keys(batch, keys, versions))
        ctx.report(rows=stats["rows"], fraction=min(0.99, stats["rows"] / unit["weight"]))
    out = sum_by_keys(pa.concat_tables(parts), keys, versions) if parts else None
    size = write_parquet(out, work(cfg) / f"counts_{month}.parquet") if out is not None else 0
    return {"rows": stats["rows"], "bytes": size}


def labels(ctx, unit):
    cfg, jc = ctx.cfg, ctx.jcfg
    versions = jc["count_versions"]
    t = sum_by_keys(pa.concat_tables(pq.read_table(f) for f in sorted(work(cfg).glob("counts_*.parquet"))),
                    ["author", "side"], versions).to_pandas()
    users = pd.DataFrame(index=pd.Index(t["author"].unique(), name="author"))
    for v in versions:
        p = t.pivot_table(index="author", columns="side", values=v, aggfunc="sum", fill_value=0)
        p = p.reindex(columns=["sanders", "clinton"], fill_value=0).reindex(users.index, fill_value=0)
        n = p["sanders"] + p["clinton"]
        share = p["sanders"] / n.where(n > 0)
        lab = pd.Series("mixed", index=users.index)
        lab[share >= jc["sanders_share_at_least"]] = "sanders"
        lab[share <= jc["clinton_share_at_most"]] = "clinton"
        lab[~(n >= jc["min_comments"])] = "unlabelled"
        users[f"n_sanders_{v}"], users[f"n_clinton_{v}"] = p["sanders"].astype(int), p["clinton"].astype(int)
        users[f"share_{v}"], users[f"label_{v}"] = share, lab
    users = users.reset_index()
    out = data_path(cfg, "within_camp")
    write_parquet(pa.Table.from_pandas(users, preserve_index=False), out / "users.parquet")
    return {"rows": len(users), "bytes": 0}


def weekly_month(ctx, unit):
    cfg, jc, month = ctx.cfg, ctx.jcfg, unit["id"]
    users = pq.read_table(data_path(cfg, "within_camp") / "users.parquet", columns=["author"] + [
        f"label_{v}" for v in jc["count_versions"]])
    lab_cols = [f"label_{v}" for v in jc["count_versions"]]
    any_label = pa.array(np.zeros(users.num_rows, dtype=bool))
    for c in lab_cols:
        any_label = pc.or_(any_label, pc.is_in(users[c], pa.array(["sanders", "clinton"])))
    authors = users["author"].filter(any_label).combine_chunks()
    venue = pa.array(jc["venue_subs"])
    stats: Counter = Counter()
    parts, keys = [], ["week_start", "author"]
    for t in iter_comments(cfg, month, ["author", "subreddit", "created_utc"], ctx.batch_rows, stats):
        t = t.filter(pc.is_in(t["author"], authors))
        if not t.num_rows:
            continue
        t = with_week_start(t)
        sub = pc.cast(t["subreddit"], pa.string())
        batch = pa.table({"week_start": t["week_start"], "author": t["author"],
                          "politics": pc.cast(pc.equal(sub, jc["venue_subs"][0]), pa.int64()),
                          "pd": pc.cast(pc.equal(sub, jc["venue_subs"][1]), pa.int64()),
                          "any": pa.array(np.ones(t.num_rows, dtype=np.int64))})
        parts.append(sum_by_keys(batch, keys, ["politics", "pd", "any"]))
        if len(parts) >= 8:
            parts = [sum_by_keys(pa.concat_tables(parts), keys, ["politics", "pd", "any"])]
        ctx.report(rows=stats["rows"], fraction=min(0.99, stats["rows"] / unit["weight"]))
    out = sum_by_keys(pa.concat_tables(parts), keys, ["politics", "pd", "any"]) if parts else None
    size = write_parquet(out, work(cfg) / f"weekly_{month}.parquet") if out is not None else 0
    return {"rows": stats["rows"], "bytes": size}


def report(ctx, unit):
    cfg, jc = ctx.cfg, ctx.jcfg
    versions, primary = jc["count_versions"], jc["primary_version"]
    users = pq.read_table(data_path(cfg, "within_camp") / "users.parquet").to_pandas()
    res = data_path(cfg, "results") / "within_camp"
    res.mkdir(parents=True, exist_ok=True)

    # weekly counts
    w = sum_by_keys(pa.concat_tables(pq.read_table(f) for f in sorted(work(cfg).glob("weekly_*.parquet"))),
                    ["week_start", "author"], ["politics", "pd", "any"]).to_pandas()
    w["week_start"] = pd.to_datetime(w["week_start"]).dt.tz_localize(None).dt.date
    rows = []
    for v in versions:
        lab = users.set_index("author")[f"label_{v}"]
        ww = w.assign(side=w["author"].map(lab))
        ww = ww[ww["side"].isin(["sanders", "clinton"])]
        for (week, side), g in ww.groupby(["week_start", "side"]):
            rows.append({"week_start": week, "version": v, "side": side, "politics": int((g["politics"] > 0).sum()),
                         "PoliticalDiscussion": int((g["pd"] > 0).sum()),
                         "venue_either": int(((g["politics"] > 0) | (g["pd"] > 0)).sum()),
                         "total_any_subreddit": int((g["any"] > 0).sum())})
    weekly = pd.DataFrame(rows).sort_values(["version", "side", "week_start"])
    weekly.to_csv(res / "weekly_counts.csv", index=False)

    # plot (primary version)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    pw = weekly[weekly["version"] == primary]
    for ax, col, title in zip(axes, ["politics", "PoliticalDiscussion", "total_any_subreddit"],
                              ["r/politics", "r/PoliticalDiscussion", "any subreddit"]):
        for side, colour in (("sanders", "#c0392b"), ("clinton", "#2471a3")):
            s = pw[pw["side"] == side]
            ax.plot(pd.to_datetime(s["week_start"]), s[col], label=side, color=colour)
        for name, day in jc["calendar"].items():
            ax.axvline(pd.Timestamp(day), color="grey", ls="--", lw=0.8)
        ax.set_ylabel("labelled users active")
        ax.set_title(f"{title} ({primary})")
    for name, day in jc["calendar"].items():
        axes[0].text(pd.Timestamp(day), axes[0].get_ylim()[1], name, rotation=90, va="top", ha="right", fontsize=7)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(res / "weekly_counts.png", dpi=120)
    plt.close(fig)

    # cross-tab with the existing posting-based leaning, and version comparisons
    lines = [f"# Within-camp setup (generated)", "",
             f"Sanders subs {jc['sanders_subs']}, Clinton subs {jc['clinton_subs']}. Classified at share >= "
             f"{jc['sanders_share_at_least']} (sanders) / <= {jc['clinton_share_at_most']} (clinton), minimum "
             f"{jc['min_comments']} comments. **Thresholds are not final.** Primary version: {primary}.", "",
             f"{len(users):,} users commented in the camp subs.", "", "## Users per label and version", ""]
    cnt = pd.DataFrame({v: users[f"label_{v}"].value_counts() for v in versions}).fillna(0).astype(int)
    cnt.to_csv(res / "label_counts.csv")
    lines += [cnt.to_string(), ""]
    ref = pq.read_table(data_path(cfg, "user_profiles") / "user_summary.parquet", columns=["author", "posting_lean"]).to_pandas()
    scored_path = data_path(cfg, "user_profiles") / "user_summary_scored.parquet"
    refs = {"posting_lean (all party subs, >=5 comments)": ref.set_index("author")["posting_lean"]}
    if scored_path.exists():
        sc = pq.read_table(scored_path, columns=["author", "lean_accepted"]).to_pandas().set_index("author")
        refs["lean_accepted (58 supporter subs, >=10 comments, score>=1)"] = sc["lean_accepted"]
    for name, series in refs.items():
        x = pd.crosstab(users.set_index("author")[f"label_{primary}"], series.reindex(users["author"]).fillna("no_data").values)
        x.to_csv(res / f"crosstab_{name.split(' ')[0]}.csv")
        lines += [f"## {primary} label x {name}", "", x.to_string(), ""]
        left = "left" if "left" in x.columns else None
        if left:
            for side in ("sanders", "clinton"):
                if side in x.index:
                    lines.append(f"{side}-side users also classed dem (left): {int(x.loc[side, left]):,} of {int(x.loc[side].sum()):,}")
            lines.append("")
    lines += ["## Version against version", "",
              "`only_a`: labelled (sanders/clinton/mixed) under a but not b; `flips`: sanders under one, clinton under the other.", ""]
    cmp_rows = []
    for i, a in enumerate(versions):
        for b in versions[i + 1:]:
            la, lb = users[f"label_{a}"], users[f"label_{b}"]
            both = (la != "unlabelled") & (lb != "unlabelled")
            cmp_rows.append({"a": a, "b": b, "labelled_both": int(both.sum()), "same": int((la == lb)[both].sum()),
                             "only_a": int(((la != "unlabelled") & (lb == "unlabelled")).sum()),
                             "only_b": int(((la == "unlabelled") & (lb != "unlabelled")).sum()),
                             "flips": int(((la == "sanders") & (lb == "clinton") | (la == "clinton") & (lb == "sanders")).sum())})
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(res / "version_comparison.csv", index=False)
    lines += [cmp_df.to_string(index=False), ""]
    (res / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return {"rows": len(weekly), "bytes": 0}

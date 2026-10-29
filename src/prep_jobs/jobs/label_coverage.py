"""Job H, label_coverage: is the labelled pool big enough where we measure?

Labels come from ``label_versions/user_counts.parquet`` (the specs in ``label_specs``). Per week it counts
labelled left/right users active in the venue subs and in each supporter sub, the share of cleaned venue
edges with a labelled author or target, and labelled users inside each thread sample. Weeks where either
side has fewer than ``min_users`` labelled users in the venue are flagged. Nothing here chooses a rule.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.prep_jobs.config import data_path, job_cfg, runs_dir  # noqa: E402
from src.prep_jobs.data import label_counts, party_map, supporter_subs  # noqa: E402
from src.prep_jobs.jobs.clean_edges_and_sampling import weeks_of  # noqa: E402
from src.prep_jobs.jobs.label_versions import VERSION_COLS, md_table  # noqa: E402
from src.prep_jobs.util import read_json, write_json, write_parquet  # noqa: E402

NAME = "label_coverage"
DEADLINE_CUTS: list[str] = []
DEPENDS_ON = ["thread_user", "clean_edges_and_sampling", "label_versions"]  # label_versions added: it supplies the labels


def spec_name(s) -> str:
    return f"{s['version']}_{s['min_comments']}_{int(round(s['share'] * 100))}"


def work(cfg):
    return runs_dir(cfg) / NAME / "work"


def phases(cfg):
    jc = job_cfg(cfg, NAME)
    mod = "src.prep_jobs.jobs.label_coverage"
    return [
        {"name": "labels", "fn": f"{mod}:build_labels", "units": [{"id": "labels", "weight": 50}], "workers": 1},
        {"name": "weeks", "fn": f"{mod}:week_unit", "units": [{"id": w, "weight": 10} for w in weeks_of(cfg)],
         "needs_all_prior": True},
        {"name": "report", "fn": f"{mod}:report", "units": [{"id": "report", "weight": 30}], "workers": 1,
         "needs_all_prior": True},
    ]


def build_labels(ctx, unit):
    cfg, jc = ctx.cfg, ctx.jcfg
    party = party_map(cfg)
    df = pq.read_table(data_path(cfg, "label_versions") / "user_counts.parquet").to_pandas()
    df["party"] = df["subreddit"].map(party)
    out = {}
    for s in jc["label_specs"]:
        col = VERSION_COLS[s["version"]]
        p = df.pivot_table(index="author", columns="party", values=col, aggfunc="sum", fill_value=0)
        p = p.reindex(columns=["rep", "dem"], fill_value=0)
        out[spec_name(s)] = label_counts(p["rep"], p["dem"], s["min_comments"], s["share"])["label"]
    labels = pd.DataFrame(out).reset_index()
    write_parquet(pa.Table.from_pandas(labels, preserve_index=False), work(cfg) / "labels.parquet")
    return {"rows": len(labels), "bytes": 0}


def week_unit(ctx, unit):
    cfg, jc, week = ctx.cfg, ctx.jcfg, unit["id"]
    labels = pd.read_parquet(work(cfg) / "labels.parquet").set_index("author")
    venue = jc["venue_subs"]
    sup = supporter_subs(cfg, job_cfg(cfg, "label_versions"))
    row: dict = {"week_start": week}
    tu = pq.read_table(data_path(cfg, "thread_user") / f"week={week}" / "part.parquet",
                       columns=["subreddit", "author"]).to_pandas().drop_duplicates()
    ed = pq.read_table(data_path(cfg, "edges_clean") / f"week={week}" / "part.parquet",
                       columns=["source", "target", "subreddit"]).to_pandas()
    ed = ed[ed["subreddit"].isin(venue)]
    row["venue_clean_edges"] = len(ed)
    for s in jc["label_specs"]:
        name = spec_name(s)
        lab = labels[name]
        sides = lab.where(lab.isin(["left", "right"]))
        a = tu.assign(side=tu["author"].map(sides))
        for side in ("left", "right"):
            ids = a[a["side"] == side]
            for v in venue:
                row[f"{name}_{side}_{v}"] = int(ids[ids["subreddit"] == v]["author"].nunique())
            row[f"{name}_{side}_venue"] = int(ids[ids["subreddit"].isin(venue)]["author"].nunique())
            for sub in sup:
                row[f"{name}_{side}_sub_{sub}"] = int(ids[ids["subreddit"] == sub]["author"].nunique())
        ls, lt = ed["source"].map(sides).notna(), ed["target"].map(sides).notna()
        row[f"{name}_edge_share_any_labelled"] = float((ls | lt).mean()) if len(ed) else float("nan")
        row[f"{name}_edge_share_both_labelled"] = float((ls & lt).mean()) if len(ed) else float("nan")
        for rate in (r for r in cfg["jobs"]["clean_edges_and_sampling"]["sampling"]["rates"]):
            f = data_path(cfg, "edges_sampled") / f"rate={rate}" / f"week={week}" / "part.parquet"
            se = pq.read_table(f, columns=["source", "target", "subreddit"]).to_pandas()
            se = se[se["subreddit"].isin(venue)]
            users = pd.concat([se["source"], se["target"]]).drop_duplicates()
            sd = users.map(sides)
            for side in ("left", "right"):
                row[f"{name}_{side}_in_sample_{rate}"] = int((sd == side).sum())
    write_json(row, work(cfg) / "weeks" / f"{week}.json")
    return {"rows": len(tu) + len(ed), "bytes": 0}


def report(ctx, unit):
    cfg, jc = ctx.cfg, ctx.jcfg
    res = data_path(cfg, "results") / "label_coverage"
    res.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([read_json(f) for f in sorted((work(cfg) / "weeks").glob("*.json"))])
    for s in jc["label_specs"]:
        n = spec_name(s)
        df[f"{n}_flag_low"] = (df[f"{n}_left_venue"] < jc["min_users"]) | (df[f"{n}_right_venue"] < jc["min_users"])
    df.to_csv(res / "weekly.csv", index=False)
    rates = cfg["jobs"]["clean_edges_and_sampling"]["sampling"]["rates"]
    lines = ["# Label coverage by week (generated by the label_coverage job)", "",
             f"Labelled left/right users active in {jc['venue_subs']} per week; flagged when either side has fewer than "
             f"{jc['min_users']}. 'Active' = at least one counted comment (thread_user table). Edge shares use the "
             "cleaned edges in the venue subs. Sample counts are labelled users appearing as author or target in the "
             "venue edges of each thread sample.", ""]
    fig, axes = plt.subplots(len(jc["label_specs"]), 1, figsize=(10, 3.6 * len(jc["label_specs"])), sharex=True, squeeze=False)
    for ax, s in zip(axes[:, 0], jc["label_specs"]):
        n = spec_name(s)
        x = pd.to_datetime(df["week_start"])
        ax.plot(x, df[f"{n}_left_venue"], label="left (dem)", color="#2471a3")
        ax.plot(x, df[f"{n}_right_venue"], label="right (rep)", color="#c0392b")
        ax.axhline(jc["min_users"], color="grey", ls="--", lw=0.8)
        ax.set_title(f"{n}: labelled users active in politics + PoliticalDiscussion")
        ax.set_ylabel("users per week")
        ax.legend()
        low = df[df[f"{n}_flag_low"]]
        summary = {"weeks": len(df), "weeks_flagged": len(low),
                   "median_left_venue": float(df[f"{n}_left_venue"].median()), "median_right_venue": float(df[f"{n}_right_venue"].median()),
                   "min_left_venue": int(df[f"{n}_left_venue"].min()), "min_right_venue": int(df[f"{n}_right_venue"].min()),
                   "median_edge_share_any": float(df[f"{n}_edge_share_any_labelled"].median()),
                   "median_edge_share_both": float(df[f"{n}_edge_share_both_labelled"].median())}
        for r in rates:
            summary[f"median_left_in_{r}"] = float(df[f"{n}_left_in_sample_{r}"].median())
            summary[f"median_right_in_{r}"] = float(df[f"{n}_right_in_sample_{r}"].median())
        lines += [f"## {n}", "", md_table(pd.DataFrame([summary]), "{:.3f}"), "",
                  "Flagged weeks: " + (", ".join(low["week_start"]) if len(low) else "none"), ""]
    fig.tight_layout()
    fig.savefig(res / "labelled_users_over_time.png", dpi=120)
    plt.close(fig)
    (res / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return {"rows": len(df), "bytes": 0}

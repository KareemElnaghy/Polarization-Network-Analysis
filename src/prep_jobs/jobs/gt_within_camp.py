"""Job T4, gt_within_camp: both ground truths for Sanders-leaning vs Clinton-leaning users, per week.

Groups come from Job E (``within_camp/users.parquet``, count version ``within_camp.count_version``, default
score_ge_1): ``sanders`` and ``clinton`` labels only. Both measures are symmetric in the two groups, so which one
plays the "rep" slot is arbitrary (``side_for_sanders``). Venue: shared (r/politics + r/PoliticalDiscussion),
full week, heavy posters excluded, contraction filter on. Output ``results/ground_truth/within_camp_weekly.parquet``
and ``within_camp.png`` with the primary calendar.
"""

from __future__ import annotations

import gc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from src.gt_runs.cells import PhraseWeek, gt1_cell, gt2_cell, keep_phrase_mask, read_phrase_week, read_thread_user  # noqa: E402
from src.gt_runs.config import gt_cfg, gt_path, labels_cfg, venues  # noqa: E402
from src.gt_runs.labels import AuthorIndex, heavy_set  # noqa: E402
from src.prep_jobs.config import data_path, runs_dir  # noqa: E402
from src.prep_jobs.jobs.clean_edges_and_sampling import weeks_of  # noqa: E402
from src.prep_jobs.jobs.phrase_counts import load_authors  # noqa: E402
from src.prep_jobs.util import read_json, write_json, write_parquet  # noqa: E402

NAME = "gt_within_camp"
DEADLINE_CUTS: list[str] = []
CAL = ["Iowa", "New Hampshire", "California primary", "Sanders endorses Clinton"]


def phases(cfg):
    mod = "src.prep_jobs.jobs.gt_within_camp"
    return [
        {"name": "weeks", "fn": f"{mod}:week_unit", "units": [{"id": w, "weight": 100} for w in weeks_of(cfg)]},
        {"name": "finalise", "fn": f"{mod}:finalise", "units": [{"id": "finalise", "weight": 50}], "workers": 1,
         "needs_all_prior": True},
    ]


def camp_sides(cfg, g) -> dict[str, str]:
    col = f"label_{g['within_camp']['count_version']}"
    u = pq.read_table(data_path(cfg, "within_camp") / "users.parquet", columns=["author", col]).to_pandas()
    u = u[u[col].isin(["sanders", "clinton"])]
    sanders = g["within_camp"]["side_for_sanders"]
    other = "dem" if sanders == "rep" else "rep"
    return {a: (sanders if l == "sanders" else other) for a, l in zip(u["author"], u[col])}


def week_unit(ctx, unit):
    cfg, g, lab, week = ctx.cfg, gt_cfg(), labels_cfg(), unit["id"]
    subs = venues(g)["shared"]
    sides = camp_sides(cfg, g)
    heavy = heavy_set(gt_path(g, "labels"), week) if lab["exclude_heavy_posters"] else None
    tu = read_thread_user(cfg, week)
    one = gt1_cell(tu, subs, sides, heavy, None, g["gt1"]["n_perm"], g["seed"], lab["min_users_per_side"])
    del tu
    aidx = AuthorIndex(load_authors(cfg)[0])
    keep, _ = keep_phrase_mask(cfg, g["gt2"]["exclude_tokens"])
    pw = PhraseWeek(read_phrase_week(cfg, week), aidx)
    a, p, c = pw.rows(subs)
    p2 = g["gt2"]
    two = gt2_cell(a, p, c, aidx.side_vector(sides), keep, aidx.flag_vector(heavy) if heavy else None, len(keep),
                   p2["min_phrases"], p2["n_perm"], p2["n_boot"], g["seed"], lab["min_users_per_side"])
    row = {"week_start": week, "gt1_n_sanders": one["n_rep"] if g["within_camp"]["side_for_sanders"] == "rep" else one["n_dem"],
           "gt1_n_clinton": one["n_dem"] if g["within_camp"]["side_for_sanders"] == "rep" else one["n_rep"],
           "gt1_n_threads": one["n_threads"], "gt1_observed": one["observed"], "gt1_null_mean": one["null_mean"],
           "gt1_excess": one["excess"], "gt1_low_n": one["low_n"],
           "gt2_n_group_a": two["n_rep"], "gt2_n_group_b": two["n_dem"], "gt2_pi": two["pi"], "gt2_null_mean": two["null_mean"],
           "gt2_null_sd": two["null_sd"], "gt2_boot_lo": two["boot_lo"], "gt2_boot_hi": two["boot_hi"], "gt2_low_n": two["low_n"]}
    write_json(row, runs_dir(cfg) / NAME / "work" / f"{week}.json")
    del pw
    gc.collect()
    return {"rows": 1, "bytes": 0}


def finalise(ctx, unit):
    cfg, g = ctx.cfg, gt_cfg()
    df = pd.DataFrame([read_json(f) for f in sorted((runs_dir(cfg) / NAME / "work").glob("*.json"))])
    out = gt_path(g, "results")
    size = write_parquet(pa.Table.from_pandas(df, preserve_index=False), out / "within_camp_weekly.parquet")
    x = pd.to_datetime(df["week_start"])
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(x, df["gt1_excess"], color="#7d3c98")
    axes[0].set_ylabel("GT#1 excess (dissimilarity)")
    axes[1].plot(x, df["gt2_pi"], color="#117a65")
    axes[1].fill_between(x, df["gt2_boot_lo"], df["gt2_boot_hi"], alpha=0.25, color="#117a65")
    axes[1].axhline(0.5, color="grey", ls="--", lw=0.8)
    axes[1].set_ylabel("GT#2 pi (leave-out)")
    for ax in axes:
        for name in CAL:
            ax.axvline(pd.Timestamp(g["events"][name]), color="grey", ls=":", lw=0.9)
    for name in CAL:
        axes[0].text(pd.Timestamp(g["events"][name]), axes[0].get_ylim()[1], name, rotation=90, va="top", ha="right", fontsize=7)
    axes[0].set_title("Sanders-leaning vs Clinton-leaning users in r/politics + r/PoliticalDiscussion")
    fig.tight_layout()
    fig.savefig(out / "within_camp.png", dpi=120)
    plt.close(fig)
    return {"rows": len(df), "bytes": size}

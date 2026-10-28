"""Score check and score-aware posting-based leaning.

1. Score and retrieval-lag distributions (from ``score_lag_scan.py`` output) for party-tagged
   subreddits and r/politics.
2. Per user per subreddit counts three ways: all, accepted (score >= 1), log-weighted
   (ln(1 + max(score, 0))), from ``data/processed/user_profiles/`` (``user_profiles.py merge``).
3. Posting-based leaning from supporter-community party subreddits, one label per count version.
4. Self-declaration and linked-domain coverage, and their agreement with each version.

Usage:
    python experiments/phase2_data_pipeline_check/score_aware_labels.py --scan-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from user_signal_coverage import NEWS_DOMAINS, declared_leaning  # noqa: E402

ROOT = HERE.parents[1]
PROFILES = ROOT / "data" / "processed" / "user_profiles"
OUT = ROOT / "results" / "score_aware"
# Party-tagged subs that are not supporter communities (ask-the-other-side, anti, or cross-party).
EXCLUDED = {"AskTrumpSupporters", "enoughsandersspam", "RepublicansForSanders", "AskDemocrats", "AskThe_Donald"}
MIN_COUNT, RIGHT_AT, LEFT_AT = 10, 0.75, 0.25
VERSIONS = {"all": "n_comments", "accepted": "n_accepted", "weighted": "w_log"}
PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def party_map() -> dict[str, str]:
    out = {}
    for line in (ROOT / "data" / "raw" / "subreddits_metadata.json").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["party"]:
                out[row["subreddit"]] = row["party"]
    return out


# ------------------------------------------------------------------ 1. score and lag
def score_lag(scan_dir: Path, party: dict[str, str]) -> None:
    df = pd.concat(pd.read_parquet(f) for f in sorted(scan_dir.glob("scan_*.parquet")))
    df["lag_h"] = (df["retrieved_on"] - df["created_utc"]) / 3600
    df["group"] = np.where(df["subreddit"] == "politics", "politics",
                           np.where(df["subreddit"].isin(EXCLUDED), "party (non-supporter)", "party (supporter)"))
    groups = {"party-tagged, all": df[df.subreddit != "politics"],
              "party-tagged, supporter subs": df[df.group == "party (supporter)"],
              "politics": df[df.subreddit == "politics"]}
    big = df.groupby("subreddit").size()
    for sub in big[big >= 50_000].index:
        if sub != "politics":
            groups[sub] = df[df.subreddit == sub]
    rows = []
    for name, g in groups.items():
        s, lag = g["score"], g["lag_h"]
        row = {"group": name, "comments": len(g), "share_score_1": (s == 1).mean(), "share_le_0": (s <= 0).mean(),
               "share_ge_2": (s >= 2).mean(), "mean_score": s.mean()}
        row.update({f"score_p{p}": np.percentile(s, p) for p in PCTS})
        row.update({"retrieved_missing": (g["retrieved_on"] == 0).mean(), "lag_negative": (lag < 0).mean(),
                    "lag_share_lt_1h": (lag < 1).mean(), "lag_share_lt_1d": (lag < 24).mean(),
                    "lag_share_ge_7d": (lag >= 168).mean()})
        row.update({f"lag_h_p{p}": np.percentile(lag, p) for p in (10, 50, 90)})
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "score_lag_by_group.csv", index=False)
    print(tab[["group", "comments", "share_score_1", "share_le_0", "score_p50", "score_p90", "score_p99",
               "lag_h_p10", "lag_h_p50", "lag_h_p90", "lag_share_lt_1d", "lag_share_ge_7d"]].round(3).to_string(index=False))
    # does score depend on how long after posting the comment was retrieved?
    pol = df[df.retrieved_on > 0].copy()
    pol["lag_bucket"] = pd.cut(pol["lag_h"], [-1e9, 1, 6, 24, 168, 24 * 30, 1e9],
                               labels=["<1h", "1-6h", "6-24h", "1-7d", "7-30d", ">30d"])
    pol["scope"] = np.where(pol.subreddit == "politics", "politics", "party-tagged")
    b = pol.groupby(["scope", "lag_bucket"], observed=True).agg(
        comments=("score", "size"), share_score_1=("score", lambda s: (s == 1).mean()),
        median_score=("score", "median"), mean_score=("score", "mean"), p90_score=("score", lambda s: np.percentile(s, 90)))
    b.to_csv(OUT / "score_by_lag_bucket.csv")
    print("\n", b.round(3).to_string())
    # same, by month created (retrieval date can be much later for early months)
    pol["month"] = pd.to_datetime(pol["created_utc"], unit="s").dt.strftime("%Y-%m")
    m = pol.groupby(["scope", "month"]).agg(comments=("score", "size"), share_score_1=("score", lambda s: (s == 1).mean()),
                                            median_lag_h=("lag_h", "median"), median_score=("score", "median"))
    m.to_csv(OUT / "score_lag_by_month.csv")
    print("\n", m.round(3).to_string())


# ------------------------------------------------------------------ 3. labels
def label(rep: pd.Series, dem: pd.Series) -> pd.DataFrame:
    n = rep + dem
    share = rep / n.where(n > 0)
    lab = pd.Series("mixed", index=n.index)
    lab[share >= RIGHT_AT] = "right"
    lab[share <= LEFT_AT] = "left"
    lab[~(n >= MIN_COUNT)] = "unlabelled"
    return pd.DataFrame({"n": n, "rep_share": share, "label": lab})


def build_labels(act: pd.DataFrame, party: dict[str, str]) -> pd.DataFrame:
    sup = act[act["subreddit"].map(party).notna() & ~act["subreddit"].isin(EXCLUDED)]
    sup = sup.assign(party=sup["subreddit"].map(party))
    out = pd.DataFrame(index=pd.Index(sup["author"].unique(), name="author"))
    for v, col in VERSIONS.items():
        p = sup.pivot_table(index="author", columns="party", values=col, aggfunc="sum", fill_value=0)
        p = p.reindex(columns=["dem", "rep"], fill_value=0).reindex(out.index, fill_value=0)
        lab = label(p["rep"], p["dem"])
        out[f"n_{v}"] = lab["n"]
        out[f"rep_share_{v}"] = lab["rep_share"]
        out[f"lean_{v}"] = lab["label"]
    out["n_rep_all"] = sup[sup.party == "rep"].groupby("author")["n_comments"].sum().reindex(out.index, fill_value=0)
    out["n_dem_all"] = sup[sup.party == "dem"].groupby("author")["n_comments"].sum().reindex(out.index, fill_value=0)
    return out


def agreement(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in (("all", "accepted"), ("all", "weighted"), ("accepted", "weighted")):
        la, lb = lab[f"lean_{a}"], lab[f"lean_{b}"]
        both = (la != "unlabelled") & (lb != "unlabelled")
        clear = both & la.isin(["left", "right"]) & lb.isin(["left", "right"])
        flips = clear & (la != lb)
        rows.append({"pair": f"{a} vs {b}", "labelled_in_both": int(both.sum()),
                     "same_label": int((la == lb)[both].sum()), "same_label_rate": float((la == lb)[both].mean()),
                     "left_right_both": int(clear.sum()), "left_right_flips": int(flips.sum()),
                     "only_in_first": int(((la != "unlabelled") & (lb == "unlabelled")).sum()),
                     "only_in_second": int(((la == "unlabelled") & (lb != "unlabelled")).sum())})
    allthree = (lab[[f"lean_{v}" for v in VERSIONS]] != "unlabelled").all(axis=1)
    same3 = allthree & (lab["lean_all"] == lab["lean_accepted"]) & (lab["lean_all"] == lab["lean_weighted"])
    rows.append({"pair": "all three", "labelled_in_both": int(allthree.sum()), "same_label": int(same3.sum()),
                 "same_label_rate": float(same3.sum() / allthree.sum())})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-dir", type=Path, required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    party = party_map()
    print("== 1. score and lag")
    score_lag(args.scan_dir, party)

    act = pd.read_parquet(PROFILES / "user_subreddit_activity.parquet")
    decl = pd.read_parquet(PROFILES / "user_subreddit_declarations.parquet")
    dom = pd.read_parquet(PROFILES / "user_subreddit_domains.parquet")

    # 2. per user per subreddit three counts (+ declarations and links)
    d_sub = decl.groupby(["author", "subreddit"]).agg(
        n_declarations=("label", "size"), declared_labels=("label", lambda s: ",".join(sorted(set(s)))))
    l_sub = dom.groupby(["author", "subreddit"]).agg(n_links=("n_links", "sum"), n_domains=("domain", "nunique"))
    us = act.set_index(["author", "subreddit"]).join(d_sub).join(l_sub).reset_index()
    us[["n_declarations", "n_links", "n_domains"]] = us[["n_declarations", "n_links", "n_domains"]].fillna(0).astype(int)
    us.to_parquet(PROFILES / "user_subreddit_table_scored.parquet", index=False)

    # 3. labels
    print("\n== 3. supporter-community labels (min %d, %.0f%%)" % (MIN_COUNT, RIGHT_AT * 100))
    sup_subs = sorted(s for s in party if s not in EXCLUDED)
    present = act[act.subreddit.isin(sup_subs)]
    print(f"{len(sup_subs)} supporter subs; {present.author.nunique():,} users with any activity; "
          f"{present.n_comments.sum():,} comments")
    lab = build_labels(act, party)
    counts = pd.DataFrame({v: lab[f"lean_{v}"].value_counts() for v in VERSIONS}).reindex(
        ["left", "right", "mixed", "unlabelled"]).fillna(0).astype(int)
    counts.loc["labelled (left+right+mixed)"] = counts.loc[["left", "right", "mixed"]].sum()
    counts.loc["labelled left/right"] = counts.loc[["left", "right"]].sum()
    counts.to_csv(OUT / "label_counts.csv")
    print(counts.to_string())
    agr = agreement(lab)
    agr.to_csv(OUT / "label_agreement.csv", index=False)
    print(agr.round(4).to_string(index=False))
    # common population: same users (>= MIN_COUNT comments under 'all'), only the shares differ
    pop = lab[lab["n_all"] >= MIN_COUNT]
    ct = {f"{a} vs {b}": pd.crosstab(pop[f"lean_{a}"], pop[f"lean_{b}"]) for a, b in (("all", "accepted"), ("all", "weighted"))}
    for k, v in ct.items():
        v.to_csv(OUT / f"crosstab_{k.replace(' ', '_')}.csv")
        print(f"\n{k} (users with >= {MIN_COUNT} comments, rows = first)\n", v.to_string())
    # score tells us something: users whose label is lost because their comments score <= 0
    lost = (lab["n_all"] >= MIN_COUNT) & (lab["n_accepted"] < MIN_COUNT)
    print(f"\nusers with >= {MIN_COUNT} comments but < {MIN_COUNT} accepted: {int(lost.sum()):,} of {int((lab.n_all >= MIN_COUNT).sum()):,}")

    # 4. declarations and domains vs each version
    users = act.groupby("author").agg(n_comments=("n_comments", "sum"), n_subreddits=("subreddit", "nunique"),
                                      score_sum=("score_sum", "sum"))
    users["score_mean"] = users.pop("score_sum") / users["n_comments"]
    users = users.join(lab).join(declared_leaning(decl))
    links = dom.groupby("author").agg(n_links=("n_links", "sum"), n_domains=("domain", "nunique"))
    top = dom.groupby(["author", "domain"])["n_links"].sum().reset_index().sort_values(
        ["author", "n_links", "domain"], ascending=[True, False, True]).drop_duplicates("author")
    users = users.join(links).join(top.set_index("author")["domain"].rename("top_domain"))
    for v in VERSIONS:
        users[f"lean_{v}"] = users[f"lean_{v}"].fillna("no_supporter_activity")
    users["declared_lean"] = users["declared_lean"].fillna("none")
    for c in ("n_declarations", "n_vote_decl", "n_identity_decl", "n_links", "n_domains"):
        users[c] = users[c].fillna(0).astype(int)
    users.reset_index().to_parquet(PROFILES / "user_summary_scored.parquet", index=False)

    n = len(users)
    cov = [("all users with a comment", n),
           ("any self-declaration", int((users.n_declarations > 0).sum())),
           ("  vote declaration", int((users.n_vote_decl > 0).sum())),
           ("  identity declaration", int((users.n_identity_decl > 0).sum())),
           ("  single non-conflicting declared label", int(users.declared_lean.isin(["left", "right", "third"]).sum())),
           ("at least one linked URL", int((users.n_links > 0).sum()))]
    for v in VERSIONS:
        cov.append((f"posting label, {v} (left/right/mixed)", int(users[f"lean_{v}"].isin(["left", "right", "mixed"]).sum())))
    cov = pd.DataFrame(cov, columns=["method", "users"])
    cov["share_of_users"] = cov["users"] / n
    cov.to_csv(OUT / "coverage.csv", index=False)
    print("\n== 4. coverage\n", cov.to_string(index=False))

    rows = []
    for name, col in (("any declaration", "declared_lean"), ("vote declaration", "vote_lean"),
                      ("identity declaration", "identity_lean")):
        d = users[col] if col != "declared_lean" else users[col].where(users[col] != "none")
        for v in VERSIONS:
            p = users[f"lean_{v}"]
            has = d.notna() & p.isin(["left", "right", "mixed"])
            clear = d.isin(["left", "right"]) & p.isin(["left", "right"])
            rows.append({"declaration": name, "posting_version": v, "declarers_labelled": int(has.sum()),
                         "left_right_both": int(clear.sum()), "agree": int((d == p)[clear].sum()),
                         "agreement_rate": float((d == p)[clear].mean())})
    agr4 = pd.DataFrame(rows)
    agr4.to_csv(OUT / "declaration_agreement.csv", index=False)
    print(agr4.round(4).to_string(index=False))
    for v in ("all", "weighted"):
        x = pd.crosstab(users.loc[users.declared_lean != "none", "declared_lean"], users[f"lean_{v}"], margins=True)
        x.to_csv(OUT / f"crosstab_declared_vs_{v}.csv")
        print(f"\ndeclared x posting ({v})\n", x.to_string())

    tot = dom.groupby("domain").agg(links=("n_links", "sum"), users=("author", "nunique")).sort_values("links", ascending=False)
    tot.head(300).to_csv(OUT / "top_domains_all.csv")
    news = tot[tot.index.isin(NEWS_DOMAINS)].head(30)
    news.to_csv(OUT / "top30_news_domains.csv")
    print("\n", news.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

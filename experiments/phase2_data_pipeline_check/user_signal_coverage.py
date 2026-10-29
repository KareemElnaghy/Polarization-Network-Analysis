"""Per-user coverage of each leaning signal and agreement with posting-based leaning.

Reads ``data/processed/user_profiles/`` (from ``src/dataset/user_profiles.py merge``) and writes
per-user tables to ``data/processed/user_profiles/`` and summary tables to ``results/user_signals/``.

Posting-based leaning: over a user's comments in party-tagged subreddits (``party`` in
``subreddits_metadata.json``), ``rep_share = rep / (rep + dem)``. With at least MIN_TAGGED such
comments, the user is ``right`` (rep_share >= 0.75), ``left`` (<= 0.25) or ``mixed``; with fewer
they are ``unclassified``.

Usage:
    python experiments/phase2_data_pipeline_check/user_signal_coverage.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "data" / "processed" / "user_profiles"
OUT = ROOT / "results" / "user_signals"
MIN_TAGGED = 5
RIGHT_AT, LEFT_AT = 0.75, 0.25

# Curated by hand from the 300 most-linked domains: outlets that publish news or commentary.
NEWS_DOMAINS = {
    "nytimes.com", "washingtonpost.com", "cnn.com", "politico.com", "theguardian.com", "huffingtonpost.com",
    "huffpost.com", "thehill.com", "breitbart.com", "npr.org", "theatlantic.com", "wsj.com", "businessinsider.com",
    "foxnews.com", "time.com", "go.com", "reuters.com", "dailymail.co.uk", "cbsnews.com", "bloomberg.com",
    "nbcnews.com", "slate.com", "vox.com", "forbes.com", "washingtontimes.com", "latimes.com", "thedailybeast.com",
    "usatoday.com", "dailykos.com", "dailycaller.com", "salon.com", "msnbc.com", "theintercept.com",
    "motherjones.com", "independent.co.uk", "bbc.com", "bbc.co.uk", "telegraph.co.uk", "nydailynews.com",
    "nypost.com", "thinkprogress.org", "fortune.com", "yahoo.com", "nationalreview.com", "zerohedge.com",
    "newsweek.com", "cnbc.com", "ibtimes.com", "usnews.com", "pbs.org", "mediaite.com", "c-span.org",
    "buzzfeed.com", "rt.com", "newyorker.com", "vice.com", "nymag.com", "alternet.org", "economist.com",
    "thenation.com", "talkingpointsmemo.com", "washingtonexaminer.com", "thegatewaypundit.com", "inquisitr.com",
    "ap.org", "reason.com", "aljazeera.com", "counterpunch.org", "wnd.com", "rawstory.com", "freebeacon.com",
    "usuncut.com", "chicagotribune.com", "rollingstone.com",
}
# fivethirtyeight/realclearpolitics are poll aggregators, politifact/snopes/factcheck are fact-checkers: not counted as news.


def posting_leaning(activity: pd.DataFrame) -> pd.DataFrame:
    party = {}
    for line in (ROOT / "data" / "raw" / "subreddits_metadata.json").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["party"]:
                party[row["subreddit"]] = row["party"]
    a = activity.assign(party=activity["subreddit"].map(party))
    tagged = a[a["party"].notna()].pivot_table(index="author", columns="party", values="n_comments",
                                               aggfunc="sum", fill_value=0)
    tagged = tagged.reindex(columns=["dem", "rep"], fill_value=0).rename(columns={"dem": "n_dem", "rep": "n_rep"})
    tagged["n_tagged"] = tagged["n_dem"] + tagged["n_rep"]
    tagged["rep_share"] = tagged["n_rep"] / tagged["n_tagged"]
    tagged["posting_lean"] = "mixed"
    tagged.loc[tagged["rep_share"] >= RIGHT_AT, "posting_lean"] = "right"
    tagged.loc[tagged["rep_share"] <= LEFT_AT, "posting_lean"] = "left"
    tagged.loc[tagged["n_tagged"] < MIN_TAGGED, "posting_lean"] = "unclassified"
    return tagged


def declared_leaning(decl: pd.DataFrame) -> pd.DataFrame:
    g = decl.groupby("author")
    labels = g["label"].agg(lambda s: set(s))
    out = pd.DataFrame({
        "n_declarations": g.size(),
        "n_vote_decl": decl[decl["kind"] == "vote"].groupby("author").size(),
        "n_identity_decl": decl[decl["kind"] == "identity"].groupby("author").size(),
        "declared_lean": labels.map(lambda s: next(iter(s)) if len(s) == 1 else "conflict"),
    })
    for kind in ("vote", "identity"):
        sub = decl[decl["kind"] == kind].groupby("author")["label"].agg(lambda s: set(s))
        out[f"{kind}_lean"] = sub.map(lambda s: next(iter(s)) if len(s) == 1 else "conflict")
    return out.fillna({"n_vote_decl": 0, "n_identity_decl": 0})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    act = pd.read_parquet(PROFILES / "user_subreddit_activity.parquet")
    decl = pd.read_parquet(PROFILES / "user_subreddit_declarations.parquet")
    dom = pd.read_parquet(PROFILES / "user_subreddit_domains.parquet")

    # --- per-user table
    users = act.groupby("author").agg(n_comments=("n_comments", "sum"), score_sum=("score_sum", "sum"),
                                      n_subreddits=("subreddit", "nunique"))
    users["score_mean"] = users["score_sum"] / users["n_comments"]
    users = users.drop(columns="score_sum")
    users = users.join(posting_leaning(act)).join(declared_leaning(decl))
    links = dom.groupby("author").agg(n_links=("n_links", "sum"), n_domains=("domain", "nunique"))
    top = dom.groupby(["author", "domain"])["n_links"].sum().reset_index().sort_values(
        ["author", "n_links", "domain"], ascending=[True, False, True]).drop_duplicates("author")
    links = links.join(top.set_index("author")["domain"].rename("top_domain"))
    users = users.join(links)
    users["posting_lean"] = users["posting_lean"].fillna("no_tagged_activity")
    users["declared_lean"] = users["declared_lean"].fillna("none")
    for c in ("n_declarations", "n_vote_decl", "n_identity_decl", "n_links", "n_domains"):
        users[c] = users[c].fillna(0).astype(int)
    users.reset_index().to_parquet(PROFILES / "user_summary.parquet", index=False)

    # --- per-user-subreddit table (activity + declarations + link counts)
    d_sub = decl.groupby(["author", "subreddit"]).agg(
        n_declarations=("label", "size"), declared_labels=("label", lambda s: ",".join(sorted(set(s)))))
    l_sub = dom.groupby(["author", "subreddit"]).agg(n_links=("n_links", "sum"), n_domains=("domain", "nunique"))
    us = act.set_index(["author", "subreddit"]).join(d_sub).join(l_sub).drop(columns="score_sum").reset_index()
    us[["n_declarations", "n_links", "n_domains"]] = us[["n_declarations", "n_links", "n_domains"]].fillna(0).astype(int)
    us.to_parquet(PROFILES / "user_subreddit_table.parquet", index=False)

    # --- coverage
    n = len(users)
    cov = pd.DataFrame([
        ("all users with a comment", n),
        ("flair (field does not exist in the raw data)", 0),
        ("any self-declaration", int((users.n_declarations > 0).sum())),
        ("  vote declaration", int((users.n_vote_decl > 0).sum())),
        ("  identity declaration", int((users.n_identity_decl > 0).sum())),
        ("  declaration, single non-conflicting label", int(users.declared_lean.isin(["left", "right", "third"]).sum())),
        ("at least one linked URL", int((users.n_links > 0).sum())),
        ("mean comment score (field exists)", n),
        ("posting-based leaning, classified (>=%d comments in party subs)" % MIN_TAGGED,
         int(users.posting_lean.isin(["left", "right", "mixed"]).sum())),
        ("  of which left/right (not mixed)", int(users.posting_lean.isin(["left", "right"]).sum())),
        ("any activity in a party-tagged sub", int((users.posting_lean != "no_tagged_activity").sum())),
    ], columns=["method", "users"])
    cov["share_of_users"] = cov["users"] / n
    cov.to_csv(OUT / "coverage.csv", index=False)
    print(cov.to_string(index=False))

    # --- agreement
    def crosstab(df, col_label, name):
        ct = pd.crosstab(df["declared_lean"] if col_label == "declared_lean" else df[col_label], df["posting_lean"],
                         margins=True)
        ct.to_csv(OUT / f"crosstab_{name}.csv")
        return ct

    rows = {}
    for name, col in (("any_declaration", "declared_lean"), ("vote_declaration", "vote_lean"),
                      ("identity_declaration", "identity_lean")):
        df = users[users[col].notna()] if col != "declared_lean" else users[users.declared_lean != "none"]
        df = df.assign(declared_lean=df[col])
        print(f"\n== declared ({name}) x posting-based leaning (users)")
        print(crosstab(df, "declared_lean", name).to_string())
        clear = df[df.declared_lean.isin(["left", "right"]) & df.posting_lean.isin(["left", "right"])]
        agree = (clear.declared_lean == clear.posting_lean).mean() if len(clear) else float("nan")
        rows[name] = {"users_both_clear": len(clear), "agree": int((clear.declared_lean == clear.posting_lean).sum()),
                      "agreement_rate": agree}
    pd.DataFrame(rows).T.to_csv(OUT / "agreement.csv")
    print(pd.DataFrame(rows).T.to_string())

    # --- score and link use by posting leaning
    by = users[users.posting_lean != "no_tagged_activity"].groupby("posting_lean").agg(
        users=("n_comments", "size"), mean_score=("score_mean", "mean"), median_score=("score_mean", "median"),
        share_with_links=("n_links", lambda s: (s > 0).mean()), share_declaring=("n_declarations", lambda s: (s > 0).mean()))
    by.to_csv(OUT / "by_posting_lean.csv")
    print("\n", by.to_string())

    # --- domains
    tot = dom.groupby("domain").agg(links=("n_links", "sum"), users=("author", "nunique")).sort_values(
        "links", ascending=False)
    tot.head(300).to_csv(OUT / "top_domains_all.csv")
    news = tot[tot.index.isin(NEWS_DOMAINS)].head(30)
    news.to_csv(OUT / "top30_news_domains.csv")
    print("\n", news.to_string())
    print("\nnews links share of all links: %.3f" % (tot[tot.index.isin(NEWS_DOMAINS)].links.sum() / tot.links.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

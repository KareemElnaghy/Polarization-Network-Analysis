"""Per-user, per-subreddit profile signals from the parsed comments.

For every (author, subreddit) in ``data/interim/comments/`` it collects:

* comment count (all, accepted = score >= 1, and weighted by ln(1 + max(score, 0))) and mean comment score;
* self-declarations in the text: "I voted for / I'm voting for / I supported" + a 2016
  candidate, and "as a Republican/Democrat/conservative/liberal/progressive";
* counts of linked domains.

Author flair is not collected: the raw Politosphere records have no flair field
(see ``results/flair_check/summary.md``).

Self-declaration rules: lines quoted with ``>`` are ignored, and a match is dropped if its
sentence contains "not", "never", "if" or an "n't" contraction.

Outputs (per month) go to ``data/interim/user_profiles/``; ``merge`` combines them into
``data/processed/user_profiles/``.

Usage:
    python src/dataset/user_profiles.py extract --workers 4
    python src/dataset/user_profiles.py merge
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMENTS_DIR = PROJECT_ROOT / "data" / "interim" / "comments"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "user_profiles"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "user_profiles"
BATCH_SIZE = 200_000

CANDIDATE_LABEL = {"trump": "right", "clinton": "left", "sanders": "left", "johnson": "third", "stein": "third"}
IDENTITY_LABEL = {"republican": "right", "conservative": "right",
                  "democrat": "left", "liberal": "left", "progressive": "left"}
_CANDIDATE_ALIASES = {"hillary": "clinton", "bernie": "sanders"}

_APOS = "['’]"
VOTE_RE = re.compile(
    rf"\bI(?:{_APOS}m| am| was)?[ \t]+(?:voted for|voting for|supported)[ \t]+"
    r"(?:(?:Donald|Hillary|Bernie|Gary|Jill|Mr\.?|Mrs\.?)[ \t]+)?(Trump|Clinton|Hillary|Bernie|Sanders|Johnson|Stein)\b",
    re.IGNORECASE)
IDENTITY_RE = re.compile(r"\bas an? (Republican|Democrat|conservative|liberal|progressive)\b", re.IGNORECASE)
GUARD_RE = re.compile(rf"\b(?:not|never|if)\b|n{_APOS}t\b", re.IGNORECASE)
SENTENCE_BREAK = re.compile(r"[.!?\n]")
QUOTED_LINE = re.compile(r"^\s*(?:>|&gt;)")
URL_RE = re.compile(r"https?://[^\s<>()\[\]\"']+", re.IGNORECASE)

# Superset of the two patterns above, in RE2 syntax, to skip most comments before the Python pass.
PREFILTER = (r"(?i)(?:voted|voting|supported).{0,30}(?:trump|clinton|hillary|bernie|sanders|johnson|stein)"
             r"|as an? (?:republican|democrat|conservative|liberal|progressive)")

_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac", "edu"}
_STRIP_PREFIXES = ("www.", "m.", "mobile.", "amp.")


def _sentence(text: str, start: int, end: int) -> str:
    left = max((m.end() for m in SENTENCE_BREAK.finditer(text, 0, start)), default=0)
    right = SENTENCE_BREAK.search(text, end)
    return text[left:right.start() if right else len(text)]


def find_declarations(body: str) -> list[tuple[str, str, str]]:
    """Return ``(kind, label, match)`` for each self-declaration in ``body``.

    ``kind`` is ``vote`` or ``identity``; ``label`` is ``left``/``right``/``third``.
    """
    text = "\n".join(line for line in body.split("\n") if not QUOTED_LINE.match(line))
    found = []
    for kind, regex in (("vote", VOTE_RE), ("identity", IDENTITY_RE)):
        for m in regex.finditer(text):
            if GUARD_RE.search(_sentence(text, m.start(), m.end())):
                continue
            word = m.group(1).lower()
            if kind == "vote":
                label = CANDIDATE_LABEL[_CANDIDATE_ALIASES.get(word, word)]
            else:
                label = IDENTITY_LABEL[word]
            found.append((kind, label, m.group(0)))
    return found


def link_domain(url: str) -> str | None:
    """Registered-style domain of ``url`` (``www.``/``m.`` removed, subdomains collapsed)."""
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.lower().replace("\\", "").rstrip(".")  # markdown-escaped dots: reddit\.com
    for prefix in _STRIP_PREFIXES:
        if host.startswith(prefix):
            host = host[len(prefix):]
    parts = host.split(".")
    if len(parts) < 2:
        return None
    keep = 3 if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in _SECOND_LEVEL else 2
    return ".".join(parts[-keep:])


def extract_month(path: Path, out_dir: Path) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    month = path.stem.removeprefix("comments_")
    activity: list[pa.Table] = []
    decls: list[tuple] = []
    domains: Counter = Counter()
    rows = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=["author", "subreddit", "created_utc", "body", "score"]):
        t = pa.Table.from_batches([batch])
        rows += t.num_rows
        t = t.set_column(1, "subreddit", t["subreddit"].cast(pa.string()))
        score = pc.cast(t["score"], pa.int64())
        act = pa.table({
            "author": t["author"], "subreddit": t["subreddit"], "score": score,
            "accepted": pc.cast(pc.greater_equal(score, 1), pa.int64()),
            "weight": pc.ln(pc.add(pc.cast(pc.max_element_wise(score, 0), pa.float64()), 1.0)),
        })
        activity.append(act.group_by(["author", "subreddit"]).aggregate(
            [("score", "sum"), ("score", "count"), ("accepted", "sum"), ("weight", "sum")]))
        body = t["body"]
        hits = t.filter(pc.match_substring_regex(body, PREFILTER))
        for author, sub, ts, text in zip(*(hits[c].to_pylist() for c in ("author", "subreddit", "created_utc", "body"))):
            for kind, label, match in find_declarations(text):
                decls.append((author, sub, ts, kind, label, match))
        links = t.filter(pc.match_substring(body, "http"))
        for author, sub, text in zip(*(links[c].to_pylist() for c in ("author", "subreddit", "body"))):
            for url in URL_RE.findall(text):
                d = link_domain(url)
                if d:
                    domains[(author, sub, d)] += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    act = pa.concat_tables(activity).group_by(["author", "subreddit"]).aggregate(
        [("score_sum", "sum"), ("score_count", "sum"), ("accepted_sum", "sum"), ("weight_sum", "sum")])
    act = act.rename_columns(["author", "subreddit", "score_sum", "n_comments", "n_accepted", "w_log"])
    pq.write_table(act, out_dir / f"activity_{month}.parquet")
    pq.write_table(pa.table({
        "author": [d[0] for d in decls], "subreddit": [d[1] for d in decls],
        "created_utc": pa.array([d[2] for d in decls], pa.timestamp("s", tz="UTC")),
        "kind": [d[3] for d in decls], "label": [d[4] for d in decls], "match": [d[5] for d in decls]}),
        out_dir / f"declarations_{month}.parquet")
    pq.write_table(pa.table({
        "author": [k[0] for k in domains], "subreddit": [k[1] for k in domains],
        "domain": [k[2] for k in domains], "n_links": pa.array(list(domains.values()), pa.int32())}),
        out_dir / f"domains_{month}.parquet")
    return month, rows, time.perf_counter() - t0


def merge(interim: Path = INTERIM_DIR, out_dir: Path = PROCESSED_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def read(prefix: str) -> pa.Table:
        return pa.concat_tables(pq.read_table(f) for f in sorted(interim.glob(f"{prefix}_*.parquet")))

    act = read("activity").group_by(["author", "subreddit"]).aggregate(
        [("score_sum", "sum"), ("n_comments", "sum"), ("n_accepted", "sum"), ("w_log", "sum")])
    act = act.rename_columns(["author", "subreddit", "score_sum", "n_comments", "n_accepted", "w_log"])
    act = act.append_column("score_mean", pc.divide(pc.cast(act["score_sum"], pa.float64()), act["n_comments"]))
    pq.write_table(act, out_dir / "user_subreddit_activity.parquet")
    dom = read("domains").group_by(["author", "subreddit", "domain"]).aggregate([("n_links", "sum")])
    pq.write_table(dom.rename_columns(["author", "subreddit", "domain", "n_links"]),
                   out_dir / "user_subreddit_domains.parquet")
    pq.write_table(read("declarations"), out_dir / "user_subreddit_declarations.parquet")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("extract", "merge"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--months", nargs="*", help="e.g. 2016-06 (default: all)")
    args = ap.parse_args()
    if args.command == "merge":
        merge()
        return 0
    files = sorted(COMMENTS_DIR.glob("comments_*.parquet"))
    if args.months:
        files = [f for f in files if f.stem.removeprefix("comments_") in args.months]
    with ProcessPoolExecutor(args.workers) as pool:
        for month, rows, secs in pool.map(extract_month, files, [INTERIM_DIR] * len(files)):
            print(f"{month}: {rows:,} comments in {secs:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Draw ~150 shared-venue edges for human labelling. No GT, no camp labels, no usernames.
Stratified by cheap lexical cue (agree-ish / disagree-ish / other) and spread over weeks. Writes the blank sheet
(for Kareem and Yasmina) and a private key file mapping pilot_id -> comment_id."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.algorithms.llm.data import load_shared

N, SEED, OUT = 150, 0, Path("data/llm/pilot")
AG = re.compile(r"^\W*(yes|yeah|exactly|this|true|agreed?|correct|right|indeed|well said|you're right)\b", re.I)
DIS = re.compile(r"^\W*(no\b|nope|wrong|false|not true|that's not|you're wrong|lol no|bullshit|nonsense|incorrect|actually,)|\b(you are wrong|idiot|stupid)\b", re.I)

df = load_shared(0.05)
df = df[df.body.str.len().between(10, 1200) & (df.parent.str.len() <= 1200)]
df["cue"] = np.where(df.body.map(lambda s: bool(AG.search(s))), "agree_cue",
                     np.where(df.body.map(lambda s: bool(DIS.search(s))), "disagree_cue", "other"))
rng = np.random.default_rng(SEED)
parts = []
for cue, k in [("agree_cue", N // 3), ("disagree_cue", N // 3), ("other", N - 2 * (N // 3))]:
    g = df[df.cue == cue]
    g = g.assign(_r=g.groupby("week").cumcount() + rng.random(len(g)) * 0.5)  # spread across weeks
    parts.append(g.sample(frac=1, random_state=SEED).assign(_r=lambda x: x.groupby("week").cumcount()).sort_values("_r").head(k))
s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
s.insert(0, "pilot_id", [f"P{i:03d}" for i in range(len(s))])
OUT.mkdir(parents=True, exist_ok=True)
s[["pilot_id", "parent", "body"]].rename(columns={"parent": "parent_comment", "body": "reply"}).assign(
    label_agree_neutral_disagree="").to_csv(OUT / "pilot_sheet_blank.csv", index=False)
s[["pilot_id", "comment_id", "week", "cue"]].to_csv(OUT / "pilot_key_PRIVATE.csv", index=False)
print(len(s), s.cue.value_counts().to_dict(), s.week.nunique(), "weeks")

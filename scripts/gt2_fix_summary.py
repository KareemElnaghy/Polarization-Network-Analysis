"""Writes results/gt2_fix_options/summary.md and series.png from the stage files."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from gt2_fix_report import GT, OUT, baseline, md, table

L = lambda s: pd.read_json(OUT / f"{s}.jsonl", lines=True)
s1, s2, s3 = L("s1"), L("s2"), L("s3")
cols = ["real", "null_mean", "null_sd", "boot_lo", "boot_hi", "frac_z>2", "dem_users", "rep_users", "rho_gold", "rho_gt1", "sec_per_cell"]
six = sorted(s1.week_start.unique())
twelve = six + sorted(set(s2.week_start) - set(six))
all12 = pd.concat([s1, s2]).drop_duplicates(["variant", "label_set", "source", "week_start"])
all12 = all12[all12.week_start.isin(twelve)]
t1 = table(s1, [])[cols]
t2 = table(all12[all12.variant.isin(["base", "A_equal", "B_phr20", "C300", "AB", "ABD", "ABC300", "AD", "BD"])], [])[cols]
t3 = table(s3, ["r0.01", "r0.05", "r0.2"])
t3a = t3[["real", "null_mean", "null_sd", "boot_lo", "boot_hi", "frac_z>2", "dem_users", "rep_users", "rho_gold", "rho_gt1", "sec_per_cell"]]
t3b = t3[[c for c in t3.columns if c.startswith(("rho_full", "real_r", "null_r", "users_r"))]]
auc = pd.read_csv(OUT / "auc_stage1.csv")
conc = pd.read_csv(OUT / "concentration.csv")
f = pd.read_parquet(GT / "gt2_weekly_full.parquet")
f = f[(f.venue == "shared") & (f.label_set == "main")][["week_start", "pi", "null_mean"]]
x = conc.merge(f, on="week_start")
rho_c = spearmanr(x.top1_share, x.pi)[0]
rho_c2 = spearmanr(x.top1_share, x.pi - x.null_mean)[0]
sm = s3[(s3.label_set == "main")]
sm["month"] = sm.week_start.str[:7]
mon = sm[sm.source == "full"].pivot_table(index="month", columns="variant", values="pi")
ns = {s: sm[(sm.variant == v) & (sm.source == s)].pi.notna().sum() for v in ["AB", "A_equal"] for s in ["r0.01", "r0.05", "r0.2"] for _ in [0]} if False else None
usable = sm.groupby(["variant", "source"]).pi.apply(lambda x: int(x.notna().sum())).unstack()
base = baseline()

fig, ax = plt.subplots(2, 1, figsize=(10, 7))
for v, st in [("A_equal", "-"), ("AB", "--")]:
    z = sm[(sm.variant == v) & (sm.source == "full")].sort_values("week_start")
    ax[0].plot(pd.to_datetime(z.week_start), z.pi, st, label=f"{v} full")
    z = sm[(sm.variant == v) & (sm.source == "r0.2")].sort_values("week_start")
    ax[1].plot(pd.to_datetime(z.week_start), z.pi, st, label=f"{v} 20% sample")
ax[0].set_title("Fixed estimator, full week (shared, main labels)"); ax[0].legend(); ax[0].axhline(0.5, color="grey", lw=.5)
ax[1].legend(); ax[1].axhline(0.5, color="grey", lw=.5)
fig.tight_layout(); fig.savefig(OUT / "series.png", dpi=110)

txt = f"""# GT#2 fix options

Exploration only: the main T2/T3 outputs are untouched. Shared venue, main labels, heavy posters excluded,
contraction filter on, n_perm 100, n_boot 100 (main only). Code: `src/gt_runs/fix_options.py`,
`scripts/gt2_fix_options.py`, `scripts/gt2_fix_report.py`, `scripts/gt2_fix_summary.py`. Plot: `series.png`.

Variants: **A** every user counts once (party profile = mean of user frequencies; leave-out otherwise unchanged);
**B** only phrases used by >= 20 distinct users in the cell; **C300/C1000** cap each user at 300/1000 phrases;
**D** pool the trailing 4 weeks; **E** held-out-user classifier AUC (cross-check only).
Columns: `real`/`null_*` are means over weeks; `boot_lo/hi` mean 95% band; `frac_z>2` share of weeks where real is
more than 2 null sd above the null mean; `dem_users/rep_users` usable users per side (>= 20 phrases);
`rho_gold` Spearman of the main-label series with the series computed from self-declared (gold) labels;
`rho_gt1` Spearman with the GT#1 excess series; `sec_per_cell` runtime of one cell (under heavy load, so only
relative sizes mean anything).

**Baseline (current main output):** real 0.428, below the null in 51 of 53 weeks; Spearman of full week vs 1% / 5% / 20% sample:
{base['0.01'][0]:.3f} / {base['0.05'][0]:.3f} / {base['0.2'][0]:.3f}.

## Mechanism
Share of phrases held by the top 1% of users is stable (mean {conc.top1_share.mean():.3f}, range {conc.top1_share.min():.3f}-{conc.top1_share.max():.3f}).
Its Spearman with the main pi across 53 weeks is {rho_c:.2f} (with pi minus null: {rho_c2:.2f}). So concentration does not explain
week-to-week movement; it is a constant offset in every week, and it matches the finding that the volume weighting is the problem:
removing the weighting (A) lifts every week above the null, capping (C) helps only partly (C300 0.480, C1000 0.451 vs 0.426 base
on the same weeks), and pooling more weeks (D) makes it worse (0.331) because it adds more volume from the same heavy users.

## Stage 1: six weeks (Election, primaries 3/7 and 1/25, convention 7/18, quiet 4/18 and 8/8)
{md(t1)}

E, held-out-user classifier (5-fold by user, logistic regression on frequencies, 3 shuffled-label nulls):

{md(auc.set_index('week_start'))}

The classifier separates the camps (AUC 0.71-0.76 against a null of 0.50), so the signal is real and the volume-weighted
leave-out estimate is the part that is off.

## Stage 2: twelve weeks, combinations
{md(t2)}

B adds nothing once A is on (AB is A with a smaller vocabulary); ABC300 equals AB exactly because normalising each user to
frequencies removes the cap; D adds cost without changing the level (ABD and AD are the same as AB and A); B or D without A stay below the null.

## Stage 3: all 53 weeks (A and A+B; full week and samples)
{md(t3a)}

{md(t3b)}

Weeks with a usable estimate (>= 50 users per side), out of 53:

{md(usable)}

Monthly mean of the full-week series:

{md(mon)}

## Criteria
| | A (equal weight) | A+B (equal weight + phrase >= 20 users) |
|---|---|---|
| (a) real clearly above null | yes: all 53 weeks, z > 2 | yes: all 53 weeks |
| (b) null near 0.5 | yes: 0.500, sd about 0.0003 | yes |
| (c) full vs 20% sample clearly above 0.278 | **0.69** (5%: 0.37, 1%: 0.24) | 0.62 (5%: 0.28, 1%: not usable) |
| (d) sensible pattern | local highs in the 3/7 primary week, 6/13 (California), 10/17 (October) and 11/7 (election); low and flat in July-September | similar |
| extra: Spearman with GT#1 / gold | 0.32 / 0.65 | 0.51 / 0.68 |
| usable weeks at 5% / 1% | 52 / 42 | 35 / 3 |

Caveat on (d): the series is flat in absolute terms (0.506-0.521). Equal weighting makes each noisy short user count the same, so the level
is a diluted partisanship measure, not comparable to the volume-weighted pi in size; only the ranking and the null comparison are meaningful.

## Recommendation
Use **A (equal-weight users)** as the GT#2 estimator, with no phrase threshold. It is the only change that
moves the estimate off its bias (every week above a null that stays at 0.500), it gives the best agreement with the samples at
every rate (0.69 at 20%, 0.37 at 5%, 0.24 at 1%) and keeps 52 of 53 weeks usable at 5%. A+B agrees a bit better with GT#1 (0.51 vs 0.32)
but loses the small samples, because 20 distinct users per phrase is too strict in a 5% sample. If B is wanted for the full week only,
the threshold would need to scale with the sample rate. Capping and the 4-week window are not worth their cost.
Report GT#2 with a note that its level is not on the old scale, and use the rank agreement with GT#1 and gold as the validity check.
Next step (not done): rerun T2/T3 with A, and add the sample-scaled variant of B if wanted.
"""
(OUT / "summary.md").write_text(txt, encoding="utf-8")
print(txt)

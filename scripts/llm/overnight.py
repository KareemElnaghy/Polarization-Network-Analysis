"""Unattended supervisor for the 5% extraction: logs progress, restarts the extraction if it dies, runs checkpoint evaluations
(Pipeline A, B, camp recovery, rq1_compare), stops the extraction at the cutoff, runs the final evaluation, scores the robustness
model on the pilot, writes a draft report and commits. Never changes the prompt, model or spec.
usage: python scripts/llm/overnight.py --stop-at 07:30 [--first-eval-pairs 30000] [--eval-every-h 2]"""
import argparse
import datetime as dt
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psutil

ap = argparse.ArgumentParser()
ap.add_argument("--stop-at", default="07:30", help="local HH:MM (next occurrence)")
ap.add_argument("--first-eval-pairs", type=int, default=30000)
ap.add_argument("--eval-every-h", type=float, default=2.0)
ap.add_argument("--log-every-min", type=float, default=30)
ap.add_argument("--robust-model", default="gemma3:12b")
a = ap.parse_args()
PY = sys.executable
LOG = Path("logs_llm/overnight.log")
Path("logs_llm").mkdir(exist_ok=True)


def log(msg):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def latest(pattern):
    d = sorted(glob.glob(pattern))
    return d[-1] if d else None


def status():
    r = latest("runs/llm/*_tier5pct")
    try:
        return r, json.load(open(Path(r) / "status.json"))
    except Exception:
        return r, {}


def extraction_alive():
    return any("run_extraction.py" in " ".join(p.info["cmdline"] or []) for p in psutil.process_iter(["cmdline"]))


def server_up():
    try:
        import requests
        return requests.get("http://localhost:11434/api/version", timeout=5).ok
    except Exception:
        return False


def start_server():
    log("starting Ollama server (parallel 4)")
    subprocess.Popen(["powershell", "-NoProfile", "-File", "scripts/llm/start_ollama.ps1", "-Parallel", "4"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=0x08000000)
    time.sleep(30)


def start_extraction():
    log("starting/resuming extraction (same command; finished pairs are cached)")
    out, err = open("logs_llm/tier5pct.log", "a"), open("logs_llm/tier5pct.err", "a")
    subprocess.Popen([PY, "scripts/llm/run_extraction.py", "--rate", "0.05", "--budget", "1000000000", "--name", "tier5pct"],
                     stdout=out, stderr=err, creationflags=0x08000000)


def sh(cmd, logfile):
    with open(logfile, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(cmd)}\n")
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


def evaluate(tag):
    """A, B, camp recovery, rq1_compare on the newest extraction run. CPU only."""
    run, st = status()
    lf = f"logs_llm/overnight_eval_{tag}.log"
    log(f"checkpoint evaluation '{tag}' on {run} ({st.get('done')} pairs done)")
    if sh([PY, "scripts/llm/run_pipeline_a.py", run, "--name", f"pipeA_{tag}"], lf):
        return log(f"Pipeline A failed for {tag}, see {lf}")
    if sh([PY, "scripts/llm/run_pipeline_b.py", run, "--name", f"pipeB_{tag}"], lf):
        return log(f"Pipeline B failed for {tag}, see {lf}")
    A, B = latest(f"runs/llm/*_pipeA_{tag}"), latest(f"runs/llm/*_pipeB_{tag}")
    sh([PY, "scripts/llm/eval_camp_recovery.py", B], lf)
    sh([PY, "scripts/llm/rq1_compare.py", A, B, "--out", f"results/llm/rq1_5pct/{tag}", "--label", tag], lf)
    log(f"checkpoint '{tag}' done: A={A} B={B}")
    return A, B


def md(df, nd=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(nd)
    h = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return h + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)) + "\n"


def write_report(tag, A, B, run_status, robust_note):
    d = Path(f"results/llm/rq1_5pct/{tag}")
    corr = pd.read_csv(d / f"rq1_{tag}_correlations.csv")
    delta = pd.read_csv(d / f"rq1_{tag}_delta.csv")
    summ = json.load(open(d / f"rq1_{tag}_summary.json"))
    prim = summ["PRIMARY_verdict_full_gt2"]
    txt = f"""# RQ1 at the 5% sample, partial coverage (auto-drafted {dt.datetime.now():%Y-%m-%d %H:%M}; needs human reading)

Spec: `docs/eval_spec.md` (eval-spec-v1) plus clarification D-E3 (logged before any 5% number). Primary model `qwen3.5:9b-q4_K_M`, prompt v2, frozen (D-M7).
Extraction run `{run_status[0]}`: **{run_status[1].get('done')} of {run_status[1].get('to_run')} unique pairs processed ({100 * run_status[1].get('done', 0) / max(run_status[1].get('to_run', 1), 1):.1f}% of the 5% tier)**,
thread-prefix ordering (whole threads, all weeks), {run_status[1].get('failed')} failures, state `{run_status[1].get('state')}`.
Pipeline A run `{A}`; Pipeline B run `{B}`. Coverage: {summ['coverage']}.

**This is a partial 5% sample: thinner than a true 5% tier, so power is limited. PRIMARY truth = full-week GT; the 5%-matched columns are secondary (computed on all 5% threads, not on the processed prefix).**

## Correlations with ground truth
{md(corr)}
## Delta rho (B minus A), 4-week block bootstrap
{md(delta)}
A vs B correlation: {summ['A_vs_B_rho']:.3f}.

## Decision (rule: B must beat A on GT#2 with the delta CI excluding 0 AND rho_B > 0)
- PRIMARY (full-week GT#2): **{prim}**
- SECONDARY (5%-matched GT#2): {summ['SECONDARY_verdict_matched_gt2']}

## Camp recovery
See `runs/llm/*_pipeB_{tag}` (eval_camp_recovery output in `logs_llm/overnight_eval_{tag}.log`).

## Robustness model
{robust_note}

## Honest reading (to be completed by a human)
- Apply the rule above literally; a pipeline that is only less negative is not a winner.
- Coverage is partial; weeks with thin data are flagged low_n and Pipeline B weeks can be NaN. Report the number of usable B weeks every time.
- A positive result here would still be one sample, one model, and 1 prompt; the Tier 0 null (docs/reports/rq1_tier0_b230_report.md) stays the baseline.
"""
    Path("docs/reports/rq1_5pct_report.md").write_text(txt, encoding="utf-8")
    log("draft report written: docs/reports/rq1_5pct_report.md")


def commit(msg):
    sh(["git", "add", "docs", "scripts", "configs", "results/llm"], "logs_llm/overnight_git.log")
    sh(["git", "commit", "-m", msg + "\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"], "logs_llm/overnight_git.log")
    if sh(["git", "push", "origin", "llm_implementation"], "logs_llm/overnight_git.log"):
        log("git push failed (offline?); committed locally")


now = dt.datetime.now()
h, m = map(int, a.stop_at.split(":"))
cutoff = now.replace(hour=h, minute=m, second=0, microsecond=0)
cutoff += dt.timedelta(days=1) if cutoff <= now else dt.timedelta(0)
log(f"overnight supervisor started; cutoff {cutoff:%Y-%m-%d %H:%M}; first eval at {a.first_eval_pairs} pairs, then every {a.eval_every_h} h")
last_log = last_eval = 0.0
n_eval = 0
while dt.datetime.now() < cutoff:
    run, st = status()
    if not extraction_alive() and st.get("state") != "done":
        if not server_up():
            start_server()
        start_extraction()
        time.sleep(60)
        continue
    if time.time() - last_log > a.log_every_min * 60:
        log(f"progress: done {st.get('done')} / {st.get('to_run')}, failed {st.get('failed')}, eta_h {st.get('eta_h', 0):.1f}, run {run}")
        last_log = time.time()
    due = (n_eval == 0 and st.get("done", 0) >= a.first_eval_pairs) or (n_eval > 0 and time.time() - last_eval > a.eval_every_h * 3600)
    if due:
        n_eval += 1
        evaluate(f"5pct_ckpt{n_eval}")
        last_eval = time.time()
        commit(f"5% checkpoint {n_eval} results ({st.get('done')} pairs)")
    time.sleep(60)

log("cutoff reached: stopping extraction")
run, _ = status()
(Path(run) / "STOP").touch()
t0 = time.time()
while extraction_alive() and time.time() - t0 < 1200:
    time.sleep(10)
run, st = status()
log(f"extraction stopped: done {st.get('done')} / {st.get('to_run')}, state {st.get('state')}")
res = evaluate("5pct_final")
robust = "Not run (final evaluation failed)."
if res:
    lf = "logs_llm/overnight_robust.log"
    log(f"scoring {a.robust_model} on the pilot (prompt v2, dev then held-out)")
    sh([PY, "scripts/llm/pilot_run.py", a.robust_model, "--prompt", "v2", "--split", "dev"], lf)
    sh([PY, "scripts/llm/pilot_run.py", a.robust_model, "--prompt", "v2", "--split", "heldout", "--final"], lf)
    robust = (f"`{a.robust_model}` pilot scoring (prompt v2, dev + held-out, same split) is in `{lf}`; compare with the primary's combined 0.72 accuracy "
              f"(D-M7). The 5% robustness run was NOT started automatically; it is a sensitivity analysis and runs only if the pilot is not clearly worse.")
    write_report("5pct_final", res[0], res[1], status(), robust)
with open("docs/LOG.md", "a", encoding="utf-8") as f:
    f.write(f"\n- {dt.datetime.now():%Y-%m-%d %H:%M} overnight supervisor finished: extraction stopped at cutoff ({st.get('done')} pairs), final evaluation "
            f"and draft report written; details in logs_llm/overnight.log.\n")
commit("5% final evaluation, draft report and overnight log")
log("supervisor finished")

Measuring political polarization in online discussions by comparing two LLM-based
approaches: direct text-based scoring vs. routing interactions through a signed
social network and measuring its structural properties.

## Background

Most work on polarization either (a) scores political polarization directly from
text using LLMs, or (b) builds signed social networks (friend/foe edges inferred
from interactions) and measures structural properties like balance or frustration.
This project asks whether (b) adds anything over (a) — and under what conditions.

**Core research question (RQ1):** Does a structure-aware pipeline (LLM sign
inference → signed graph → frustration score) track a ground-truth polarization
proxy better than a direct LLM text-scoring pipeline?

Secondary questions (RQ2–RQ5, see `docs/decisions.md` for full scope) ask when
structure helps most (window size, density, polarization level), how sign-inference
quality affects results, whether either pipeline improves forecasting, and what
drives polarization shifts.

**Dataset:** Reddit Politosphere (reply-edge comment network, 2008–2019, with
subreddit-level Democratic/Republican tags used as a ground-truth proxy).
Full dataset and method details are in the project proposal (see `docs/`).

**Scope note:** This is a 2-person, 2-month project. RQ5, a second dataset
(PolitiSky24), and several proposed extensions (multi-camp detection, GNNs,
alternative LLMs) are explicitly out of scope for this timeline — see
`docs/decisions.md` → "Scope cuts" for what was cut and why.

## Project structure

- `src/dataset/` — data ingestion, graph construction, ground-truth proxy (owned by Yasmina)
- `src/algorithms/` — synthetic validation, both pipelines, evaluation metrics (owned by Kareem)
- `src/common/` — shared data schemas both sides depend on (joint)
- `experiments/` — scripts/notebooks organized by phase and RQ
- `docs/decisions.md` — design decisions (ground-truth rule, null-model definition, scope cuts)

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
# requires a local Ollama instance running (see config/config.yaml for model name)
```

Run tests: `pytest`

---

## Current Status

> Update this section at the start of each work session or milestone phase.
> Keep only the current entry at the top — move older entries down under
> "Status history" so this stays scannable.

**As of: 2026-10-19 — Midway milestone (Week 1 of 2)**

- **Phase:** Midway sprint — building the full pipeline on a small Politosphere
  slice (1–2 months) to answer RQ1 only.
- **In progress:**
  - Yasmina — downloading Politosphere months, building reply-edge network, designing
    the ground-truth mapping rule from subreddit tags
  - Kareem — Pipeline A/B scaffolding, parameterized synthetic graph generator,
    local Ollama setup + throughput testing
- **Not started yet:** integration run of both pipelines on real data, agreement
  metrics + null-model control, midway slides
- **Blockers / open questions:** none yet
- **Next milestone:** Midway presentation — [insert date], showing a working
  pipeline on a small real slice with preliminary RQ1 signal

### Status history

<!-- Move previous "Current Status" entries here as they're superseded -->
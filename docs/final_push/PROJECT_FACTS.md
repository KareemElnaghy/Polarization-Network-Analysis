# Project facts (fill in before starting the agent)

The agent reads this file first. Anything left as `TODO` gets detected by the agent where it can (`nvidia-smi`, `free -h`, and so on) or asked about once. Wrong facts here cost more time than missing ones, so delete anything you are unsure of.

## Deadline

- Final milestone due (date and time, Cairo): `TODO e.g. 2026-11-30 23:59`
- Presentation date, if different: `TODO`
- Hard stop for new experiments, after which we only write up (suggested: deadline minus 4 days): `TODO`

## Machines

| Name used in the plan | OS | CPU / RAM | GPU / VRAM | Holds the data? | Notes |
|---|---|---|---|---|---|
| `WIN` | Windows 11 | i9-11900F / 64 GB | RTX 3090 / 24 GB | **yes**: `data/`, `runs/`, `extractions.sqlite` | the GPU is sometimes used by other projects |
| `UBU` | Ubuntu | `TODO` / `TODO` (16 GB?) | `TODO` / `TODO` (16 GB?) | no | new; needs the repo cloned and a venv |
| `MAC` | macOS (Apple Silicon) | `TODO` / 24 GB unified | MPS only | partial (dev copies) | analysis and writing |

> We are not sure whether "16 GB / 24 GB" refers to the PCs' RAM or their GPU memory. Correct the table if needed. The agent verifies the real values with `nvidia-smi`.

## Connectivity between machines

- Same LAN? `TODO yes/no`
- SSH from the agent's machine to the others? `TODO yes/no (if yes, host names and users)`
- Shared folder or drive for large files (shards, exports)? `TODO e.g. \\WIN\share, an external SSD, or none`
- Internet on all machines (pip, Hugging Face downloads)? `TODO`

## Where the agent runs

- The agent runs on: `TODO WIN (recommended, it holds the data)`
- People available to run commands on the other machines: `TODO Kareem: UBU+WIN, Yasmina: MAC`

## Scope preferences (pre-filled with the plan's defaults; change them if you disagree)

- Extra years: **2017 first, then 2015**, only after the 2016 density ladder is done.
- External datasets: SRQ only if tweet text can be obtained (1-hour check); PolitiSky24 is a stretch goal; Conflict-Bias-Eval is cut unless everything else is done.
- The robustness LLM (`gemma3:12b`) runs only if the faster server works.

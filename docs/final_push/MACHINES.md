# Machines: who does what, and how results come back

## Roles

| Machine | Main jobs | Why |
|---|---|---|
| **WIN** (3090 24 GB, 64 GB RAM): the data host and the agent's home | Phase 0; distillation export; G1.6 server benchmark; encoder E1 training; encoder inference shard A; data jobs for other years; Pipeline B shards; merging all shards | It holds `data/`, the cache and the GT files. It has the strongest GPU and the most RAM. |
| **UBU** (16 GB-class GPU; verify) | Encoder E2 training; C2 embeddings; encoder inference shard B; Pipeline B shards (CPU) | A second GPU doubles the throughput of training and inference. Linux makes CUDA and pip simple. |
| **MAC** (24 GB unified) | Phase 1-CPU analyses (A1–A8) that need only exported Parquet; Pipeline B shards for the 1%, 5% and 20% tiers; figures; writing | Free CPU and no GPU contention. It is where the writing happens anyway. |

**GPU contention on WIN.** Before each GPU job, check `nvidia-smi` for other processes (handbook §2.2: other projects have used this GPU). If the GPU is busy, put the job in the queue in `STATUS.md` and move UBU work forward.

## Timeline per machine (when everything goes to plan)

```
          D1                D2                D3                D4          D5          D6+
WIN  P0 | G1.6 bench | G1.1 export | E1 train ......| select+checks | encode shard A ...| B shards | years/data jobs
UBU       setup      | E2 train ...........| C2 embed+head | encode shard B ......| B shards (full density)
MAC       P1.1 P1.3 P1.4 | P1.2 (B on halves) | P1.5 P1.6 P1.7 P1.8 | density-ladder stats, figures | writing
```

## What has to move between machines

| From → to | What | Size (estimate) | How |
|---|---|---|---|
| WIN → UBU | `data/llm/distill/distill_v1.parquet` | about 1 GB | shared folder / scp / SSD |
| WIN → UBU | full-venue input shards: Parquet with pair_hash, the two truncated texts, week, thread, endpoints (`scripts/final/export_encode_inputs.py --weeks 28-53`) | 5–8 GB | same |
| UBU → WIN | `encoder_outputs_UBU.sqlite`, the C2 head plus embeddings cache (or just the C2 weights per pair_hash as Parquet) | 1–3 GB | same |
| WIN → MAC | for the analyses: per-edge Parquet (week, thread, endpoints, labels joined *only* in the `analysis/` export, cache outputs), plus the run folders | under 1 GB per density | same |
| MAC/UBU → WIN | Pipeline B shard run folders (`runs/llm/<ts>_pipeB_<density>_w<from>-<to>`) | small | same, or commit `weekly_B.csv` only |

Move code only through git. Never hand-copy `.py` files. Every machine runs a commit SHA that is recorded in its manifests.

## Shard and merge protocol

1. **Encoder inference shards.** Split by week: WIN takes weeks 1–27 and UBU takes 28–53 at every density. Adjust the split from measured throughput so both finish together. Each shard writes `encoder_outputs_<machine>.sqlite`. Merge on WIN with `scripts/final/merge_stores.py --into data/llm/encoder_outputs.sqlite <shards…>`, which uses `INSERT OR IGNORE` on the primary key. After merging, assert that the row count equals the union of unique hashes and that every week of the requested density is covered.
2. **Pipeline B, C1 and C2 shards.** Add `--weeks` to the runners if it is missing. Each shard produces a normal run folder. `scripts/final/merge_pipeline_runs.py` concatenates `weekly_*.csv` and `camp_assignments.parquet`, checks that there are no duplicate weeks and that 53/53 are present, and writes a merged run folder whose manifest lists the shard folders and their SHAs. `rq1_compare.py` then runs on the merged folder as usual.
3. **Determinism.** Seeds depend on (week, source, null index), not on the shard, so a sharded run gives the same numbers as an unsharded one. Add a test: run 3 weeks sharded 1+2 against unsharded, and require identical output.

## Long-job hygiene (as in the existing framework)

- Run detached (`Start-Process … -WindowStyle Hidden` on WIN; `nohup … &` or `systemd-run --user` on UBU), log to `logs_llm/<job>.log`, make the job resumable, and honour a `STOP` file.
- Keep WIN awake during long runs (`powercfg /change standby-timeout-ac 0`, and restore the setting afterwards).
- Write a heartbeat line at least every 5 minutes with the number done, the rate and the ETA. `STATUS.md` quotes the latest one.
- After any crash: resume. Never start fresh unless the output is corrupt, and keep corrupt outputs renamed rather than deleted.

# Lab notebook (Kareem, LLM side)

## 2026-10-30
- Phase R recon done and approved. Findings in the Phase R message; samples are not nested.
- Phase 1 started. Ollama 0.40.11 had no `~/.ollama/id_ed25519`, so `ollama pull` failed with "open id_ed25519: no such file". Generated a key with `ssh-keygen -t ed25519` and the pulls worked (3-4 MB/s: llama3.1:8b ~25 min, qwen3:8b ~25 min, gemma3:12b ~40 min).
- Built `src/algorithms/llm/{schema,backends,store,extract,data}.py`, `configs/llm/llm.yaml`; 10 mock-backend tests pass.
- Pilot sheet drawn: 150 edges, 50 per lexical-cue stratum (agree-ish / disagree-ish / other), all 53 weeks.

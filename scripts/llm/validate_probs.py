"""Are Ollama token logprobs under constrained decoding meaningful? Compare p(label) from logprobs (T=0) with the
frequency of each label over k samples at T=0.7, on 100 pilot edges. usage: python scripts/llm/validate_probs.py MODEL"""
import json
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from src.algorithms.llm.backends import OllamaBackend, probs_from_logprobs, validate
from src.algorithms.llm.data import load_shared
from src.algorithms.llm.schema import AGREEMENT

model, K = sys.argv[1], 10
cfg = yaml.safe_load(open("configs/llm/llm.yaml")) | {"model": model}
b = OllamaBackend(cfg)
key = pd.read_csv("data/llm/pilot/pilot_key_PRIVATE.csv")
e = load_shared(0.05)
e = e[e.comment_id.isin(key.comment_id)].head(100)
rows = []
for r in e.itertuples():
    j = b._chat(r.parent, r.body, 0.0, 0, True)
    lp = j.get("logprobs") or j["message"].get("logprobs")
    p = probs_from_logprobs(lp) if lp else None
    freq = dict.fromkeys(AGREEMENT, 0.0)
    for i in range(K):
        try:
            freq[validate(json.loads(b._chat(r.parent, r.body, 0.7, 1 + i, False)["message"]["content"]))["agreement"]] += 1 / K
        except Exception:
            pass
    rows.append({"has_lp": lp is not None, "p": p, "freq": freq, "greedy": json.loads(j["message"]["content"])["agreement"]})
d = pd.DataFrame(rows)
ok = d[d.p.notna()]
out = {"model": model, "n": len(d), "logprobs_returned": float(d.has_lp.mean()), "probs_parsed": float(d.p.notna().mean())}
if len(ok):
    P = np.array([[x[k] for k in AGREEMENT] for x in ok.p]); F = np.array([[x[k] for k in AGREEMENT] for x in ok.freq])
    out |= {"mean_max_p": float(P.max(1).mean()), "mae_vs_freq": float(np.abs(P - F).mean()),
            "argmax_matches_greedy": float(np.mean([AGREEMENT[i] == g for i, g in zip(P.argmax(1), ok.greedy)])),
            **{f"spearman_{k}": float(spearmanr(P[:, i], F[:, i])[0]) for i, k in enumerate(AGREEMENT)}}
print(json.dumps(out, indent=1))
open(f"results/llm/bench/probs_{model.replace(':', '_')}.json", "w").write(json.dumps(out))

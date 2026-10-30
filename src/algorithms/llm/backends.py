"""Backend-agnostic LLM interface. A backend maps (parent, body) -> Result; probabilities may be None."""
import hashlib
import json
import time
from dataclasses import dataclass

import requests

from .schema import AGREEMENT, SCHEMA, SYSTEM_PROMPT, user_message


class InvalidOutput(Exception):
    pass


@dataclass
class Result:
    fields: dict            # parsed extraction (validated)
    probs: dict | None      # {"agree":p,"neutral":p,"disagree":p} or None
    latency: float
    n_tokens: int = 0


def validate(obj) -> dict:
    """Raise InvalidOutput unless obj matches SCHEMA."""
    if not isinstance(obj, dict) or list(obj) != list(SCHEMA["properties"]):
        raise InvalidOutput(f"bad keys: {obj}")
    for k, spec in SCHEMA["properties"].items():
        v = obj[k]
        ok = isinstance(v, bool) if spec["type"] == "boolean" else v in spec["enum"]
        if not ok:
            raise InvalidOutput(f"bad value {k}={v!r}")
    return obj


def normalise(raw: dict) -> dict | None:
    """Normalise masses over the three labels; None when there is no mass at all."""
    s = sum(raw.get(k, 0.0) for k in AGREEMENT)
    return {k: raw.get(k, 0.0) / s for k in AGREEMENT} if s > 0 else None


def label_of_token(tok: str) -> str | None:
    """Map a first value token to a label by prefix ("dis"->disagree, "ag"->agree, "neu"->neutral)."""
    t = tok.strip().lower()
    for lab in AGREEMENT:
        if t and lab.startswith(t) and len(t) >= 2 or t == lab:
            return lab
    return None


def probs_from_logprobs(content: list[dict]) -> dict | None:
    """Find the token after '"agreement": "' (first field) and normalise its top_logprobs over the labels."""
    text = ""
    for i, tok in enumerate(content):
        if text.replace(" ", "").endswith('"agreement":"'):
            raw: dict = {}
            for alt in [{"token": tok["token"], "logprob": tok["logprob"]}] + list(tok.get("top_logprobs", [])):
                lab = label_of_token(alt["token"])
                if lab:
                    raw[lab] = max(raw.get(lab, 0.0), 2.718281828 ** alt["logprob"])
            return normalise(raw)
        text += tok["token"]
    return None


class OllamaBackend:
    def __init__(self, cfg: dict):
        self.cfg, self.url = cfg, cfg["base_url"].rstrip("/")
        self.session = requests.Session()

    def info(self) -> dict:
        d = self.session.post(f"{self.url}/api/show", json={"model": self.cfg["model"]}, timeout=60).json()["details"]
        v = self.session.get(f"{self.url}/api/version", timeout=10).json()["version"]
        return {"quantization": d.get("quantization_level", "?"), "ollama_version": v, "family": d.get("family")}

    def _chat(self, parent: str, body: str, temperature: float, seed: int, logprobs: bool) -> dict:
        c = self.cfg
        req = {"model": c["model"], "stream": False, "format": SCHEMA, "think": False,  # thinking off (D-M2)
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_message(parent, body, c["max_chars"])}],
               "options": {"temperature": temperature, "seed": seed, "num_predict": c["num_predict"]}}
        if logprobs:
            req.update(logprobs=True, top_logprobs=c["top_logprobs"])
        r = self.session.post(f"{self.url}/api/chat", json=req, timeout=300)
        r.raise_for_status()
        return r.json()

    def extract(self, parent: str, body: str) -> Result:
        c, t0 = self.cfg, time.time()
        want_lp = c["prob_method"] == "logprobs"
        j = self._chat(parent, body, c["temperature"], c["seed"], want_lp)
        try:
            fields = validate(json.loads(j["message"]["content"]))
        except (json.JSONDecodeError, KeyError) as e:
            raise InvalidOutput(str(e))
        probs = probs_from_logprobs(j.get("logprobs") or j["message"].get("logprobs") or []) if want_lp else None
        if c["prob_method"] == "samples":
            counts = {k: 0.0 for k in AGREEMENT}
            for i in range(c["k_samples"]):
                try:
                    a = validate(json.loads(self._chat(parent, body, 0.7, c["seed"] + 1 + i, False)["message"]["content"]))
                    counts[a["agreement"]] += 1
                except (InvalidOutput, json.JSONDecodeError, KeyError):
                    pass
            probs = normalise(counts)
        return Result(fields, probs, time.time() - t0, j.get("eval_count", 0))


class MockBackend:
    """Deterministic fake. `fail_first` makes the first n calls per text return invalid output (retry tests)."""

    def __init__(self, cfg: dict | None = None, fail_first: int = 0):
        self.cfg, self.fail_first, self.calls, self._seen = cfg or {}, fail_first, 0, {}

    def info(self) -> dict:
        return {"quantization": "mock", "ollama_version": "mock", "family": "mock"}

    def extract(self, parent: str, body: str) -> Result:
        self.calls += 1
        n = self._seen[body] = self._seen.get(body, 0) + 1
        if n <= self.fail_first:
            raise InvalidOutput("mock invalid")
        h = int(hashlib.sha1(body.encode()).hexdigest(), 16)
        a = AGREEMENT[h % 3]
        p = {k: 0.1 for k in AGREEMENT}
        p[a] = 0.8
        return Result({"agreement": a, "emotional_charge": ["low", "medium", "high"][h // 3 % 3],
                       "hostility": ["none", "mild", "strong"][h // 9 % 3], "personal_attack": bool(h // 27 % 2)},
                      p, 0.0, 10)

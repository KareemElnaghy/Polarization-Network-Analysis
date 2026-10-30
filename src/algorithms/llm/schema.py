"""Extraction schema and prompt. The system prompt is static and first so Ollama can reuse its KV prefix."""
import hashlib

AGREEMENT = ["agree", "neutral", "disagree"]
# Field order matters: `agreement` first so its token probabilities are interpretable.
# TODO(verify): attribute set against Chowdhury et al. 2026 (stance, affective tone, agreement).
SCHEMA = {
    "type": "object",
    "properties": {
        "agreement": {"type": "string", "enum": AGREEMENT},
        "emotional_charge": {"type": "string", "enum": ["low", "medium", "high"]},
        "hostility": {"type": "string", "enum": ["none", "mild", "strong"]},
        "personal_attack": {"type": "boolean"},
    },
    "required": ["agreement", "emotional_charge", "hostility", "personal_attack"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You analyse one reply to a Reddit comment. You are given the PARENT comment and the REPLY.
Return JSON with exactly these fields, in this order:
- agreement: the reply's stance toward the position of the parent comment: "agree", "neutral" (no clear stance, off-topic, a question, or mixed), or "disagree".
- emotional_charge: "low", "medium" or "high": how emotionally charged the reply is.
- hostility: "none", "mild" or "strong": hostility of the reply toward the parent comment's author or position.
- personal_attack: true if the reply attacks the parent's author as a person (insults, name-calling), not just their argument.
Judge only the text shown. Answer with the JSON object only."""

FORBIDDEN_FIELDS = ("source", "target", "subreddit", "link_id", "side", "label", "gt", "author")


def clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n]


def user_message(parent: str, body: str, max_chars: dict) -> str:
    return f"PARENT:\n{clip(parent, max_chars['parent'])}\n\nREPLY:\n{clip(body, max_chars['body'])}"


def pair_hash(parent: str, body: str, max_chars: dict) -> str:
    """Identity of the text the model actually sees (after truncation); identical pairs share one call."""
    return hashlib.sha1(user_message(parent, body, max_chars).encode()).hexdigest()


def was_truncated(parent: str, body: str, max_chars: dict) -> bool:
    return len(parent) > max_chars["parent"] or len(body) > max_chars["body"]

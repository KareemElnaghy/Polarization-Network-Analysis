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

PROMPTS = {"v1": """You analyse one reply to a Reddit comment. You are given the PARENT comment and the REPLY.
Return JSON with exactly these fields, in this order:
- agreement: the reply's stance toward the position of the parent comment: "agree", "neutral" (no clear stance, off-topic, a question, or mixed), or "disagree".
- emotional_charge: "low", "medium" or "high": how emotionally charged the reply is.
- hostility: "none", "mild" or "strong": hostility of the reply toward the parent comment's author or position.
- personal_attack: true if the reply attacks the parent's author as a person (insults, name-calling), not just their argument.
Judge only the text shown. Answer with the JSON object only.""",
    "v2": """You analyse one reply to a Reddit comment. You are given the PARENT comment and the REPLY.
Output one line of compact JSON (no spaces or line breaks) with exactly these fields, in this order:
agreement, emotional_charge, hostility, personal_attack.

agreement = does the REPLY side with the position of the PARENT comment?
- "agree": the reply endorses, supports, confirms or extends the parent's point. This holds even if the reply is angry or insulting about someone else (a candidate, a party, a group): negative words alone do not mean disagreement.
- "disagree": the reply rejects, corrects or argues against the parent's own point, politely or rudely.
- "neutral": no clear stance toward the parent's point: a question, a request for a source, an off-topic remark, or an even mix.
Sarcasm counts as its intended meaning.
Examples (invented):
Parent: "Both parties take money from the same donors." Reply: "Exactly, and those clowns in Congress will never fix it." -> agree (hostile words, but aimed at a third party)
Parent: "Voter fraud decided the primary." Reply: "There is no evidence of that. Every audit matched the counts." -> disagree (polite)
Parent: "The poll shows her ahead by six." Reply: "Which poll is this?" -> neutral

emotional_charge = "low", "medium" or "high": how emotionally charged the reply is.
hostility = "none", "mild" or "strong": hostility of the reply toward the PARENT comment's author or position (not toward third parties).
personal_attack = true only if the reply insults or name-calls the parent's author as a person.
Judge only the text shown.""",
}
SYSTEM_PROMPT = PROMPTS["v1"]  # default for callers that do not pass a version


def system_prompt(version: str) -> str:
    return PROMPTS[version]


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

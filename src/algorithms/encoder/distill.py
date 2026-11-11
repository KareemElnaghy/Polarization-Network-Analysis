"""Distillation-dataset helpers (SPEC_ENCODER section 2). No labels, GT or eval imports (label firewall D-B5)."""
import hashlib


def split_of(thread: str) -> str:
    """Thread-level split: int(sha1(thread)) % 20 -> 0 test (5%), 1 val (5%), else train (90%).
    int() is taken over the sha1 hex digest. Threads never straddle splits."""
    k = int(hashlib.sha1(str(thread).encode()).hexdigest(), 16) % 20
    return "test" if k == 0 else "val" if k == 1 else "train"

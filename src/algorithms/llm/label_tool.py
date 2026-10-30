"""Terminal labelling tool for the pilot sheet. Shows parent then reply only; a/n/d label, s skip, b back, q quit.
Saves data/llm/pilot/labels_<rater>.csv after every label and resumes at the first unlabelled item.
usage: python -m src.algorithms.llm.label_tool --rater kareem"""
import argparse
import csv
import os
import sys
import termios
import textwrap
import tty

KEYS = {"a": "agree", "n": "neutral", "d": "disagree"}


def read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1).lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def load_labels(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as f:
        return {r["pilot_id"]: r["label"] for r in csv.DictReader(f)}


def save_labels(path: str, ids: list, labels: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:  # atomic: a crash never leaves a half-written file
        w = csv.writer(f)
        w.writerow(["pilot_id", "label"])
        w.writerows((i, labels[i]) for i in ids if i in labels)
    os.replace(tmp, path)


def first_unlabelled(ids: list, labels: dict, start: int = 0) -> int | None:
    for k in list(range(start, len(ids))) + list(range(0, start)):
        if ids[k] not in labels:
            return k
    return None


def show(i, n, done, parent, reply, current):
    w = lambda s: "\n".join(textwrap.fill(p, 100) for p in s.splitlines())
    print("\033[2J\033[H", end="")
    print(f"Item {i + 1} / {n}    labelled {done} / {n}" + (f"    (current: {current})" if current else ""))
    print("\n=== PARENT COMMENT ===\n" + w(parent) + "\n\n=== REPLY ===\n" + w(reply))
    print("\nReply's stance toward the parent's position:  [a] agree   [n] neutral   [d] disagree   [s] skip   [b] back   [q] quit")


def run(sheet: str, out: str, key=read_key, display=show) -> None:
    with open(sheet, newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["pilot_id"] for r in rows]
    labels = {k: v for k, v in load_labels(out).items() if k in ids}
    pos = first_unlabelled(ids, labels) or 0
    while True:
        r = rows[pos]
        display(pos, len(rows), len(labels), r["parent_comment"], r["reply"], labels.get(r["pilot_id"]))
        k = key()
        if k in KEYS:
            labels[r["pilot_id"]] = KEYS[k]
            save_labels(out, ids, labels)
            if len(labels) == len(rows):
                print(f"\nAll {len(rows)} items labelled. Saved to {out}")
                return
            pos = first_unlabelled(ids, labels, pos + 1)  # next unlabelled, wrapping around
        elif k == "s":
            pos = (pos + 1) % len(rows)
        elif k == "b":
            pos = max(pos - 1, 0)
        elif k in ("q", "\x03"):
            print(f"\nStopped. {len(labels)} / {len(rows)} saved to {out}")
            return


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rater", required=True)
    ap.add_argument("--sheet", default="data/llm/pilot/pilot_sheet_blank.csv")
    a = ap.parse_args()
    run(a.sheet, os.path.join(os.path.dirname(a.sheet), f"labels_{a.rater}.csv"))

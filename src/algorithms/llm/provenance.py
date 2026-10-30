"""Immutable run folders + results registry (results/llm/INDEX.md and index.csv)."""
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()


def new_run(name: str, cfg: dict, inputs: list[str], root: str = "runs/llm", **extra) -> Path:
    """Create runs/llm/<timestamp>_<name>/ (never reuses a folder) and write manifest.json with status=running."""
    d = Path(root) / f"{time.strftime('%Y%m%dT%H%M%S')}_{name}"
    d.mkdir(parents=True, exist_ok=False)
    m = {"run_id": d.name, "status": "running", "start": time.time(), "git_sha": _git("rev-parse", "HEAD"),
         "git_dirty": bool(_git("status", "--porcelain")), "config": cfg, "seed": cfg.get("seed"),
         "model": cfg.get("model"), "prompt_version": cfg.get("prompt_version"),
         "inputs": {p: sha256(p) for p in inputs if Path(p).is_file()}, "hardware": platform.platform() + " " + platform.machine(),
         **extra}
    (d / "manifest.json").write_text(json.dumps(m, indent=1, default=str))
    return d


def finish_run(d: Path, status: str, headline: dict, purpose: str = "", index_dir: str = "results/llm") -> None:
    """Hash outputs, mark status (done/partial/failed), append to the registry. Failed/negative runs are kept."""
    mf = d / "manifest.json"
    m = json.loads(mf.read_text())
    m.update(status=status, end=time.time(), headline=headline,
             outputs={str(p.relative_to(d)): sha256(p) for p in sorted(d.rglob("*")) if p.is_file() and p.name != "manifest.json"})
    mf.write_text(json.dumps(m, indent=1, default=str))
    idx = Path(index_dir)
    idx.mkdir(parents=True, exist_ok=True)
    row = {"run_id": m["run_id"], "purpose": purpose, "model": m.get("model"), "rate": m.get("sample_rate"),
           "status": status, "headline": json.dumps(headline), "path": str(d)}
    new = not (idx / "index.csv").exists()
    with open(idx / "index.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    rows = list(csv.DictReader(open(idx / "index.csv")))
    (idx / "INDEX.md").write_text("# Results registry\n\n| run | purpose | model | rate | status | headline |\n|---|---|---|---|---|---|\n" +
                                  "\n".join(f"| [{r['run_id']}]({r['path']}) | {r['purpose']} | {r['model']} | {r['rate']} | {r['status']} | {r['headline']} |" for r in rows) + "\n")

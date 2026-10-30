"""Append-only SQLite extraction store. One row per (text pair, model, quantization, prompt version, temperature, seed)."""
import json
import sqlite3
import time
from pathlib import Path

KEY = ("pair_hash", "model", "quant", "prompt_version", "temperature", "seed")
SQL = """CREATE TABLE IF NOT EXISTS extractions(
 pair_hash TEXT, model TEXT, quant TEXT, prompt_version TEXT, temperature REAL, seed INTEGER,
 fields TEXT, p_agree REAL, p_neutral REAL, p_disagree REAL, latency REAL, retries INTEGER, n_tokens INTEGER,
 prob_method TEXT, created REAL, PRIMARY KEY(pair_hash, model, quant, prompt_version, temperature, seed))"""


class Store:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(SQL)

    def have(self, ctx: dict) -> set[str]:
        q = "SELECT pair_hash FROM extractions WHERE model=? AND quant=? AND prompt_version=? AND temperature=? AND seed=?"
        return {r[0] for r in self.db.execute(q, (ctx["model"], ctx["quant"], ctx["prompt_version"], ctx["temperature"], ctx["seed"]))}

    def put(self, h: str, ctx: dict, res, retries: int) -> None:
        p = res.probs or {}
        self.db.execute("INSERT OR IGNORE INTO extractions VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (h, ctx["model"], ctx["quant"], ctx["prompt_version"], ctx["temperature"], ctx["seed"],
                         json.dumps(res.fields), p.get("agree"), p.get("neutral"), p.get("disagree"),
                         res.latency, retries, res.n_tokens, ctx["prob_method"], time.time()))
        self.db.commit()

    def table(self, ctx: dict):
        import pandas as pd
        q = "SELECT * FROM extractions WHERE model=? AND quant=? AND prompt_version=? AND temperature=? AND seed=?"
        df = pd.read_sql(q, self.db, params=(ctx["model"], ctx["quant"], ctx["prompt_version"], ctx["temperature"], ctx["seed"]))
        return df.join(pd.json_normalize(df.pop("fields").map(json.loads)))

    def export_parquet(self, out_dir: str) -> str:
        """Dump all rows to a Parquet shard with a sha256 manifest line. Never deletes anything."""
        import hashlib
        import pandas as pd
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        df = pd.read_sql("SELECT * FROM extractions", self.db)
        f = out / f"extractions_{time.strftime('%Y%m%dT%H%M%S')}.parquet"
        df.to_parquet(f)
        with open(out / "MANIFEST.tsv", "a") as m:
            m.write(f"{f.name}\t{len(df)}\t{hashlib.sha256(f.read_bytes()).hexdigest()}\n")
        return str(f)

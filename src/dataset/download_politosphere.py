"""Download the Reddit Politosphere dataset (Zenodo record 5851729) into data/raw/.

Resumable, checksum-verified, re-runnable: files already present with a correct
MD5 are skipped, partial downloads (<name>.part) are resumed with HTTP Range.

Usage:
    python src/dataset/download_politosphere.py                 # everything
    python src/dataset/download_politosphere.py --only a.bz2 b.csv
    python src/dataset/download_politosphere.py --list          # list and exit
"""

from __future__ import annotations

import argparse
import csv
import email.utils
import hashlib
import http.client
import logging
import os
import queue
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

RECORD_ID = "5851729"
API = "https://zenodo.org/api/records"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEST = PROJECT_ROOT / "data" / "raw"

MAX_ATTEMPTS = 10
BASE_DELAY = 5.0
MAX_DELAY = 300.0
TIMEOUT = (30, 120)  # (connect, read) seconds
WORKERS = 3
CHUNK = 1 << 20
SPACE_MARGIN = 1.30

log = logging.getLogger("politosphere")


class RetryableError(Exception):
    def __init__(self, msg: str, retry_after: float | None = None):
        super().__init__(msg)
        self.retry_after = retry_after


class FatalError(Exception):
    pass


@dataclass
class RemoteFile:
    name: str
    size: int
    md5: str
    url: str


@dataclass
class Result:
    name: str
    size: int
    md5: str
    status: str  # ok | skipped | failed
    attempts: int
    bytes_downloaded: int
    error: str = ""


# --------------------------------------------------------------------------- #
# Zenodo metadata
# --------------------------------------------------------------------------- #

def get_json(session: requests.Session, url: str) -> dict:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                raise RetryableError(f"HTTP {r.status_code}", parse_retry_after(r))
            r.raise_for_status()
            return r.json()
        except (requests.Timeout, requests.ConnectionError, RetryableError) as e:
            if attempt == MAX_ATTEMPTS:
                raise
            delay = backoff(attempt, getattr(e, "retry_after", None))
            log.warning("metadata %s attempt %d failed: %s; retry in %.0fs", url, attempt, e, delay)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def resolve_record(session: requests.Session, record_id: str) -> dict:
    """Return the latest version of a record (works for concept or version IDs)."""
    rec = get_json(session, f"{API}/{record_id}")
    latest_url = rec.get("links", {}).get("latest")
    if latest_url:
        latest = get_json(session, latest_url)
        if str(latest["id"]) != str(rec["id"]):
            log.info("record %s resolved to latest version %s", record_id, latest["id"])
        rec = latest
    return rec


def list_files(rec: dict) -> list[RemoteFile]:
    files = []
    for f in rec["files"]:
        algo, _, digest = f["checksum"].partition(":")
        if algo != "md5":
            raise FatalError(f"unsupported checksum algorithm {algo} for {f['key']}")
        files.append(RemoteFile(f["key"], int(f["size"]), digest, f["links"]["self"]))
    return sorted(files, key=lambda f: f.name)


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #

def backoff(attempt: int, retry_after: float | None = None) -> float:
    if retry_after is not None:
        return min(max(retry_after, 1.0), MAX_DELAY)
    delay = min(BASE_DELAY * 2 ** (attempt - 1), MAX_DELAY)
    return delay * random.uniform(0.8, 1.2)


def parse_retry_after(r: requests.Response) -> float | None:
    value = r.headers.get("Retry-After")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        dt = email.utils.parsedate_to_datetime(value)
        return max(dt.timestamp() - time.time(), 0.0) if dt else None


def md5_of(path: Path, h=None) -> "hashlib._Hash":
    h = h or hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK * 8), b""):
            h.update(block)
    return h


def is_complete(path: Path, rf: RemoteFile) -> bool:
    return path.exists() and path.stat().st_size == rf.size and md5_of(path).hexdigest() == rf.md5


class Slots:
    """Fixed tqdm line positions so concurrent bars don't overwrite each other."""

    def __init__(self, n: int):
        self.q: queue.Queue[int] = queue.Queue()
        for i in range(n):
            self.q.put(i)

    def acquire(self) -> int:
        return self.q.get()

    def release(self, i: int) -> None:
        self.q.put(i)


def attempt_download(session: requests.Session, rf: RemoteFile, part: Path, slot: int,
                     tty: bool) -> int:
    """One download attempt into `part`. Returns bytes transferred. Raises on failure."""
    offset = part.stat().st_size if part.exists() else 0
    if offset > rf.size:
        log.warning("%s: .part larger than expected (%d > %d), restarting", rf.name, offset, rf.size)
        part.unlink()
        offset = 0

    transferred = 0
    if offset < rf.size:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with session.get(rf.url, headers=headers, stream=True, timeout=TIMEOUT) as r:
            if r.status_code == 429 or r.status_code >= 500:
                raise RetryableError(f"HTTP {r.status_code}", parse_retry_after(r))
            if r.status_code == 416:  # range past end: .part is already full-length
                pass
            elif r.status_code not in (200, 206):
                raise FatalError(f"HTTP {r.status_code}")
            else:
                if offset and r.status_code == 200:
                    log.warning("%s: server ignored Range, restarting from 0", rf.name)
                    offset = 0
                mode = "ab" if offset else "wb"
                with open(part, mode) as fh, tqdm(
                    total=rf.size, initial=offset, unit="B", unit_scale=True, unit_divisor=1024,
                    desc=rf.name[:24].ljust(24), position=slot, leave=False,
                    mininterval=0.5 if tty else 30, file=sys.stderr,
                ) as bar:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
                        transferred += len(chunk)
                        bar.update(len(chunk))

    got = part.stat().st_size
    if got < rf.size:
        raise RetryableError(f"incomplete read: {got}/{rf.size} bytes")
    digest = md5_of(part).hexdigest()
    if got != rf.size or digest != rf.md5:
        part.unlink()  # corrupt: next attempt starts from zero
        raise RetryableError(f"checksum mismatch: got {digest} ({got} B), expected {rf.md5}")
    return transferred


def download_file(session: requests.Session, rf: RemoteFile, dest: Path, slots: Slots,
                  tty: bool) -> Result:
    final = dest / rf.name
    part = dest / (rf.name + ".part")

    if is_complete(final, rf):
        log.info("%s: already present with correct checksum, skipping", rf.name)
        return Result(rf.name, rf.size, rf.md5, "skipped", 0, 0)
    if final.exists():
        log.warning("%s: existing file has wrong size/checksum, re-downloading", rf.name)
        final.unlink()

    total_bytes = 0
    last_err = ""
    slot = slots.acquire()
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            start_offset = part.stat().st_size if part.exists() else 0
            log.info("%s: attempt %d/%d starting at byte %d", rf.name, attempt, MAX_ATTEMPTS, start_offset)
            try:
                n = attempt_download(session, rf, part, slot, tty)
                total_bytes += n
                os.replace(part, final)
                log.info("%s: OK attempt %d, %d bytes this attempt, checksum verified", rf.name, attempt, n)
                return Result(rf.name, rf.size, rf.md5, "ok", attempt, total_bytes)
            except FatalError as e:
                last_err = str(e)
                log.error("%s: attempt %d fatal error: %s", rf.name, attempt, e)
                return Result(rf.name, rf.size, rf.md5, "failed", attempt, total_bytes, last_err)
            except (RetryableError, requests.Timeout, requests.ConnectionError,
                    requests.exceptions.ChunkedEncodingError, requests.exceptions.ContentDecodingError,
                    http.client.IncompleteRead, ConnectionError, TimeoutError) as e:
                now = part.stat().st_size if part.exists() else 0
                total_bytes += max(now - start_offset, 0)
                last_err = f"{type(e).__name__}: {e}"
                if attempt == MAX_ATTEMPTS:
                    log.error("%s: attempt %d failed: %s; bytes on disk %d; giving up",
                              rf.name, attempt, last_err, now)
                    break
                delay = backoff(attempt, getattr(e, "retry_after", None))
                log.warning("%s: attempt %d failed: %s; bytes on disk %d; retry in %.0fs",
                            rf.name, attempt, last_err, now, delay)
                time.sleep(delay)
    finally:
        slots.release(slot)
    return Result(rf.name, rf.size, rf.md5, "failed", MAX_ATTEMPTS, total_bytes, last_err)


# --------------------------------------------------------------------------- #
# Manifest / summary
# --------------------------------------------------------------------------- #

MANIFEST_FIELDS = ["file", "size", "checksum", "status", "attempts", "bytes_downloaded", "error"]


def write_manifest(path: Path, results: list[Result]) -> None:
    rows: dict[str, dict] = {}
    if path.exists():  # keep rows for files not touched in this run (e.g. --only)
        with open(path, newline="") as fh:
            rows = {row["file"]: row for row in csv.DictReader(fh)}
    for r in results:
        if r.status == "skipped" and rows.get(r.name, {}).get("status") == "ok":
            continue  # keep the attempts/bytes recorded when it was downloaded
        status = "ok" if r.status in ("ok", "skipped") else "failed"
        rows[r.name] = {"file": r.name, "size": r.size, "checksum": f"md5:{r.md5}", "status": status,
                        "attempts": r.attempts, "bytes_downloaded": r.bytes_downloaded, "error": r.error}
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for name in sorted(rows):
            w.writerow({k: rows[name].get(k, "") for k in MANIFEST_FIELDS})
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", default=RECORD_ID)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--only", nargs="+", metavar="FILE", help="download only these file names")
    ap.add_argument("--list", action="store_true", help="list files and free space, then exit")
    ap.add_argument("--workers", type=int, default=WORKERS)
    args = ap.parse_args()

    dest: Path = args.dest
    dest.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(dest / "download.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s %(message)s"))
    log.addHandler(fh)

    session = requests.Session()
    session.headers["User-Agent"] = "politosphere-downloader/1.0 (research)"
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers + 1, pool_maxsize=args.workers + 1)
    session.mount("https://", adapter)

    rec = resolve_record(session, args.record)
    files = list_files(rec)
    if args.only:
        wanted = set(args.only)
        missing = wanted - {f.name for f in files}
        if missing:
            print(f"Unknown file(s) in record {rec['id']}: {sorted(missing)}", file=sys.stderr)
            return 2
        files = [f for f in files if f.name in wanted]

    total = sum(f.size for f in files)
    free = shutil.disk_usage(dest).free
    print(f"Record {rec['id']} (concept {rec.get('conceptrecid')}): {len(files)} files")
    print(f"{'file':40s} {'size':>12s}")
    for f in files:
        print(f"{f.name:40s} {f.size / 1e6:9.1f} MB")
    print(f"Total: {total / 1e9:.2f} GB | free at {dest}: {free / 1e9:.2f} GB "
          f"| required with 30% margin: {total * SPACE_MARGIN / 1e9:.2f} GB")
    log.info("record %s: %d files, %d bytes, free %d bytes", rec["id"], len(files), total, free)

    already = sum((dest / f.name).stat().st_size for f in files if (dest / f.name).exists())
    if free + already < total * SPACE_MARGIN:
        print("ABORT: not enough free disk space.", file=sys.stderr)
        log.error("abort: insufficient disk space")
        return 3
    if args.list:
        return 0

    tty = sys.stderr.isatty()
    slots = Slots(args.workers)
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="dl") as pool:
        results = list(pool.map(lambda f: download_file(session, f, dest, slots, tty), files))

    write_manifest(dest / "manifest.csv", results)

    ok = [r for r in results if r.status == "ok"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]
    gb = sum(r.bytes_downloaded for r in results) / 1e9
    summary = (f"\nSummary: {len(ok)} downloaded, {len(skipped)} already present, "
               f"{len(failed)} failed | {gb:.2f} GB downloaded this run | "
               f"{len(ok) + len(skipped)}/{len(files)} files complete")
    print(summary)
    log.info(summary.strip())
    if failed:
        print("Failed files:")
        for r in failed:
            print(f"  {r.name}: {r.error}")
        print("Rerun just these with:\n  python src/dataset/download_politosphere.py --only "
              + " ".join(r.name for r in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

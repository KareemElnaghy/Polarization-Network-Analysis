"""Load guard: keeps the prep jobs from taking the shared PC down.

A governor thread samples system-wide CPU, RAM and free disk. Every job process runs one, so jobs
see the same numbers and back off together. ``allowed(base)`` is what the dispatch loop consults:
the number of units that may run now (0 means paused).
"""

from __future__ import annotations

import random
import shutil
import threading
import time
from pathlib import Path

import psutil


class Governor:
    def __init__(self, guard: dict, output_dir: Path, runs_dir: Path, job_dir: Path) -> None:
        self.g = guard
        self.output_dir = Path(output_dir)
        self.runs_dir = Path(runs_dir)
        self.job_dir = Path(job_dir)
        self.state = "ok"             # ok | throttled | paused
        self.reason = ""
        self.factor = 1.0
        self.cpu = 0.0
        self.ram = 0.0
        self.free_disk_gb = 0.0
        self._high_since: float | None = None
        self._resume_at = 0.0
        self._last_halve = 0.0
        self._ram_paused = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.sample()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.wait(self.g["sample_seconds"]):
            self.sample()

    def sample(self) -> None:
        g, now = self.g, time.time()
        ram = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=None)
        usage = shutil.disk_usage(self.output_dir if self.output_dir.exists() else self.output_dir.parent)
        free_gb, free_pct = usage.free / 1e9, 100 * usage.free / usage.total
        with self._lock:
            self.cpu, self.ram, self.free_disk_gb = cpu, ram, free_gb
            if ram > g["ram_pause_percent"]:
                self._high_since = self._high_since or now
                if now - self._high_since >= g["ram_pause_seconds"]:
                    self._ram_paused = True
            else:
                self._high_since = None
            if self._ram_paused and ram < g["ram_resume_percent"]:
                self._ram_paused = False
                self._resume_at = now + random.uniform(0, g["resume_jitter_seconds"])
            if ram > g["ram_throttle_percent"] and now - self._last_halve >= g["sample_seconds"] * 2:
                self.factor = max(0.05, self.factor / 2)
                self._last_halve = now
            elif ram < g["ram_restore_percent"] and self.factor < 1.0:
                self.factor = min(1.0, self.factor * 2)

            if (self.runs_dir / "PAUSE_ALL").exists():
                self.state, self.reason = "paused", "runs/PAUSE_ALL exists"
            elif free_gb < g["min_free_disk_gb"] or free_pct < g["min_free_disk_percent"]:
                self.state = "paused"
                self.reason = f"free disk {free_gb:.0f} GB ({free_pct:.0f}%) is below the limit"
            elif self._ram_paused:
                self.state, self.reason = "paused", f"RAM {ram:.0f}% for over {g['ram_pause_seconds']} s"
            elif now < self._resume_at:
                self.state, self.reason = "paused", "resuming after RAM pause (jitter)"
            elif self.factor < 1.0:
                self.state, self.reason = "throttled", f"RAM {ram:.0f}%: workers x{self.factor:g}"
            else:
                self.state, self.reason = "ok", ""

    def allowed(self, base: int) -> int:
        """Units that may run at once given the current load (0 while paused)."""
        with self._lock:
            if self.state == "paused":
                return 0
            return max(1, int(base * self.factor))

    def snapshot(self) -> dict:
        with self._lock:
            return {"state": self.state, "reason": self.reason, "factor": self.factor, "cpu": self.cpu,
                    "ram_percent": self.ram, "free_disk_gb": self.free_disk_gb}

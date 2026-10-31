"""Throughput sweep over OLLAMA_NUM_PARALLEL on Windows: restarts the server per setting (start_ollama.ps1), warms the model, runs
bench.py and samples VRAM. usage: python scripts/llm/sweep_parallel.py MODEL [--slots 4,8,16] [--n 200]"""
import argparse
import subprocess
import sys
import threading
import time

import requests

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--slots", default="4,8,16")
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()


def vram():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()[0])


for p in map(int, a.slots.split(",")):
    with open(f"logs_llm/start_{p}.out", "w") as f:
        subprocess.run(["powershell", "-NoProfile", "-File", "scripts/llm/start_ollama.ps1", "-Parallel", str(p)], stdout=f, stderr=f,
                       timeout=120)
    requests.post("http://localhost:11434/api/generate", json={"model": a.model, "prompt": "hi", "stream": False, "think": False,
                                                              "options": {"num_predict": 2}}, timeout=300)
    loaded, peak, stop = vram(), [0], threading.Event()

    def watch():
        while not stop.is_set():
            peak[0] = max(peak[0], vram())
            time.sleep(2)
    threading.Thread(target=watch, daemon=True).start()
    out = subprocess.run([sys.executable, "scripts/llm/bench.py", a.model, "--parallel", str(p), "--n", str(a.n), "--tag", f"_sweep{p}"],
                         capture_output=True, text=True)
    stop.set()
    line = [l for l in out.stdout.splitlines() if "edges_per_s" in l]
    print(f"== OLLAMA_NUM_PARALLEL={p} | VRAM loaded {loaded} MiB, peak {peak[0]} MiB\n{line[0] if line else out.stdout[-500:] + out.stderr[-500:]}", flush=True)

"""Relaunch the 5% extraction if it dies (finished pairs are cached, so nothing is lost). Stop by creating a STOP file in the
newest runs/llm/*_tier5pct folder. Logs to logs_llm/watch_5pct.log."""
import glob, subprocess, sys, time

PY = r".venv\Scripts\python.exe"
CMD = [PY, "scripts/llm/run_extraction.py", "--rate", "0.05", "--budget", "1000000000", "--name", "tier5pct"]
log = open("logs_llm/watch_5pct.log", "a", buffering=1)
out, err = open("logs_llm/tier5pct_resume.out", "a"), open("logs_llm/tier5pct.err", "a")
for attempt in range(1, 31):
    log.write(f"{time.ctime()} start attempt {attempt}\n")
    rc = subprocess.Popen(CMD, stdout=out, stderr=err, creationflags=0x08000000).wait()
    runs = sorted(glob.glob("runs/llm/*_tier5pct"))
    stopped = bool(runs) and bool(glob.glob(runs[-1] + "/STOP"))
    log.write(f"{time.ctime()} process exited rc={rc} stop_file={stopped}\n")
    if rc == 0 or stopped:
        break
    time.sleep(30)
log.write(f"{time.ctime()} watcher finished\n")

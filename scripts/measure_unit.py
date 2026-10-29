import sys, time, json, importlib, gc
from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prep_jobs.config import load_config, runs_dir
from src.prep_jobs.util import peak_memory_gb, write_json
job, week = sys.argv[1], sys.argv[2]
cfg = load_config()
class Ctx:
    def __init__(s): s.cfg=cfg; s.jcfg=cfg["jobs"][job]; s.batch_rows=250000; s.attempt=1; s.job=job
    def report(s,*a,**k): pass
    def log(s,m): print("log:",m)
    def over_limit(s): return False
    def shrink(s): pass
mod = importlib.import_module(f"src.prep_jobs.jobs.{job}")
t=time.time()
r = mod.week_unit(Ctx(), {"id": week, "weight": 1})
out = {"job": job, "week": week, "seconds": round(time.time()-t,1), "peak_gb": round(peak_memory_gb(),2), "result": r}
print(json.dumps(out))
p = runs_dir(cfg) / "gt_measure.json"
try: d = json.loads(p.read_text())
except Exception: d = {}
d[job] = out; write_json(d, p)

"""Print the worker allocation plan for the prep jobs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prep_jobs.allocation import format_plan, plan  # noqa: E402
from src.prep_jobs.config import load_config  # noqa: E402

if __name__ == "__main__":
    print(format_plan(plan(load_config())))

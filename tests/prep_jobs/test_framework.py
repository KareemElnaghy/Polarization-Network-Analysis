import json
import os

import pytest
import yaml

from src.prep_jobs import runner
from src.prep_jobs.config import SOURCE_ROOT


@pytest.fixture
def project(tmp_path, monkeypatch):
    cfg = yaml.safe_load((SOURCE_ROOT / "config" / "prep_jobs.yaml").read_text())
    cfg["jobs"]["dummy"].update({"seconds": 3, "units": 3})
    cfg["framework"]["heartbeat_seconds"] = 1
    cfg["framework"]["backoff_seconds"] = 0.1
    cfg_path = tmp_path / "prep_jobs.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("PREP_PROJECT_ROOT", str(tmp_path))
    return tmp_path, cfg, cfg_path


def run(cfg_path, resume=False, workers=2):
    return runner.main(["dummy", "--workers", str(workers), "--config", str(cfg_path)] + (["--resume"] if resume else []))


def status(root):
    return json.loads((root / "runs" / "dummy" / "status.json").read_text())


def test_job_runs_and_resume_skips_finished_units(project):
    root, cfg, cfg_path = project
    assert run(cfg_path) == 0
    st = status(root)
    assert st["state"] == "done" and st["units_done"] == st["units_total"] == 4 and st["percent"] == 100
    assert len(list((root / "runs" / "dummy" / "checkpoints").glob("*.done"))) == 4
    log_before = (root / "runs" / "dummy" / "job.log").read_text().count("start sleep__")
    assert run(cfg_path, resume=True) == 0
    assert (root / "runs" / "dummy" / "job.log").read_text().count("start sleep__") == log_before  # nothing re-run


def test_second_run_without_resume_refuses(project):
    root, cfg, cfg_path = project
    run(cfg_path)
    run(cfg_path)
    assert status(root)["state"] == "failed" and "--resume" in status(root)["last_error"]


def test_failed_unit_is_retried_then_succeeds(project):
    root, cfg, cfg_path = project
    cfg["jobs"]["dummy"].update({"flaky_unit": "u1", "flaky_attempts": 2})
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert run(cfg_path) == 0
    assert status(root)["state"] == "done"
    ck = json.loads((root / "runs" / "dummy" / "checkpoints" / "sleep__u1.done").read_text())
    assert ck["attempt"] == 2
    assert (root / "runs" / "dummy" / "errors" / "sleep__u1.txt").exists()


def test_unit_failing_every_time_is_marked_failed_and_job_continues(project):
    root, cfg, cfg_path = project
    cfg["jobs"]["dummy"].update({"flaky_unit": "u1", "flaky_attempts": 99})
    cfg_path.write_text(yaml.safe_dump(cfg))
    assert run(cfg_path) == 1
    st = status(root)
    assert st["state"] == "done_with_failures" and "sleep__u1" in st["failed_units"]
    assert st["units_done"] == 2  # the other two sleep units; the finish phase is skipped after failures


def test_stop_file_stops_cleanly(project):
    root, cfg, cfg_path = project
    (root / "runs" / "dummy").mkdir(parents=True)
    (root / "runs" / "dummy" / "STOP").touch()
    assert run(cfg_path) == 0
    assert status(root)["state"] == "stopped"


def test_lock_blocks_a_second_copy(project):
    root, cfg, cfg_path = project
    d = root / "runs" / "dummy"
    d.mkdir(parents=True)
    (d / "lock").write_text(str(os.getppid()))  # another live process
    with pytest.raises(SystemExit) as exc:
        runner.acquire_lock(d)
    assert "already running" in str(exc.value)
    (d / "lock").write_text("999999999")  # a dead one: the lock is taken over
    runner.acquire_lock(d)
    assert (d / "lock").read_text() == str(os.getpid())

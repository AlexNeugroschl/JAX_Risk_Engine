"""
The component demo (demos/demo_components.py) runs every section to completion, on any date;
the HTTP demos' job polling stops at every final status.

I-28: the var_es section's swap once had no `evaluation_date`, so it was scheduled off ORE's
wall-clock today while the pillars stayed at `EVAL_DATE`, and was refused on every day but
2026-07-30. I-65 moved the sections out of `engine` modules' `__main__` blocks.
"""
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from demos import demo_profile_small, demo_structured

DEMOS = Path(__file__).resolve().parent.parent / "demos"


class TestComponentDemosRun:
    @pytest.mark.slow
    def test_every_section_runs(self):
        proc = subprocess.run(
            [sys.executable, str(DEMOS / "demo_components.py")],
            capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        for section in ("simulation", "swap", "european", "bermudan", "american", "greeks", "var_es"):
            assert f"===== {section} =====" in proc.stdout
        assert "Base (t=0) NPV:" in proc.stdout
        assert "VaR_95:" in proc.stdout


@pytest.mark.parametrize("demo", [demo_structured, demo_profile_small], ids=lambda m: m.__name__.split(".")[-1])
class TestTheHttpDemosPolling:
    """Until 2026-10-06 each HTTP demo polled until `done` or `failed`, so a job whose worker
    died (`interrupted`, final too) was polled forever. Each demo polls with its own code
    (`demo_api.py` too, which runs as a script on import and is not tested here)."""

    @pytest.fixture
    def statuses(self, monkeypatch):
        """The job's statuses, one per poll; the requests go nowhere and nothing sleeps."""
        polls = []
        submit = httpx.Response(202, json={"job_id": "j"}, request=httpx.Request("POST", "http://demo"))

        def get(url, **_):
            return httpx.Response(200, json=polls.pop(0), request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: submit)
        monkeypatch.setattr(httpx, "get", get)
        monkeypatch.setattr(time, "sleep", lambda seconds: None)
        return polls

    def test_a_done_job_returns_its_result(self, statuses, demo):
        statuses += [{"status": "pending"}, {"status": "running"}, {"status": "done", "result": {"x": 1}}]
        assert demo.submit_and_wait({}) == {"x": 1}

    @pytest.mark.parametrize("final", ["failed", "interrupted"])
    def test_a_job_that_did_not_finish_raises(self, statuses, demo, final):
        statuses += [{"status": "running"}, {"status": final, "error": "why"}]
        with pytest.raises(RuntimeError, match=f"pricing job {final}:\nwhy"):
            demo.submit_and_wait({})
        assert not statuses


class TestTheProfilingDemosSwitches:
    """`demo_profile_small.py`'s switches reach the server it starts (which is not started)."""

    @pytest.fixture
    def server_environment(self, monkeypatch):
        seen = {}

        def popen(*args, env=None, **kwargs):
            seen.update(env)
            return type("Process", (), {"pid": 0})()

        monkeypatch.setattr(subprocess, "Popen", popen)
        monkeypatch.setattr(demo_profile_small, "wait_until_healthy", lambda: None)
        monkeypatch.delenv("JAX_RISK_PROFILE_PHASE", raising=False)
        monkeypatch.delenv("JAX_RISK_PROFILE_WARMUP", raising=False)

        def started(argv):
            args = demo_profile_small.parse_args(argv)
            demo_profile_small.start_server(args.cold, args.disk_cache, args.phase)
            return seen
        return started

    def test_the_default_traces_a_repeat_of_the_whole_job(self, server_environment):
        env = server_environment([])
        assert env["JAX_RISK_PROFILE_WARMUP"] == "1" and "JAX_RISK_PROFILE_PHASE" not in env

    def test_a_phase_is_traced_alone(self, server_environment):
        """`--phase` traces one phase of the job (`engine.api.worker._profiled`)."""
        env = server_environment(["--cold", "--phase", "pricing"])
        assert env["JAX_RISK_PROFILE_PHASE"] == "pricing" and "JAX_RISK_PROFILE_WARMUP" not in env

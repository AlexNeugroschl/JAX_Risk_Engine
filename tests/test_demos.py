"""
The component demo (demos/demo_components.py) runs every section to completion, on any date;
the HTTP demos' job polling (demos/demo_http.py) stops at every final status.

I-28: the var_es section's swap once had no `evaluation_date`, so it was scheduled off ORE's
wall-clock today while the pillars stayed at `EVAL_DATE`, and was refused on every day but
2026-07-30. I-65 moved the sections out of `engine` modules' `__main__` blocks.
"""
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from demos import demo_http

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


class TestTheHttpDemosPolling:
    """Until roadmap 2.3 each HTTP demo polled until `done` or `failed`, so a job whose worker
    died (`interrupted`, final too) was polled forever."""

    @pytest.fixture
    def statuses(self, monkeypatch):
        """The job's statuses, one per poll; the requests go nowhere."""
        polls = []
        submit = httpx.Response(202, json={"job_id": "j"}, request=httpx.Request("POST", demo_http.API_BASE))

        def get(url, **_):
            return httpx.Response(200, json=polls.pop(0), request=httpx.Request("GET", url))

        monkeypatch.setattr(demo_http.httpx, "post", lambda *args, **kwargs: submit)
        monkeypatch.setattr(demo_http.httpx, "get", get)
        return polls

    def test_a_done_job_returns_its_result(self, statuses):
        statuses += [{"status": "pending"}, {"status": "running"}, {"status": "done", "result": {"x": 1}}]
        assert demo_http.submit_and_wait({}, poll_seconds=0.0) == {"x": 1}

    @pytest.mark.parametrize("final", ["failed", "interrupted"])
    def test_a_job_that_did_not_finish_raises(self, statuses, final):
        statuses += [{"status": "running"}, {"status": final, "error": "why"}]
        with pytest.raises(RuntimeError, match=f"pricing job {final}:\nwhy"):
            demo_http.submit_and_wait({}, poll_seconds=0.0)
        assert not statuses

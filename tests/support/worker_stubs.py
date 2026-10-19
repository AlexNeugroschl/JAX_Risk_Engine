"""
An engine worker with a stub pricer, for the process-level tests of tests/test_engine_worker.py:
the real queue loop, lock and parent watch (`engine.api.worker.serve`) around a pricer that
echoes its job, passing `"phases"` phase boundaries (`engine.api.worker.phase_boundary`, where a
cancelled job stops) and sleeping `"sleep"` seconds before each, so a test can catch a job
`running`, or cancel it mid-run, without pricing anything.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def echo_pricer(job):
    """`{"echo": <the request>, "kind": <the job's kind>}` after `"phases"` phase boundaries
    (default 0), each after sleeping the request's `"sleep"` seconds (default 0; once with no
    phases)."""
    from engine.api.worker import JobResult, phase_boundary

    body = json.loads(job.request)
    for n in range(max(int(body.get("phases", 0)), 1)):
        time.sleep(float(body.get("sleep", 0.0)))
        if body.get("phases"):
            phase_boundary(f"stub{n}")
    return JobResult(json.dumps({"echo": body, "kind": job.kind}))


def stub_worker_command(queue_path, lock_path, parent_pid=None) -> list:
    """The command line of a stub worker for `queue_path` under the host lock `lock_path`, from
    any working directory."""
    code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); from tests.support.worker_stubs import main; main()"
    command = [sys.executable, "-c", code, str(queue_path), str(lock_path)]
    return command + ([str(parent_pid)] if parent_pid is not None else [])


def failing_worker_command() -> list:
    """The command line of a worker that cannot start: it exits at once with an error, its
    traceback on stderr (a broken install, I-77)."""
    return [sys.executable, "-c", "raise SystemExit('the engine cannot be imported: a broken install')"]


def main() -> None:
    from engine.api.worker import serve

    queue_path, lock_path, *parent = sys.argv[1:]
    sys.exit(serve(queue_path, lock_path=lock_path, price=echo_pricer,
                   parent_pid=int(parent[0]) if parent else None))

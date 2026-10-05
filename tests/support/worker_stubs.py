"""
An engine worker with a stub pricer, for the process-level tests of tests/test_engine_worker.py:
the real queue loop, lock and parent watch (`engine.api.worker.serve`) around a pricer that
echoes its job and sleeps `"sleep"` seconds, so a test can catch a job `running` without
pricing anything.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def echo_pricer(request: bytes) -> str:
    """`{"echo": <the request>}` after sleeping the request's `"sleep"` seconds (default 0)."""
    body = json.loads(request)
    time.sleep(float(body.get("sleep", 0.0)))
    return json.dumps({"echo": body})


def stub_worker_command(queue_path, parent_pid=None) -> list:
    """The command line of a stub worker for `queue_path`, from any working directory."""
    code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); from tests.support.worker_stubs import main; main()"
    command = [sys.executable, "-c", code, str(queue_path)]
    return command + ([str(parent_pid)] if parent_pid is not None else [])


def main() -> None:
    from engine.api.worker import serve

    queue_path, *parent = sys.argv[1:]
    sys.exit(serve(queue_path, price=echo_pricer, parent_pid=int(parent[0]) if parent else None))

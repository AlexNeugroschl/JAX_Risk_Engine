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
    phases).

    For a worker set of several hosts (`distributed_main`): each host first writes a file
    `<job id>.<host index>` into the request's `"ran_in"` directory; `"fail"` raises a
    `ValueError` (a `bad-terms` failure) on every host; `"host_failure": [host, marker]` raises a
    `RuntimeError` (an `infrastructure` one) on that host while the file `marker` does not exist,
    creating it, so the job fails its first run there and passes when run again."""
    from engine.api.worker import JobResult, phase_boundary

    body = json.loads(job.request)
    if "ran_in" in body:
        import jax

        (Path(body["ran_in"]) / f"{job.id}.{jax.process_index()}").write_text("ran")
    if body.get("fail"):
        raise ValueError(f"the stub's terms: {body['fail']}")
    if "host_failure" in body:
        import jax

        host, marker = body["host_failure"]
        if jax.process_index() == host and not Path(marker).exists():
            Path(marker).write_text("failed once")
            raise RuntimeError(f"host {host}'s device failed")
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


def distributed_worker_command(queue_path, lock_path, real: bool = False) -> list:
    """The command line of one worker of a set (`distributed_main`), with the stub pricer or,
    `real`, the engine's; its host is the environment's (`engine.api.hosts.CLUSTER_ENV`)."""
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
            f"from tests.support.worker_stubs import distributed_main; distributed_main()")
    return [sys.executable, "-c", code, str(queue_path), str(lock_path)] + (["real"] if real else [])


def distributed_main() -> None:
    """One worker of a set on several hosts, with the stub pricer (or the engine's, given
    `real`): the set serves the queue until no job is pending (the primary's `drain`), then
    exits."""
    from engine.api.worker import serve

    queue_path, lock_path, *real = sys.argv[1:]
    sys.exit(serve(queue_path, lock_path=lock_path, price=None if real else echo_pricer, distributed=True,
                   drain=True, poll_seconds=0.01))

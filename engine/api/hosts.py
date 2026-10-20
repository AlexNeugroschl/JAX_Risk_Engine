"""
The hosts an engine worker runs a job on (I-61; docs/planning/details/precision.md §11.3): its
own host alone, or every host of a TPU pod slice (or any `jax.distributed` cluster), one worker
per host, all running the same job at the same time.

Under SPMD every host must run the same program, so the hosts cannot each claim from the queue
as one host's worker does. The **primary** host (JAX's process 0) alone opens the queue: it
claims each job and hands it to the others (`share`), decides at each phase boundary whether a
cancel has reached it (`agree`), learns how the job ended on every host (`outcomes`), and writes
the row; it alone purges finished jobs. Every message is a broadcast from the primary through
JAX's own collectives (`jax.experimental.multihost_utils`), in arrays of fixed shapes, so the
messages compile two small programs once and nothing per job.

    hosts = LOCAL                        # one host: every call returns its argument
    hosts = distributed_hosts(environ)   # jax.distributed.initialize; this host's index

**Failures.** A job's failure of a class the request determines (`bad-terms`,
`missing-market-data`, `unsupported-product`, `numerical-failure`) is raised the same way on
every host, which all report it (`outcomes`). Any other exception may be one host's alone (its
device, its memory) while the other hosts wait in a collective of the job, so that host stops
its process (`fail_stop`); JAX's distributed service then stops every other host of the set
(measured 2026-10-19: "Terminating process because the JAX distributed service detected fatal
errors"). Restarted, the primary puts the job back in the queue once and runs it again whole
(`engine.api.job_queue.JobQueue.requeue_running`).

`distributed_hosts` reads the cluster from `JAX_RISK_COORDINATOR` (`host:port` of process 0),
`JAX_RISK_NUM_PROCESSES` and `JAX_RISK_PROCESS_ID`, all three or none: none lets JAX detect the
cluster itself, as it does on a Cloud TPU VM. A CPU cluster (the tests) also needs JAX's CPU
collectives, `JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo`.
"""
import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from engine.api.job_queue import RUNNING, Job

#: A command from the primary host (`Hosts.share`).
IDLE, JOB, STOP = 0, 1, 2
#: How a job ended on a host (`Hosts.outcomes`).
DONE, FAILED, CANCELLED = 0, 1, 2
OUTCOME_NAMES = {DONE: "done", FAILED: "failed", CANCELLED: "cancelled"}
#: The bytes of a job's message broadcast at once: one compiled broadcast for every job, a body
#: of any size sent in as many chunks as it needs.
CHUNK_BYTES = 1 << 20
#: The exit status of a host that stopped its process on a failure of its own (`fail_stop`).
EXIT_HOST_FAILED = 4
#: The cluster's variables (see the module docstring).
CLUSTER_ENV = ("JAX_RISK_COORDINATOR", "JAX_RISK_NUM_PROCESSES", "JAX_RISK_PROCESS_ID")


@dataclass(frozen=True)
class Command:
    """What every host does next: wait (`IDLE`), run `job` (`JOB`) or exit (`STOP`)."""
    kind: int
    job: Optional[Job] = None


@dataclass(frozen=True)
class Hosts:
    """This host's place among the hosts that run each job: `index` of `count` (JAX's process
    index and count). `count == 1` is one host, where nothing is shared."""
    index: int = 0
    count: int = 1

    @property
    def primary(self) -> bool:
        """Whether this host owns the queue (JAX's process 0)."""
        return self.index == 0

    def share(self, command: Optional[Command]) -> Command:
        """The primary's `command`, on every host (the others pass None)."""
        if self.count == 1:
            return command
        import numpy as np

        payload = _encode(command.job) if self.primary and command.kind == JOB else b""
        kind, size = (command.kind, len(payload)) if self.primary else (0, 0)
        kind, size = (int(v) for v in self._broadcast(np.array([kind, size], dtype=np.int64)))
        if kind != JOB:
            return Command(kind)
        chunks = []
        for start in range(0, size, CHUNK_BYTES):
            chunk = np.zeros(CHUNK_BYTES, dtype=np.uint8)
            if self.primary:
                part = np.frombuffer(payload[start:start + CHUNK_BYTES], dtype=np.uint8)
                chunk[:part.size] = part
            chunks.append(self._broadcast(chunk)[:min(CHUNK_BYTES, size - start)].tobytes())
        return Command(JOB, command.job if self.primary else _decode(b"".join(chunks)))

    def agree(self, flag: bool) -> bool:
        """The primary's `flag`, on every host: the hosts decide together (a cancel at a phase
        boundary)."""
        if self.count == 1:
            return flag
        import numpy as np

        return bool(self._broadcast(np.array([int(flag)], dtype=np.int64))[0])

    def outcomes(self, outcome: int) -> Tuple[int, ...]:
        """Every host's `outcome` of the job just run (`DONE`, `FAILED`, `CANCELLED`), by host
        index."""
        if self.count == 1:
            return (outcome,)
        import numpy as np
        from jax.experimental import multihost_utils

        gathered = multihost_utils.process_allgather(np.array([outcome], dtype=np.int64))
        return tuple(int(v) for v in np.asarray(gathered).reshape(-1))

    def _broadcast(self, values):
        from jax.experimental import multihost_utils
        import numpy as np

        return np.asarray(multihost_utils.broadcast_one_to_all(values, is_source=self.primary))


#: One host: nothing is shared.
LOCAL = Hosts()


def distributed_hosts(environ: Optional[Mapping[str, str]] = None) -> Hosts:
    """Join the cluster (`jax.distributed.initialize`, before any other JAX work) and return this
    host's place in it (see the module docstring)."""
    import jax

    environ = os.environ if environ is None else environ
    given = [environ.get(name) for name in CLUSTER_ENV]
    if any(given) and not all(given):
        raise ValueError(f"set all of {', '.join(CLUSTER_ENV)}, or none to let JAX detect the cluster")
    if all(given):
        coordinator, processes, process = given
        jax.distributed.initialize(coordinator, int(processes), int(process))
    else:
        jax.distributed.initialize()
    return Hosts(jax.process_index(), jax.process_count())


def fail_stop(exc: BaseException) -> None:
    """Stop this host's process after a failure that may be its own (see the module docstring):
    the traceback to stderr, then the exit, which stops every other host of the set."""
    sys.stderr.write("".join(traceback.format_exception(exc)))
    sys.stderr.write(f"engine worker: a failure of this host alone stops the worker set (exit {EXIT_HOST_FAILED}); "
                     f"the primary runs the job again once restarted\n")
    sys.stderr.flush()
    os._exit(EXIT_HOST_FAILED)


def _encode(job: Job) -> bytes:
    """A job as one message: its id and kind as a JSON line, then its request body."""
    return json.dumps({"id": job.id, "kind": job.kind}).encode("utf-8") + b"\n" + bytes(job.request)


def _decode(message: bytes) -> Job:
    head, request = message.split(b"\n", 1)
    fields = json.loads(head.decode("utf-8"))
    return Job(id=fields["id"], kind=fields["kind"], status=RUNNING, request=request)

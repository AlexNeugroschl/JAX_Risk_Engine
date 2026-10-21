"""Process failures must stay observable on Windows and Linux (I-27)."""
import io
import os
import sys
import time
import pytest

from demos.benchmark_reporting import launch_worker


def test_unicode_output_is_retained_with_an_ascii_terminal(tmp_path, monkeypatch):
    terminal_bytes = io.BytesIO()
    terminal = io.TextIOWrapper(terminal_bytes, encoding="ascii", errors="strict")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    command = [sys.executable, "-c", "for i in range(40): print(chr(0x1f9ee)*2048)"]
    with monkeypatch.context() as context:
        context.setattr(sys, "stdout", terminal)
        code = launch_worker(command, tmp_path, 5.)
    terminal.flush()
    assert code == 0
    assert (tmp_path / "worker.log").read_text(encoding="utf-8") == (chr(0x1f9ee)*2048 + "\n")*40
    assert b"\\U0001f9ee" in terminal_bytes.getvalue()


def test_worker_output_is_utf8_even_with_an_inherited_ascii_encoding(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    assert launch_worker([sys.executable, "-c", "print(chr(0x1f9ee))"], tmp_path, 5.) == 0
    assert (tmp_path / "worker.log").read_text(encoding="utf-8") == chr(0x1f9ee) + "\n"


def test_a_closed_terminal_does_not_block_the_persistent_log(tmp_path, monkeypatch):
    class ClosedTerminal:
        encoding = "utf-8"
        def write(self, value):
            raise OSError("terminal disconnected")
        def flush(self):
            pass
    with monkeypatch.context() as context:
        context.setattr(sys, "stdout", ClosedTerminal())
        code = launch_worker([sys.executable, "-c", "for i in range(40): print('x'*8192)"], tmp_path, 5.)
    assert code == 0
    assert (tmp_path / "worker.log").read_text(encoding="utf-8") == ("x"*8192 + "\n")*40


def test_timeout_stops_descendants_that_inherit_the_output_pipe(tmp_path):
    code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(12)']); print('spawned',flush=True); time.sleep(12)"
    start = time.monotonic()
    assert launch_worker([sys.executable, "-c", code], tmp_path, 1.5) == 124
    assert time.monotonic() - start < 6.
    assert "spawned" in (tmp_path / "worker.log").read_text(encoding="utf-8")


def test_normal_exit_closes_descendants_that_still_hold_the_output_pipe(tmp_path):
    code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(12)']); print('done',flush=True)"
    start = time.monotonic()
    assert launch_worker([sys.executable, "-c", code], tmp_path, 5.) == 0
    assert time.monotonic() - start < 6.
    assert (tmp_path / "worker.log").read_text(encoding="utf-8").strip() == "done"


@pytest.mark.skipif(os.name != "nt", reason="Windows job assignment race")
def test_windows_worker_cannot_spawn_before_job_assignment(tmp_path, monkeypatch):
    from demos.worker_process import WindowsJob
    attach = WindowsJob.attach
    def delayed(self, process):
        time.sleep(1.)
        attach(self, process)
    monkeypatch.setattr(WindowsJob, "attach", delayed)
    test_timeout_stops_descendants_that_inherit_the_output_pipe(tmp_path)

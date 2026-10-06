"""
The engine's process defaults for accelerators (roadmap 2.2, I-79; `engine/__init__.py`):

  * importing `engine` turns XLA's GPU preallocation off, makes matrix products run at their
    operands' precision, and excludes GPU kernels that are not deterministic, each unless the
    environment says otherwise (either XLA determinism flag), and keeps any other XLA flags;
  * a served API keeps its own JAX on the CPU, so only the engine worker opens an accelerator;
    an API process that has already opened one is left on it, with a warning;
  * on an accelerator (skipped on the CPU, which has neither problem): a float32 matrix product
    is float32, not TensorFloat-32 or bfloat16, and the AD Greeks repeat bit for bit.

The process checks start a fresh interpreter: JAX reads these settings when a process first
uses a device, so they cannot be changed in the test's own process.
"""
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import xla_bridge

from engine.api import app as api_app
from engine.risk.greeks import portfolio_greeks
from engine.valuation.config import LgmSwaptionEngineConfig, PricingConfig
from tests.support import portfolio as shared
from tests.support.worker_stubs import ROOT

PREALLOCATE, MATMUL, XLA_FLAGS = "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_DEFAULT_MATMUL_PRECISION", "XLA_FLAGS"
DETERMINISTIC = "--xla_gpu_exclude_nondeterministic_ops=true"
SETTINGS = "import os, engine, jax; print(os.environ.get({!r}), jax.config.jax_default_matmul_precision)"

on_an_accelerator = pytest.mark.skipif(jax.default_backend() == "cpu", reason="an accelerator's default, not the CPU's")


def _run(code: str, **env) -> str:
    environ = {k: v for k, v in os.environ.items() if k not in (PREALLOCATE, MATMUL, XLA_FLAGS, "JAX_PLATFORMS")}
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**environ, **env},
                               capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return completed.stdout.strip().splitlines()[-1]


class TestProcessDefaults:
    def test_importing_the_engine_sets_them(self):
        assert _run(f"import os, engine; print(os.environ[{PREALLOCATE!r}])") == "false"
        assert _run(SETTINGS.format(XLA_FLAGS)) == f"{DETERMINISTIC} highest"

    def test_they_hold_when_jax_was_imported_first(self):
        assert _run("import jax; " + SETTINGS.format(XLA_FLAGS)) == f"{DETERMINISTIC} highest"

    @pytest.mark.parametrize("flags", ["--xla_gpu_deterministic_ops=true", "--xla_gpu_exclude_nondeterministic_ops=false"])
    def test_an_explicit_setting_wins(self, flags):
        assert _run(f"import os, engine; print(os.environ[{PREALLOCATE!r}])", **{PREALLOCATE: "true"}) == "true"
        assert _run(SETTINGS.format(XLA_FLAGS), **{XLA_FLAGS: flags, MATMUL: "tensorfloat32"}) == f"{flags} tensorfloat32"

    def test_other_xla_flags_are_kept(self):
        flags = "--xla_force_host_platform_device_count=2"
        assert _run(SETTINGS.format(XLA_FLAGS), **{XLA_FLAGS: flags}) == f"{flags} {DETERMINISTIC} highest"


class TestTheApiKeepsJaxOnTheCpu:
    def test_a_served_api_uses_only_the_cpu(self):
        """Through the app's start-up (`with TestClient`), as uvicorn runs it: `/version`, which
        asks JAX for its backend, opens no backend but the CPU."""
        code = """
import json, jax
from fastapi.testclient import TestClient
from jax._src import xla_bridge
from engine.api.app import app
with TestClient(app) as client:
    version = client.get("/version").json()
print(json.dumps({"platforms": jax.config.jax_platforms, "backends": sorted(xla_bridge.backends()),
                  "version": version["jax_backend"]}))
"""
        seen = json.loads(_run(code))
        assert seen["platforms"] == "cpu" and seen["backends"] == ["cpu"], seen
        assert seen["version"].startswith("cpu ")

    def test_a_process_with_no_device_yet_is_pinned(self, monkeypatch):
        updates = []
        monkeypatch.setattr(xla_bridge, "backends_are_initialized", lambda: False)
        monkeypatch.setattr(api_app.jax.config, "update", lambda *args: updates.append(args))
        api_app.keep_jax_on_the_cpu()
        assert updates == [("jax_platforms", "cpu")]

    def test_a_process_already_on_an_accelerator_is_left_there_with_a_warning(self, monkeypatch):
        updates = []
        monkeypatch.setattr(xla_bridge, "backends_are_initialized", lambda: True)
        monkeypatch.setattr(api_app.jax, "default_backend", lambda: "gpu")
        monkeypatch.setattr(api_app.jax.config, "update", lambda *args: updates.append(args))
        with pytest.warns(RuntimeWarning, match="'gpu' backend"):
            api_app.keep_jax_on_the_cpu()
        assert updates == []

    def test_a_process_already_on_the_cpu_is_left_quietly(self, monkeypatch, recwarn):
        monkeypatch.setattr(xla_bridge, "backends_are_initialized", lambda: True)
        monkeypatch.setattr(api_app.jax, "default_backend", lambda: "cpu")
        api_app.keep_jax_on_the_cpu()
        assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


@on_an_accelerator
class TestOnAnAccelerator:
    def test_a_float32_matrix_product_is_float32(self):
        """TensorFloat-32 keeps 10 mantissa bits (relative error ~1e-3), bfloat16 7; float32
        keeps 23 (~1e-7 for a sum of 256 products)."""
        a, b = jax.random.normal(jax.random.key(0), (2, 256, 256))
        got = np.asarray(jnp.asarray(a, jnp.float32) @ jnp.asarray(b, jnp.float32), dtype=np.float64)
        exact = np.asarray(a, np.float64) @ np.asarray(b, np.float64)
        assert np.max(np.abs(got - exact)) / np.max(np.abs(exact)) < 1e-5

    def test_the_greeks_repeat_bit_for_bit(self):
        """The AD Greeks' scatter-adds are the kernels a GPU would otherwise run with atomics
        (an ulp apart between runs without the default, measured on an RTX 5060)."""
        fast = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)
        trades, market = [shared.trades()["bermudan-payer-physical"]], shared.market()
        first, second = (portfolio_greeks(trades, market, "USD", PricingConfig(bermudan=fast))[0] for _ in range(2))
        assert first.keys() == second.keys()
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)

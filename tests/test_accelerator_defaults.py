"""
Device settings and who sets them (decision A-22; I-79, I-80):

  * importing `engine` sets no device setting, only x64: no GPU preallocation variable, no
    XLA flag and no default matrix-product precision, whether JAX was imported first or not;
  * the engine worker runs deterministic GPU kernels: its environment appends the XLA flag,
    keeps the other flags and yields to either determinism flag set by the operator; a demo's
    server turns GPU preallocation off unless its environment says otherwise; the test
    processes have both;
  * every matrix product of the engine's JAX code states its precision
    (`engine.precision.matmul`), checked by tracing the pipelines, AD's transposes included;
  * a served API keeps its own JAX on the CPU, so only the engine worker opens an accelerator;
    an API process that has already opened one is left on it, with a warning;
  * on an accelerator (skipped on the CPU, which has neither problem): a float32 matrix product
    through the engine is float32, not TensorFloat-32 or bfloat16, and the AD Greeks repeat bit
    for bit.

The import checks start a fresh interpreter: the test process's own environment is set by
`tests/conftest.py`.
"""
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest
from jax._src import source_info_util, xla_bridge
from jax._src.lax.lax import dot_general_p

from demos import demo_profile_small, demo_structured
from engine.api import app as api_app
from engine.api import worker
from engine.market_simulation.config import CamConfig, HullWhiteConfig, LgmConfig
from engine.precision import Precision, matmul, product_precision
from engine.pricing.config import LgmSwaptionEngineConfig, PricingConfig
from engine.risk.greeks.ad import portfolio_greeks
from engine.risk.market import MarketRiskRequest, monte_carlo_scenarios, run_market_risk
from engine.run import GreeksConfig, PortfolioRequest, RunConfig, price_portfolio
from tests import market_risk_support as mr
from tests.support import portfolio as shared
from tests.support.worker_stubs import ROOT

PREALLOCATE, MATMUL, XLA_FLAGS = "XLA_PYTHON_CLIENT_PREALLOCATE", "JAX_DEFAULT_MATMUL_PRECISION", "XLA_FLAGS"
DETERMINISTIC = worker.DETERMINISTIC_KERNELS_FLAG
ENGINE = ROOT / "engine"
HELPER = ENGINE / "precision" / "products.py"
FAST = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)

on_an_accelerator = pytest.mark.skipif(jax.default_backend() == "cpu", reason="an accelerator's default, not the CPU's")


def _run(code: str, **env) -> str:
    environ = {k: v for k, v in os.environ.items() if k not in (PREALLOCATE, MATMUL, XLA_FLAGS, "JAX_PLATFORMS")}
    completed = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**environ, **env},
                               capture_output=True, text=True, timeout=300)
    assert completed.returncode == 0, completed.stderr[-4000:]
    return completed.stdout.strip().splitlines()[-1]


class TestImportingTheEngine:
    SETTINGS = (f"import os, engine, jax; print([os.environ.get({PREALLOCATE!r}), os.environ.get({XLA_FLAGS!r}), "
                "jax.config.jax_default_matmul_precision, jax.config.jax_enable_x64])")

    @pytest.mark.parametrize("before", ["", "import jax; "], ids=["engine first", "jax first"])
    def test_sets_x64_and_no_device_setting(self, before):
        assert _run(before + self.SETTINGS) == "[None, None, None, True]"


class TestTheWorkerRunsDeterministicKernels:
    def test_the_flag_is_added(self):
        assert worker.deterministic_kernels_environment({}) == {XLA_FLAGS: DETERMINISTIC}

    def test_other_xla_flags_are_kept(self):
        flags = "--xla_force_host_platform_device_count=2"
        assert worker.deterministic_kernels_environment({XLA_FLAGS: flags}) == {XLA_FLAGS: f"{flags} {DETERMINISTIC}"}

    @pytest.mark.parametrize("flags", ["--xla_gpu_deterministic_ops=true", "--xla_gpu_exclude_nondeterministic_ops=false"])
    def test_an_explicit_determinism_flag_wins(self, flags):
        assert worker.deterministic_kernels_environment({XLA_FLAGS: flags}) == {}

    def test_main_sets_it_before_serving(self, monkeypatch, tmp_path):
        """`main` puts it in the process's environment before `serve` imports JAX, beside the
        compilation cache's variables."""
        monkeypatch.setattr(os, "environ", {"PATH": os.environ.get("PATH", "")})
        seen = {}
        monkeypatch.setattr(worker, "serve", lambda *args, **kwargs: seen.update(os.environ) or 0)
        with pytest.raises(SystemExit) as exit_:
            worker.main(["--queue", str(tmp_path / "jobs.sqlite3")])
        assert exit_.value.code == 0
        assert seen[XLA_FLAGS] == DETERMINISTIC
        assert Path(seen["JAX_COMPILATION_CACHE_DIR"]) == tmp_path.resolve() / worker.COMPILATION_CACHE_DIRNAME

    def test_the_test_processes_run_them_too(self):
        """`tests/conftest.py`: the tests price in-process, on a GPU shared by every test
        process."""
        assert "deterministic_ops" in os.environ[XLA_FLAGS]
        assert PREALLOCATE in os.environ


#: The demos that start a server, each starting it in its own way. `demo_api.py` does the same
#: but runs as a script on import.
DEMO_SERVERS = {"demo_structured": lambda: demo_structured.start_server(),
                "demo_profile_small": lambda: demo_profile_small.start_server(cold=True, disk_cache=True)}


class TestADemoServerSharesTheGpu:
    """The environment each demo starts its server in (the server is not started)."""

    @pytest.fixture
    def server_environment(self, monkeypatch):
        def started(start, environ):
            seen = {}

            def popen(*args, env=None, **kwargs):
                seen.update(env)
                return type("Process", (), {"pid": 0})()

            monkeypatch.setattr(os, "environ", dict(environ))
            monkeypatch.setattr(subprocess, "Popen", popen)
            for module in (demo_structured, demo_profile_small):
                monkeypatch.setattr(module, "wait_until_healthy", lambda: None)
            start()
            return seen
        return started

    @pytest.mark.parametrize("demo", DEMO_SERVERS)
    def test_preallocation_is_off(self, server_environment, demo):
        env = server_environment(DEMO_SERVERS[demo], {"PATH": "x"})
        assert env[PREALLOCATE] == "false" and env["PATH"] == "x"

    @pytest.mark.parametrize("demo", DEMO_SERVERS)
    def test_an_explicit_setting_wins(self, server_environment, demo):
        assert server_environment(DEMO_SERVERS[demo], {PREALLOCATE: "true"})[PREALLOCATE] == "true"


def _engine_line() -> str:
    """The innermost line of engine code that binds the current product, the helper's caller
    for a product through `engine.precision.matmul`, or "outside the engine". A transpose AD generates is the original product's, from the source information
    JAX sets while transposing; otherwise the Python stack (JAX's own captured traceback stops
    at `jnp.matmul`'s jit boundary)."""
    context = source_info_util._source_info_context.context.traceback
    frames = [(f.file_name, f.start_line) for f in source_info_util.user_frames(context)] if context else []
    frame = sys._getframe()
    while frame is not None:
        frames.append((frame.f_code.co_filename, frame.f_lineno))
        frame = frame.f_back
    for file_name, line in frames:
        path = Path(file_name)
        if path.is_relative_to(ENGINE) and path != HELPER:
            return f"{path.relative_to(ROOT).as_posix()}:{line}"
    return "outside the engine"


@pytest.fixture
def products(monkeypatch):
    """Every matrix product (`dot_general`) bound while the test runs, traced, transposed by AD
    or run eagerly: (engine line, operand dtype, precision). JAX's caches are cleared first, so
    every program is traced again rather than served from this process's caches (the
    persistent cache is keyed on the traced program, so it saves only the compile). `jnp.matmul`
    is jitted itself, so a second product of the same shapes and precision is a cache hit and
    not recorded again; a product that states another precision never shares its entry."""
    seen = []
    bind = dot_general_p.bind

    def recording(*args, **params):
        seen.append((_engine_line(), jnp.result_type(*args[:2]), params["precision"]))
        return bind(*args, **params)

    jax.clear_caches()
    monkeypatch.setattr(dot_general_p, "bind", recording)
    return seen


class TestEveryMatrixProductStatesItsPrecision:
    @pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
    def test_full_precision_for_float64_and_float32(self, dtype):
        assert product_precision(dtype) == jax.lax.Precision.HIGHEST
        a = jnp.arange(6, dtype=dtype).reshape(2, 3)
        jaxpr = jax.make_jaxpr(matmul)(a, a.T)
        (eqn,) = [e for e in jaxpr.eqns if e.primitive is dot_general_p]
        assert eqn.params["precision"] == (jax.lax.Precision.HIGHEST,) * 2
        np.testing.assert_array_equal(matmul(a, a.T), np.asarray(a) @ np.asarray(a).T)

    @pytest.mark.parametrize("name", ["float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"])
    def test_a_format_not_enabled_for_compute_is_refused(self, name):
        with pytest.raises(ValueError, match=rf"{name} is not enabled yet \(F-07\)"):
            product_precision(jnp.dtype(name))

    @pytest.mark.slow
    def test_in_every_pipeline(self, products):
        """The simulation (LGM and Hull-White, its calibration and the Brownian bridge), the
        scenario market, every product's path pricing (the Bermudan's and American's per-path
        recalibration included), today's values, the AD Greeks and market risk, at float64 and
        float32. A product without its compute format's precision would run at the device's
        default; one outside the engine's code would be a product the engine cannot vouch for.
        Red first: with the nine products bare (before 2026-10-06), it named `cam.py:501`,
        `random.py:115`, `ore_lgm.py:270` and `bermudan_swaption.py:697, 700, 701` (the modules'
        names then), each at float64 and float32."""
        assert jax.config.jax_default_matmul_precision is None  # nothing fills it in for them
        trades, market = list(shared.trades().values()), shared.market()
        dates = tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (6, 12))
        lgm = CamConfig(dates=dates, base_currency="USD", samples=32, seed=3,
                        ir={"USD": LgmConfig(0.03, 0.01, ("1Y", "2Y", "5Y"), ("9Y", "8Y", "5Y"))})
        hull_white = dataclasses.replace(lgm, ir={"USD": HullWhiteConfig(0.03, 0.01, ("1Y", "2Y", "5Y"),
                                                                          ("9Y", "8Y", "5Y"))})
        run = RunConfig(simulation=lgm, pricing=PricingConfig(bermudan=FAST, american=FAST),
                        greeks=GreeksConfig(method="AD"))
        price_portfolio(PortfolioRequest(market=market, trades=trades, config=run, compute_greeks=True))
        price_portfolio(PortfolioRequest(market=market, trades=trades, config=dataclasses.replace(
            run, simulation=hull_white, precision=Precision.throughout("float32"))))
        scenarios = monte_carlo_scenarios(mr.factors(), mr.covariance(), 10, 16, seed=3)
        run_market_risk(MarketRiskRequest([mr.swap(), mr.european(), mr.bermudan(), mr.bond()], mr.market(),
                                          scenarios, mr.PRICING, precision=Precision.throughout("float32")))

        wrong = sorted({(line, str(dtype), str(precision)) for line, dtype, precision in products
                        if line == "outside the engine" or precision != (product_precision(dtype),) * 2})
        assert not wrong, f"{len(wrong)} matrix products without their stated precision: {wrong}"
        lines = {line.rsplit(":", 1)[0] for line, _, _ in products}
        assert lines == {"engine/market_simulation/paths.py", "engine/market_simulation/sobol.py",
                         "engine/calibration/ore_lgm.py", "engine/pricing/lgm_grid.py"}, lines
        assert {str(dtype) for _, dtype, _ in products} == {"float64", "float32"}


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
        """Through the engine's product (`engine.precision.matmul`), with no process-wide
        setting: TensorFloat-32 keeps 10 mantissa bits (relative error ~1e-3), bfloat16 7;
        float32 keeps 23 (~1e-7 for a sum of 256 products)."""
        a, b = jax.random.normal(jax.random.key(0), (2, 256, 256))
        got = np.asarray(matmul(jnp.asarray(a, jnp.float32), jnp.asarray(b, jnp.float32)), dtype=np.float64)
        exact = np.asarray(a, np.float64) @ np.asarray(b, np.float64)
        assert np.max(np.abs(got - exact)) / np.max(np.abs(exact)) < 1e-5

    def test_the_greeks_repeat_bit_for_bit(self):
        """The AD Greeks' scatter-adds are the kernels a GPU would otherwise run with atomics
        (an ulp apart between runs without the flag, measured on an RTX 5060); the test
        processes run with it, as the engine worker does."""
        trades, market = [shared.trades()["bermudan-payer-physical"]], shared.market()
        first, second = (portfolio_greeks(trades, market, "USD", PricingConfig(bermudan=FAST))[0] for _ in range(2))
        assert first.keys() == second.keys()
        for key in first:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)

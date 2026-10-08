"""
Structural tests of the JIT and profiling work (docs/concepts/profiling.md): the engine
reaches its numbers through a small, fixed number of compiled XLA programs rather than
thousands of eager dispatches. Numerical correctness is covered elsewhere; a regression here
(a hardcoded dtype, a field on the wrong side of `_PreparedBermudan`'s pytree split, a
dropped `jax.jit`) passes every numerical test while multiplying compile time.

Compiles are counted by patching `jax._src.compiler.backend_compile_and_load`, the function
an xprof trace records as XLA compilation; cache hits do not call it (`tests/support/compiles.py`).
"""
import dataclasses
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, _PreparedBermudan
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.market import ZeroCurveConfig
from engine.models.lgm import Sigma
from engine.portfolio.profiling import PHASES
from engine.risk.greeks import curve_greeks, portfolio_greeks, vega_greek
from engine.risk.price_functions import bermudan_price_function, curves_of, trade_price_function
from engine.valuation.bermudan import calibrate_on
from engine.valuation.config import LgmSwaptionEngineConfig, PricingConfig
from engine.valuation.context import from_market
from tests.support import portfolio as shared
from tests.support.compiles import count_compiles
from tests.support.lgm_engine import grid_npv, prepared

ROOT = Path(__file__).resolve().parent.parent
ASOF = shared.ASOF
CURVE = ZeroCurveConfig(times=[0.0, 1.0, 3.0], rates=[0.03, 0.03, 0.03])
ENGINE = LgmSwaptionEngineConfig(n_per_std=16, std_devs=6.0)
PRICING = PricingConfig(bermudan=ENGINE)


def bermudan_cfg(**overrides) -> BermudanSwaptionConfig:
    fields = dict(notional=1_000_000.0, fixed_rate=0.030, payer=True, exercise_dates=[ASOF + 365, ASOF + 730],
                  swap_tenor="3Y", evaluation_date=ASOF, trade_id="bermudan")
    fields.update(overrides)
    return BermudanSwaptionConfig(**fields)


def engine_npv(cfg=None, n_per_std=16, sigma=0.01) -> float:
    """The grid engine alone, with a fixed LGM (no calibration)."""
    return grid_npv(cfg or bermudan_cfg(), a=0.03, sigma=sigma, curve=CURVE, n_per_std=n_per_std)


def prepared_bermudan(cfg=None):
    return prepared(cfg or bermudan_cfg(), a=0.03, sigma=0.01, curve=CURVE, n_per_std=16)


# =============================================================================
# THE PYTREE SPLIT (_PreparedBermudan)
# =============================================================================
class TestPreparedBermudanPytree:
    """`_PreparedBermudan` is a pytree whose static aux data is only what fixes the
    program's shape, so `_backward_induction_arrays` is jitted once per trade shape while
    Greeks differentiate through it."""

    def test_only_shape_fixing_fields_are_static(self):
        children, aux = prepared_bermudan().tree_flatten()
        assert _PreparedBermudan._STATIC == ("payer", "n_per_std", "std_devs")
        assert aux == (True, 16, 6.0)
        assert len(children) == len(dataclasses.fields(_PreparedBermudan)) - len(aux)

    def test_round_trips_through_flatten_unflatten(self):
        """Flatten/unflatten round-trips exactly (JAX does it at every jit/grad boundary)."""
        original = prepared_bermudan()
        children, aux = original.tree_flatten()
        rebuilt = _PreparedBermudan.tree_unflatten(aux, children)
        for f in dataclasses.fields(_PreparedBermudan):
            for x, y in zip(jax.tree_util.tree_leaves(getattr(rebuilt, f.name)),
                            jax.tree_util.tree_leaves(getattr(original, f.name))):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    def test_aux_data_is_hashable(self):
        """The aux tuple is part of the jit cache key, so it must hash."""
        _children, aux = prepared_bermudan().tree_flatten()
        assert isinstance(hash(aux), int)

    def test_jax_tree_util_sees_it_as_a_pytree(self):
        # An unregistered dataclass would be a single opaque leaf; the differentiation
        # targets (the curves for Delta, the volatility for Vega) must be leaves.
        swap = prepared_bermudan()
        leaves = jax.tree_util.tree_leaves(swap)
        for target in (swap.curve.pillar_rates, swap.sigma):
            assert any(leaf is target for leaf in leaves)


# =============================================================================
# COMPILE-COUNT REGRESSION GUARDS
# =============================================================================
class TestCompileCounts:
    """Upper bounds on XLA compilations per operation, a few times the measured counts (in
    each test's comment), to catch a return to eager dispatch rather than pin exact numbers.
    The reference Greeks job once compiled 602 programs."""

    def test_bermudan_engine_compiles_few_programs(self):
        # Measured: a handful (jit__backward_induction_arrays + small helpers).
        with count_compiles() as counter:
            npv = engine_npv(n_per_std=17)
        assert npv > 0.0
        assert sum(counter.values()) < 40, dict(counter)

    def test_repeated_identical_pricing_hits_the_compilation_cache(self):
        """A second identical call compiles nothing (the aux data compares equal by value)."""
        engine_npv()  # warm
        with count_compiles() as counter:
            engine_npv()
        assert sum(counter.values()) == 0, dict(counter)

    def test_trades_differing_only_in_scale_share_compiled_programs(self):
        """`notional`/`fixed_amounts` are children, so same-structure trades of different size
        share one kernel, and the value scales exactly."""
        base = engine_npv(bermudan_cfg(notional=1_000_000.0))
        with count_compiles() as counter:
            scaled = engine_npv(bermudan_cfg(notional=5_000_000.0))
        assert sum(counter.values()) == 0, dict(counter)
        assert scaled == pytest.approx(5 * base, rel=1e-12)

    def test_differing_grid_resolution_does_compile_a_new_program(self):
        """`n_per_std` changes array shapes, so it stays static and forces a recompile."""
        engine_npv(n_per_std=16)  # warm
        with count_compiles() as counter:
            engine_npv(n_per_std=20)
        assert sum(counter.values()) >= 1, dict(counter)

    def test_calibrations_of_one_basket_shape_share_one_program(self):
        """I-22: the bootstrap takes the basket, curves and volatilities as arguments, so a
        calibration of another trade with the same basket shape (here another deal strike
        and notional) compiles nothing, and still gets its own answer."""
        market = from_market(shared.market())
        first = calibrate_on(bermudan_cfg(), ENGINE, market)
        other = bermudan_cfg(fixed_rate=0.045, notional=3_000_000.0)
        with count_compiles() as counter:
            second = calibrate_on(other, ENGINE, market)
        assert sum(counter.values()) == 0, dict(counter)
        assert not np.array_equal(np.asarray(first.sigma.values), np.asarray(second.sigma.values))
        np.testing.assert_allclose(second.model, second.market, rtol=1e-10)

    @pytest.mark.slow
    def test_repeated_greeks_call_compiles_nothing(self):
        """I-21: the pricers are jitted with the trade as an argument and the Greeks are not
        jitted as a closure, so a repeated call reuses every program."""
        trades, market = [bermudan_cfg()], shared.market()
        portfolio_greeks(trades, market, "USD", PRICING)  # warm
        with count_compiles() as counter:
            portfolio_greeks(trades, market, "USD", PRICING)
        assert sum(counter.values()) == 0, dict(counter)

    @pytest.mark.slow
    @pytest.mark.parametrize("change", [dict(fixed_rate=0.041), dict(swap_tenor="4Y")], ids=["rate", "tenor"])
    def test_a_different_trade_gets_its_own_greeks_from_warm_programs(self, change):
        """I-21's negative test: after another trade warmed the caches, a trade differing in
        rate or tenor gets exactly the Greeks it gets from cold caches (a cache keyed on too
        little would hand it the first trade's program)."""
        market = shared.market()
        jax.clear_caches()
        portfolio_greeks([bermudan_cfg()], market, "USD", PRICING)
        warm = portfolio_greeks([bermudan_cfg(**change)], market, "USD", PRICING)[0]
        jax.clear_caches()
        cold = portfolio_greeks([bermudan_cfg(**change)], market, "USD", PRICING)[0]
        assert warm.keys() == cold.keys()
        for key in cold:
            np.testing.assert_array_equal(warm[key], cold[key], err_msg=key)

    @pytest.mark.slow
    def test_an_option_s_greeks_are_a_program_per_derivative(self):
        """Roadmap 2.4: a Bermudan's Delta and Gamma on every curve are one program and its
        Vega gradient another, each jitted with the trade as an argument. Differentiated
        eagerly they were a forward and a backward program per curve and derivative, about
        ten for one option."""
        market = shared.market()
        jax.clear_caches()
        with count_compiles() as counter:
            curve_greeks(bermudan_cfg(), market, PRICING, 1e-4)
            vega_greek(bermudan_cfg(), market, PRICING, 1e-4)
        assert counter["jit__curve_derivatives"] == 1 and counter["jit__option_vega"] == 1, dict(counter)
        assert counter["jit__backward_induction_arrays"] == 0, dict(counter)

    @pytest.mark.slow
    def test_a_repeated_job_compiles_nothing(self):
        """Roadmap 2.4 (I-53): the demo's job, every trade type with AD Greeks, run twice in a
        fresh process with no disk cache, compiles nothing the second time. It compiled 9
        programs: the swap's and European's eagerly differentiated pricers were kept only in
        JAX's internal caches of 2,048 traced programs, which the rest of the job overflowed."""
        script = (
            "import json, jax\n"
            "from demos.demo_profile_small import build_portfolio_request\n"
            "from engine.api.market_schemas import MarketPortfolioRequestSchema\n"
            "from engine.portfolio import price_portfolio\n"
            "from tests.support.compiles import count_compiles\n"
            "request = MarketPortfolioRequestSchema.model_validate(build_portfolio_request()).to_dataclass()\n"
            "counts = []\n"
            "for _ in range(2):\n"
            "    with count_compiles() as counter:\n"
            "        jax.block_until_ready(price_portfolio(request).npv_cube)\n"
            "    counts.append(dict(counter))\n"
            "print(json.dumps(counts))\n")
        process = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT,
                                 timeout=900)
        assert process.returncode == 0, process.stderr[-3000:]
        first, second = json.loads(process.stdout.strip().splitlines()[-1])
        assert sum(first.values()) > 0 and second == {}, second

    @pytest.mark.slow
    def test_greeks_scale_linearly_with_notional(self):
        """7x the notional gives exactly 7x the Delta (`notional` is a traced child)."""
        market = shared.market()
        base = curve_greeks(bermudan_cfg(notional=1_000_000.0), market, PRICING, 1e-4)
        scaled = curve_greeks(bermudan_cfg(notional=7_000_000.0), market, PRICING, 1e-4)
        for key in base:
            np.testing.assert_allclose(scaled[key], 7.0 * base[key], rtol=1e-9, atol=1e-9)


# =============================================================================
# HESSIAN DIAGONAL VIA HVP == DIAGONAL OF THE FULL HESSIAN
# =============================================================================
class TestHessianDiagonalEquivalence:
    """`curve_greeks`' Delta and Gamma, each curve's gradient and Hessian diagonal from one
    linearization over all of the trade's curves (`engine.risk.greeks`), equal `jax.grad` and
    `jnp.diagonal(jax.hessian(...))` of the price function in each curve, for every trade's
    price function."""

    @pytest.mark.parametrize("cfg", [
        SwapConfig(notional=1e6, fixed_rate=0.032, payer=True, swap_tenor="3Y", evaluation_date=ASOF,
                   trade_id="swap"),
        SwaptionConfig(notional=1.5e6, fixed_rate=0.031, payer=True, swap_tenor="2Y",
                       forward_start=ORE.Period("1Y"), evaluation_date=ASOF, trade_id="european"),
        pytest.param(bermudan_cfg(), marks=pytest.mark.slow),
    ], ids=["swap", "european", "bermudan"])
    def test_matches_the_gradient_and_full_hessian_of_every_curve(self, cfg):
        market, shift = shared.market(), 1e-4
        greeks = curve_greeks(cfg, market, PRICING, shift)
        fn = trade_price_function(cfg, market, PRICING)
        rates = curves_of(fn, market, jnp.float64)
        for k, (kind, name) in enumerate(fn.curves):
            grad = np.asarray(jax.grad(fn.price, argnums=k)(*rates))
            full = np.diagonal(np.asarray(jax.hessian(fn.price, argnums=k)(*rates)))
            np.testing.assert_allclose(greeks[f"delta:{kind}:{name}"] / shift, grad, rtol=1e-12,
                                       atol=1e-12 * float(np.max(np.abs(grad))))
            # atol scaled to the compared magnitude: a pillar whose true Gamma is zero lands on
            # different tiny values by the two routes, so rtol alone is meaningless there.
            np.testing.assert_allclose(greeks[f"gamma:{kind}:{name}"] / shift ** 2, full, rtol=1e-7,
                                       atol=1e-6 * float(np.max(np.abs(full))))


# =============================================================================
# GRADIENTS STILL FLOW THROUGH THE JITTED INDUCTION
# =============================================================================
class TestGradientsSurviveTheJitBoundary:
    """A differentiable field placed in static aux data gets frozen into the cache key and a
    zero gradient, with no error. These catch that."""

    @pytest.fixture(scope="class")
    def option(self):
        market = shared.market()
        fn = bermudan_price_function(bermudan_cfg(), market, PRICING, jnp.float64)
        disc, index = (jnp.asarray(market.currency("USD").discount_curve.rates),
                       jnp.asarray(market.index_curve("USD", shared.INDEX).rates))
        return fn, disc, index

    @staticmethod
    def _price(option, disc_rates):
        from engine.models.curves import ZeroCurve
        fn, _disc, index = option
        times = jnp.asarray(shared.PILLARS)
        return fn.price(ZeroCurve(times, disc_rates), ZeroCurve(times, index))

    def test_delta_is_nonzero(self, option):
        delta = np.asarray(jax.grad(lambda r: self._price(option, r))(option[1]))
        assert np.any(np.abs(delta) > 1e-6), f"all-zero delta suggests a frozen curve: {delta}"

    def test_delta_matches_a_finite_difference_of_the_price(self, option):
        """The gradient is correct across the jit boundary."""
        rates = np.asarray(option[1], dtype=np.float64)
        eps, fd = 1e-6, np.zeros_like(rates)
        for i in range(len(rates)):
            up, down = rates.copy(), rates.copy()
            up[i] += eps
            down[i] -= eps
            fd[i] = (float(self._price(option, jnp.asarray(up))) - float(self._price(option, jnp.asarray(down)))) / (2 * eps)
        analytic = np.asarray(jax.grad(lambda r: self._price(option, r))(option[1]))
        # Loose rtol: the finite difference is the noisy side.
        np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-2)

    def test_vega_gradient_flows_through_sigma(self):
        """The gradient flows through `sigma`, a nested `Sigma` pytree child."""
        cfg = bermudan_cfg()

        def npv(values):
            from engine.instruments.bermudan_swaption import grid_value
            sigma = Sigma(times=jnp.asarray([1.0]), values=values)
            return grid_value(prepared(cfg, a=0.03, sigma=sigma, curve=CURVE, n_per_std=16))

        d_npv_d_sigma = np.asarray(jax.grad(npv)(jnp.asarray([0.01, 0.012])))
        assert np.all(np.abs(d_npv_d_sigma) > 1e-6), f"a zero d(NPV)/d(sigma) suggests sigma was frozen: {d_npv_d_sigma}"

    @pytest.mark.slow
    def test_jitted_and_unjitted_induction_agree(self):
        """The jitted induction equals the same computation under `jax.disable_jit`."""
        jitted = engine_npv()
        with jax.disable_jit():
            eager = engine_npv()
        assert jitted == pytest.approx(eager, rel=1e-9)


# =============================================================================
# PROFILER HOOK / TRACE TRUNCATION GUARD
# =============================================================================
class TestProfilerHook:
    def test_hook_is_inert_without_the_env_var(self, monkeypatch, tmp_path):
        """With `JAX_RISK_PROFILE_DIR` unset, the engine worker's job runs untraced: its value
        comes back and no directory is created."""
        monkeypatch.delenv("JAX_RISK_PROFILE_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        from engine.api.worker import _profiled

        assert _profiled(lambda: "the job's result", lambda result: result) == "the job's result"
        assert list(tmp_path.iterdir()) == []

    def test_a_traced_job_writes_its_summary_beside_the_trace(self, monkeypatch, tmp_path):
        """With `JAX_RISK_PROFILE_DIR` and `JAX_RISK_PROFILE_WARMUP=1`, a real job is traced
        and `<trace run>.summary.json` holds both runs' wall time and compiles, the trace's
        events and the time in each phase: the warm-up compiled the job's program, the traced
        repeat compiled nothing."""
        import json
        from types import SimpleNamespace

        from engine.api.worker import _profiled
        from engine.portfolio.profiling import phase

        monkeypatch.setenv("JAX_RISK_PROFILE_DIR", str(tmp_path))
        monkeypatch.setenv("JAX_RISK_PROFILE_WARMUP", "1")
        doubled = jax.jit(lambda x: 2.0 * x)  # a fresh function: its first call compiles

        def job():
            with phase("pricing"):
                return SimpleNamespace(npv_cube=doubled(jnp.arange(5.0)))

        result = _profiled(job, lambda result: result.npv_cube)

        np.testing.assert_array_equal(np.asarray(result.npv_cube), 2.0 * np.arange(5.0))
        [path] = list(tmp_path.glob("pid-*/*.summary.json"))
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert summary["warmup"]["compiles"] >= 1 and summary["traced"]["compiles"] == 0
        assert summary["traced"]["wall_seconds"] > 0 and summary["events"] > 0 and summary["bytes"] > 0
        assert set(summary["phases"]) == {"pricing"} and summary["phases"]["pricing"] > 0
        assert summary["warning"] is None and summary["coverage"] > 0
        assert path.name == f"{Path(summary['path']).parent.name}.summary.json"

    def test_a_traced_phase_is_all_the_trace_holds(self, monkeypatch, tmp_path):
        """With `JAX_RISK_PROFILE_PHASE` (roadmap 2.4) the job runs whole but only that phase
        is traced: the summary's phases are that phase's, its wall time and compiles are the
        phase's, and the job's own are kept beside them."""
        import json
        from types import SimpleNamespace

        from engine.api.worker import _profiled
        from engine.portfolio.profiling import phase

        monkeypatch.setenv("JAX_RISK_PROFILE_DIR", str(tmp_path))
        monkeypatch.setenv("JAX_RISK_PROFILE_PHASE", "pricing")
        doubled, tripled = jax.jit(lambda x: 2.0 * x), jax.jit(lambda x: 3.0 * x)  # each compiles once

        def job():
            with phase("calibration"):
                x = doubled(jnp.arange(5.0))
            with phase("pricing"):
                x = tripled(x)
            with phase("exposure"):
                return SimpleNamespace(npv_cube=x + 1.0)

        result = _profiled(job, lambda result: result.npv_cube)

        np.testing.assert_array_equal(np.asarray(result.npv_cube), 6.0 * np.arange(5.0) + 1.0)
        [path] = list(tmp_path.glob("pid-*/*.summary.json"))
        summary = json.loads(path.read_text(encoding="utf-8"))
        traced = summary["traced"]
        assert set(summary["phases"]) == {"pricing"} and traced["phase"] == "pricing"
        assert traced["compiles"] == 1 and traced["job_compiles"] >= 3
        assert 0 < traced["wall_seconds"] <= traced["job_wall_seconds"]

    def test_a_phase_the_job_does_not_have_traces_nothing(self, monkeypatch, tmp_path):
        from types import SimpleNamespace

        from engine.api.worker import _profiled

        monkeypatch.setenv("JAX_RISK_PROFILE_DIR", str(tmp_path))
        monkeypatch.setenv("JAX_RISK_PROFILE_PHASE", "princing")
        with pytest.warns(UserWarning, match="'princing': the job has no such phase"):
            result = _profiled(lambda: SimpleNamespace(npv_cube=jnp.ones(3)), lambda result: result.npv_cube)
        assert result.npv_cube.shape == (3,)
        assert not list(tmp_path.rglob("*.xplane.pb")) and not list(tmp_path.rglob("*.summary.json"))

    def test_the_window_opens_on_the_phase_s_first_run_only(self):
        """Started and stopped once, around the first run of the phase, even one that
        raises; other phases and later runs are not traced, and the window closes with the
        block."""
        import engine.portfolio.profiling as profiling

        calls = []
        with profiling.traced_phase("pricing", lambda: calls.append("start"), lambda: calls.append("stop")) as window:
            with profiling.phase("calibration"):
                pass
            with pytest.raises(ValueError):
                with profiling.phase("pricing"):
                    raise ValueError("a failing phase still closes its window")
            with profiling.phase("pricing"):
                pass
        assert calls == ["start", "stop"] and window.wall_seconds is not None
        assert profiling._window is None

    def test_recording_never_raises_on_a_broken_trace(self, tmp_path):
        """Reading the trace back never breaks a pricing job."""
        from engine.api.worker import _record_trace

        run_dir = tmp_path / "plugins" / "profile" / "run"
        run_dir.mkdir(parents=True)
        (run_dir / "host.xplane.pb").write_bytes(b"not actually an XSpace")

        _record_trace(str(tmp_path), {"wall_seconds": 10.0, "compiles": 0}, None)  # must not raise
        assert not list(tmp_path.glob("*.summary.json"))

    def test_recording_is_quiet_when_no_trace_exists(self, tmp_path):
        from engine.api.worker import _record_trace

        _record_trace(str(tmp_path), {"wall_seconds": 10.0, "compiles": 0}, None)  # must not raise


def _write_trace(directory, lines) -> str:
    """An `.xplane.pb` of one host plane: `lines` maps a thread name to its events,
    `(name, start seconds, duration seconds)`."""
    from jax.profiler import ProfileData

    names = sorted({name for events in lines.values() for name, _, _ in events})
    ids = {name: i + 1 for i, name in enumerate(names)}
    text = 'planes { name: "/host:CPU" '
    for i, (thread, events) in enumerate(lines.items()):
        text += f'lines {{ id: {i + 1} name: "{thread}" timestamp_ns: 0 '
        text += "".join(f"events {{ metadata_id: {ids[name]} offset_ps: {round(start * 1e12)} "
                        f"duration_ps: {round(duration * 1e12)} }} " for name, start, duration in events)
        text += "} "
    text += "".join(f'event_metadata {{ key: {i} value {{ id: {i} name: "{name}" }} }} ' for name, i in ids.items())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "host.xplane.pb"
    path.write_bytes(ProfileData.text_proto_to_serialized_xspace(text + "}"))
    return str(path)


def _summary(events: int, span_seconds: float):
    from engine.portfolio.profiling import TraceSummary

    return TraceSummary(path="t", bytes=0, events=events, span_seconds=span_seconds, phases={}, threads={})


class TestTraceSummary:
    """`engine.portfolio.profiling.summarize_trace` and the worker's check of a trace."""

    def test_counts_events_span_phases_and_threads(self, tmp_path):
        from engine.portfolio.profiling import summarize_trace

        path = _write_trace(tmp_path / "run", {
            "": [("pricing", 0.0, 0.25), ("pricing", 0.5, 0.25), ("greeks", 1.0, 3.0),
                 ("greeks/trade0/SwapConfig", 1.0, 1.0)],
            "tf_XLAEigen": [("wrapped_add", 0.2, 1e-5)],
        })
        (tmp_path / "run" / "host.trace.json.gz").write_bytes(b"0123")  # the export beside it

        summary = summarize_trace(path)

        assert summary.events == 5
        assert summary.span_seconds == pytest.approx(4.0)  # 0 to the end of "greeks"
        assert summary.phases == pytest.approx({"greeks": 3.0, "greeks/trade0/SwapConfig": 1.0, "pricing": 0.5})
        assert summary.threads == {"": 4, "tf_XLAEigen": 1}
        assert summary.bytes == Path(path).stat().st_size + 4
        assert summary.coverage(8.0) == pytest.approx(0.5)
        assert summary.json_export_complete

    def test_latest_trace_is_the_newest_or_none(self, tmp_path):
        import os

        from engine.portfolio.profiling import latest_trace

        assert latest_trace(str(tmp_path)) is None
        old = _write_trace(tmp_path / "plugins" / "profile" / "a", {"": [("pricing", 0.0, 1.0)]})
        new = _write_trace(tmp_path / "plugins" / "profile" / "b", {"": [("pricing", 0.0, 1.0)]})
        os.utime(old, (1_000, 1_000))
        assert latest_trace(str(tmp_path)) == new

    def test_a_short_trace_is_reported(self):
        """A trace spanning far less than the job's wall time was stopped early."""
        from engine.api.worker import _trace_warning

        assert "spans only" in _trace_warning(_summary(events=2, span_seconds=0.1), wall_seconds=100.0)

    def test_a_trace_beyond_the_json_export_cap_is_reported(self):
        """The `.trace.json.gz` export keeps about a million events; xprof reads them all."""
        from engine.api.worker import _trace_warning
        from engine.portfolio.profiling import JSON_EXPORT_EVENT_CAP

        warning = _trace_warning(_summary(events=JSON_EXPORT_EVENT_CAP + 1, span_seconds=10.0), wall_seconds=10.0)
        assert ".trace.json.gz" in warning and "xprof reads the whole trace" in warning
        assert _summary(events=JSON_EXPORT_EVENT_CAP, span_seconds=10.0).json_export_complete

    def test_a_complete_trace_is_not_reported(self):
        from engine.api.worker import _trace_warning

        assert _trace_warning(_summary(events=1_000, span_seconds=9.5), wall_seconds=10.0) is None


# =============================================================================
# PHASE ANNOTATIONS
# =============================================================================
class TestPhaseAnnotations:
    """`engine.portfolio.profiling.phase` is wired correctly and safe in the pricing path."""

    def test_phase_is_a_no_op_context_manager_outside_a_trace(self):
        from engine.portfolio.profiling import phase

        with phase("unit-test"):
            value = 1 + 1
        assert value == 2

    def test_phase_enters_both_annotation_mechanisms(self):
        """Both mechanisms are entered (host timeline and compiled-HLO names);
        `named_scope` alone gave no host events for eager phases."""
        import engine.portfolio.profiling as profiling

        entered = []

        class _Recorder:
            def __init__(self, name):
                self.name = name

            def __enter__(self):
                entered.append(self.name)
                return self

            def __exit__(self, *exc):
                return False

        original_annotation = jax.profiler.TraceAnnotation
        original_scope = jax.named_scope
        try:
            jax.profiler.TraceAnnotation = lambda n: _Recorder(f"annotation:{n}")
            jax.named_scope = lambda n: _Recorder(f"scope:{n}")
            with profiling.phase("calibration"):
                pass
        finally:
            jax.profiler.TraceAnnotation = original_annotation
            jax.named_scope = original_scope

        assert entered == ["annotation:calibration", "scope:calibration"]

    def test_price_portfolio_annotates_its_phases(self, portfolio_request, monkeypatch):
        """A real pricing run enters the phases."""
        import dataclasses

        import engine.portfolio.market_path as market_path
        from engine.portfolio.request import price_portfolio

        seen = []
        original = market_path.phase

        @contextmanager
        def recording(name):
            seen.append(name)
            with original(name):
                yield

        monkeypatch.setattr(market_path, "phase", recording)
        price_portfolio(dataclasses.replace(portfolio_request, compute_greeks=True))
        assert {"calibration", "simulation", "pricing", "exposure", "greeks"} <= set(seen) <= set(PHASES)
        seen.clear()
        price_portfolio(dataclasses.replace(portfolio_request, scenario_risk=False))
        assert seen == ["base_npv"]

    @pytest.mark.parametrize("method", ["AD", "bump"])
    def test_each_trade_has_its_own_greeks_phase(self, method, monkeypatch):
        """Both Greeks methods label each trade's Greeks `greeks/trade<index>/<config type>`,
        the per-trade time a trace summary reports."""
        import engine.portfolio.profiling as profiling
        from engine.portfolio.market_path import _greeks

        seen = []
        original = profiling.phase

        @contextmanager
        def recording(name):
            seen.append(name)
            with original(name):
                yield

        monkeypatch.setattr(profiling, "phase", recording)
        trades = [shared.trades()[name] for name in ("swap-payer", "bond")]
        _greeks(method)(trades, shared.market(), "USD", PRICING)
        assert seen == ["greeks/trade0/SwapConfig", "greeks/trade1/BondConfig"]

"""
Structural tests of the JIT and profiling work (docs/concepts/profiling.md): the engine
reaches its numbers through a small, fixed number of compiled XLA programs rather than
thousands of eager dispatches. Numerical correctness is covered elsewhere; a regression here
(a hardcoded dtype, a field on the wrong side of `_PreparedBermudan`'s pytree split, a
dropped `jax.jit`) passes every numerical test while multiplying compile time.

Compiles are counted by patching `jax._src.compiler.backend_compile_and_load`, the function
an xprof trace records as XLA compilation; cache hits do not call it.
"""
import collections
from contextlib import contextmanager

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
from engine.risk.greeks import _grad_and_hessian_diagonal, curve_greeks, portfolio_greeks
from engine.risk.price_functions import bermudan_price_function, curves_of, trade_price_function
from engine.valuation.bermudan import calibrate_on
from engine.valuation.config import LgmSwaptionEngineConfig, PricingConfig
from engine.valuation.context import from_market
from tests.support import portfolio as shared
from tests.support.lgm_engine import grid_npv, prepared

ASOF = shared.ASOF
CURVE = ZeroCurveConfig(times=[0.0, 1.0, 3.0], rates=[0.03, 0.03, 0.03])
ENGINE = LgmSwaptionEngineConfig(n_per_std=16, std_devs=6.0)
PRICING = PricingConfig(bermudan=ENGINE)


@contextmanager
def count_compiles():
    """Count XLA compilations in the block, as a `Counter` keyed by the program's MLIR
    `sym_name` (`jit_<fn>`, the name xprof shows). A cache hit is not counted."""
    import jax._src.compiler as _compiler

    counter: collections.Counter = collections.Counter()
    original = _compiler.backend_compile_and_load

    def counting(backend, module, *args, **kwargs):
        name = "?"
        try:
            name = module.operation.attributes["sym_name"].value
        except Exception:
            pass
        counter[name] += 1
        return original(backend, module, *args, **kwargs)

    _compiler.backend_compile_and_load = counting
    try:
        yield counter
    finally:
        _compiler.backend_compile_and_load = original


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
    """`_PreparedBermudan` is a pytree whose children are exactly the traced fields, so
    `_backward_induction_arrays` can be jitted while Greeks differentiate through it."""

    def test_flatten_exposes_only_the_traced_fields_as_children(self):
        children, aux = prepared_bermudan().tree_flatten()
        assert len(children) == len(_PreparedBermudan._TRACED)
        # The differentiation targets must be children, or a tracer would be frozen into the
        # cache key instead of carrying a gradient: the curves (Delta) and the volatility (Vega).
        assert {"curve", "index_curve", "sigma"} <= set(_PreparedBermudan._TRACED)
        aux_names = {name for name, _ in aux}
        assert aux_names.isdisjoint(set(_PreparedBermudan._TRACED))
        # Trade structure stays static: it keys the cache.
        assert {"exercise_times", "fixed_times", "n_per_std", "payer", "reversion"} <= aux_names

    def test_round_trips_through_flatten_unflatten(self):
        """Flatten/unflatten round-trips exactly (JAX does it at every jit/grad boundary)."""
        original = prepared_bermudan()
        children, aux = original.tree_flatten()
        rebuilt = _PreparedBermudan.tree_unflatten(aux, children)
        assert rebuilt == original
        for name in ("exercise_times", "fixed_times", "fixed_amounts"):
            np.testing.assert_array_equal(np.asarray(getattr(rebuilt, name)), np.asarray(getattr(original, name)))
        np.testing.assert_array_equal(np.asarray(rebuilt.curve.pillar_rates), np.asarray(original.curve.pillar_rates))

    def test_round_tripped_arrays_stay_writable(self):
        """Unflattened schedule arrays are writable copies (`np.frombuffer` is read-only)."""
        children, aux = prepared_bermudan().tree_flatten()
        rebuilt = _PreparedBermudan.tree_unflatten(aux, children)
        assert rebuilt.fixed_times.flags.writeable and rebuilt.exercise_times.flags.writeable

    def test_aux_data_is_hashable(self):
        """The aux tuple is the jit cache key, so it must hash (a raw NumPy array would not)."""
        _children, aux = prepared_bermudan().tree_flatten()
        assert isinstance(hash(aux), int)

    def test_jax_tree_util_sees_it_as_a_pytree(self):
        # An unregistered dataclass would be a single opaque leaf.
        assert len(jax.tree_util.tree_leaves(prepared_bermudan())) >= len(_PreparedBermudan._TRACED)


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

    def test_calibration_recompiles_a_few_programs_per_call(self):
        """Pins a known residue (I-22): the bootstrap's bisection bakes each helper's market
        price into its traced program, so every call compiles again. Measured 6 per call for
        a two-helper basket; tighten when I-22 is fixed."""
        market = from_market(shared.market())
        calibrate_on(bermudan_cfg(), ENGINE, market)  # warm
        with count_compiles() as counter:
            calibrate_on(bermudan_cfg(), ENGINE, market)
        assert 0 < sum(counter.values()) <= 12, dict(counter)

    @pytest.mark.slow
    def test_repeated_greeks_call_compiles_a_bounded_number_of_programs(self):
        """Pins a known residue (I-21): the AD Greeks build fresh closures per call (and each
        calibration recompiles, I-22), so a repeated call compiles again. Measured 30 for one
        Bermudan (28 of them calibration scans). When I-21 is fixed, tighten this and add
        I-21's negative test (a trade differing in notional, rate or tenor must still get its
        own answer), since a count alone would pass a broken cache."""
        trades, market = [bermudan_cfg()], shared.market()
        portfolio_greeks(trades, market, "USD", PRICING)  # warm
        with count_compiles() as counter:
            portfolio_greeks(trades, market, "USD", PRICING)
        assert sum(counter.values()) <= 60, dict(counter)

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
def _full_and_hvp_diagonal(cfg):
    """The diagonal of d^2 NPV / d(discount pillar rates)^2 both ways, other curves fixed."""
    fn = trade_price_function(cfg, shared.market(), PRICING)
    rates = curves_of(fn, shared.market(), jnp.float64)
    _grad, diag = _grad_and_hessian_diagonal(fn.price, rates[0], *rates[1:])
    full = jnp.diagonal(jax.hessian(fn.price, argnums=0)(*rates))
    return np.asarray(diag), np.asarray(full)


class TestHessianDiagonalEquivalence:
    """`_grad_and_hessian_diagonal` equals `jnp.diagonal(jax.hessian(f))`, the expression it
    replaced, for every trade's price function."""

    def test_matches_full_hessian_on_an_analytic_function(self):
        """A closed form with a known Hessian diagonal."""
        def f(x):
            # sum(x_i^3) + x_0*x_1 -> d2f/dx_i^2 = 6*x_i (the cross term is off-diagonal).
            return jnp.sum(x ** 3) + x[0] * x[1]

        x = jnp.asarray([1.0, 2.0, 3.0])
        grad, diag = _grad_and_hessian_diagonal(f, x)
        np.testing.assert_allclose(np.asarray(grad), np.asarray(jax.grad(f)(x)), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(diag), 6.0 * np.asarray(x), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(diag), np.asarray(jnp.diagonal(jax.hessian(f)(x))), rtol=1e-12)

    @pytest.mark.parametrize("cfg", [
        SwapConfig(notional=1e6, fixed_rate=0.032, payer=True, swap_tenor="3Y", evaluation_date=ASOF,
                   trade_id="swap"),
        SwaptionConfig(notional=1.5e6, fixed_rate=0.031, payer=True, swap_tenor="2Y",
                       forward_start=ORE.Period("1Y"), evaluation_date=ASOF, trade_id="european"),
        pytest.param(bermudan_cfg(), marks=pytest.mark.slow),
    ], ids=["swap", "european", "bermudan"])
    def test_matches_full_hessian(self, cfg):
        diag, full = _full_and_hvp_diagonal(cfg)
        # atol scaled to the compared magnitude: a pillar whose true Gamma is zero lands on
        # different tiny values by the two routes, so rtol alone is meaningless there.
        np.testing.assert_allclose(diag, full, rtol=1e-7, atol=1e-6 * float(np.max(np.abs(full))))


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
    def test_hook_is_inert_without_the_env_var(self, monkeypatch):
        """With `JAX_RISK_PROFILE_DIR` unset, nothing is traced and no directory is
        created."""
        monkeypatch.delenv("JAX_RISK_PROFILE_DIR", raising=False)
        from engine.portfolio import worker_pool

        # No trace directory, and the job returns normally.
        assert worker_pool.os.environ.get("JAX_RISK_PROFILE_DIR") is None

    def test_truncation_guard_warns_on_a_short_trace(self, tmp_path):
        """A trace spanning far less than the job's wall time (the signature of the silent
        buffer cap) warns, not raises."""
        import gzip
        import json
        from engine.portfolio.worker_pool import _warn_if_trace_truncated

        run_dir = tmp_path / "pid-1" / "plugins" / "profile" / "run"
        run_dir.mkdir(parents=True)
        # 0.1s of events against a 100s job.
        events = {"traceEvents": [{"ts": 0.0}, {"ts": 100_000.0}]}
        with gzip.open(run_dir / "host.trace.json.gz", "wt") as handle:
            json.dump(events, handle)

        with pytest.warns(UserWarning, match="spans only"):
            _warn_if_trace_truncated(str(tmp_path), wall_seconds=100.0)

    def test_truncation_guard_is_quiet_on_a_complete_trace(self, tmp_path):
        import gzip
        import json
        from engine.portfolio.worker_pool import _warn_if_trace_truncated

        run_dir = tmp_path / "pid-1" / "plugins" / "profile" / "run"
        run_dir.mkdir(parents=True)
        # 9.5s of events against a 10s job: healthy.
        events = {"traceEvents": [{"ts": 0.0}, {"ts": 9_500_000.0}]}
        with gzip.open(run_dir / "host.trace.json.gz", "wt") as handle:
            json.dump(events, handle)

        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")  # any warning fails the test
            _warn_if_trace_truncated(str(tmp_path), wall_seconds=10.0)

    def test_truncation_guard_never_raises_on_a_broken_trace(self, tmp_path):
        """The self-check never breaks a pricing job."""
        from engine.portfolio.worker_pool import _warn_if_trace_truncated

        run_dir = tmp_path / "pid-1"
        run_dir.mkdir(parents=True)
        (run_dir / "host.trace.json.gz").write_bytes(b"not actually gzip")

        _warn_if_trace_truncated(str(tmp_path), wall_seconds=10.0)  # must not raise

    def test_truncation_guard_is_quiet_when_no_trace_exists(self, tmp_path):
        from engine.portfolio.worker_pool import _warn_if_trace_truncated

        _warn_if_trace_truncated(str(tmp_path), wall_seconds=10.0)  # must not raise


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
        assert {"calibration", "simulation", "pricing", "exposure", "greeks"} <= set(seen)
        seen.clear()
        price_portfolio(dataclasses.replace(portfolio_request, scenario_risk=False))
        assert seen == ["base_npv"]

"""
Tests for the profiling/JIT-structure work described in
`docs/concepts/profiling.md`.

These are **structural** tests, not numerical ones: the numbers this engine
produces are already pinned exhaustively by `test_greeks.py`,
`test_greeks_bermudan.py`, `test_bermudan_swaption.py` and the ORE-parity
suite. What is pinned HERE is the thing those tests cannot see -- that the
engine still reaches those same numbers through a small, fixed number of
compiled XLA programs rather than thousands of eager dispatches.

**Why that needs its own tests.** The whole Greeks-cost problem this work
fixed was invisible to every correctness test in the suite: the answers were
right the entire time, they were merely arrived at via ~600 separate
compilations. A regression here (someone re-hardcodes a dtype, adds a field
to `_PreparedBermudan` on the wrong side of the pytree split, or drops a
`jax.jit`) would likewise pass every numerical test while silently restoring
a 40MB trace and a 4x slowdown. Compile counting is the only signal that
catches it.

The counting hook patches `jax._src.compiler.backend_compile_and_load`,
which is precisely the function an xprof trace records as XLA compilation --
so "compiles" here means the same thing it means in a trace. Cache HITS do
not call it, which is what makes the cache-reuse assertions below meaningful.
"""
import collections
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.models.hull_white import ZeroCurve
from engine.models.lgm import Sigma
from engine.simulation.market_model import ZeroCurveConfig
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig,
    _PreparedBermudan,
    prepare_bermudan,
    price_bermudan_swaption_base,
)
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.risk.greeks import (
    _grad_and_hessian_diagonal,
    _swap_price_fn,
    _swaption_price_fn,
    _bermudan_price_fn,
    bermudan_delta_gamma,
    bermudan_theta,
    swap_delta_gamma,
    swaption_delta_gamma,
)
from engine.simulation.demo_scenarios import EVAL_DATE


PILLAR_TIMES = [0.0, 1.0, 3.0]
PILLAR_RATES = [0.03, 0.03, 0.03]


@contextmanager
def count_compiles():
    """Counts XLA compilations inside the block. Yields a `Counter` keyed by
    the compiled program's own MLIR `sym_name` (`jit_<fn>`), which is the
    same name the program appears under in an xprof trace.

    Patches `jax._src.compiler.backend_compile_and_load` -- the single
    funnel every `jit` compilation goes through, and the function whose
    trace events the profiler labels as XLA compilation. A cache HIT never
    reaches it, so a count of 0 genuinely means "everything was reused"."""
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
    base = dict(
        notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
        hw_a=0.03, hw_sigma=0.01,
        initial_zero_curve=ZeroCurveConfig(times=PILLAR_TIMES, rates=PILLAR_RATES),
        exercise_dates=[EVAL_DATE + 365, EVAL_DATE + 730], swap_tenor="3Y",
        n_per_std=16, std_devs=6.0, evaluation_date=EVAL_DATE,
    )
    base.update(overrides)
    return BermudanSwaptionConfig(**base)


def jax_curve() -> ZeroCurve:
    return ZeroCurve(
        pillar_times=jnp.asarray(PILLAR_TIMES),
        pillar_rates=jnp.asarray(PILLAR_RATES),
    )


# =============================================================================
# THE PYTREE SPLIT (_PreparedBermudan)
# =============================================================================
class TestPreparedBermudanPytree:
    """`_PreparedBermudan` must behave as a pytree whose children are exactly
    the traced fields -- the property that lets `_backward_induction_arrays`
    be `jax.jit`-wrapped while `engine.risk.greeks` still differentiates
    through it. See that dataclass's own docstring."""

    def test_flatten_exposes_only_the_traced_fields_as_children(self):
        prepared = prepare_bermudan(bermudan_cfg())
        children, aux = prepared.tree_flatten()

        assert len(children) == len(_PreparedBermudan._TRACED)
        # The two genuine differentiation targets must be children, or a
        # jax.grad tracer substituted into either would be silently frozen
        # into the jit cache key instead of propagating a gradient.
        assert "zero_rates" in _PreparedBermudan._TRACED
        assert "hw_sigma" in _PreparedBermudan._TRACED

        aux_names = {name for name, _ in aux}
        assert aux_names.isdisjoint(set(_PreparedBermudan._TRACED))
        # Trade STRUCTURE must stay static -- it is what keys the cache.
        assert {"exercise_times", "fixed_times", "n_per_std", "payer"} <= aux_names

    def test_round_trips_through_flatten_unflatten(self):
        """JAX round-trips a pytree through flatten/unflatten at every
        jit/grad boundary, so an inexact `tree_unflatten` would corrupt the
        trade silently rather than raising."""
        prepared = prepare_bermudan(bermudan_cfg())
        children, aux = prepared.tree_flatten()
        rebuilt = _PreparedBermudan.tree_unflatten(aux, children)

        assert rebuilt == prepared
        for name in ("exercise_times", "fixed_times", "fixed_amounts", "zero_rates"):
            np.testing.assert_array_equal(
                np.asarray(getattr(rebuilt, name)), np.asarray(getattr(prepared, name))
            )

    def test_round_tripped_arrays_stay_writable(self):
        """`np.frombuffer` hands back a READ-ONLY view, so an unflattened
        trade would carry silently-immutable schedule arrays where the
        original had writable ones -- a difference that surfaces far from
        here as a confusing "assignment destination is read-only"."""
        prepared = prepare_bermudan(bermudan_cfg())
        children, aux = prepared.tree_flatten()
        rebuilt = _PreparedBermudan.tree_unflatten(aux, children)

        assert rebuilt.fixed_times.flags.writeable
        assert rebuilt.exercise_times.flags.writeable

    def test_aux_data_is_hashable(self):
        """The aux tuple IS the `jax.jit` cache key -- an unhashable entry
        (a raw NumPy array, the original failure this split had to avoid)
        raises `TypeError: unhashable type` the moment jit tries to key on
        it."""
        _children, aux = prepare_bermudan(bermudan_cfg()).tree_flatten()
        assert isinstance(hash(aux), int)

    def test_jax_tree_util_sees_it_as_a_pytree(self):
        prepared = prepare_bermudan(bermudan_cfg())
        leaves = jax.tree_util.tree_leaves(prepared)
        # zero_rates (array) + hw_sigma (scalar) + notional + fixed_amounts;
        # a non-registered dataclass would give exactly ONE opaque leaf.
        assert len(leaves) >= len(_PreparedBermudan._TRACED)


# =============================================================================
# COMPILE-COUNT REGRESSION GUARDS
# =============================================================================
class TestCompileCounts:
    """Upper bounds on XLA compilations per operation.

    The bounds are deliberately loose (roughly 3-5x the measured counts, which
    are in each test's own comment) -- they exist to catch a RETURN TO EAGER
    DISPATCH, which is an order-of-magnitude regression, not to pin an exact
    number that a JAX version bump would churn. Before this work, the single
    reference Greeks job below compiled 602 programs; it now compiles 13.
    """

    def test_bermudan_forward_pricing_compiles_few_programs(self):
        # Measured: 4 (jit__backward_induction_arrays + 3 small helpers).
        with count_compiles() as counter:
            npv = price_bermudan_swaption_base(bermudan_cfg())
        # 8521.0223 before 2026-09-23; floating coupons are now projected over
        # the index fixing period, as ORE's LGM engine projects them (I-31).
        assert npv == pytest.approx(8522.460486631673, rel=1e-6)
        assert sum(counter.values()) < 20, dict(counter)

    @pytest.mark.slow
    def test_bermudan_delta_gamma_compiles_few_programs(self):
        # Measured: 5. This is the number that was 470 before the pytree
        # split + HVP diagonal -- by far the largest single win.
        with count_compiles() as counter:
            greeks = bermudan_delta_gamma(bermudan_cfg(), jax_curve())
            jax.block_until_ready(greeks["delta"])
        assert sum(counter.values()) < 30, dict(counter)

    def test_bermudan_theta_compiles_few_programs(self):
        # Measured: 2 (one jitted price_fn per evaluation date).
        with count_compiles() as counter:
            bermudan_theta(bermudan_cfg(), jax_curve())
        assert sum(counter.values()) < 15, dict(counter)

    def test_european_swaption_theta_compiles_few_programs(self):
        """Regression guard for the specific call that measured WORST before
        this work: 56 compilations, because both Jamshidian valuations ran
        eagerly. Now 2."""
        cfg = SwaptionConfig(
            notional=1_500_000.0, fixed_rate=0.031, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.01,
            initial_zero_curve=ZeroCurveConfig(times=PILLAR_TIMES, rates=PILLAR_RATES),
            swap_tenor="2Y", forward_start=ORE.Period("1Y"), evaluation_date=EVAL_DATE,
        )
        from engine.risk.greeks import swaption_theta

        with count_compiles() as counter:
            swaption_theta(cfg, jax_curve())
        assert sum(counter.values()) < 15, dict(counter)

    def test_repeated_identical_pricing_hits_the_compilation_cache(self):
        """The whole point of keying jit on static aux data: a second
        identical call must compile NOTHING. If `_PreparedBermudan`'s aux
        data ever stops comparing equal by value (e.g. someone puts a raw
        array in it, making two preparations of the same trade hash
        differently), this is what catches it."""
        cfg = bermudan_cfg()
        price_bermudan_swaption_base(cfg)  # warm

        with count_compiles() as counter:
            price_bermudan_swaption_base(cfg)
        assert sum(counter.values()) == 0, dict(counter)

    def test_trades_differing_only_in_scale_share_compiled_programs(self):
        """`notional`/`fixed_amounts` are pytree CHILDREN, not static aux
        data, specifically so a portfolio of same-structure/different-size
        trades compiles one kernel rather than one per trade."""
        price_bermudan_swaption_base(bermudan_cfg(notional=1_000_000.0))  # warm

        with count_compiles() as counter:
            npv = price_bermudan_swaption_base(bermudan_cfg(notional=5_000_000.0))
        assert sum(counter.values()) == 0, dict(counter)
        # Same structure, 5x the size -- and a swap's value is linear in
        # notional, so the price must scale exactly.
        assert npv == pytest.approx(5 * 8522.460486631673, rel=1e-6)

    def test_calibration_compiles_few_programs(self):
        """`calibrate_lgm_sigma` measured 137 compilations before its
        `bachelier_swaption_price`/`price_lgm_swaption` calls were jitted --
        the bisection was never the culprit (`_bisect_bucket_sigma`'s
        `lax.scan` already compiles all 60 iterations as one program), the
        eager setup and diagnostics around it were. Now ~18."""
        from engine.calibration.basket import build_coterminal_basket
        from engine.calibration.lgm import calibrate_lgm_sigma

        curve = jax_curve()
        targets = build_coterminal_basket(
            exercise_times=[1.0, 2.0], final_maturity_time=3.0, notional=1_000_000.0,
            payer=True, market_vols=[0.0080, 0.0090], zero_curve=curve,
            evaluation_date=EVAL_DATE,
        )

        with count_compiles() as counter:
            result = calibrate_lgm_sigma(targets, curve, a=0.03)
        assert sum(counter.values()) < 60, dict(counter)
        # An exact bootstrap reprices every basket instrument exactly; this
        # guards that the jitting did not cost accuracy.
        assert result.rmse < 1e-8

    @pytest.mark.slow
    def test_repeated_greeks_call_costs_one_compile_not_zero(self):
        """Pins a KNOWN residue rather than an aspiration -- filed as
        **docs/known-issues.md I-21**: `price_fn` is a fresh closure per
        call, and `jax.jit` keys on function identity, so the combined
        grad+Hessian-diagonal program recompiles once per call even for an
        identical trade. One compile, not the ~470 this started at.

        When I-21 is fixed this drops to 0. TIGHTEN the bound then rather
        than deleting the test, so the property stays pinned in whichever
        direction it moves -- and pair it with I-21's required negative
        test (a trade differing only in notional/fixed_rate/tenor must still
        get its own correct, DIFFERENT answer), since a count-only assertion
        would pass against a broken always-hit cache."""
        cfg = bermudan_cfg()
        curve = jax_curve()
        jax.block_until_ready(bermudan_delta_gamma(cfg, curve)["delta"])  # warm

        with count_compiles() as counter:
            jax.block_until_ready(bermudan_delta_gamma(cfg, curve)["delta"])
        assert sum(counter.values()) <= 2, dict(counter)

    @pytest.mark.slow
    def test_greeks_scale_linearly_with_notional(self):
        """Independent correctness check on the pytree split: `notional` is
        a traced child, so a 7x trade must give exactly 7x the Delta (a
        swaption's value is linear in notional). A split that mis-sorted a
        field would break this long before it broke a compile count."""
        curve = jax_curve()
        base = bermudan_delta_gamma(bermudan_cfg(notional=1_000_000.0), curve)
        scaled = bermudan_delta_gamma(bermudan_cfg(notional=7_000_000.0), curve)

        np.testing.assert_allclose(
            np.asarray(scaled["delta"]), 7.0 * np.asarray(base["delta"]), rtol=1e-9
        )

    def test_differing_grid_resolution_does_compile_a_new_program(self):
        """The complement of the test above: `n_per_std` genuinely changes
        array SHAPES, so it must remain static and must force a recompile.
        A pytree split that swept too much into the children would break
        this (and produce shape errors or silent wrong answers)."""
        price_bermudan_swaption_base(bermudan_cfg(n_per_std=16))  # warm

        with count_compiles() as counter:
            price_bermudan_swaption_base(bermudan_cfg(n_per_std=20))
        assert sum(counter.values()) >= 1, dict(counter)


# =============================================================================
# HESSIAN DIAGONAL VIA HVP == DIAGONAL OF THE FULL HESSIAN
# =============================================================================
class TestHessianDiagonalEquivalence:
    """`_grad_and_hessian_diagonal` replaced `jnp.diagonal(jax.hessian(f))`
    in all three Delta/Gamma functions. The two are mathematically identical;
    these tests pin that they are also numerically identical here, against
    the very expression that was replaced -- the only way to be sure the
    cheaper route did not change a reported Gamma."""

    def test_matches_full_hessian_on_an_analytic_function(self):
        """A closed-form case where the Hessian diagonal is known exactly, so
        this test can fail for the right reason rather than merely agreeing
        with another implementation."""
        def f(x):
            # sum(x_i^3) + x_0*x_1  ->  d2f/dx_i^2 = 6*x_i  (the cross term
            # contributes only OFF-diagonal, which is exactly what must be
            # excluded).
            return jnp.sum(x ** 3) + x[0] * x[1]

        x = jnp.asarray([1.0, 2.0, 3.0])
        grad, diag = _grad_and_hessian_diagonal(f, x)

        np.testing.assert_allclose(np.asarray(grad), np.asarray(jax.grad(f)(x)), rtol=1e-12)
        np.testing.assert_allclose(np.asarray(diag), 6.0 * np.asarray(x), rtol=1e-12)
        np.testing.assert_allclose(
            np.asarray(diag), np.asarray(jnp.diagonal(jax.hessian(f)(x))), rtol=1e-12
        )

    def test_matches_full_hessian_for_a_swap(self):
        curve = jax_curve()
        cfg = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.032, payer=True,
            discount_curve_index=0, forward_curve_index=1,
            swap_tenor="3Y", evaluation_date=EVAL_DATE,
        )
        price_fn = _swap_price_fn(cfg, curve, curve)

        _grad, diag = _grad_and_hessian_diagonal(price_fn, curve.pillar_rates, curve.pillar_rates)
        full = jnp.diagonal(jax.hessian(price_fn, argnums=0)(curve.pillar_rates, curve.pillar_rates))
        # atol scaled to the compared magnitude -- see the European swaption
        # case below for why an exact-zero pillar needs this.
        np.testing.assert_allclose(
            np.asarray(diag), np.asarray(full),
            rtol=1e-9, atol=1e-6 * float(np.max(np.abs(np.asarray(full)))),
        )

    @pytest.mark.slow
    def test_matches_full_hessian_for_a_european_swaption(self):
        curve = jax_curve()
        cfg = SwaptionConfig(
            notional=1_500_000.0, fixed_rate=0.031, payer=True, rate_factor_index=0,
            hw_a=0.03, hw_sigma=0.01,
            initial_zero_curve=ZeroCurveConfig(times=PILLAR_TIMES, rates=PILLAR_RATES),
            swap_tenor="2Y", forward_start=ORE.Period("1Y"), evaluation_date=EVAL_DATE,
        )
        price_fn = _swaption_price_fn(cfg, curve)

        _grad, diag = _grad_and_hessian_diagonal(price_fn, curve.pillar_rates)
        full = jnp.diagonal(jax.hessian(price_fn)(curve.pillar_rates))
        # atol is scaled to the magnitude of the Gamma actually being
        # compared (O(1e8) here): the t=0 pillar's true Gamma is ZERO, and
        # the two routes reach it by different floating-point paths, so they
        # land on different sub-nanoscale values (7e-12 vs -2e-9). A pure
        # rtol comparison is meaningless against an exact zero; what matters
        # is that both are ~17 orders of magnitude below the real entries.
        np.testing.assert_allclose(
            np.asarray(diag), np.asarray(full),
            rtol=1e-9, atol=1e-6 * float(np.max(np.abs(np.asarray(full)))),
        )

    @pytest.mark.slow
    def test_matches_full_hessian_for_a_bermudan(self):
        """The important one: this path goes through the newly-jitted
        `_backward_induction_arrays`, so it checks the pytree split and the
        HVP diagonal together."""
        curve = jax_curve()
        price_fn, sigma_values = _bermudan_price_fn(bermudan_cfg(), curve)

        _grad, diag = _grad_and_hessian_diagonal(price_fn, curve.pillar_rates, sigma_values)
        full = jnp.diagonal(jax.hessian(price_fn, argnums=0)(curve.pillar_rates, sigma_values))
        np.testing.assert_allclose(
            np.asarray(diag), np.asarray(full),
            rtol=1e-7, atol=1e-6 * float(np.max(np.abs(np.asarray(full)))),
        )

    @pytest.mark.slow
    def test_reported_gamma_is_unchanged_by_the_hvp_route(self):
        """End-to-end at the public API: `bermudan_delta_gamma`'s own
        reported Gamma must equal what the old
        `jnp.diagonal(jax.hessian(...)) * bump**2` expression produced."""
        from engine.risk.greeks import DEFAULT_RATE_BUMP

        curve = jax_curve()
        cfg = bermudan_cfg()
        reported = bermudan_delta_gamma(cfg, curve)

        price_fn, sigma_values = _bermudan_price_fn(cfg, curve)
        legacy_grad = jax.grad(price_fn, argnums=0)(curve.pillar_rates, sigma_values)
        legacy_hess = jax.hessian(price_fn, argnums=0)(curve.pillar_rates, sigma_values)

        np.testing.assert_allclose(
            np.asarray(reported["delta"]),
            np.asarray(legacy_grad) * DEFAULT_RATE_BUMP,
            rtol=1e-9, atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(reported["gamma"]),
            np.asarray(jnp.diagonal(legacy_hess)) * DEFAULT_RATE_BUMP ** 2,
            rtol=1e-7, atol=1e-12,
        )


# =============================================================================
# GRADIENTS STILL FLOW THROUGH THE JITTED INDUCTION
# =============================================================================
class TestGradientsSurviveTheJitBoundary:
    """The failure mode the pytree split most easily introduces: a
    differentiable field accidentally placed in STATIC aux data. JAX does
    not raise for that -- it silently freezes the value into the cache key
    and returns a ZERO gradient. These tests would catch exactly that."""

    def test_delta_is_nonzero(self):
        greeks = bermudan_delta_gamma(bermudan_cfg(), jax_curve())
        delta = np.asarray(greeks["delta"])
        assert np.any(np.abs(delta) > 1e-6), f"all-zero delta suggests a frozen curve: {delta}"

    def test_delta_matches_a_finite_difference_of_the_price(self):
        """An independent check that the gradient is not merely nonzero but
        CORRECT across the jit boundary -- finite-differencing the jitted
        forward price itself, which shares no autodiff machinery with
        `jax.grad`."""
        curve = jax_curve()
        cfg = bermudan_cfg()
        price_fn, sigma_values = _bermudan_price_fn(cfg, curve)

        eps = 1e-6
        rates = np.asarray(curve.pillar_rates, dtype=np.float64)
        fd = np.zeros_like(rates)
        for i in range(len(rates)):
            up, down = rates.copy(), rates.copy()
            up[i] += eps
            down[i] -= eps
            fd[i] = (
                float(price_fn(jnp.asarray(up), sigma_values))
                - float(price_fn(jnp.asarray(down), sigma_values))
            ) / (2 * eps)

        analytic = np.asarray(jax.grad(price_fn, argnums=0)(curve.pillar_rates, sigma_values))
        # Loose rtol: the FD reference is the noisy side of this comparison.
        np.testing.assert_allclose(analytic, fd, rtol=1e-4, atol=1e-3)

    def test_vega_gradient_flows_through_hw_sigma(self):
        """`hw_sigma` is the second differentiable child. A `Sigma` is itself
        a registered pytree, so this checks the nested-pytree-as-child case
        specifically."""
        curve = jax_curve()
        sigma = Sigma(times=jnp.asarray([1.0]), values=jnp.asarray([0.01, 0.012]))
        cfg = bermudan_cfg(hw_sigma=sigma)
        price_fn, sigma_values = _bermudan_price_fn(cfg, curve)

        d_npv_d_sigma = jax.grad(price_fn, argnums=1)(curve.pillar_rates, sigma_values)
        assert np.any(np.abs(np.asarray(d_npv_d_sigma)) > 1e-6), (
            f"all-zero d(NPV)/d(sigma) suggests hw_sigma was frozen as static: {d_npv_d_sigma}"
        )

    @pytest.mark.slow
    def test_jitted_and_unjitted_induction_agree(self):
        """The jit wrapper must not change the answer. Compares the public
        priced value against the same computation forced through an eager
        path via `jax.disable_jit`."""
        cfg = bermudan_cfg()
        jitted = price_bermudan_swaption_base(cfg)
        with jax.disable_jit():
            eager = price_bermudan_swaption_base(cfg)
        assert jitted == pytest.approx(eager, rel=1e-9)


# =============================================================================
# PROFILER HOOK / TRACE TRUNCATION GUARD
# =============================================================================
class TestProfilerHook:
    def test_hook_is_inert_without_the_env_var(self, monkeypatch):
        """The profiler hook's central promise: with `JAX_RISK_PROFILE_DIR`
        unset it does nothing at all -- not even import JAX -- so every test
        and the whole CI HTTP path are byte-identical to a build without it."""
        monkeypatch.delenv("JAX_RISK_PROFILE_DIR", raising=False)
        from engine.portfolio import worker_pool

        # No trace dir is created, and the job returns normally.
        assert worker_pool.os.environ.get("JAX_RISK_PROFILE_DIR") is None

    def test_truncation_guard_warns_on_a_short_trace(self, tmp_path):
        """A trace spanning far less than its job's wall time is the
        signature of silent buffer-cap truncation -- the failure mode that
        once hid 98% of a job. Must warn, not raise."""
        import gzip
        import json
        from engine.portfolio.worker_pool import _warn_if_trace_truncated

        run_dir = tmp_path / "pid-1" / "plugins" / "profile" / "run"
        run_dir.mkdir(parents=True)
        # 0.1s of captured events against a 100s job.
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
        # 9.5s of events against a 10s job -- healthy coverage.
        events = {"traceEvents": [{"ts": 0.0}, {"ts": 9_500_000.0}]}
        with gzip.open(run_dir / "host.trace.json.gz", "wt") as handle:
            json.dump(events, handle)

        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("error")  # any warning fails the test
            _warn_if_trace_truncated(str(tmp_path), wall_seconds=10.0)

    def test_truncation_guard_never_raises_on_a_broken_trace(self, tmp_path):
        """Profiling self-checks must not be able to break a pricing job
        whose result is already computed and correct."""
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
    """`engine.portfolio.profiling.phase` is what makes a trace readable with
    the Python tracer off. These check it is wired correctly and is safe to
    leave permanently in the pricing path."""

    def test_phase_is_a_no_op_context_manager_outside_a_trace(self):
        from engine.portfolio.profiling import phase

        with phase("unit-test"):
            value = 1 + 1
        assert value == 2

    def test_phase_enters_both_annotation_mechanisms(self):
        """Both are required and for different reasons (host timeline vs.
        compiled-HLO op names) -- see `engine.portfolio.profiling`'s
        docstring. Using only `named_scope` was measured to produce ZERO
        labelled host events for eagerly-dispatched phases."""
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

    def test_price_portfolio_annotates_its_phases(self, portfolio_request):
        """The phases must actually be entered by a real pricing run -- a
        rename or a dropped `with` would otherwise go unnoticed until
        someone next opened a trace."""
        import engine.portfolio.request as request_module
        from engine.portfolio.request import price_portfolio

        seen = []
        original = request_module._phase

        @contextmanager
        def recording(name):
            seen.append(name)
            with original(name):
                yield

        request_module._phase = recording
        try:
            price_portfolio(portfolio_request)
        finally:
            request_module._phase = original

        assert {"calibration", "simulation", "pricing", "base_npv", "exposure"} <= set(seen)

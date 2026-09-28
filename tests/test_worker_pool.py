"""
`engine.portfolio.worker_pool`: per-precision process pools.

tests/test_portfolio_entrypoint.py::TestPricePortfolioConcurrency shows `price_portfolio`
stays correct when two threads race in one process (serialized by `_PRICING_LOCK`, so never
overlapping). `TestWorkerPoolConcurrency` shows the pools give real concurrent execution
across precision tiers, with correct results.

Each worker is a spawned process that imports JAX and ORE from scratch, so spawn, import and
first compile dominate this file's run time, not the small portfolios.
"""
import os
import time
from concurrent.futures import Future, wait as futures_wait

import jax.numpy as jnp
import numpy as np
import pytest

from engine.portfolio import PortfolioRequest, PrecisionConfig, price_portfolio
from engine.portfolio.worker_pool import (
    _freeze_trade, _pool_for, _run_pricing_job, shutdown_pools, submit_pricing_job,
)
from tests.test_portfolio_entrypoint import _build_trades, _sim_config


def _make_request(precision: PrecisionConfig) -> PortfolioRequest:
    """A small, quick-to-compile two-trade portfolio from test_portfolio_entrypoint's
    helpers."""
    swap_cfg, swaption_cfg, bermudan_cfg, american_cfg = _build_trades()
    trades = [swap_cfg, swaption_cfg]
    sim_config = _sim_config([swap_cfg, swaption_cfg, bermudan_cfg, american_cfg])
    return PortfolioRequest(market=sim_config, trades=trades, precision=precision)


def _timed_job(frozen_request: PortfolioRequest):
    """Picklable pool task: runs `_run_pricing_job` and returns the worker-side monotonic
    start/end times and `os.getpid()`, so a test can assert where and when the work ran
    (distinct PIDs are the mechanism; overlapping intervals the symptom, which depends on the
    scheduler; I-15)."""
    start = time.monotonic()
    result = _run_pricing_job(frozen_request)
    end = time.monotonic()
    return start, end, result, os.getpid()


def _sleep_job(seconds: float):
    """Picklable pool task that sleeps for `seconds` and returns its PID and interval. A
    warm pricing job finishes in ~15ms, too quickly to occupy a second worker (I-15); this
    tests the pool's dispatch without JAX."""
    start = time.monotonic()
    time.sleep(seconds)
    return start, time.monotonic(), os.getpid()


def _submit_timed(request: PortfolioRequest, pool_size: int = 2) -> "Future":
    """Submit `_timed_job` routed and frozen as `submit_pricing_job` does, to observe
    worker-side timing."""
    from dataclasses import replace
    pool = _pool_for(request.precision.simulation, pool_size=pool_size)
    frozen_trades = [_freeze_trade(cfg) for cfg in request.trades]
    frozen_request = replace(request, trades=frozen_trades)
    return pool.submit(_timed_job, frozen_request)


@pytest.fixture(scope="module", autouse=True)
def _cleanup_pools():
    yield
    # Shut down the pools so later modules do not inherit idle workers.
    shutdown_pools(wait=True)


@pytest.mark.slow
class TestSubmitPricingJobRouting:
    """`submit_pricing_job` routes by `request.precision.simulation`, returns a
    `Future[PortfolioResult]`, and matches `price_portfolio` called directly (so the
    ORE-date freeze/thaw round trip is lossless)."""

    def test_float64_job_returns_correct_result(self):
        precision = PrecisionConfig(simulation=64, pricing=64, risk=64)
        request = _make_request(precision)
        future = submit_pricing_job(request)
        assert isinstance(future, Future)
        result = future.result(timeout=120)

        ref = price_portfolio(_make_request(precision))
        assert result.npv_cube.dtype == jnp.float64
        np.testing.assert_allclose(np.asarray(result.npv_cube), np.asarray(ref.npv_cube), rtol=1e-9)
        np.testing.assert_allclose(result.base_npv, ref.base_npv, rtol=1e-9)

    def test_float32_job_returns_correct_result(self):
        precision = PrecisionConfig(simulation=32, pricing=32, risk=32)
        request = _make_request(precision)
        future = submit_pricing_job(request)
        result = future.result(timeout=120)

        ref = price_portfolio(_make_request(precision))
        assert result.npv_cube.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(result.npv_cube), np.asarray(ref.npv_cube), rtol=1e-3)
        np.testing.assert_allclose(result.base_npv, ref.base_npv, rtol=1e-3)

    def test_mixed_simulation_pricing_precision_routes_by_simulation_only(self):
        """simulation=32 selects the float32 pool even when pricing/risk ask for 64;
        `price_portfolio` re-enables x64 inside the worker, so pricing=64 still works."""
        precision = PrecisionConfig(simulation=32, pricing=64, risk=64)
        request = _make_request(precision)
        result = submit_pricing_job(request).result(timeout=120)

        ref = price_portfolio(_make_request(precision))
        assert result.npv_cube.dtype == jnp.float64  # pricing=64 honored
        np.testing.assert_allclose(np.asarray(result.npv_cube), np.asarray(ref.npv_cube), rtol=1e-6)


@pytest.mark.slow
class TestWorkerPoolConcurrency:
    """A float32-tier and a float64-tier job submitted together both give correct results
    (against sequential references) and overlap in wall-clock time, measured inside the
    workers. Repeated over several pairs, since a single timing observation is weak."""

    def _make_requests(self):
        precision_a = PrecisionConfig(simulation=64, pricing=64, risk=64)
        precision_b = PrecisionConfig(simulation=32, pricing=32, risk=32)
        return _make_request(precision_a), _make_request(precision_b)

    def test_cross_tier_jobs_correct_and_concurrent(self):
        # Sequential references outside the pool (values, not just dtypes).
        precision_a = PrecisionConfig(simulation=64, pricing=64, risk=64)
        precision_b = PrecisionConfig(simulation=32, pricing=32, risk=32)
        ref_a = price_portfolio(_make_request(precision_a))
        ref_b = price_portfolio(_make_request(precision_b))

        # Prime both pools once, unmeasured, so spawn and first-compile time do not dominate
        # the timed jobs.
        futures_wait([_submit_timed(_make_request(precision_a)), _submit_timed(_make_request(precision_b))])

        NUM_ROUNDS = 4
        intervals_a = []
        intervals_b = []
        for _ in range(NUM_ROUNDS):
            req_a, req_b = self._make_requests()
            fut_a = _submit_timed(req_a)
            fut_b = _submit_timed(req_b)
            start_a, end_a, result_a, _ = fut_a.result(timeout=120)
            start_b, end_b, result_b, _ = fut_b.result(timeout=120)

            assert result_a.npv_cube.dtype == jnp.float64
            assert result_b.npv_cube.dtype == jnp.float32
            np.testing.assert_allclose(np.asarray(result_a.npv_cube), np.asarray(ref_a.npv_cube), rtol=1e-9)
            np.testing.assert_allclose(np.asarray(result_b.npv_cube), np.asarray(ref_b.npv_cube), rtol=1e-3)
            np.testing.assert_allclose(result_a.base_npv, ref_a.base_npv, rtol=1e-9)
            np.testing.assert_allclose(result_b.base_npv, ref_b.base_npv, rtol=1e-3)

            intervals_a.append((start_a, end_a))
            intervals_b.append((start_b, end_b))

        def overlaps(iv1, iv2) -> bool:
            return iv1[0] < iv2[1] and iv2[0] < iv1[1]

        overlap_count = sum(
            1 for ia, ib in zip(intervals_a, intervals_b) if overlaps(ia, ib)
        )
        assert overlap_count >= 1, (
            f"expected at least one of {NUM_ROUNDS} concurrently-submitted "
            f"float32-tier/float64-tier job pairs to show genuine wall-clock "
            f"overlap (proving real cross-process concurrency), but none did. "
            f"intervals_a={intervals_a} intervals_b={intervals_b}"
        )

    def test_same_tier_jobs_also_overlap_across_pool_workers(self):
        """Two float64 jobs on a 2-worker pool run on two processes concurrently.

        Regression (I-15): with the warm tiny portfolio each job took ~15ms, so job 1 often
        finished before job 2 was dispatched and both ran on one process; the old overlap
        assertion failed intermittently although the pool was fine. `_sleep_job` lasts long
        enough to need a second worker, making distinct PIDs and overlap deterministic. Real
        pricing concurrency is covered by `test_cross_tier_jobs_correct_and_concurrent`.
        """
        pool = _pool_for(64, pool_size=2)
        # Prime both workers so spawn time is outside the measured interval.
        futures_wait([pool.submit(_sleep_job, 0.05) for _ in range(2)])

        JOB_SECONDS = 1.0
        fut_1 = pool.submit(_sleep_job, JOB_SECONDS)
        fut_2 = pool.submit(_sleep_job, JOB_SECONDS)
        start_1, end_1, pid_1 = fut_1.result(timeout=120)
        start_2, end_2, pid_2 = fut_2.result(timeout=120)

        assert pid_1 != pid_2, (
            f"both jobs ran in worker process {pid_1} -- a 2-worker pool "
            f"serialized two concurrently-submitted same-tier jobs, so the "
            f"tier has no real concurrency"
        )

        # Both intervals use the same monotonic clock inside the workers, so overlap is real.
        assert start_1 < end_2 and start_2 < end_1, (
            f"two {JOB_SECONDS}s jobs on distinct workers did not overlap in "
            f"wall-clock time: [{start_1}, {end_1}] vs [{start_2}, {end_2}]"
        )


class TestTradeFreezingRoundTrip:
    """`_freeze_trade` makes every config picklable, including ORE dates inside a nested
    dataclass (a bond's `CouponPeriod`s), which once failed."""

    def test_coupon_bond_survives_pickling(self):
        import pickle

        import ORE

        from engine.instruments.treasury import BondConfig, CouponPeriod
        from engine.portfolio.worker_pool import _freeze_trade, _thaw_trade
        from engine.simulation.market_model import ZeroCurveConfig

        bond = BondConfig(
            face_amount=100_000.0,
            maturity_date=ORE.Date(1, 1, 2028),
            evaluation_date=ORE.Date(1, 1, 2026),
            initial_zero_curve=ZeroCurveConfig(times=[0.0, 1.0, 5.0], rates=[0.03, 0.03, 0.03]),
            coupon_rate=0.04,
            coupon_schedule=(
                CouponPeriod(ORE.Date(1, 1, 2026), ORE.Date(1, 1, 2027)),
                CouponPeriod(ORE.Date(1, 1, 2027), ORE.Date(1, 1, 2028)),
            ),
        )
        restored = _thaw_trade(pickle.loads(pickle.dumps(_freeze_trade(bond))))
        assert restored == bond


class TestPoolsSpawnOnEveryPlatform:
    """I-33: forked workers hang once the parent has run JAX (Linux defaults to fork;
    Windows always spawns, which is why it went unseen). A stub records the pool's
    construction, so this checks the spawn context on any platform without a process."""

    def test_pool_is_built_with_a_spawn_context(self, monkeypatch):
        from engine.portfolio import worker_pool

        built = {}

        class _RecordingExecutor:
            def __init__(self, **kwargs):
                built.update(kwargs)

        monkeypatch.setattr(worker_pool, "ProcessPoolExecutor", _RecordingExecutor)
        monkeypatch.setattr(worker_pool, "_POOLS", {})
        worker_pool._pool_for(64)
        assert built["mp_context"].get_start_method() == "spawn"

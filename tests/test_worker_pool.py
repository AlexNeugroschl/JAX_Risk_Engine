"""
Tests for `engine.portfolio.worker_pool` -- the multi-process, per-precision-
-tier dispatch layer that replaces "one process, `_PRICING_LOCK`" as the
*primary* concurrency mechanism (the lock itself is unchanged, one layer
down in `engine.portfolio.request`, kept as narrower defense-in-depth).

`TestWorkerPoolConcurrency` is the load-bearing new test this whole effort
is judged by: `tests/test_portfolio_entrypoint.py::TestPricePortfolioConcurrency`
already proves `price_portfolio` is *correct* (no dtype/numeric corruption)
when two threads race inside one process, serialized behind
`_PRICING_LOCK` -- but that test, by construction, never proves genuine
overlap; the lock's whole job is to prevent overlap. This file's job is the
complementary proof: that the worker pool achieves REAL concurrent wall-clock
execution across precision tiers, which a single-process lock architecture
structurally cannot.

Runtime note: each `ProcessPoolExecutor` worker is a full spawned Python
process that re-imports `engine.portfolio`, `jax`, and `ORE` from a clean
interpreter (see `engine.portfolio.worker_pool`'s module docstring) --
Windows process-spawn + first-import + first-JIT-compile overhead dominates
this file's wall time, not the (deliberately tiny) portfolios themselves.
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
    """A small, fast-JIT-compiling portfolio -- reuses
    tests/test_portfolio_entrypoint.py's own trade/sim-config helpers (same
    portfolio TestPricePortfolioConcurrency itself prices) rather than
    inventing a second fixture, and keeps only two trades (not all four) to
    keep each job's own JIT-compile cost small -- process-spawn overhead is
    already the dominant cost in this file (see module docstring)."""
    swap_cfg, swaption_cfg, bermudan_cfg, american_cfg = _build_trades()
    trades = [swap_cfg, swaption_cfg]
    sim_config = _sim_config([swap_cfg, swaption_cfg, bermudan_cfg, american_cfg])
    return PortfolioRequest(market=sim_config, trades=trades, precision=precision)


def _timed_job(frozen_request: PortfolioRequest):
    """Top-level (picklable) helper submitted directly to a tier's pool --
    wraps `_run_pricing_job` to additionally record `time.monotonic()`
    immediately before/after the actual `price_portfolio` call, INSIDE the
    worker process, and returns both through the Future. A worker process
    can't share Python objects (e.g. a shared clock/counter) with the test
    process directly -- timestamps have to be captured on the worker side
    and shipped back, or the measurement wouldn't reflect when the work
    itself actually ran.

    Also reports `os.getpid()` so a test can assert the DISPATCH MECHANISM
    (two jobs landed on two distinct worker processes) rather than only the
    observable SYMPTOM (their wall-clock intervals overlapped). The symptom
    depends on the OS scheduler and on jobs being slow enough to still be
    running when the second one starts; the mechanism does not. See I-15."""
    start = time.monotonic()
    result = _run_pricing_job(frozen_request)
    end = time.monotonic()
    return start, end, result, os.getpid()


def _sleep_job(seconds: float):
    """Top-level (picklable) pool task that occupies a worker for a known
    duration and reports its own process id and monotonic interval.

    Used by the same-tier concurrency test to keep a worker genuinely busy:
    a warm-JIT pricing job finishes in ~15ms, which is too fast to prove
    anything about a 2-worker pool (see that test's docstring, and I-15).
    Deliberately does no JAX work -- the pool's DISPATCH behavior is what
    is under test, and real pricing concurrency is covered by
    `test_cross_tier_jobs_correct_and_concurrent`."""
    start = time.monotonic()
    time.sleep(seconds)
    return start, time.monotonic(), os.getpid()


def _submit_timed(request: PortfolioRequest, pool_size: int = 2) -> "Future":
    """Test-only submission path mirroring `submit_pricing_job`'s own
    routing/freezing (by precision.simulation, via `_freeze_trade`), but
    submitting `_timed_job` instead of the module's plain
    `_run_pricing_job`, so the test can observe genuine worker-side
    wall-clock intervals. Does not change/duplicate `submit_pricing_job`
    itself -- that function's own contract (`Future[PortfolioResult]`, no
    timing) is exercised separately below."""
    from dataclasses import replace
    pool = _pool_for(request.precision.simulation, pool_size=pool_size)
    frozen_trades = [_freeze_trade(cfg) for cfg in request.trades]
    frozen_request = replace(request, trades=frozen_trades)
    return pool.submit(_timed_job, frozen_request)


@pytest.fixture(scope="module", autouse=True)
def _cleanup_pools():
    yield
    # Tear down every pool this module spun up so later test modules (and
    # pytest's own process-exit bookkeeping) don't inherit idle worker
    # processes.
    shutdown_pools(wait=True)


class TestSubmitPricingJobRouting:
    """`submit_pricing_job`'s own public contract: routes purely by
    `request.precision.simulation`, returns a real
    `concurrent.futures.Future[PortfolioResult]`, and produces output
    identical to calling `price_portfolio` directly (sequentially, in this
    test process) on the equivalent request -- proving the freeze/thaw
    round-trip (ORE.Date/Period -> str -> ORE.Date/Period, see
    `engine.portfolio.worker_pool`'s module docstring) is lossless."""

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
        """precision.simulation=32 selects the float32-tier pool even when
        pricing/risk request 64 -- routing is purely by `simulation` (see
        engine.portfolio.worker_pool's module docstring's "Routing"
        section); price_portfolio's own re-enable-x64-after-generate_paths
        logic (unchanged, inside the worker) is what makes pricing=64 still
        work correctly on a float32-tier worker."""
        precision = PrecisionConfig(simulation=32, pricing=64, risk=64)
        request = _make_request(precision)
        result = submit_pricing_job(request).result(timeout=120)

        ref = price_portfolio(_make_request(precision))
        assert result.npv_cube.dtype == jnp.float64  # pricing=64 honored
        np.testing.assert_allclose(np.asarray(result.npv_cube), np.asarray(ref.npv_cube), rtol=1e-6)


class TestWorkerPoolConcurrency:
    """The load-bearing proof: submitting a float32-tier job and a
    float64-tier job concurrently gives (a) both jobs correct,
    uncorrupted, dtype-appropriate output, matching a sequential
    price_portfolio reference, AND (b) genuine wall-clock overlap between
    their [start, end] intervals, timestamped INSIDE the worker processes
    -- proving real concurrency, not just correctness-under-contention
    (which TestPricePortfolioConcurrency in test_portfolio_entrypoint.py
    already proves for the single-process, lock-serialized case).

    Repeats across multiple pairs (not just one shot) since a timing-based
    concurrency assertion that only sometimes observes overlap is a weak
    guarantee -- reported honestly if genuine overlap turns out to be hard
    to reproduce reliably on this specific Windows dev machine (see this
    class's own test bodies for what "reliably" means here in practice).
    """

    def _make_requests(self):
        precision_a = PrecisionConfig(simulation=64, pricing=64, risk=64)
        precision_b = PrecisionConfig(simulation=32, pricing=32, risk=32)
        return _make_request(precision_a), _make_request(precision_b)

    def test_cross_tier_jobs_correct_and_concurrent(self):
        # Sequential references, computed OUTSIDE the pool -- the numeric
        # ground truth each worker-produced result is cross-checked
        # against (dtype alone wouldn't catch numeric corruption).
        precision_a = PrecisionConfig(simulation=64, pricing=64, risk=64)
        precision_b = PrecisionConfig(simulation=32, pricing=32, risk=32)
        ref_a = price_portfolio(_make_request(precision_a))
        ref_b = price_portfolio(_make_request(precision_b))

        # Prime both pools once, unmeasured -- each worker's first job pays
        # process-spawn + first-JIT-compile cost, which would otherwise
        # dominate the timed submissions below and make overlap look like
        # pure spawn-latency coincidence rather than genuine concurrent
        # pricing work. A correctness-under-first-use guarantee is already
        # covered by TestSubmitPricingJobRouting above.
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
        """Two SAME-tier (both float64) jobs, submitted to a 2-worker pool,
        must run on two worker PROCESSES concurrently -- this is what
        "N workers per tier = N devices of that tier running genuinely
        concurrently" (see worker_pool's module docstring) means in
        practice, distinct from the cross-tier case above.

        **Why this test measures a deliberately SLOW job (I-15).** It
        previously primed the pool and then submitted the same tiny
        portfolio the other tests use. After priming, that portfolio's JIT
        cache is warm and each job completes in ~15ms -- so job 1 routinely
        finished before the executor even handed job 2 to a worker. Both
        jobs then ran on ONE process, and the old `overlap_count >= 1`
        assertion failed intermittently even though the pool was behaving
        correctly. The pool was never the problem: the probe is what was
        unsound, since two jobs that never coexist in time cannot
        demonstrate concurrency at all.

        `_sleep_job` below makes the work last long enough that a second
        worker is genuinely required, which turns both the mechanism
        (distinct PIDs) and the payoff (wall-clock overlap) into
        deterministic assertions rather than races against the scheduler.
        It sleeps rather than pricing because what is under test here is
        the POOL's dispatch behavior; `test_cross_tier_jobs_correct_and_
        concurrent` above already covers concurrency with real pricing work
        plus numerical correctness.
        """
        pool = _pool_for(64, pool_size=2)
        # Prime BOTH workers, so process-spawn cost (seconds on Windows --
        # see module docstring) is not inside the measured interval and
        # cannot itself serialize the pair.
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

        # The two intervals are measured inside their own workers against
        # the same monotonic clock, so overlap here is genuine wall-clock
        # concurrency, not an artifact of submission order.
        assert start_1 < end_2 and start_2 < end_1, (
            f"two {JOB_SECONDS}s jobs on distinct workers did not overlap in "
            f"wall-clock time: [{start_1}, {end_1}] vs [{start_2}, {end_2}]"
        )


class TestTradeFreezingRoundTrip:
    """`_freeze_trade` must make every trade config picklable, including
    ORE values nested inside another dataclass. A coupon bond's
    `CouponPeriod`s hold `ORE.Date`s (unpicklable SWIG objects), and before
    nested dataclasses were frozen, submitting one to the pool failed."""

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

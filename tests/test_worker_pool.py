"""
`engine.portfolio.worker_pool`: the process pool behind the HTTP jobs, one for every precision
since roadmap 1.4 (before it, one pool per simulation precision).

A job gives exactly what `price_portfolio` gives in this process, at float64 and at float32
(the freeze/thaw round trip is lossless and every worker runs with x64 on); the worker re-runs
the request's validation; two jobs run on two workers at once.
tests/test_portfolio_entrypoint.py::TestPricePortfolioConcurrency shows `price_portfolio`
stays correct when two threads run it at once in one process.

Each worker is a spawned process that imports JAX and ORE from scratch, so spawn, import and
first compile dominate this file's run time, not the small portfolios.
"""
import os
import time
from concurrent.futures import Future, wait as futures_wait

import jax.numpy as jnp
import numpy as np
import pytest

import ORE

from engine.portfolio import (
    CamConfig, HullWhiteConfig, PortfolioRequest, Precision, RunConfig, price_portfolio,
)
from engine.portfolio.worker_pool import _pool, shutdown_pool, submit_pricing_job
from tests.support import portfolio as shared


def _make_request(precision: Precision = Precision()) -> PortfolioRequest:
    """A small, quick-to-compile two-trade portfolio (a swap and a European) on the shared
    market, simulated by the Hull-White model."""
    trades = [shared.trades()[n] for n in ("swap-payer", "european-payer")]
    simulation = CamConfig(dates=tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (6, 12)),
                           base_currency="USD", ir={"USD": HullWhiteConfig(0.03, 0.01)}, samples=64, seed=5)
    return PortfolioRequest(market=shared.market(), trades=trades,
                            config=RunConfig(simulation=simulation, precision=precision))


def _sleep_job(seconds: float):
    """Picklable pool task that sleeps for `seconds` and returns its PID and interval. A
    warm pricing job finishes in ~15ms, too quickly to occupy a second worker (I-15); this
    tests the pool's dispatch without JAX."""
    start = time.monotonic()
    time.sleep(seconds)
    return start, time.monotonic(), os.getpid()


@pytest.fixture(scope="module", autouse=True)
def _cleanup_pool():
    yield
    # Shut down the pool so later modules do not inherit idle workers.
    shutdown_pool(wait=True)


@pytest.mark.slow
class TestSubmitPricingJob:
    """`submit_pricing_job` returns a `Future[PortfolioResult]` equal to `price_portfolio`
    called directly, bit for bit, at every precision."""

    @pytest.mark.parametrize("precision", [Precision(), Precision.throughout("float32")], ids=["float64", "float32"])
    def test_a_job_equals_the_direct_call(self, precision):
        future = submit_pricing_job(_make_request(precision))
        assert isinstance(future, Future)
        result = future.result(timeout=120)
        ref = price_portfolio(_make_request(precision))
        # Before I-71's fix a float32-tier worker turned x64 off and priced in float32 what
        # this process priced in float64: paths deep in the tail came back 0.0.
        assert result.npv_cube.dtype == ref.npv_cube.dtype == jnp.float64
        np.testing.assert_array_equal(np.asarray(result.npv_cube), np.asarray(ref.npv_cube))
        assert result.base_npv == ref.base_npv

    def test_the_two_precisions_differ(self):
        """The float32 job is priced in float32, not collapsed onto the float64 one."""
        a, b = (submit_pricing_job(_make_request(p)) for p in (Precision(), Precision.throughout("float32")))
        assert not np.array_equal(np.asarray(a.result(timeout=120).npv_cube), np.asarray(b.result(timeout=120).npv_cube))

    def test_the_worker_re_runs_the_validation(self):
        """A request the route would refuse fails its own job, naming the reason, in the
        worker (`price_portfolio` validates before any work)."""
        request = _make_request()
        request.config = RunConfig(precision=request.config.precision)  # scenario risk without a simulation
        with pytest.raises(ValueError, match="scenario_risk needs config.simulation"):
            submit_pricing_job(request).result(timeout=120)


@pytest.mark.slow
class TestWorkerPoolConcurrency:
    def test_two_jobs_run_on_two_workers_at_once(self):
        """Two jobs on a 2-worker pool run on two processes concurrently.

        Regression (I-15): with the warm tiny portfolio each job took ~15ms, so job 1 often
        finished before job 2 was dispatched and both ran on one process; an overlap
        assertion on pricing jobs failed intermittently although the pool was fine (I-27).
        `_sleep_job` lasts long enough to need a second worker, making distinct PIDs and
        overlap deterministic; the pricing jobs' values are `TestSubmitPricingJob`'s.
        """
        # A fresh pool: one left by the pricing tests above may hold a single started worker,
        # which Python's executor reuses rather than spawning the second (the test then failed
        # deterministically after them, passing alone).
        shutdown_pool(wait=True)
        pool = _pool(pool_size=2)
        # Prime both workers so spawn time is outside the measured interval.
        futures_wait([pool.submit(_sleep_job, 0.05) for _ in range(2)])

        JOB_SECONDS = 1.0
        fut_1 = pool.submit(_sleep_job, JOB_SECONDS)
        fut_2 = pool.submit(_sleep_job, JOB_SECONDS)
        start_1, end_1, pid_1 = fut_1.result(timeout=120)
        start_2, end_2, pid_2 = fut_2.result(timeout=120)

        assert pid_1 != pid_2, (
            f"both jobs ran in worker process {pid_1} -- a 2-worker pool serialized two "
            f"concurrently-submitted jobs"
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

        bond = BondConfig(
            trade_id="bond",
            face_amount=100_000.0,
            maturity_date=ORE.Date(1, 1, 2028),
            evaluation_date=ORE.Date(1, 1, 2026),
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
        monkeypatch.setattr(worker_pool, "_POOL", None)
        worker_pool._pool()
        assert built["mp_context"].get_start_method() == "spawn"

"""
The replay of ORE's calibrations (tests/support/ore_calibration_replay.py, I-126): ORE's debug
log read as it writes it, and the engine's side of the replay, which moves nothing when ORE's
calibrations are exact and refuses a market ORE's log does not have.

The replay against ORE itself is the parity sweep's market-risk test on seed 18
(tests/test_ore_parity_sweep.py), where one of ORE's calibrations stops 1.8e-6 short.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from engine.calibration.lgm import bootstrap
from engine.market_data.curves import with_node_deviation
from engine.risk.greeks.price_functions import curves_on, trade_price_function
from engine.risk.market.revaluation import factor_indices
from tests.support.ore_calibration_replay import calibration_tables, ores_fit_correction
from tests.support.parity_sweep import FIXED_SAMPLE, case as sweep_case


def _details(rows, *, prefix="DEBUG    [2026-Nov-10 21:05:41.529292]       (OREData/ored/model/lgmbuilder.cpp:253) : "):
    """`getCalibrationDetails`' table (OREData/ored/model/utilities.cpp) as ORE's log writes it
    (`DLOGGERSTREAM`, OREData/ored/model/lgmbuilder.cpp): every line after the log's prefix,
    `setw(14)` columns at six significant digits, the table ended by the parameters beyond the
    last expiry, which name the header's last column again."""
    header = "".join(f"{h:>14}" for h in ("time", "modelVol", "marketVol", "(diff)", "modelValue", "marketValue",
                                           "(diff)", "irlgm1fAlpha", "irlgm1fKappa"))
    lines = [prefix + "Calibration details (with time grid = calibration swaption expiries):",
             prefix + f"{'#':>3}{header}{'irlgm1fHwSigma':>16}"]
    for j, (market, diff) in enumerate(rows):
        numbers = (0.5 * (j + 1), 0.0085, 0.0085, 1e-9, market + diff, market, diff, 0.009, 0.01)
        lines.append(prefix + f"{j:>3}" + "".join(f"{v:>14.6g}" for v in numbers) + f"{0.009:>16.6g}")
    lines.append(prefix + "t >= 2.5: irlgm1fAlpha = 0.009 irlgm1fKappa = 0.01 irlgm1fHwSigma = 0.009")
    return lines


def test_the_log_is_read_as_ore_writes_it():
    """Each table's helpers in order, their market value and model less market value; lines of
    other messages in and between the tables are not helpers."""
    first = [(0.0123456, 1.2e-15), (0.0234567, -2.22e-08)]
    second = [(0.0111111, 0.0), (0.0222222, 4e-13), (0.0333333, -1.8e-06)]
    lines = (["noise 0 1 2 3 4 5 6 7 8 9 10"] + _details(first) + ["DLOG Skipping calibration as nothing has changed"]
             + _details(second))
    tables = calibration_tables(lines)
    assert [t.shape for t in tables] == [(2, 2), (3, 2)]
    np.testing.assert_allclose(tables[0], first, rtol=1e-6)
    np.testing.assert_allclose(tables[1], second, rtol=1e-6)


#: The engine's side builds and calibrates an option: 5 s or more (`slow`).
slow = pytest.mark.slow


@pytest.fixture(scope="module")
def calibrated():
    """A calibrated option of the sweep's fixed sample, its case, and its helpers' market values on
    the base market and in the first three scenarios."""
    for seed in range(FIXED_SAMPLE):
        c = sweep_case(seed)
        for cfg in c.trades:
            engine = {"bermudan_swaption": c.pricing.bermudan, "american_swaption": c.pricing.american}.get(cfg.product)
            if engine is None or not engine.calibrates or cfg.is_expired():
                continue
            fn = trade_price_function(cfg, c.market, c.pricing, jnp.float64)
            model, calibration = fn.calibration.model, fn.terms.terms.calibration
            base, quotes = curves_on(fn.nodes, fn.rates), jnp.asarray(fn.quotes)
            factors = c.market_risk.scenarios.factors
            own = [factors.slice_of(i) for i in factor_indices(cfg, factors)]
            shifts = np.asarray(c.market_risk.scenarios.shifts)
            markets = []
            for s in (None, 0, 1, 2):
                curves = base if s is None else tuple(with_node_deviation(k, jnp.asarray(shifts[s][o], jnp.float64))
                                                      for k, o in zip(base, own))
                markets.append(np.asarray(bootstrap(calibration.basket, *model.market(calibration, curves, quotes),
                                                    calibration.dt, model.reversion, model.solver)[1]))
            return c, cfg, markets
    pytest.fail("the fixed sample has no calibrated option")


@slow
def test_exact_calibrations_move_nothing(calibrated):
    """With ORE's helpers all exact, the engine's fit is ORE's: no correction."""
    c, cfg, markets = calibrated
    tables = [np.stack([m, np.zeros_like(m)], axis=-1) for m in markets]
    replay = ores_fit_correction(cfg, c.market, c.pricing, c.market_risk.scenarios.factors,
                                 c.market_risk.scenarios.shifts, (0, 1, 2), tables)
    assert np.max(np.abs(replay.correction)) <= 1e-13 * cfg.notional
    assert np.max(replay.ore_miss) == 0.0 and replay.base_miss == 0.0


@slow
def test_a_helper_ore_stopped_short_of_moves_the_pnl_of_its_scenario_alone(calibrated):
    """A miss of 1e-6 on the last helper in scenario 1: that scenario's P&L moves, the others'
    do not, and the miss reported is ORE's."""
    c, cfg, markets = calibrated
    tables = [np.stack([m, np.zeros_like(m)], axis=-1) for m in markets]
    tables[2][-1, 1] = 1e-6 * tables[2][-1, 0]
    replay = ores_fit_correction(cfg, c.market, c.pricing, c.market_risk.scenarios.factors,
                                 c.market_risk.scenarios.shifts, (0, 1, 2), tables)
    assert np.max(np.abs(replay.correction[[0, 2]])) <= 1e-13 * cfg.notional
    assert abs(replay.correction[1]) > 1e-12 * cfg.notional
    assert replay.ore_miss[1] == pytest.approx(1e-6)


@slow
def test_a_market_ores_log_does_not_have_is_refused(calibrated):
    c, cfg, markets = calibrated
    tables = [np.stack([m * 1.001, np.zeros_like(m)], axis=-1) for m in markets]
    with pytest.raises(AssertionError, match="no calibration of the base market"):
        ores_fit_correction(cfg, c.market, c.pricing, c.market_risk.scenarios.factors,
                            c.market_risk.scenarios.shifts, (0,), tables)

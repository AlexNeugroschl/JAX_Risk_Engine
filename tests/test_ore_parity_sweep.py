"""
ORE parity beyond the shared portfolio (I-116): seeded random markets, portfolios and settings
(tests/support/parity_sweep.py) against ORE run in-process, for every analytic the oracles reach.

    today's values   ORE's NPV analytic (tests/support/ore_npv_oracle.py): every trade, and every
                     trade ORE refuses to build refused by the engine too
    sensitivities    ORE's sensitivity analytic (tests/support/ore_sensitivity_oracle.py): every
                     bump Greek of every trade, Theta across the case's roll (a day, a weekend,
                     onto a coupon, fixing or exercise, months), the case's tenors and shift
                     sizes; the AD Greeks of the smooth trades as the shift vanishes
    market risk      ORE's own historical simulation VaR analytic (tests/support/
                     ore_market_risk_oracle.py), end to end: each trade's P&L in every scenario
                     (Monte Carlo or historical, calibrated options recalibrated), the VaR

**Tiers.** The fixed sample (`parity_sweep.FIXED_SAMPLE` seeds, each enumerated choice in turn)
runs in the fast tier, the rest of the sweep (`parity_sweep.FULL_SWEEP` seeds) in the full tier
(`slow`). A failure names its seed and prints the case (`SweepCase.describe`); `pytest
tests/test_ore_parity_sweep.py -k seed-17` reruns it.

**Tolerances**, each of the trade's notional (a face amount for a bond), set from the sweep's
measured worst cases (2026-10-18, 40 seeds): today's values 2.7e-12 (options, ORE's calibration
converging to its Levenberg-Marquardt tolerance) and 1.4e-13 (the rest); bump Greeks 3.4e-11
(Bermudans and Americans, recalibrated under every shift) and 1.5e-13; market-risk P&L 5e-13. A
relative tolerance would not do: a deep out-of-the-money option is worth 1e-40 of its notional.
The AD Greeks are ORE's as the shift vanishes: ORE's forward-difference Delta less half its
Gamma, and its Gamma, differ from the derivatives by the shift's third- and fourth-order terms,
1.1e-2 of a curve's largest for a European days from expiry at a 2.5bp shift; extrapolated over
the shift (Richardson: ORE's Greeks at the case's shift and the bump Greeks at half of it, the
latter shown equal to ORE's at the case's shift), they are the AD Greeks to 2.5e-6 of each curve's
largest, the tolerance 1e-5 of it or 1e-10 of the notional, whichever is larger.

**Where ORE's calibration stops short** (I-126). ORE's Levenberg-Marquardt can stop a few 1e-6
from a helper of a recalibrated option's basket (its finite-difference step straddling the
root), which moves the option's P&L by up to ~2e-6 of its notional, where the engine's fit is
exact. Where a calibrated option's P&L misses in a scenario, the market-risk test reruns ORE on
the trade alone at its debug log level, reads where each of its calibrations ended, and adds to
the engine's P&L what ending there moves it by (tests/support/ore_calibration_replay.py); the
comparison is then at the same tolerance, VaR and ES the engine's statistics on the corrected
P&L, and the test warns, naming each scenario. Only ORE's own misses change the engine's
targets, so an engine error still fails. Seed 18's scenario 9 is one (1.8e-6 on one helper,
3.3e-10 of the notional; 4e-13 against ORE's calibration).

**Refusals.** Where ORE refuses a trade (its NPV analytic leaves it out) or fails an analytic on
a calibration that misses its basket under a shift or a scenario, the engine refuses it too, a
`CalibrationError` (decision A-24); the test asserts that.

**Found by these sweeps and fixed on 2026-10-18** (each test below red on the code before):
a deal-strike helper struck at the exact rate where ORE's builder writes it to six decimals
(1e-7 of a deep in-the-money Bermudan); each helper ending on the underlying's unadjusted
maturity where ORE ends it on its last payment date (1e-3 of a helper's value under ACT/ACT
(ICMA), 1.7e-4 of the option's); an American without a calibrating helper priced on its starting
volatility where ORE refuses it; Theta adding back the coupons paid in the roll, which ORE's
analytic does not (17,000 on a swap whose coupon paid on the Theta date); and the Theta market
reading the swaption volatilities from a Saturday where ORE's floating matrix reads them from
the Monday (a European's Theta 1.06 against ORE's -4.94). The sweep's own generator found two
of its faults the same way: fixings drawn per trade where ORE holds one per index and date, and
as-of dates on holidays. **Found in ORE and recorded** (I-117): on a holiday as-of date, and for
a deal strike ORE's fallback moves on an ACT/ACT (ICMA) swap index, ORE's sensitivity analytic is
inconsistent with itself (its base value off its own scenarios'); on a Theta date where an
American has no calibrating helper ORE values it at 0, where the engine refuses the run; and
under a 5bp volatility shift ORE's optimiser can miss a 32-helper basket the engine fits to
2e-15, failing the analytic. The generator draws none of them.

**ES** is compared with ORE's `RiskStatistics` on ORE's P&L: ORE's historical simulation report
averages the tail with the VaR scenario included, `RiskStatistics` (and the engine) without it
(decision X-12); the report's VaR is the engine's wherever it is defined (at least `1 / p`
scenarios, `N p` not whole, a loss), the test asserting it.
"""
import dataclasses
import warnings

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.calibration.lgm import CalibrationError
from engine.pricing.cube import value_today
from engine.risk.greeks.ad import portfolio_greeks
from engine.risk.greeks.bump import portfolio_sensitivities
from engine.risk.market import MarketRiskRequest, run_market_risk
from engine.risk.market.var_es import compute_risk_metrics
from engine.run import GreeksConfig
from tests.support.ore_calibration_replay import calibration_tables, ores_fit_correction
from tests.support.ore_market_risk_oracle import ore_market_risk_run, ore_trade_pnl_with_log
from tests.support.ore_npv_oracle import ore_npv_run
from tests.support.ore_parity_matrix import cells
from tests.support.ore_sensitivity_oracle import ore_sensitivity_run
from tests.support.parity_sweep import fixed_sample, full_sweep

#: Of the trade's notional (see the module docstring).
TODAY = {"option": 2e-11, "linear": 1e-12}
GREEKS = {"option": 1e-10, "linear": 1e-12}
PNL = 1e-11
#: Of each curve's largest AD Greek, and of the notional (see the module docstring).
AD_RTOL, AD_ATOL = 1e-5, 1e-10
SMOOTH = ("swap", "european_swaption", "bond")
CASES = [(case, False) for case in fixed_sample()] + [(case, True) for case in full_sweep()]


def _params(declare, tolerance: str):
    """One parameter per case, each declaring its own cells (`declare(case)`), the full sweep
    `slow`."""
    return [pytest.param(case, id=case.id, marks=[pytest.mark.ore_parity(declare(case), tolerance=tolerance)]
                         + ([pytest.mark.slow] if slow else [])) for case, slow in CASES]


def _scale(cfg) -> float:
    return float(getattr(cfg, "notional", None) or cfg.face_amount)


def _kind(cfg) -> str:
    return "option" if cfg.product in ("bermudan_swaption", "american_swaption") else "linear"


def _built(case):
    """ORE's NPV run of the case, and the trades it builds: one it refuses is refused by the
    engine too (asserted), and stays out of the other analytics, as ORE leaves it out."""
    ore = ore_npv_run(case.market, case.trades, case.pricing)
    for cfg in case.trades:
        if cfg.trade_id in ore.refused:
            with pytest.raises(CalibrationError):
                value_today([cfg], case.market, "USD", case.pricing)
    return ore, [cfg for cfg in case.trades if cfg.trade_id not in ore.refused]


def _ore_or_refusal(ore, engine):
    """ORE's run (`ore()`), or None where ORE fails it on a calibration that misses its basket
    (under a shift or a scenario), the engine's run (`engine()`) then refused too."""
    try:
        return ore()
    except RuntimeError as error:
        if "calibration target function value" not in str(error):
            raise
    with pytest.raises(CalibrationError):
        engine()
    return None


@pytest.mark.parametrize("case", _params(lambda c: cells("npv", c.trades, c.pricing),
                                         "2e-11 of the notional (Bermudans, Americans), 1e-12 (the rest); a trade "
                                         "ORE refuses to build refused"))
def test_todays_values_are_ores(case):
    ore, built = _built(case)
    values = value_today(built, case.market, "USD", case.pricing)
    for cfg, value in zip(built, values):
        assert abs(value - ore.npv[cfg.trade_id]) <= TODAY[_kind(cfg)] * _scale(cfg), \
            f"{cfg.trade_id}: {value} against ORE's {ore.npv[cfg.trade_id]}\n{case.describe()}"


@pytest.mark.parametrize("case", _params(
    lambda c: cells("sensitivities", c.trades, c.pricing, greeks=GreeksConfig("Bump", c.sensitivity))
    + cells("sensitivities", [t for t in c.trades if t.product in SMOOTH], c.pricing,
            greeks=GreeksConfig("AD", c.sensitivity)),
    "bump Greeks 1e-10 of the notional (Bermudans, Americans), 1e-12 (the rest); AD as the shift vanishes "
    "(Richardson), 1e-5 of each curve's largest"))
def test_the_greeks_are_ores(case):
    _, built = _built(case)
    ore = _ore_or_refusal(lambda: ore_sensitivity_run(case.market, built, case.pricing, case.sensitivity),
                          lambda: portfolio_sensitivities(built, case.market, "USD", case.pricing, case.sensitivity))
    if ore is None:
        return
    bump = portfolio_sensitivities(built, case.market, "USD", case.pricing, case.sensitivity)
    for i, cfg in enumerate(built):
        theirs, mine = ore.greeks[cfg.trade_id], bump[i]
        assert mine.keys() == theirs.keys(), f"{cfg.trade_id}\n{case.describe()}"
        for key in mine:
            np.testing.assert_allclose(mine[key], theirs[key], rtol=0.0, atol=GREEKS[_kind(cfg)] * _scale(cfg),
                                       err_msg=f"{cfg.trade_id} {key}\n{case.describe()}")
    smooth = [cfg for cfg in built if cfg.product in SMOOTH]
    ad = portfolio_greeks(smooth, case.market, "USD", case.pricing, case.sensitivity)
    half = portfolio_sensitivities(smooth, case.market, "USD", case.pricing,
                                   dataclasses.replace(case.sensitivity, curve_shift=case.sensitivity.curve_shift / 2))
    for i, cfg in enumerate(smooth):
        theirs, mine = ore.greeks[cfg.trade_id], ad[i]
        assert mine.keys() == theirs.keys(), cfg.trade_id
        for key in (k for k in mine if k.startswith("delta")):
            gamma = "gamma" + key[len("delta"):]
            # Richardson over the shift: ORE's Greeks at h, the bump Greeks (ORE's, above) at h / 2.
            delta = (8.0 * (half[i][key] - 0.5 * half[i][gamma]) - (theirs[key] - 0.5 * theirs[gamma])) / 3.0
            curvature = (16.0 * half[i][gamma] - theirs[gamma]) / 3.0
            for name, value, expected in ((key, mine[key], delta), (gamma, mine[gamma], curvature)):
                atol = max(AD_RTOL * float(np.max(np.abs(expected))), AD_ATOL * _scale(cfg))
                np.testing.assert_allclose(value, expected, rtol=0.0, atol=atol,
                                           err_msg=f"{cfg.trade_id} {name}\n{case.describe()}")
        np.testing.assert_allclose(mine["theta"], theirs["theta"], rtol=0.0, atol=GREEKS["linear"] * _scale(cfg),
                                   err_msg=f"{cfg.trade_id} theta\n{case.describe()}")


def _market_risk_request(case):
    risk = case.market_risk
    return MarketRiskRequest(list(case.trades), case.market, risk.scenarios, case.pricing, quantiles=risk.quantiles)


@pytest.mark.parametrize("case", _params(lambda c: cells("market_risk", market_risk=_market_risk_request(c)),
                                         "each trade's P&L in every scenario 1e-11 of the notional, against ORE's "
                                         "own calibration where it stops short; VaR 1e-9 relative to ORE's report; "
                                         "ES to ORE's RiskStatistics on ORE's P&L"))
def test_market_risk_is_ores(case):
    _, built = _built(case)
    risk = case.market_risk
    request = MarketRiskRequest(built, case.market, risk.scenarios, case.pricing, quantiles=risk.quantiles)
    ore = _ore_or_refusal(lambda: ore_market_risk_run(case.market, built, risk.history, risk.dates, risk.horizon_days,
                                                      [1.0 - q for q in risk.quantiles], case.pricing),
                          lambda: run_market_risk(request))
    if ore is None:
        return
    result = run_market_risk(request)
    pnl = np.array(result.pnl)
    assert pnl.shape == ore.pnl.shape, case.describe()
    replayed = []
    for j, cfg in enumerate(built):
        missed = np.flatnonzero(np.abs(pnl[:, j] - ore.pnl[:, j]) > PNL * _scale(cfg))
        if missed.size and _kind(cfg) == "option":
            pnl[missed, j] += _on_ores_calibrations(case, cfg, missed, ore.pnl[:, j], replayed)
        np.testing.assert_allclose(pnl[:, j], ore.pnl[:, j], rtol=0.0, atol=PNL * _scale(cfg),
                                   err_msg=f"{cfg.trade_id}\n{case.describe()}")
    figures = result.risk
    if replayed:
        # The engine's own statistics, on its P&L corrected to ORE's calibrations.
        figures = {key: float(np.asarray(value).reshape(())) for key, value in compute_risk_metrics(
            jnp.asarray(pnl)[:, None, :], jnp.asarray(0.0), percentiles=risk.quantiles, include_diagnostics=False).items()}
        warnings.warn("compared against ORE's calibrations where they stopped short (I-126): " + "; ".join(replayed))
    stats = ORE.RiskStatistics()
    stats.add(ORE.DoubleVector([float(v) for v in ore.portfolio_pnl]))
    for q in risk.quantiles:
        label = f"{q * 100:g}"
        reported = -ore.quantile[1.0 - q]
        # The report's quantile is the P&L, a gain unfloored; the engine's VaR floors it at zero as
        # `RiskStatistics` (X-12). The sweep's counts and levels never make N p whole, where the two
        # can read different scenarios (test_ore_parity_settings.py).
        if np.isfinite(reported):
            assert figures[f"VaR_{label}"] == pytest.approx(max(reported, 0.0), rel=1e-9), \
                f"VaR {label}\n{case.describe()}"
        else:  # fewer than 1 / p scenarios: ORE's report has no quantile, the engine the worst P&L's
            assert pnl.shape[0] * (1.0 - q) < 1.0
        assert figures[f"VaR_{label}"] == pytest.approx(stats.valueAtRisk(q), rel=1e-9)
        if np.isfinite(figures[f"ES_{label}"]):  # else an empty tail, where RiskStatistics raises
            assert figures[f"ES_{label}"] == pytest.approx(stats.expectedShortfall(q), rel=1e-9), \
                f"ES {label}\n{case.describe()}"


def _on_ores_calibrations(case, cfg, scenarios, ore_pnl, replayed) -> np.ndarray:
    """How `cfg`'s P&L moves in `scenarios` with its calibrations ending where ORE's did
    (tests/support/ore_calibration_replay.py, I-126), read from ORE's debug log of the trade's run
    alone, whose P&L must be the portfolio run's; the replay's description joins `replayed`."""
    risk = case.market_risk
    alone, log = ore_trade_pnl_with_log(case.market, cfg, risk.history, risk.dates, risk.horizon_days, case.pricing)
    np.testing.assert_allclose(alone, ore_pnl, rtol=0.0, atol=PNL * _scale(cfg),
                               err_msg=f"{cfg.trade_id}: ORE alone is not ORE in the portfolio\n{case.describe()}")
    replay = ores_fit_correction(cfg, case.market, case.pricing, risk.scenarios.factors, risk.scenarios.shifts,
                                 scenarios, calibration_tables(log))
    replayed.append(replay.describe(cfg.trade_id))
    return replay.correction

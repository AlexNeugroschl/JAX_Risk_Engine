"""
ORE parity of the settings each analytic accepts (I-116), on the shared portfolio's sloped
two-curve market (tests/support/portfolio.py), each against ORE run in-process; and a
regression test for each difference the seeded sweeps found (tests/test_ore_parity_sweep.py),
red on the code before its fix (2026-10-18).

    today's values   a deal strike as ORE's builder writes it; each helper's term the
                     underlying's last payment date; an American without a calibrating helper
                     refused as ORE refuses it
    sensitivities    the sensitivity settings (tenors, shift sizes, Theta horizon); each
                     calibration strategy, `ShiftHorizon` and a short option, recalibrated under
                     every shift; Theta onto a coupon (none added back, as in ORE) and across a
                     weekend (the volatilities from the Monday, as ORE's floating matrix)
    market risk      the calibrated options at ORE's default engine (the sweeps' grids are
                     coarser), recalibrated under every historical scenario by ORE's own analytic

Tolerances as tests/test_ore_parity_sweep.py's, of the notional; measured worst cases in each
docstring (2026-10-18).
"""
import dataclasses

import numpy as np
import ORE
import pytest

from engine.calibration.ore_lgm import CalibrationError, SwapIndexConventions
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, exercisable_dates, underlying_swap
from engine.instruments.european_swaption import SwaptionConfig
from engine.pricing.config import LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.cube import value_today
from engine.risk.market import MarketRiskRequest, RateRiskFactors, historical_scenarios, run_market_risk
from engine.risk.greeks.bump import SensitivityConfig, portfolio_sensitivities
from engine.run import GreeksConfig
from tests.support import portfolio as shared
from tests.support.ore_market_risk_oracle import business_days_before, ore_market_risk_run
from tests.support.ore_npv_oracle import ore_npv_run
from tests.support.ore_parity_matrix import cells
from tests.support.ore_sensitivity_oracle import ore_sensitivity_run

#: The option engines on a coarse grid, ORE's run of several cheap.
FAST = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)
TRADES = shared.trades()
OPTIONS = [cfg for cfg in TRADES.values() if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig))]
EUROPEANS = [cfg for cfg in TRADES.values() if isinstance(cfg, SwaptionConfig)]
#: Few tenors: every option's Greeks on them, at a fraction of ORE's 13.
TENORS = ("1Y", "3Y", "7Y", "10Y")
TOLERANCE = {"option": 1e-10, "linear": 1e-12}


def _scale(cfg) -> float:
    return float(getattr(cfg, "notional", None) or cfg.face_amount)


def _assert_greeks_are_ores(trades, pricing, config):
    ore = ore_sensitivity_run(shared.market(), trades, pricing, config)
    mine = portfolio_sensitivities(trades, shared.market(), "USD", pricing, config)
    for i, cfg in enumerate(trades):
        kind = "option" if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)) else "linear"
        assert mine[i].keys() == ore.greeks[cfg.trade_id].keys(), cfg.trade_id
        for key in mine[i]:
            np.testing.assert_allclose(mine[i][key], ore.greeks[cfg.trade_id][key], rtol=0.0,
                                       atol=TOLERANCE[kind] * _scale(cfg), err_msg=f"{cfg.trade_id} {key}")
    return ore, mine


# --- today's values: the sweeps' findings --------------------------------------------------------

def _deep_in_the_money_bermudan(**overrides) -> BermudanSwaptionConfig:
    """A payer Bermudan struck 300bp under the forward on an underlying maturing on a Saturday
    (2033-02-05), its deal strike with more than six decimals."""
    underlying = dict(notional=4e6, fixed_rate=0.0123456789, payer=True, effective_date=ORE.Date(5, 2, 2027),
                      maturity_date=ORE.Date(5, 2, 2033), evaluation_date=shared.ASOF, trade_id="bermudan-itm")
    probe = BermudanSwaptionConfig(**underlying, exercise_dates=[ORE.Date(5, 2, 2028)])
    return dataclasses.replace(probe, exercise_dates=exercisable_dates(probe)[1:], **overrides)


DEAL_STRIKE = PricingConfig(bermudan=dataclasses.replace(FAST, strategy="CoterminalDealStrike"))
ICMA_INDEX = PricingConfig(bermudan=dataclasses.replace(
    FAST, swap_index=SwapIndexConventions(fixed_day_counter="ACT/ACT (ICMA)")))


@pytest.mark.ore_parity(cells("npv", [_deep_in_the_money_bermudan()], DEAL_STRIKE),
                        cells("npv", [_deep_in_the_money_bermudan()], ICMA_INDEX),
                        tolerance="1e-12 of the notional")
@pytest.mark.parametrize("pricing", [DEAL_STRIKE, ICMA_INDEX], ids=["deal-strike", "icma-swap-index"])
def test_a_calibrated_bermudans_helpers_are_ores(pricing):
    """Its deal strike as ORE's builder writes it, six decimals (`calibration_strike`: the exact
    rate left this option 2.5e-11 of its notional off ORE's; a deep in-the-money one of the
    sweep 7.9e-9), and each helper ending on the underlying's last payment date, 2033-02-07, not
    its Saturday maturity (`underlying_maturity`: 2.3e-7 under ACT/ACT (ICMA)). Measured within
    5.2e-15 of the notional."""
    cfg = _deep_in_the_money_bermudan()
    ore = ore_npv_run(shared.market(), [cfg], pricing).npv[cfg.trade_id]
    assert abs(value_today([cfg], shared.market(), "USD", pricing)[0] - ore) <= 1e-12 * cfg.notional


def _closing_american() -> AmericanSwaptionConfig:
    """A seasoned American whose window, opened before the as-of date, closes 40 days after it,
    before the first date of a 6M reference calibration grid."""
    cfg = AmericanSwaptionConfig(notional=1e6, fixed_rate=0.04, payer=True, effective_date=ORE.Date(3, 2, 2026),
                                 maturity_date=ORE.Date(3, 2, 2031), first_exercise_date=shared.ASOF - 30,
                                 last_exercise_date=shared.ASOF + 40, evaluation_date=shared.ASOF,
                                 trade_id="american-closing")
    fixed = [ORE.as_floating_rate_coupon(c).fixingDate() for c in underlying_swap(cfg).floatingLeg()]
    return dataclasses.replace(cfg, fixings={d: 0.0335 for d in fixed if d < shared.ASOF})


COARSE_GRID = PricingConfig(american=dataclasses.replace(FAST, reference_calibration_grid="100,6M"))


@pytest.mark.ore_parity(cells("npv", [_closing_american()], COARSE_GRID), tolerance="refused, as ORE refuses it")
def test_an_american_without_a_calibrating_helper_is_refused_as_ore_refuses_it():
    """ORE's `LgmBuilder` has no calibrating swaption ("no calibrating swaption provided") and
    refuses the trade; until 2026-10-18 the engine priced it on its starting volatility
    (16,035.70)."""
    cfg = _closing_american()
    refused = ore_npv_run(shared.market(), [cfg, TRADES["swap-payer"]], COARSE_GRID).refused
    assert "no calibrating swaption" in refused[cfg.trade_id]
    with pytest.raises(CalibrationError, match="no calibration helper"):
        value_today([cfg], shared.market(), "USD", COARSE_GRID)


# --- sensitivities ------------------------------------------------------------------------------

LINEAR = [TRADES["swap-payer"], TRADES["swap-receiver-seasoned-icma"], TRADES["bond"]]
SETTINGS = SensitivityConfig(curve_tenors=("6M", "2Y", "5Y", "12Y", "30Y"), curve_shift=5e-4, vol_shift=1e-3,
                             theta_days=7)


@pytest.mark.ore_parity(cells("sensitivities", LINEAR + EUROPEANS, greeks=GreeksConfig("Bump", SETTINGS)),
                        tolerance="1e-12 of the notional")
def test_the_sensitivity_settings_are_ores():
    """Other tenors, a 5bp curve shift, a 10bp volatility shift and a week's Theta: every Greek of
    the swaps, bond and Europeans. Measured within 2.7e-15 of the notional."""
    _assert_greeks_are_ores(LINEAR + EUROPEANS, PricingConfig(), SETTINGS)


ENGINES = {
    "atm-shift-0": dataclasses.replace(FAST, strategy="CoterminalATM", shift_horizon=0.0),
    "deal-strike-shift-1": dataclasses.replace(FAST, shift_horizon=1.0, reversion=0.02),
    "uncalibrated": dataclasses.replace(FAST, calibration="None", volatility=0.008),
}
SHORT_OPTIONS = [dataclasses.replace(cfg, long_short="Short", trade_id=f"{cfg.trade_id}-short") for cfg in OPTIONS]
SMALL = SensitivityConfig(curve_tenors=TENORS)


@pytest.mark.ore_parity(*(cells("sensitivities", SHORT_OPTIONS, PricingConfig(bermudan=e, american=e),
                                greeks=GreeksConfig("Bump", SMALL)) for e in ENGINES.values()),
                        tolerance="1e-10 of the notional")
@pytest.mark.parametrize("engine", list(ENGINES.values()), ids=list(ENGINES))
def test_each_calibration_strategy_and_shift_horizon_is_ores(engine):
    """The shared portfolio's Bermudans and American booked short, recalibrated under every shift
    by each strategy, at `ShiftHorizon` 0 and 1, and uncalibrated. Measured within 3.6e-14 of the
    notional."""
    _assert_greeks_are_ores(SHORT_OPTIONS, PricingConfig(bermudan=engine, american=engine), SMALL)


ROLL_ONTO_A_COUPON = SensitivityConfig(curve_tenors=TENORS, theta_days=16)


@pytest.mark.ore_parity(cells("sensitivities", LINEAR, greeks=GreeksConfig("Bump", ROLL_ONTO_A_COUPON)),
                        tolerance="1e-12 of the notional")
def test_theta_onto_a_coupon_adds_nothing_back_as_ores():
    """A 16-day roll onto the bond's coupon of Saturday 2026-08-15: ORE's period flows are read
    after the roll, where the coupon has occurred, so it leaves Theta as it leaves the NPV; until
    2026-10-18 the engine added it back (2,059.90 where ORE's is -16,690.10). Measured within
    5.2e-16 of the face."""
    ore, mine = _assert_greeks_are_ores(LINEAR, PricingConfig(), ROLL_ONTO_A_COUPON)
    assert -18_750.0 < float(mine[2]["theta"]) < -15_000.0   # minus the coupon, plus 16 days' carry


ACROSS_A_WEEKEND = SensitivityConfig(curve_tenors=TENORS, theta_days=2)


@pytest.mark.ore_parity(cells("sensitivities", EUROPEANS, greeks=GreeksConfig("Bump", ACROSS_A_WEEKEND)),
                        tolerance="1e-12 of the notional")
def test_theta_across_a_weekend_reads_the_volatilities_from_the_monday_as_ore():
    """From Thursday 2026-07-30 to Saturday 2026-08-01: ORE's simulation-market volatility matrix
    floats to the first business day, Monday 2026-08-03, where each option's variance starts
    (`floating_volatility_date`); until 2026-10-18 the engine started it on the Saturday (the
    Europeans' Theta -119.84 and -11.11 where ORE's are -194.98 and -31.13). Measured within
    2.8e-16 of the notional."""
    _assert_greeks_are_ores(EUROPEANS, PricingConfig(), ACROSS_A_WEEKEND)


# --- market risk --------------------------------------------------------------------------------

DEFAULT_ENGINE = PricingConfig(bermudan=shared.ENGINE, american=shared.ENGINE)


def _history(days: int, seed: int = 2026) -> np.ndarray:
    """`days` levels of the shared market's factors, a 5bp-a-day random walk ending at today's."""
    factors = RateRiskFactors.from_market(shared.market())
    steps = np.random.default_rng(seed).normal(0.0, 5e-4, size=(days - 1, factors.size))
    walk = np.concatenate([np.zeros((1, factors.size)), np.cumsum(steps, axis=0)])
    return factors.base_rates()[None, :] + walk - walk[-1][None, :]


def _market_risk(trades, history, quantile, pricing=PricingConfig()):
    """ORE's HISTSIM_VAR run and the engine's of `trades` on the one-day moves of `history`."""
    scenarios = historical_scenarios(RateRiskFactors.from_market(shared.market()), history, horizon_days=1)
    ore = ore_market_risk_run(shared.market(), trades, history, business_days_before(shared.ASOF, len(history)),
                              1, [1.0 - quantile], pricing)
    return ore, run_market_risk(MarketRiskRequest(trades, shared.market(), scenarios, pricing,
                                                  quantiles=(quantile,)))


#: Five one-day moves whose two worst are losses; 70%, a quantile whose N p (1.5) is not whole.
HISTORY = _history(6, seed=4)
QUANTILE = 0.7


@pytest.mark.ore_parity(cells("market_risk", market_risk=MarketRiskRequest(
                            OPTIONS, shared.market(), historical_scenarios(
                                RateRiskFactors.from_market(shared.market()), HISTORY, horizon_days=1),
                            DEFAULT_ENGINE, quantiles=(QUANTILE,))),
                        tolerance="P&L 1e-10 of the notional; VaR 1e-9 relative")
def test_market_risk_of_options_at_ores_default_engine_is_ores():
    """The Bermudans and American on ORE's default 30-point grid (the sweeps' are coarser), each
    recalibrated under every move by ORE's HISTSIM_VAR analytic, as by `run_market_risk`."""
    ore, result = _market_risk(OPTIONS, HISTORY, QUANTILE, DEFAULT_ENGINE)
    pnl = np.asarray(result.pnl)
    assert pnl.shape == ore.pnl.shape == (5, len(OPTIONS))
    for j, cfg in enumerate(OPTIONS):
        np.testing.assert_allclose(pnl[:, j], ore.pnl[:, j], rtol=0.0, atol=1e-10 * cfg.notional,
                                   err_msg=cfg.trade_id)
    assert np.sort(ore.portfolio_pnl)[1] < 0.0   # a loss: the VaR is not floored
    assert result.risk[f"VaR_{QUANTILE * 100:g}"] == pytest.approx(-ore.quantile[1.0 - QUANTILE], rel=1e-9)


def test_at_a_whole_n_p_the_var_is_risk_statistics_not_ores_report():
    """ORE's report reads the (floor(N p) + 1)-th smallest P&L, `RiskStatistics` and the engine
    the ceil(N (1 - c))-th, which at a whole N p is the one below where 1 - c rounds low in
    binary: 90% of ten moves, ORE's second, `RiskStatistics`' the worst (decision X-12;
    measured 2026-10-18, as at 80% of five; 95% of 20 and 97.5% of 40 round high and agree)."""
    swap = TRADES["swap-payer"]
    ore, result = _market_risk([swap], _history(11, seed=10), 0.9)
    ranked = np.sort(ore.portfolio_pnl)
    assert ore.quantile[1.0 - 0.9] == pytest.approx(ranked[1], rel=1e-12)
    stats = ORE.RiskStatistics()
    stats.add(ORE.DoubleVector([float(v) for v in ore.portfolio_pnl]))
    assert result.risk["VaR_90"] == pytest.approx(stats.valueAtRisk(0.9), rel=1e-9)
    assert result.risk["VaR_90"] == pytest.approx(-ranked[0], rel=1e-9)

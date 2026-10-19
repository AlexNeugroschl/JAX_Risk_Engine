"""
The engine's assembled exposure pipeline against ORE's own exposure simulation (I-50; docs/planning/details/ore-parity-validation.md, layers L3 and L4, gate V-4), through
`tests/support/ore_xva_oracle.py`, on the shared portfolio's sloped two-curve market
(tests/support/portfolio.py), under both interest-rate models calibrated to a tenor basket:

  * the oracle is ORE: its in-memory run of ORE's `Examples/Exposure` swap equals ORE's own
    file-driven run of the example;
  * V-4: ORE's scenario dump is the engine's simulation market on ORE's paths. Each path's
    state, read from ORE's numeraire, rebuilds every discount and index curve ORE simulated
    (which also checks the calibration ORE's `CrossAssetModelBuilder` ran against the engine's);
  * L3: on ORE's paths every trade's NPV equals ORE's cube, cell by cell, and t=0 (ORE's `T0`)
    is the engine's value on the simulation market of the as-of date; a Bermudan or American
    is the trade as ORE built it on the as-of date, its LGM calibrated there and kept on every
    path (I-49), at ORE's default `ShiftHorizon` (I-32), calibrated or not; an option booked
    `Short` (ORE's `LongShort`) is minus the long one in ORE's cube and in the engine's, and
    equals ORE's (I-101);
  * ORE's exposure definitions: the engine's profiles of ORE's own cube are ORE's reports;
  * L4 (statistical, `slow`): the engine's simulation and exposure against ORE's, each on its
    own random numbers (decision X-1), within four combined standard errors.

Measured (2026-10-07, ORE 1.8.16, both models): the dumped curves rebuild to 1.2e-11
relative; swaps, Europeans and the bond equal ORE's cube to 3.0e-11 of each trade's largest
value; T0 to 3.9e-11 relative; the profiles of ORE's cube to 1.8e-13. Since 2026-10-12 (I-49)
the calibrated Bermudans and the American equal ORE's cube to 1.9e-11 of their largest value.

Found by this test and fixed on 2026-10-12 (I-49): the engine recalibrated every option's LGM on
every path and date, where ORE's exposure simulation keeps the model it calibrated when it built
the trade on the simulation market of the as-of date (its `XvaAnalytic` gives its valuation
engine no model builders): the options were worth 0.5-4% more than ORE's on the paths, and near
the exercise boundary one cell in 160 took the other decision (a cash-settled Bermudan ORE
exercised, 0, the engine did not). Calibrated on today's market rather than the simulation
market's, the kept model leaves 1e-5.

Found by this test and fixed on 2026-10-12 (I-32): on a path date the engine measured the shift
horizon and opened an American's window from that date, where ORE keeps the as-of build's: up to
4.2e-5 of the Bermudan's largest value and 1.1e-3 of the American's, with the LGM uncalibrated.

Found by this test and fixed on 2026-10-07: the simulation market held each curve's tenor
points at the tenors' times from the simulation date, where ORE holds them at the times from the
as-of date (`engine.market_simulation.scenario_market`), up to 0.3% of a swap's path values; and the
exposure started from today's value, where ORE starts from the simulation market's (`T0`).
"""
import csv
import dataclasses
import os
import shutil
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.calibration.lgm import SwapIndexConventions
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, exercisable_dates
from engine.instruments.bermudan_swaption import underlying_swap as option_underlying
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market_data.curves import ZeroCurve, discount
from engine.market_data.market import CurrencyMarket, Market, index_name
from engine.models.lgm import Sigma
from engine.market_simulation.config import CamConfig, HullWhiteConfig, LgmConfig, build_cross_asset_model, simulate
from engine.market_simulation.scenario_market import build_scenario_market
from engine.pricing.config import LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.context import simulation_market_today
from engine.pricing.cube import trade_maturity, value_paths, value_today
from engine.risk.counterparty.exposure import exposure_profile, netting_set_profile
from engine.run import PortfolioRequest, RunConfig, price_portfolio
from tests.support import portfolio as shared
from tests.support.ore_inputs import DISCOUNT_INDEX, report_columns, run_app, _scratch_dir
from tests.support.ore_parity_matrix import cells
from tests.support.ore_xva_oracle import NETTING_SET_PROFILE, implied_states, ore_xva_run

MODELS = {"LGM": LgmConfig, "HullWhite": HullWhiteConfig}
BASKET = dict(calibration_expiries=("1Y", "2Y", "3Y"), calibration_terms=("5Y", "4Y", "3Y"))
#: Before and after the first exercises, the Europeans' expiries and the bond's coupons.
DATES = tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (3, 6, 12, 24, 36))
FAST = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)
PRICING = PricingConfig(bermudan=FAST, american=FAST)
#: The options' LGM uncalibrated, so the shift horizon is checked apart from the calibration, at
#: ORE's default `ShiftHorizon` (stated: the shift's horizon is what the test pins).
UNCALIBRATED = dataclasses.replace(FAST, calibration="None", shift_horizon=0.5)
PFE_QUANTILE = 0.95
OPTIONS = (BermudanSwaptionConfig, AmericanSwaptionConfig)
#: Cells of a trade's cube equal ORE's to this fraction of the trade's largest value.
CUBE_RTOL = 1e-10
#: The rebuilt curves, T0 and ORE's profiles of its own cube, relative.
RTOL = 1e-10


def _simulation(model: str, samples: int, seed: int = 7) -> CamConfig:
    return CamConfig(dates=DATES, base_currency="USD", ir={"USD": MODELS[model](0.03, 0.01, **BASKET)},
                     samples=samples, seed=seed)


def _trades():
    return list(shared.trades().values())


def _declared(trades, pricing=PRICING, models=tuple(MODELS), pfe_quantiles=None, samples=32):
    """The matrix cells of an exposure comparison of `trades` under each of `models`."""
    return [cell for model in models for cell in cells("exposure", trades, pricing, pfe_quantiles=pfe_quantiles,
                                                       simulation=_simulation(model, samples))]


def _shorts():
    return [dataclasses.replace(cfg, long_short="Short", trade_id=f"{cfg.trade_id}-short") for cfg in _trades()
            if isinstance(cfg, (SwaptionConfig,) + OPTIONS)]


def _on_ores_paths(trades, model: str, pricing: PricingConfig = PRICING, market=None, config=None):
    """ORE's run of `trades` on 32 paths (on the shared market under `model`, unless `market` and
    `config` say otherwise), and the engine's simulation market and cube on ORE's paths."""
    market, config = market or shared.market(), config or _simulation(model, samples=32)
    ore = ore_xva_run(market, trades, config, pricing, PFE_QUANTILE)
    cam = build_cross_asset_model(market, config)
    index_curves = {name: ("USD", ZeroCurve.from_config(curve))
                    for name, curve in market.currency("USD").index_curves.items()}
    scenarios = build_scenario_market(cam, market.asof, config.dates,
                                      jnp.asarray(implied_states(cam, ore.times, ore.numeraire)),
                                      config.curve_tenors, index_curves)
    columns = value_paths(trades, market, scenarios, "USD", pricing, config.swaption_vol_decay)
    cube = np.stack([np.asarray(c) for c in columns], axis=-1) / ore.numeraire[:, :, None]
    return ore, scenarios, cube


@pytest.fixture(scope="module", params=list(MODELS))
def on_ores_paths(request):
    """ORE's run of the shared portfolio on 32 paths, and the engine's on ORE's paths."""
    return _on_ores_paths(_trades(), request.param)


@pytest.fixture(scope="module")
def shorts_on_ores_paths():
    """ORE's run of every option of the shared portfolio, long and `Short` side by side, on 32
    paths of the LGM, and the engine's on ORE's paths."""
    longs = [cfg for cfg in _trades() if isinstance(cfg, (SwaptionConfig,) + OPTIONS)]
    trades = [t for cfg in longs
              for t in (cfg, dataclasses.replace(cfg, long_short="Short", trade_id=f"{cfg.trade_id}-short"))]
    return trades, _on_ores_paths(trades, "LGM")


@pytest.mark.ore_parity([c for model in MODELS for c in cells("calibration", models=["cross_asset_model"],
                                                             simulation=_simulation(model, 32))],
                        tolerance="every simulated curve 1e-10 relative, rebuilt from the calibrated model's states")
def test_the_scenario_dump_is_the_engines_simulation_market_on_ores_paths(on_ores_paths):
    """Gate V-4: one state per path and date (read from the numeraire) rebuilds every curve
    ORE dumped, so ORE's paths can be priced by the engine (L3)."""
    ore, scenarios, _ = on_ores_paths
    np.testing.assert_allclose(np.asarray(scenarios.numeraire), ore.numeraire, rtol=1e-14)
    np.testing.assert_allclose(np.exp(np.asarray(scenarios.discount["USD"].log_discounts))[:, :, 1:], ore.discount,
                               rtol=RTOL)
    np.testing.assert_allclose(np.exp(np.asarray(scenarios.index[shared.INDEX].log_discounts))[:, :, 1:],
                               ore.index[shared.INDEX], rtol=RTOL)
    # ORE simulates the discounting index from the discount curve: the same curve.
    np.testing.assert_allclose(ore.index[DISCOUNT_INDEX], ore.discount, rtol=1e-14)


@pytest.mark.ore_parity(_declared(_trades()), tolerance="each cell 1e-10 of the trade's largest value")
@pytest.mark.parametrize("name", list(shared.trades()))
def test_every_trades_cube_equals_ores_on_ores_paths(on_ores_paths, name):
    """L3: each cell within `CUBE_RTOL` of the trade's largest value; a Bermudan's or
    American's LGM calibrated once, where ORE builds the trade, and kept (I-49)."""
    ore, _, cube = on_ores_paths
    j = ore.trade_ids.index(name)
    scale = np.max(np.abs(ore.cube[:, :, j]))
    assert scale > 0.0, "the trade must be worth something on the paths"
    assert np.max(np.abs(cube[:, :, j] - ore.cube[:, :, j])) <= CUBE_RTOL * scale


@pytest.mark.ore_parity(_declared([c for c in _trades() if isinstance(c, OPTIONS)],
                                  PricingConfig(bermudan=UNCALIBRATED, american=UNCALIBRATED), models=("LGM",)),
                        tolerance="each cell 1e-10 of the trade's largest value")
def test_an_uncalibrated_option_equals_ores_cube_on_ores_paths():
    """L3 for Bermudans and Americans apart from their calibration: with the LGM
    uncalibrated, every cell of every option equals ORE's. ORE builds the trade once, on the
    as-of date, so on a later date its LGM is shifted at the as-of horizon (`LgmBuilder` shifts
    once) and an American's window opened no later than the day after the as-of date, so the
    path date itself is an exercise inside the window (I-32)."""
    options = [cfg for cfg in _trades() if isinstance(cfg, OPTIONS)]
    ore, _, cube = _on_ores_paths(options, "LGM", PricingConfig(bermudan=UNCALIBRATED, american=UNCALIBRATED))
    for j, name in enumerate(ore.trade_ids):
        scale = np.max(np.abs(ore.cube[:, :, j]))
        assert scale > 0.0, name
        assert np.max(np.abs(cube[:, :, j] - ore.cube[:, :, j])) <= CUBE_RTOL * scale, name


@pytest.mark.ore_parity(_declared(_shorts(), models=("LGM",)), tolerance="each cell 1e-10 of the trade's largest value")
def test_a_short_option_on_ores_paths_is_ores_short_option(shorts_on_ores_paths):
    """I-101: ORE's `OptionWrapper` multiplies a short option's NPV by -1, its exercise the
    holder's, so ORE's cube and T0 of a short option are minus the long one's; the engine's
    are too, so every short option equals ORE's cell by cell."""
    trades, (ore, _, cube) = shorts_on_ores_paths
    for j in range(0, len(trades), 2):
        long, sold = trades[j], trades[j + 1]
        scale = np.max(np.abs(ore.cube[:, :, j]))
        assert scale > 0.0, long.trade_id
        np.testing.assert_allclose(ore.cube[:, :, j + 1], -ore.cube[:, :, j], rtol=0.0, atol=1e-14 * scale,
                                   err_msg=f"ORE: {sold.trade_id}")
        assert ore.t0[j + 1] == pytest.approx(-ore.t0[j], rel=1e-14)
        np.testing.assert_array_equal(cube[:, :, j + 1], -cube[:, :, j], err_msg=f"engine: {sold.trade_id}")
        assert np.max(np.abs(cube[:, :, j + 1] - ore.cube[:, :, j + 1])) <= CUBE_RTOL * scale, sold.trade_id


# --- the settings the shared run leaves at their defaults (I-116) ---------------------------------

ICMA_3M = SwapIndexConventions(fixed_tenor="6M", fixed_day_counter="ACT/ACT (ICMA)", index_tenor_months=3)
#: Name -> (the index's tenor in months, the simulation, the option engines).
SETTINGS = {
    "lgm-uncalibrated-piecewise-6m": (6, CamConfig(
        dates=DATES, base_currency="USD", samples=32, seed=7, swaption_vol_decay="ConstantVariance",
        curve_tenors=("3M", "6M", "1Y", "2Y", "5Y", "10Y", "30Y"),
        ir={"USD": LgmConfig(0.0, Sigma(jnp.asarray([1.0, 2.0]), jnp.asarray([0.008, 0.010, 0.009])))}),
        PricingConfig(bermudan=dataclasses.replace(FAST, strategy="CoterminalATM", shift_horizon=1.0, reversion=0.02,
                                                   volatility=0.008, reference_calibration_grid="400,1M",
                                                   bootstrap_tolerance=1e-3, solver="Bisection"),
                      american=dataclasses.replace(FAST, exercise_time_steps_per_year=12,
                                                   reference_calibration_grid="100,6M"))),
    "hull-white-calibrated-3m": (3, CamConfig(
        dates=DATES, base_currency="USD", samples=32, seed=7, bootstrap_tolerance=1e-3,
        ir={"USD": HullWhiteConfig(-0.01, 0.01, swap_index=ICMA_3M, solver="Bisection", **BASKET)}),
        PricingConfig(bermudan=dataclasses.replace(FAST, swap_index=ICMA_3M),
                      american=dataclasses.replace(FAST, strategy="CoterminalATM", swap_index=ICMA_3M, shift_horizon=1.5,
                                                   reversion=0.01, volatility=0.009, bootstrap_tolerance=1e-3,
                                                   solver="Bisection"))),
}


def _market_on(index_months: int) -> Market:
    """The shared market, its forwarding curve the index of `index_months`'."""
    data = shared.market().currency("USD")
    curve = next(iter(data.index_curves.values()))
    return Market(shared.ASOF, {"USD": CurrencyMarket(data.discount_curve, {index_name("USD", index_months): curve},
                                                      data.swaption_vols)})


def _settings_trades(index_months: int):
    """A trade of each product with the settings the shared portfolio leaves at their defaults:
    on the index of `index_months`, receivers with a spread, the options on a seasoned underlying
    with its past fixings (the American's window opened before the as-of date, cash settled), a
    bond accruing ACT/365 and a bill."""
    asof = shared.ASOF
    common = dict(notional=1e6, fixed_rate=0.043, payer=False, index_tenor_months=index_months,
                  floating_spread=0.0015, evaluation_date=asof)
    underlying = dict(effective_date=ORE.Date(3, 2, 2026), maturity_date=ORE.Date(3, 2, 2033))
    probe = BermudanSwaptionConfig(**common, **underlying, exercise_dates=[ORE.Date(3, 2, 2032)], trade_id="probe")
    fixed = [ORE.as_floating_rate_coupon(c).fixingDate() for c in option_underlying(probe).floatingLeg()]
    fixings = {d: 0.033 + 0.0001 * k for k, d in enumerate(d for d in fixed if d < asof)}
    starts = [d for d in exercisable_dates(probe) if d > asof]
    return [
        SwapConfig(**common, swap_tenor="5Y", trade_id="swap"),
        SwaptionConfig(**common, swap_tenor="5Y", forward_start=ORE.Period(1, ORE.Years), trade_id="european"),
        BermudanSwaptionConfig(**common, **underlying, exercise_dates=starts[:-1], fixings=fixings,
                               trade_id="bermudan"),
        AmericanSwaptionConfig(**common, **underlying, first_exercise_date=asof - 30, last_exercise_date=starts[-2],
                               fixings=fixings, settlement="Cash", trade_id="american"),
        BondConfig(face_amount=1e6, maturity_date=ORE.Date(15, 8, 2029), coupon_rate=0.0375, accrual_day_count="ACT/365",
                   coupon_schedule=tuple(CouponPeriod(a, b) for a, b in zip(
                       [ORE.Date(15, 8, 2026 + k) for k in range(3)], [ORE.Date(15, 8, 2027 + k) for k in range(3)])),
                   evaluation_date=asof, trade_id="bond"),
        BondConfig(face_amount=1e6, maturity_date=ORE.Date(15, 1, 2027), evaluation_date=asof, trade_id="bill"),
    ]


def _settings_declared(analytic):
    out = []
    for months, simulation, pricing in SETTINGS.values():
        out += (cells("exposure", _settings_trades(months), pricing, simulation=simulation) if analytic == "exposure"
                else cells("calibration", models=["cross_asset_model"], simulation=simulation))
    return out


@pytest.fixture(scope="module", params=list(SETTINGS))
def settings_on_ores_paths(request):
    months, simulation, pricing = SETTINGS[request.param]
    trades = _settings_trades(months)
    return trades, _on_ores_paths(trades, "", pricing, _market_on(months), simulation)


@pytest.mark.ore_parity(_settings_declared("exposure"), tolerance="each cell 1e-10 of the trade's largest value")
@pytest.mark.ore_parity(_settings_declared("calibration"),
                        tolerance="every simulated curve 1e-10 relative, rebuilt from the model's states")
def test_other_settings_equal_ores_simulation_on_ores_paths(settings_on_ores_paths):
    """The simulation's and the engines' settings the shared run leaves at their defaults, and
    the trades': an uncalibrated LGM with a piecewise volatility, ConstantVariance and other
    tenors; a calibrated Hull-White model with a negative reversion, its own tolerance, the
    reference solver and a semi-annual ACT/ACT (ICMA) swap index on a 3M index; the option
    engines' strategies, shift horizons, reversions, grids and tolerances. On ORE's paths every
    simulated curve is rebuilt from the model's states and every trade's cube equals ORE's.
    Measured (2026-10-18): the curves within 2.4e-11 relative, the cubes within 3.1e-11 of each
    trade's largest value. Found by this test: the American, its window opened before the as-of
    date, was never exercised on a path, where ORE's `ExerciseBuilder` makes it exercisable from
    the day after (`contract_exercise_dates`; the cash-settled one 0.39 of its largest value off
    ORE's on 20 cells)."""
    trades, (ore, scenarios, cube) = settings_on_ores_paths
    np.testing.assert_allclose(np.exp(np.asarray(scenarios.discount["USD"].log_discounts))[:, :, 1:], ore.discount,
                               rtol=RTOL)
    for j, cfg in enumerate(trades):
        scale = np.max(np.abs(ore.cube[:, :, j]))
        assert scale > 0.0, cfg.trade_id
        assert np.max(np.abs(cube[:, :, j] - ore.cube[:, :, j])) <= CUBE_RTOL * scale, cfg.trade_id


@pytest.mark.ore_parity(_declared(_trades()), tolerance="T0 1e-10 relative")
def test_t0_is_the_value_on_the_simulation_market_of_the_as_of_date(on_ores_paths):
    """ORE's cube starts from each trade's value on the `ScenarioSimMarket` of the as-of date
    (the curves sampled at the simulation tenors), and so do the engine's profiles."""
    ore, _, _ = on_ores_paths
    market, config = shared.market(), _simulation("LGM", samples=32)
    start = value_today(_trades(), market, "USD", PRICING, simulation_market_today(market, config.curve_tenors))
    np.testing.assert_allclose(start, ore.t0, rtol=RTOL)
    assert not np.allclose(value_today(_trades(), market, "USD", PRICING), ore.t0, rtol=1e-6), \
        "today's market should differ from the simulation market's, or this test checks nothing"


@pytest.mark.ore_parity(_declared(_trades(), pfe_quantiles=(PFE_QUANTILE,)), tolerance="EPE, ENE at t=0 1e-10 relative")
def test_the_pipelines_profiles_start_from_ores_t0(on_ores_paths, request):
    """I-85: `price_portfolio`'s exposure at t=0 is ORE's, each trade's and the netting set's
    (until 2026-10-07 it started from today's value, up to 4.2% away on this portfolio)."""
    ore, _, _ = on_ores_paths
    model = request.node.callspec.params["on_ores_paths"]
    result = price_portfolio(PortfolioRequest(shared.market(), _trades(),
                                              RunConfig(simulation=_simulation(model, samples=32), pricing=PRICING),
                                              pfe_quantiles=(PFE_QUANTILE,)))
    for profile, key in zip(result.trade_exposures + [result.exposure], list(ore.trade_ids) + [NETTING_SET_PROFILE]):
        for field, column in (("epe", "EPE"), ("ene", "ENE")):
            assert float(getattr(profile, field)[0]) == pytest.approx(ore.exposure[key][column][0], rel=RTOL, abs=1e-6)


@pytest.mark.ore_parity(_declared(_trades(), pfe_quantiles=(PFE_QUANTILE,)),
                        tolerance="every profile on ORE's cube 1e-10 relative")
def test_the_profiles_of_ores_cube_are_ores_reports(on_ores_paths):
    """ORE's `ExposureCalculator` definitions (`engine.risk.counterparty.exposure`), on ORE's own cube and
    numeraire, per trade and for the netting set."""
    ore, scenarios, _ = on_ores_paths
    market, trades = shared.market(), _trades()
    p0 = discount(ZeroCurve.from_config(market.currency("USD").discount_curve), jnp.asarray(ore.times))
    common = dict(numeraire=ore.numeraire, discount=p0, times=ore.times, quantiles=(PFE_QUANTILE,), dates=ore.dates,
                  asof=market.asof)
    npv = ore.cube * ore.numeraire[:, :, None]
    profiles = {t.trade_id: exposure_profile(npv[:, :, j], ore.t0[j], maturity=trade_maturity(t), **common)
                for j, t in enumerate(trades)}
    profiles[NETTING_SET_PROFILE] = netting_set_profile(npv, ore.t0, **common)
    # ORE's report labels its dates with ActualActual (ISDA) times, not the simulation's ACT/365.
    labels = [0.0] + [ORE.ActualActual(ORE.ActualActual.ISDA).yearFraction(market.asof, d) for d in ore.dates]
    for key, profile in profiles.items():
        expected = ore.exposure[key]
        np.testing.assert_allclose(labels, expected["Time"], rtol=0, atol=1e-12)
        for field, column in (("epe", "EPE"), ("ene", "ENE"), ("ee_b", "BaselEE"), ("eee_b", "BaselEEE"),
                              ("epe_b", "TimeWeightedBaselEPE"), ("eepe_b", "TimeWeightedBaselEEPE")):
            np.testing.assert_allclose(np.asarray(getattr(profile, field)), expected[column], rtol=RTOL,
                                       atol=1e-6, err_msg=f"{key} {column}")
        np.testing.assert_allclose(np.asarray(profile.pfe["PFE_95"]), expected["PFE"], rtol=RTOL, atol=1e-6,
                                   err_msg=f"{key} PFE")


# --- L4 -------------------------------------------------------------------------------------------

def _mean_and_error(samples, axis=0):
    samples = np.asarray(samples)
    return samples.mean(axis=axis), samples.std(axis=axis, ddof=1) / np.sqrt(samples.shape[axis])


def _quantile_and_error(samples, q):
    """The quantile as ORE takes it (`sorted[floor(q (n - 1) + 0.5)]`) and a standard error from
    the order statistics one binomial standard deviation either side."""
    ordered = np.sort(np.asarray(samples), axis=0)
    n = ordered.shape[0]
    centre = int(np.floor(q * (n - 1) + 0.5))
    spread = int(np.ceil(np.sqrt(n * q * (1.0 - q))))
    low, high = ordered[max(centre - spread, 0)], ordered[min(centre + spread, n - 1)]
    return np.maximum(ordered[centre], 0.0), (np.maximum(high, 0.0) - np.maximum(low, 0.0)) / 2.0


def _assert_within(name, ours, ore, error, floor):
    gap = np.abs(np.asarray(ours) - np.asarray(ore))
    bound = 4.0 * np.asarray(error) + floor
    assert np.all(gap <= bound), f"{name}: worst {np.max(gap / bound):.2f} of four combined standard errors"


@pytest.mark.slow
@pytest.mark.ore_parity(_declared(_trades(), pfe_quantiles=(PFE_QUANTILE,), samples=2 ** 12),
                        tolerance="EPE, ENE, PFE within four combined standard errors (independent paths)")
@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("names, paths", [
    (("swap-payer", "swap-receiver-seasoned-icma", "european-payer", "european-receiver-otm-cash", "bond"), 2 ** 14),
    (("bermudan-payer-physical", "bermudan-receiver-cash", "american-payer"), 2 ** 12),
], ids=["linear-and-europeans", "options"])
def test_the_engines_simulation_and_exposure_are_ores_in_distribution(model, names, paths):
    """L4: each simulated curve's mean discount factors, and each trade's and the netting set's
    EPE, ENE and PFE on every date, within four combined standard errors of ORE's; each run on
    its own random numbers. Options on fewer paths, for ORE's time: its grid engine on every path
    and date takes about two minutes per model at 4,096 paths (2026-10-13); the engine's took
    1,024 at most until its rollback became one shared operator per step (I-83)."""
    market = shared.market()
    trades = [shared.trades()[n] for n in names]
    config = _simulation(model, samples=paths, seed=11)
    ore = ore_xva_run(market, trades, config, PRICING, PFE_QUANTILE)
    model_ = build_cross_asset_model(market, config)
    scenarios = simulate(market, config, model_)
    result = price_portfolio(PortfolioRequest(market, trades, RunConfig(simulation=config, pricing=PRICING),
                                              pfe_quantiles=(PFE_QUANTILE,)))

    for curve, ours, theirs in (("discount", scenarios.discount["USD"], ore.discount),
                                ("index", scenarios.index[shared.INDEX], ore.index[shared.INDEX])):
        ours_mean, ours_error = _mean_and_error(np.exp(np.asarray(ours.log_discounts))[:, :, 1:])
        ore_mean, ore_error = _mean_and_error(theirs)
        _assert_within(f"{curve} curve", ours_mean, ore_mean, np.hypot(ours_error, ore_error), 1e-12)

    deflated = np.asarray(result.npv_cube) / np.asarray(scenarios.numeraire)[:, :, None]
    profiles = dict(zip(result.trade_ids, result.trade_exposures))
    profiles[NETTING_SET_PROFILE] = result.exposure
    for key, profile in profiles.items():
        ours = deflated.sum(axis=-1) if key == NETTING_SET_PROFILE else deflated[:, :, result.trade_ids.index(key)]
        theirs = ore.cube.sum(axis=-1) if key == NETTING_SET_PROFILE else ore.cube[:, :, ore.trade_ids.index(key)]
        expected = ore.exposure[key]
        np.testing.assert_allclose(np.asarray(profile.epe)[0], expected["EPE"][0], rtol=RTOL, err_msg=f"{key} t=0")
        floor = 1e-9 * max(np.max(np.abs(theirs)), 1.0)
        for field, column, sign in (("epe", "EPE", 1.0), ("ene", "ENE", -1.0)):
            _, ours_error = _mean_and_error(np.maximum(sign * ours, 0.0))
            _, ore_error = _mean_and_error(np.maximum(sign * theirs, 0.0))
            _assert_within(f"{key} {column}", np.asarray(getattr(profile, field))[1:], expected[column][1:],
                           np.hypot(ours_error, ore_error), floor)
        _, ours_error = _quantile_and_error(ours, PFE_QUANTILE)
        _, ore_error = _quantile_and_error(theirs, PFE_QUANTILE)
        _assert_within(f"{key} PFE", np.asarray(profile.pfe["PFE_95"])[1:], expected["PFE"][1:],
                       np.hypot(ours_error, ore_error), floor)


# --- the oracle is ORE ----------------------------------------------------------------------------

EXAMPLES = Path(__file__).resolve().parents[1] / "reference" / "ORE" / "Examples"


@pytest.mark.slow
def test_the_oracles_in_memory_run_is_ores_file_driven_run_of_an_example():
    """The oracle's way of running ORE (`tests.support.ore_inputs.run_app` over in-memory
    inputs) is ORE's: `Examples/Exposure`'s single swap (`ore_swapflat.xml`: a five-currency
    cross-asset model calibrated to swaptions, 242 monthly dates, 1000 paths) run both ways by
    the installed ORE gives the same exposure report, to the file's rounding. (The release's own
    `ExpectedOutput` for the example was written by an earlier build and differs by up to 30% in
    EPE; ORE's file-driven run is the reference.) About 30 s; needs the `reference/ORE` checkout,
    which CI does not fetch."""
    if not EXAMPLES.is_dir():
        pytest.skip("reference/ORE checkout not present")
    inputs, exposure = EXAMPLES / "Input", EXAMPLES / "Exposure" / "Input"
    read = lambda path: path.read_text(encoding="utf-8")  # noqa: E731
    lines = lambda path: [line.strip() for line in read(path).splitlines()  # noqa: E731
                          if line.strip() and not line.startswith("#")]
    parameters = ORE.InputParameters()
    parameters.setAsOfDate("2016-02-05")
    parameters.setBaseCurrency("EUR")
    parameters.setResultsPath(_scratch_dir())
    parameters.setAllFixings(True)
    parameters.setEntireMarket(True)
    parameters.setImplyTodaysFixings(True)
    parameters.setCurveConfigs(read(inputs / "curveconfig.xml"))
    parameters.setConventions(read(inputs / "conventions.xml"))
    parameters.setTodaysMarketParams(read(inputs / "todaysmarket.xml"))
    parameters.setPricingEngine(read(inputs / "pricingengine.xml"))
    parameters.setCalendarAdjustment(read(inputs / "calendaradjustment.xml"))
    parameters.setCurrencyConfig(read(inputs / "currencies.xml"))
    parameters.setPortfolio(read(exposure / "portfolio_swapflat.xml"))
    for context in ("lgmcalibration", "fxcalibration", "eqcalibration", "pricing", "simulation"):
        parameters.setMarketConfig("libor", context)
    simulation = read(exposure / "simulation_swapflat.xml")
    parameters.setExposureSimMarketParams(simulation)
    parameters.setScenarioGeneratorData(simulation)
    parameters.setCrossAssetModelData(simulation)
    parameters.setSimulationPricingEngine(read(inputs / "pricingengine.xml"))
    parameters.setExposureBaseCurrency("EUR")
    parameters.setXvaBaseCurrency("EUR")
    parameters.setExposureObservationModel("Disable")
    parameters.setNettingSetManager(read(exposure / "netting.xml"))
    parameters.setExposureProfiles(True)
    parameters.setExposureProfilesByTrade(True)
    parameters.insertAnalytic("EXPOSURE")
    parameters.insertAnalytic("PFE")
    in_memory = report_columns(run_app(parameters, ORE.Date(5, 2, 2016), lines(inputs / "market_20160205_flat.txt"),
                                       lines(inputs / "fixings_20160205.txt"), "exposure_trade_Swap_20"),
                               "exposure_trade_Swap_20")

    root = Path(tempfile.mkdtemp(prefix="ore_example_"))
    shutil.copytree(inputs, root / "Input")
    shutil.copytree(exposure, root / "Exposure" / "Input")
    (root / "Exposure" / "Output" / "swapflat").mkdir(parents=True)
    previous, cwd = ORE.Settings.instance().evaluationDate, os.getcwd()
    try:
        os.chdir(root / "Exposure")
        file_parameters = ORE.Parameters()
        file_parameters.fromFile("Input/ore_swapflat.xml")
        ORE.OREApp(file_parameters, False).run()
    finally:
        os.chdir(cwd)
        ORE.Settings.instance().evaluationDate = previous
    with open(root / "Exposure" / "Output" / "swapflat" / "exposure_trade_Swap_20.csv", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = [h.lstrip("#") for h in rows[0]]
    for column, rounding in (("Time", 1e-6), ("EPE", 1.0), ("ENE", 1.0), ("PFE", 1.0), ("BaselEE", 1.0),
                             ("BaselEEE", 1.0), ("TimeWeightedBaselEPE", 0.01), ("TimeWeightedBaselEEPE", 0.01)):
        written = np.asarray([float(r[header.index(column)]) for r in rows[1:]])
        assert len(written) == len(in_memory[column]) == 243
        np.testing.assert_allclose(in_memory[column], written, rtol=0, atol=rounding / 2 + 1e-9, err_msg=column)

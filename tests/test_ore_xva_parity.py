"""
The engine's assembled exposure pipeline against ORE's own exposure simulation (roadmap 3.2;
I-50; docs/planning/details/ore-parity-validation.md, layers L3 and L4, gate V-4), through
`tests/support/ore_xva_oracle.py`, on the shared portfolio's sloped two-curve market
(tests/support/portfolio.py), under both interest-rate models calibrated to a tenor basket:

  * the oracle is ORE: its in-memory run of ORE's `Examples/Exposure` swap equals ORE's own
    file-driven run of the example;
  * V-4: ORE's scenario dump is the engine's simulation market on ORE's paths. Each path's
    state, read from ORE's numeraire, rebuilds every discount and index curve ORE simulated
    (which also checks the calibration ORE's `CrossAssetModelBuilder` ran against the engine's);
  * L3: on ORE's paths every trade's NPV equals ORE's cube, cell by cell, and t=0 (ORE's `T0`)
    is the engine's value on the simulation market of the as-of date;
  * ORE's exposure definitions: the engine's profiles of ORE's own cube are ORE's reports;
  * L4 (statistical, `slow`): the engine's simulation and exposure against ORE's, each on its
    own random numbers (decision X-1), within four combined standard errors.

Measured (2026-10-07, ORE 1.8.16, both models): the dumped curves rebuild to 1.2e-11
relative; swaps, Europeans and the bond equal ORE's cube to 3.0e-11 of each trade's largest
value; T0 to 3.9e-11 relative; the profiles of ORE's cube to 1.8e-13. Bermudans and Americans
are worth 0.5-4% more than ORE says on paths after t=0, before their exercise (ORE's per-path
recalibration, I-49), and the same on a path after exercising into the swap; near the exercise
boundary that moves a decision (one cell in 160: a cash-settled Bermudan ORE exercised, 0, the
engine did not). Roadmap 3.5 reproduces ORE there against this test; until then their L3 cases
are strict expected failures.

Found by this test and fixed in roadmap 3.2: the simulation market held each curve's tenor
points at the tenors' times from the simulation date, where ORE holds them at the times from the
as-of date (`engine.simulation.scenario_market`), up to 0.3% of a swap's path values; and the
exposure started from today's value, where ORE starts from the simulation market's (`T0`).
"""
import csv
import os
import shutil
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.models.curves import ZeroCurve, discount
from engine.portfolio import PortfolioRequest, RunConfig, price_portfolio
from engine.risk.exposure import exposure_profile, netting_set_profile
from engine.simulation.config import CamConfig, HullWhiteConfig, LgmConfig, build_cross_asset_model, simulate
from engine.simulation.scenario_market import build_scenario_market
from engine.valuation.config import LgmSwaptionEngineConfig, PricingConfig
from engine.valuation.context import simulation_market_today
from engine.valuation.portfolio import trade_maturity, value_paths, value_today
from tests.support import portfolio as shared
from tests.support.ore_inputs import DISCOUNT_INDEX, report_columns, run_app, _scratch_dir
from tests.support.ore_xva_oracle import NETTING_SET_PROFILE, implied_states, ore_xva_run

MODELS = {"LGM": LgmConfig, "HullWhite": HullWhiteConfig}
BASKET = dict(calibration_expiries=("1Y", "2Y", "3Y"), calibration_terms=("5Y", "4Y", "3Y"))
#: Before and after the first exercises, the Europeans' expiries and the bond's coupons.
DATES = tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (3, 6, 12, 24, 36))
FAST = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)
PRICING = PricingConfig(bermudan=FAST, american=FAST)
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


@pytest.fixture(scope="module", params=list(MODELS))
def on_ores_paths(request):
    """ORE's run on 32 paths, and the engine's simulation market and cube on ORE's paths."""
    market, trades, config = shared.market(), _trades(), _simulation(request.param, samples=32)
    ore = ore_xva_run(market, trades, config, PRICING, PFE_QUANTILE)
    model = build_cross_asset_model(market, config)
    index_curves = {name: ("USD", ZeroCurve.from_config(curve))
                    for name, curve in market.currency("USD").index_curves.items()}
    scenarios = build_scenario_market(model, market.asof, config.dates,
                                      jnp.asarray(implied_states(model, ore.times, ore.numeraire)),
                                      config.curve_tenors, index_curves)
    columns = value_paths(trades, market, scenarios, "USD", PRICING, config.swaption_vol_decay)
    cube = np.stack([np.asarray(c) for c in columns], axis=-1) / ore.numeraire[:, :, None]
    return ore, scenarios, cube


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


@pytest.mark.parametrize("name", list(shared.trades()))
def test_every_trades_cube_equals_ores_on_ores_paths(on_ores_paths, name, request):
    """L3: each cell within `CUBE_RTOL` of the trade's largest value."""
    ore, _, cube = on_ores_paths
    if isinstance(shared.trades()[name], OPTIONS):
        request.applymarker(pytest.mark.xfail(strict=True, reason="I-49: ORE's per-path recalibration (roadmap 3.5)"))
    j = ore.trade_ids.index(name)
    scale = np.max(np.abs(ore.cube[:, :, j]))
    assert scale > 0.0, "the trade must be worth something on the paths"
    assert np.max(np.abs(cube[:, :, j] - ore.cube[:, :, j])) <= CUBE_RTOL * scale


def test_t0_is_the_value_on_the_simulation_market_of_the_as_of_date(on_ores_paths):
    """ORE's cube starts from each trade's value on the `ScenarioSimMarket` of the as-of date
    (the curves sampled at the simulation tenors), and so do the engine's profiles."""
    ore, _, _ = on_ores_paths
    market, config = shared.market(), _simulation("LGM", samples=32)
    start = value_today(_trades(), market, "USD", PRICING, simulation_market_today(market, config.curve_tenors))
    np.testing.assert_allclose(start, ore.t0, rtol=RTOL)
    assert not np.allclose(value_today(_trades(), market, "USD", PRICING), ore.t0, rtol=1e-6), \
        "today's market should differ from the simulation market's, or this test checks nothing"


def test_the_pipelines_profiles_start_from_ores_t0(on_ores_paths, request):
    """I-85: `price_portfolio`'s exposure at t=0 is ORE's, each trade's and the netting set's
    (until roadmap 3.2 it started from today's value, up to 4.2% away on this portfolio)."""
    ore, _, _ = on_ores_paths
    model = request.node.callspec.params["on_ores_paths"]
    result = price_portfolio(PortfolioRequest(shared.market(), _trades(),
                                              RunConfig(simulation=_simulation(model, samples=32), pricing=PRICING),
                                              pfe_quantiles=(PFE_QUANTILE,)))
    for profile, key in zip(result.trade_exposures + [result.exposure], list(ore.trade_ids) + [NETTING_SET_PROFILE]):
        for field, column in (("epe", "EPE"), ("ene", "ENE")):
            assert float(getattr(profile, field)[0]) == pytest.approx(ore.exposure[key][column][0], rel=RTOL, abs=1e-6)


def test_the_profiles_of_ores_cube_are_ores_reports(on_ores_paths):
    """ORE's `ExposureCalculator` definitions (`engine.risk.exposure`), on ORE's own cube and
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
@pytest.mark.parametrize("model", list(MODELS))
@pytest.mark.parametrize("names, paths", [
    (("swap-payer", "swap-receiver-seasoned-icma", "european-payer", "european-receiver-otm-cash", "bond"), 2 ** 14),
    (("bermudan-payer-physical", "bermudan-receiver-cash", "american-payer"), 2 ** 10),
], ids=["linear-and-europeans", "options"])
def test_the_engines_simulation_and_exposure_are_ores_in_distribution(model, names, paths):
    """L4: each simulated curve's mean discount factors, and each trade's and the netting set's
    EPE, ENE and PFE on every date, within four combined standard errors of ORE's; each run on
    its own random numbers. Options on fewer paths: their path rollback holds 0.3-0.7 MB per
    path and date (I-83)."""
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

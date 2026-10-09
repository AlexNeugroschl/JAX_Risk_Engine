"""
One component at a time: each section runs a single engine module's public API end to end
on the shared scenarios of demos/demo_scenarios.py, and prints what it returns.

    simulation     simulate the cross-asset model with the Hull-White model per currency
                   (`engine.market_simulation.config.simulate`)
    swap           a 2Y payer swap today and on every path (`engine.pricing.cube`)
    european       a 3Y-into-2Y European swaption, Bachelier (ORE's default) and Jamshidian
    bermudan       a Bermudan swaption today and on the paths (LGM grid engine, recalibrated on
                   every path and date as ORE does; 128 paths, as that is the slow part, I-53)
    american       an American swaption today and on the paths
    greeks         a swap's and a European's Greeks, by ORE's bump-and-revalue and by AD
    var_es         the risk statistics of a swap's NPV cube (`compute_risk_metrics`)

The whole pipeline (`price_portfolio`) and market-risk VaR/ES are in demos/demo.py.

Run with: .venv/Scripts/python.exe demos/demo_components.py [section ...]   (default: all)
"""
import sys

import numpy as np
import ORE

from demo_scenarios import EVAL_DATE, demo_market, demo_simulation
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.market_simulation.config import simulate
from engine.pricing.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.cube import value_portfolio
from engine.risk.greeks.ad import portfolio_greeks
from engine.risk.greeks.bump import portfolio_sensitivities
from engine.risk.market.var_es import compute_risk_metrics

#: A coarser grid than ORE's default for the Bermudan/American engine, to keep the demo quick.
FAST = PricingConfig(bermudan=LgmSwaptionEngineConfig(n_per_std=16), american=LgmSwaptionEngineConfig(n_per_std=16))


def _usd_paths(samples=1024):
    """The USD market and its Hull-White simulation on the demo grid."""
    market = demo_market(("USD",))
    return market, simulate(market, demo_simulation("HullWhite", samples=samples, currencies=("USD",)))


def _print_mean_npv(trade, pricing=PricingConfig(), samples=1024):
    market, scenarios = _usd_paths(samples)
    valuation = value_portfolio([trade], market, scenarios, "USD", pricing)
    print(f"Base (t=0) NPV: {valuation.today[0]:,.2f}")
    cube = np.asarray(valuation.cube[:, :, 0])
    for j, t in enumerate(scenarios.times):
        print(f"  t={t:.2f}: mean NPV across paths = {cube[:, j].mean():,.2f}")


def simulation():
    market = demo_market()
    scenarios = simulate(market, demo_simulation("HullWhite", samples=1024))
    print(f"Paths x dates: {scenarios.numeraire.shape}; times {np.round(scenarios.times, 3).tolist()}")
    print(f"Curves: discount {sorted(scenarios.discount)}, index {sorted(scenarios.index)}")
    print(f"FX: {sorted(scenarios.fx)}, equities: {sorted(scenarios.equity)}")
    usd = scenarios.discount["USD"]
    print("Path 0, first date, USD discount factors to 1Y/2Y/5Y/10Y:",
          [round(float(np.exp(usd.log_discounts[0, 0, k])), 4) for k in (3, 4, 7, 9)])
    print("E[1/N(t)] (today's discount factors to each date):",
          np.round(np.mean(1.0 / np.asarray(scenarios.numeraire), axis=0), 4).tolist())


def swap():
    _print_mean_npv(SwapConfig(notional=1_000_000.0, fixed_rate=0.034, payer=True, swap_tenor="2Y",
                               evaluation_date=EVAL_DATE, trade_id="swap"))


def european():
    """Physically settled: after expiry an exercised path carries the swap it entered."""
    trade = SwaptionConfig(notional=1_000_000.0, fixed_rate=0.040, payer=True, swap_tenor="2Y",
                           forward_start=ORE.Period(3, ORE.Years), evaluation_date=EVAL_DATE, trade_id="european")
    print("-- Bachelier on the market volatility (ORE's default) --")
    _print_mean_npv(trade)
    print("-- Jamshidian on Hull-White(0.03, 0.01) --")
    _print_mean_npv(trade, PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(0.03, 0.01)))


def bermudan():
    _print_mean_npv(BermudanSwaptionConfig(
        notional=1_000_000.0, fixed_rate=0.040, payer=True, swap_tenor="5Y", evaluation_date=EVAL_DATE,
        exercise_dates=[EVAL_DATE + ORE.Period(years, ORE.Years) for years in (1, 2, 3, 4)], trade_id="bermudan"),
        FAST, samples=128)


def american():
    trade = AmericanSwaptionConfig(notional=1_000_000.0, fixed_rate=0.040, payer=True, swap_tenor="5Y",
                                   first_exercise_date=EVAL_DATE + ORE.Period(1, ORE.Years),
                                   last_exercise_date=EVAL_DATE + ORE.Period(4, ORE.Years),
                                   evaluation_date=EVAL_DATE, trade_id="american")
    print("Exercise opportunities:", len(trade.option_times(FAST.american.exercise_time_steps_per_year)))
    _print_mean_npv(trade, FAST, samples=128)


def greeks():
    market = demo_market(("USD",))
    trades = [SwapConfig(notional=1_000_000.0, fixed_rate=0.040, payer=True, swap_tenor="5Y",
                         evaluation_date=EVAL_DATE, trade_id="swap"),
              SwaptionConfig(notional=1_000_000.0, fixed_rate=0.044, payer=True, swap_tenor="5Y",
                             forward_start=ORE.Period(3, ORE.Years), evaluation_date=EVAL_DATE, trade_id="european")]
    for method, compute in (("bump (ORE's sensitivity analysis, per curve tenor)", portfolio_sensitivities),
                            ("AD (per market-curve pillar)", portfolio_greeks)):
        print(f"--- {method} ---")
        for i, greeks in compute(trades, market, "USD").items():
            print(f"{trades[i].trade_id}:")
            for key, value in greeks.items():
                value = np.asarray(value)
                shown = round(float(value), 2) if value.ndim == 0 else np.round(value, 2).tolist()
                print(f"  {key}: {shown}")


def var_es():
    market, scenarios = _usd_paths()
    trade = SwapConfig(notional=1_000_000.0, fixed_rate=0.034, payer=True, swap_tenor="2Y",
                       evaluation_date=EVAL_DATE, trade_id="swap")
    valuation = value_portfolio([trade], market, scenarios, "USD")
    metrics = compute_risk_metrics(valuation.cube, valuation.today[0], percentiles=(0.95, 0.99))
    print("Base (t=0) NPV:", valuation.today[0])
    for key, values in metrics.items():
        print(f"{key}: {[round(float(v), 2) for v in values]}")


SECTIONS = {f.__name__: f for f in (simulation, swap, european, bermudan, american, greeks, var_es)}


if __name__ == "__main__":
    names = sys.argv[1:] or list(SECTIONS)
    unknown = [n for n in names if n not in SECTIONS]
    if unknown:
        sys.exit(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")
    for name in names:
        print(f"\n===== {name} =====")
        SECTIONS[name]()

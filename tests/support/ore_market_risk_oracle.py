"""
ORE's own market-risk analytic, run in-process: its historical simulation VaR (`HISTSIM_VAR`,
`HistoricalSimulationVarAnalytic`, OREAnalytics/orea/app/analytics/varanalytic.cpp) of a
portfolio over a history of the engine's risk factors, end to end: ORE's historical scenario
generator, its simulation market, its historical P&L generator (`HistoricalPnlGenerator`, the
valuation engine with the trades' model builders, so every calibrated option is recalibrated
under every scenario) and its VaR report. The reference market risk is validated against
(tests/test_ore_parity_sweep.py; I-116).

Test tooling, not a pricer: the engine never imports it. The inputs are
`tests.support.ore_inputs.portfolio_inputs`' (the engine's market, trades and engines); what this
module adds:

  * **The simulation market** (`setHistVarSimMarketParams`): every curve at a tenor of one day
    and at each pillar of the engine's curves after t=0, linear in the zero rate, flat zero
    beyond the last (`LinearZero`, `FlatZero`), the swaption volatilities not simulated (held,
    as the engine holds them). Between its first tenor and its last pillar ORE's curve is then
    the engine's `ZeroCurve` itself (linear in the zero rate between pillars, and the one-day
    tenor on the first segment's line, so a curve sloped there reaches ORE unchanged); ORE's
    curve is flat in the first day, where no flow of a trade valued today lies. Beyond the last
    pillar the two extrapolate differently (ORE flat zero, the engine flat forward): the
    histories' curves reach beyond every flow.
  * **The scenarios** (`setScenarioReader`, a scenario file): on each history date each
    simulated curve's discount factors at the tenors, `exp(-z t)` with `z` the date's zero rate
    at the tenor's time, linear in the zero rate between the engine's pillars. ORE's return of a
    discount factor is logarithmic (`ReturnConfiguration`'s default), applied to its base
    scenario (`baseValue * exp(log(v2 / v1))`), so a scenario's zero-rate move at each tenor is
    the history's `z(d2) - z(d1)`: the engine's absolute pillar shift
    (`engine.risk.market.historical_scenarios`). The engine's one discount curve is ORE's two
    factors (`DiscountCurve/<ccy>` and the swap index's discounting `IndexCurve`, as in the
    sensitivity oracle), each given the same discount factors.
  * **The windows**: each history date `d1` and `d2 = TARGET.advance(d1, horizon business
    days)` (`HistoricalScenarioGenerator::setDates`, overlapping), so a history on consecutive
    TARGET business days gives the engine's overlapping moves `history[i + h] - history[i]`,
    in order. ORE's benchmark period names the windows' end dates.
  * **What is read**: two runs, since ORE writes the VaR report only without the trade-level
    P&L. The trade-level run's `historical_PnL` report (`tradePnl`) gives each trade's P&L per
    window; the portfolio run's gives the portfolio P&L and the `var` report, ORE's
    `Quantile_<p>` (the P&L's quantile at tail probability `p`: boost's right tail quantile of
    the P&L, `HistoricalSimulationVarCalculator::var`, NaN where fewer than `1 / p` scenarios
    leave no scenario in the tail, the report then without a row if no figure is defined) and
    `ExpectedShortfall_<p>` (the mean of the P&L at or below that quantile).

**Two ORE estimators of a tail.** ORE's `RiskStatistics`, whose conventions the engine's VaR
and ES follow (`engine.risk.market.var_es`), and this report agree on the VaR scenario but not
on the expected shortfall: `RiskStatistics::expectedShortfall` averages the P&L strictly below
minus the VaR, the report the P&L at or below the quantile, the VaR scenario included
(historicalsimulationvar.cpp). The engine's ES is `RiskStatistics`' (decision X-12); the VaR
scenario is the same `floor(N p) + 1`-th smallest P&L in both unless `N p` is a whole number:
the report then reads the `(N p + 1)`-th, `RiskStatistics` its `ceil(N (1 - c))`-th, the one
below where `1 - c` rounds low in binary (c = 0.8, 0.9; measured 2026-10-18). The report's
quantile is the P&L itself, a gain included, where `RiskStatistics`' VaR floors it at zero.

One currency, one Ibor index; what the oracle cannot express is refused by name.
"""
from __future__ import annotations

import gc
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import ORE

from engine.market_data.market import Market
from engine.pricing.config import PricingConfig
from engine.risk.market import RateRiskFactors
from tests.support.ore_inputs import (
    CCY, DISCOUNT_INDEX, curve_dates, iso, portfolio_inputs, report_columns, sim_market_xml,
)

#: The calendar of the history and of ORE's windows (`mporCalendar`).
CALENDAR = ORE.TARGET()


@dataclass(frozen=True)
class OreMarketRiskRun:
    """One ORE historical simulation VaR run (see the module docstring). Trade axes follow
    `trade_ids`, the request's order; scenario axes the windows, oldest first."""
    trade_ids: Tuple[str, ...]
    windows: Tuple[Tuple[ORE.Date, ORE.Date], ...]
    pnl: np.ndarray               # [S, N] each trade's P&L
    portfolio_pnl: np.ndarray     # [S]
    quantile: Dict[float, float]  # tail probability p -> ORE's Quantile_p of the portfolio P&L
    expected_shortfall: Dict[float, float]  # p -> ORE's ExpectedShortfall_p


def business_days_before(asof: ORE.Date, count: int) -> list:
    """`count` consecutive TARGET business days, the last before `asof`: a history's dates."""
    dates = [CALENDAR.advance(asof, ORE.Period(-1, ORE.Days))]
    while len(dates) < count:
        dates.insert(0, CALENDAR.advance(dates[0], ORE.Period(-1, ORE.Days)))
    return dates


def ore_market_risk_run(market: Market, trades: Sequence, history, dates: Sequence[ORE.Date], horizon_days: int,
                        tail_probabilities: Sequence[float], pricing: PricingConfig = PricingConfig()
                        ) -> OreMarketRiskRun:
    """ORE's historical simulation VaR of `trades` on `market` (see the module docstring).

    history: `[D, F]` the factors' levels on `dates`, in `RateRiskFactors.from_market(market)`'s
        vector order (the engine's), oldest first.
    dates: `D` consecutive TARGET business days.
    tail_probabilities: ORE's quantiles, e.g. 0.01 for a 99% VaR.

    Raises `RuntimeError` with ORE's messages if a trade or the market fails to build."""
    factors = RateRiskFactors.from_market(market)
    levels = np.asarray(history, dtype=np.float64)
    if levels.shape != (len(dates), factors.size):
        raise ValueError(f"history must be [{len(dates)}, {factors.size}]; got {levels.shape}")
    for previous, date in zip(dates, dates[1:]):
        if CALENDAR.advance(previous, ORE.Period(1, ORE.Days)) != date:
            raise ValueError(f"the history's dates must be consecutive TARGET business days; {iso(previous)} is "
                             f"followed by {iso(date)}")
    by_run = [_run(market, trades, levels, dates, horizon_days, tail_probabilities, pricing, trade_pnl)
              for trade_pnl in (True, False)]
    trade_run, portfolio_run = by_run
    pnl_rows = _rows(trade_run["historical_PnL"])
    windows = tuple(dict.fromkeys((d1, d2) for d1, d2, _, _ in pnl_rows))
    by_trade = {(d1, d2, trade): value for d1, d2, trade, value in pnl_rows}
    ids = tuple(t.trade_id for t in trades)
    pnl = np.asarray([[by_trade[(d1, d2, trade)] for trade in ids] for d1, d2 in windows])
    portfolio = _rows(portfolio_run["historical_PnL"])
    if [(d1, d2) for d1, d2, _, _ in portfolio] != list(windows):
        raise RuntimeError("ORE's trade-level and portfolio runs have different windows")
    var = portfolio_run["var"]
    row = [i for i, (c, t) in enumerate(zip(var["RiskClass"], var["RiskType"])) if (c, t) == ("All", "All")]
    if len(row) > 1:
        raise RuntimeError(f"ORE's var report has {len(row)} rows for all risk classes and types")
    # ORE writes no row when no requested figure is defined (each tail holding under one scenario).
    quantiles, shortfalls = {}, {}
    for column, values in var.items():
        for prefix, table in (("Quantile_", quantiles), ("ExpectedShortfall_", shortfalls)):
            if column.startswith(prefix):
                table[_probability(column[len(prefix):], tail_probabilities)] = \
                    float(values[row[0]]) if row else float("nan")
    return OreMarketRiskRun(ids, windows, pnl, np.asarray([v for _, _, _, v in portfolio]), quantiles, shortfalls)


def _probability(label: str, probabilities: Sequence[float]) -> float:
    """The requested probability ORE's column label (`std::to_string`, six decimals) names."""
    matches = [p for p in probabilities if abs(p - float(label)) < 5e-7]
    if len(matches) != 1:
        raise RuntimeError(f"ORE's column {label!r} names none of the requested probabilities {list(probabilities)}")
    return matches[0]


def _rows(report) -> list:
    """`(PLDate1, PLDate2, Portfolio, PLAmount)` of the report's rows over all risk classes and
    types (ORE repeats each window per risk group), in ORE's order."""
    return [(d1, d2, name, float(value)) for d1, d2, name, value, risk_class, risk_type in
            zip(report["PLDate1"], report["PLDate2"], report["Portfolio"], report["PLAmount"], report["RiskClass"],
                report["RiskType"]) if (risk_class, risk_type) == ("All", "All")]


def _run(market: Market, trades, levels: np.ndarray, dates, horizon_days: int, tail_probabilities, pricing,
         trade_pnl: bool, *, include_npv: bool = False) -> dict:
    run = portfolio_inputs(market, trades, pricing, "the market-risk oracle")
    factors = RateRiskFactors.from_market(market)
    keys = {f"discount:{CCY}": (f"DiscountCurve/{CCY}", f"IndexCurve/{DISCOUNT_INDEX}"),
            f"index:{run.index}": (f"IndexCurve/{run.index}",)}
    pillars = np.asarray(run.curves.times, dtype=np.float64)
    curve_dates(market.asof, pillars)  # whole days from the as-of date, starting at 0
    key_days = [1] + [round(t * 365) for t in pillars[1:]]
    key_times = np.asarray(key_days, dtype=np.float64) / 365.0
    header = ["Date", "Scenario", "Numeraire"] + [f"{key}/{k}" for name in factors.names for key in keys[name]
                                                  for k in range(len(key_days))]
    lines = [",".join(header)]
    for date, row in zip(dates, levels):
        cells = [iso(date), "1", "1"]
        for i, name in enumerate(factors.names):
            zero = np.interp(key_times, pillars, row[factors.slice_of(i)])
            cells += [repr(float(v)) for _ in keys[name] for v in np.exp(-zero * key_times)]
        lines.append(",".join(cells))
    parameters = run.parameters
    market_xml = sim_market_xml([f"{d}D" for d in key_days], (run.index, DISCOUNT_INDEX), run.vols,
                                simulate_vols=False, interpolation="LinearZero", extrapolation="FlatZero")
    parameters.setHistVarSimMarketParams(f"<Simulation>{market_xml}</Simulation>")
    parameters.setBenchmarkVarPeriod(f"{iso(dates[horizon_days])},{iso(dates[-1])}")
    parameters.setMporDays(horizon_days)
    parameters.setMporCalendar("TARGET")
    parameters.setMporOverlappingPeriods(True)
    parameters.setVarQuantiles(",".join(repr(float(p)) for p in tail_probabilities))
    parameters.setIncludeExpectedShortfall(True)
    parameters.setTradePnl(trade_pnl)
    parameters.insertAnalytic("HISTSIM_VAR")
    if include_npv:
        parameters.insertAnalytic("NPV")
    # The scenarios as a file ORE reads. Given the text itself, ORE keeps it as its `scenarioFile`
    # parameter, whose path the analytic then asks the filesystem about: on Linux that fails on so
    # long a name ("File name too long"). ORE holds the file open while its reader lives (on
    # Windows the directory cannot be removed then), so its objects go before the directory.
    failure = None
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scenarios.csv"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        parameters.setScenarioReader(str(path))
        app = None
        try:
            app = run.run("historical_PnL" if trade_pnl else "var")
            names = ("historical_PnL",) + (() if trade_pnl else ("var",)) + (("npv",) if include_npv else ())
            reports = {name: report_columns(app, name) for name in names}
        except BaseException as error:
            failure = error
            # A failed run's traceback owns OREApp/InputParameters too. Keep its diagnostic
            # stack, but release completed frames' locals before removing the scenario file.
            pending, seen = [error], set()
            while pending:
                current = pending.pop()
                if current is not None and id(current) not in seen:
                    seen.add(id(current))
                    traceback.clear_frames(current.__traceback__)
                    pending.extend((current.__cause__, current.__context__))
        finally:
            del app, parameters, run
            gc.collect()
    if failure is not None:
        raise failure
    return reports

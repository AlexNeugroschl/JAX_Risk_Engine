"""
ORE's t=0 values of a portfolio, run in-process: the reference the engine's today's values are
validated against beyond the shared portfolio (tests/test_ore_parity_sweep.py; I-116).

Test tooling, not a pricer: the engine never imports it. One `OREApp` run of ORE's `NPV`
analytic over in-memory inputs (`tests.support.ore_inputs.portfolio_inputs`): the engine's
market, its trades as ORE's trade XML on the engine's own schedules, and ORE's engines as the
engine's `PricingConfig` configures them (`DiscountingSwapEngine`, the Bachelier
`BlackMultiLegOptionEngine`, the LGM grid engine with its calibration, `DiscountingRiskyBondEngine`).
Each trade is ORE's trade as booked, its position and settlement included: a short option's
value is ORE's `OptionWrapper`'s, and a cash-settled Bermudan's ORE's approximation of it by the
physical one (swaption.cpp, as tests/support/portfolio.py says). What the inputs cannot express
is refused by name.

ORE builds each trade on its own (`BuildFailedTrades` off): a trade that fails to build (a
calibration missing its basket, an American without a calibrating swaption) is left out of the
report and its error logged, the other trades priced. `OreNpvRun.refused` holds each such trade
with ORE's message, the reference the engine's refusals are checked against.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Sequence

from engine.market_data.market import Market
from engine.pricing.config import PricingConfig
from tests.support.ore_inputs import log_lines, portfolio_inputs, report_columns

#: ORE's structured log message of a trade that failed to build.
_BUILD_ERROR = re.compile(r"StructuredMessage (\{.*\})")


@dataclass(frozen=True)
class OreNpvRun:
    """ORE's NPV of every trade it built, by trade id, in the trade's currency; and each trade it
    refused to build, with ORE's message."""
    npv: Dict[str, float]
    refused: Dict[str, str]


def ore_npv_run(market: Market, trades: Sequence, pricing: PricingConfig = PricingConfig()) -> OreNpvRun:
    """ORE's NPVs of `trades` on `market` (see the module docstring). Raises `RuntimeError` with
    ORE's messages if the market fails to build or no trade builds."""
    run = portfolio_inputs(market, trades, pricing, "the NPV oracle")
    run.parameters.insertAnalytic("NPV")
    report = report_columns(run.run("npv"), "npv")
    npvs = dict(zip(report["TradeId"], (float(v) for v in report["NPV"])))
    refused = {}
    for line in log_lines():
        match = _BUILD_ERROR.search(line)
        if not match:
            continue
        message = json.loads(match.group(1))
        fields = {f["name"]: f["value"] for f in message.get("sub_fields", [])}
        if message.get("group") == "Trade" and fields.get("tradeId") in {t.trade_id for t in trades}:
            refused[fields["tradeId"]] = message["message"]
    missing = [t.trade_id for t in trades if t.trade_id not in npvs and t.trade_id not in refused]
    if missing:
        raise RuntimeError(f"ORE's NPV report has no row for {missing}, and no build error says why")
    return OreNpvRun(npvs, refused)

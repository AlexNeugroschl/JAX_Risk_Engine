"""
One valuation date's market as the single-date pricers read it: the date, a discount curve per
currency and a forwarding curve per index (either curve type, measured from the date), the
normal swaption volatility each option reads, and fixings known on that date beyond the
trades' own history.

`from_market` is today's market. `simulation_market_today` is ORE's simulation market on the
as-of date, every curve sampled at the simulation tenors: where ORE's exposure starts
(`engine.run.pipeline`) and its sensitivity analysis bumps
(`engine.risk.greeks.bump`, whose Theta market is the same curves on a later date), so one set
of pricers values them all.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Sequence

import jax.numpy as jnp
import ORE

from engine.market_data.curves import DiscountCurve, ZeroCurve, log_discount
from engine.market_data.day_counts import TIME_AXIS_DAY_COUNTER
from engine.market_data.market import Market

#: (currency, option time from the date, swap length) -> normal volatility.
VolatilityLookup = Callable[[str, float, float], float]


@dataclass(frozen=True)
class PricingContext:
    """See the module docstring. `fixings[index name]` adds to each trade's history (the
    theta roll backfills the fixings of the day it skips)."""
    date: ORE.Date
    discount: Mapping[str, object]
    index: Mapping[str, object]
    volatility: VolatilityLookup
    fixings: Mapping[str, Mapping[ORE.Date, float]] = field(default_factory=dict)

    def curves(self, currency: str, index_name: str):
        if currency not in self.discount:
            raise KeyError(f"no discount curve for {currency}")
        if index_name not in self.index:
            raise KeyError(f"no forwarding curve for index {index_name!r} in {currency}")
        return self.discount[currency], self.index[index_name]


def from_market(market: Market) -> PricingContext:
    """Today's market: the zero curves as given, the vol surfaces read as ORE reads them on
    the as-of date."""
    discount: Dict[str, ZeroCurve] = {}
    index: Dict[str, ZeroCurve] = {}
    for code, data in market.currencies.items():
        discount[code] = ZeroCurve.from_config(data.discount_curve)
        index.update({name: ZeroCurve.from_config(curve) for name, curve in data.index_curves.items()})

    def volatility(currency: str, option_time: float, swap_length: float) -> float:
        return float(market.swaption_vols(currency).volatility(market.asof, option_time, swap_length))

    return PricingContext(date=market.asof, discount=discount, index=index, volatility=volatility)


def sampled_curve(curve: ZeroCurve, asof: ORE.Date, tenors: Sequence[str]) -> DiscountCurve:
    """ORE's simulation-market curve (`ScenarioSimMarket::addYieldCurve`): points at the times of
    `asof + tenor` from the as-of date, carrying the as-of curve's discount factors there, and 1
    at t=0; log-linear between them, flat forward beyond. ORE holds the points at these times as
    its reference date moves (on a simulated path, I-84, and on the sensitivity analysis's Theta
    date, `engine.risk.greeks.bump`)."""
    dates = [asof + ORE.Period(t) for t in tenors]
    times = [0.0] + [TIME_AXIS_DAY_COUNTER.yearFraction(asof, d) for d in dates]
    values = [0.0] + [float(log_discount(curve, t)) for t in times[1:]]
    return DiscountCurve(times=jnp.asarray(times), log_discounts=jnp.asarray(values))


def simulation_market_today(market: Market, tenors: Sequence[str]) -> PricingContext:
    """ORE's simulation market on the as-of date: every discount and index curve sampled at
    `tenors` (`sampled_curve`), the swaption volatilities as given. ORE's exposure simulation
    values every trade here at t=0 (the NPV cube's `T0`) and its sensitivity analysis bumps it,
    so a trade's value here differs from its value on today's market (`from_market`) by the
    curves' interpolation between the tenors."""
    discount, index = {}, {}
    for code, data in market.currencies.items():
        discount[code] = sampled_curve(ZeroCurve.from_config(data.discount_curve), market.asof, tenors)
        index.update({name: sampled_curve(ZeroCurve.from_config(c), market.asof, tenors)
                      for name, c in data.index_curves.items()})
    return PricingContext(market.asof, discount, index, from_market(market).volatility)

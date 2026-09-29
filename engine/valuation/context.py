"""
One valuation date's market as the single-date pricers read it: the date, a discount curve per
currency and a forwarding curve per index (either curve type, measured from the date), the
normal swaption volatility each option reads, and fixings known on that date beyond the
trades' own history.

`from_market` is today's market. The sensitivity analysis (`engine.risk.sensitivities`) builds
bumped and rolled contexts the same way, so one set of pricers values them all.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping

import ORE

from engine.market import Market
from engine.models.curves import ZeroCurve

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

"""
Helpers shared by the Greeks tests (tests/test_greeks.py, tests/test_greeks_bermudan.py): the
shared market with one curve pillar moved, and a comparison scaled to the compared magnitude.
"""
import dataclasses

import numpy as np

from engine.market_data.market import Market, ZeroCurveConfig
from tests.support import portfolio as shared


def bumped_market(kind: str, k: int, h: float, market: Market = None) -> Market:
    """`market` with pillar `k` of the USD discount (`kind="discount"`) or 6M index curve
    moved by `h` in zero rate."""
    market = market or shared.market()
    usd = market.currency("USD")

    def moved(curve: ZeroCurveConfig) -> ZeroCurveConfig:
        rates = list(curve.rates)
        rates[k] += h
        return ZeroCurveConfig(list(curve.times), rates)

    if kind == "discount":
        usd = dataclasses.replace(usd, discount_curve=moved(usd.discount_curve))
    else:
        usd = dataclasses.replace(usd, index_curves={shared.INDEX: moved(usd.index_curves[shared.INDEX])})
    return dataclasses.replace(market, currencies={**market.currencies, "USD": usd})


def assert_close(actual, expected, rtol):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=rtol * float(np.max(np.abs(expected))))

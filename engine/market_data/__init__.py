"""
`engine.market_data`: today's market, the t=0 inputs every valuation starts from.

    market.py      Market, CurrencyMarket, ZeroCurveConfig, the swaption volatility matrix
    curves.py      the curve primitives in JAX: ZeroCurve (today's), DiscountCurve (a path's)
    day_counts.py  the simulation's time axis (ACT/365) and the accrual day counts a trade
                   may name

Nothing is imported here, so `engine.market_data.day_counts` (ORE only) can be imported
without JAX: the TraderX path depends on it (I-05).
"""

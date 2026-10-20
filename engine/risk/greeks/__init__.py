"""
`engine.risk.greeks`: the Greeks of each trade on today's market.

    bump.py             ORE's sensitivity analysis: bump and revalue, recalibrating the
                        option models under every bump (the default method)
    ad.py               automatic differentiation of each trade's price function on the same
                        sensitivity market: ORE's Greek as the shift vanishes
    price_functions.py  each trade's t=0 price as a pure JAX function of its market (its
                        curves' node values and its volatility quotes), with its configured
                        engine and a calibrated option's calibration inside it (also what
                        `engine.risk.market` revalues)
    scenario_pnl.py     a trade's P&L under scenarios of that market, in batches: the bump
                        Greeks' shifts and market risk's shocks, one revaluation
"""

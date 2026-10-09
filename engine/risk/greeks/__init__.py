"""
`engine.risk.greeks`: the Greeks of each trade on today's market.

    bump.py             ORE's sensitivity analysis: bump and revalue, recalibrating the
                        option models under every bump (the default method)
    ad.py               automatic differentiation of each trade's price function
    price_functions.py  each trade's t=0 price as a pure JAX function of its curves' pillar
                        rates, with its configured engine (also what `engine.risk.market`
                        revalues)
"""

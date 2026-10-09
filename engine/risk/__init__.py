"""
`engine.risk`: the risk figures of a portfolio, one subpackage per kind.

    greeks/        sensitivities to today's market: ORE's bump-and-revalue (`bump.py`) and
                   automatic differentiation (`ad.py`) of the price functions both use
    counterparty/  exposure profiles over the simulation's paths (EPE, ENE, PFE)
    market/        VaR and ES by full revaluation under shocked curves, and the VaR/ES
                   estimators

No module sits here but this one (tests/test_import_layering.py).
"""

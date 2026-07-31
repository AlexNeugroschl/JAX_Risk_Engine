"""
Canonical demo/reference scenarios and shared ORE curve-construction
helpers, factored out of the individual modules' __main__ demo blocks
(where the same configuration used to be hand-copied and had already begun
drifting out of sync between engine/instruments/interest_rate_swap.py and
engine/aggregate_statistics/risk_statistics.py).

This module has no engine.instruments/engine.aggregate_statistics
dependency of its own -- it only builds SimulationConfig objects and plain
ORE curve handles, so every downstream module (and the test suite) can
import from here without a circular dependency.
"""
import jax.numpy as jnp
import numpy as np
import ORE

from engine.market_simulations import EquityConfig, RatesConfig, SimulationConfig, ZeroCurveConfig

# Shared evaluation date for every demo/test scenario in this module, so a
# single change here keeps every dependent cashflow schedule consistent.
EVAL_DATE = ORE.Date(30, 7, 2026)

# Union of both legs' accrual/payment dates for the single_currency_swap_config
# scenario below (2Y, semi-annual float / annual fixed, starting at the
# standard 2-day spot lag) -- required maturity pillars for any yield curve
# cube meant to price that swap (see engine.instruments.interest_rate_swap's
# maturity-pillar-alignment requirement).
SWAP_DEMO_MATURITIES = [
    0.010958904109589041, 0.5150684931506849, 1.010958904109589,
    1.515068493150685, 2.0136986301369864,
]


def cross_asset_demo_config() -> SimulationConfig:
    """
    Two-equity (AAPL, EUR/USD), two-rate-factor (USD, EUR) cross-asset
    scenario -- illustrates the full market simulation surface (equities,
    FX via UIP, multiple correlated rate factors, 4 output maturity
    pillars). Used by engine/market_simulations.py's own demo.
    """
    return SimulationConfig(
        time_grid=[0.0, 0.25, 0.50, 0.75, 1.0],
        scenarios=4096,
        equities=EquityConfig(
            initial_prices=[150.0, 1.10],       # AAPL, EUR/USD
            dividend_yields=[0.01, 0.00],
            rate_mapping=[
                [1.0, 0.0],                     # AAPL relies purely on USD rate (index 0)
                [1.0, -1.0],                    # EUR/USD relies on USD - EUR
            ],
        ),
        rates=RatesConfig(
            initial_rates=[0.03, 0.02],         # USD SOFR, EURIBOR
            theta=[0.03, 0.02],
            mean_reversion=[0.1, 0.15],
            maturities=[1.0, 2.0, 5.0, 10.0],   # Output curves out to 10Y
            # One curve per rate factor (USD, then EUR), each consistent
            # with that factor's own initial_rates/theta above -- matches
            # ORE's Cross-Asset Model, where every currency's Hull-White
            # process is calibrated against its own curve, never a shared one.
            initial_zero_curves=[
                ZeroCurveConfig(
                    times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
                    rates=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03],
                ),
                ZeroCurveConfig(
                    times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
                    rates=[0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
                ),
            ],
        ),
        joint_covariance=[
            [0.0400, 0.0000, 0.0010, 0.0005],   # AAPL
            [0.0000, 0.0100, 0.0002, -0.0001],  # EUR/USD
            [0.0010, 0.0002, 0.0001, 0.00008],  # USD SOFR
            [0.0005, -0.0001, 0.00008, 0.0002], # EURIBOR
        ],
    )


def single_currency_swap_demo_config() -> SimulationConfig:
    """
    One equity, two correlated USD rate factors (0 = OIS/discounting,
    1 = Euribor-style forwarding) -- the minimal multi-curve scenario used
    by engine/instruments/interest_rate_swap.py and
    engine/aggregate_statistics/risk_statistics.py's demos, and by the
    corresponding test suites' ORE cross-checks. Maturity pillars are
    SWAP_DEMO_MATURITIES: the exact accrual/payment dates of the 2Y demo
    swap built by swap_demo_config().
    """
    return SimulationConfig(
        time_grid=[0.0, 0.5, 1.0, 1.5, 2.0],
        scenarios=4096,
        equities=EquityConfig(
            initial_prices=[150.0],
            dividend_yields=[0.0],
            rate_mapping=[[1.0, 0.0]],
        ),
        rates=RatesConfig(
            initial_rates=[0.030, 0.035],
            theta=[0.030, 0.035],
            mean_reversion=[0.03, 0.03],
            maturities=SWAP_DEMO_MATURITIES,
            # Factor 0 = OIS/discounting curve, factor 1 = Euribor-style
            # forwarding curve -- genuinely distinct curves even within one
            # currency, each consistent with its own initial_rates above.
            initial_zero_curves=[
                ZeroCurveConfig(
                    times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
                    rates=[0.030, 0.030, 0.030, 0.030, 0.030, 0.030],
                ),
                ZeroCurveConfig(
                    times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0],
                    rates=[0.035, 0.035, 0.035, 0.035, 0.035, 0.035],
                ),
            ],
        ),
        joint_covariance=[
            [0.0400, 0.0000, 0.0000],
            [0.0000, 0.0001, 0.00005],
            [0.0000, 0.00005, 0.0001],
        ],
    )


def flat_yield_curves(disc_rate: float, fwd_rate: float, maturities=SWAP_DEMO_MATURITIES,
                       eval_date: ORE.Date = EVAL_DATE):
    """
    A [1, 1, len(maturities), 2] deterministic (zero-shock) yield curve
    cube from two flat ORE.FlatForward curves -- the standard "today's
    actual market, no simulated noise" base case used for VaR's t=0
    baseline and for cross-checking a pricer's output directly against
    ORE.VanillaSwap.NPV() (see tests/test_interest_rate_swap.py and
    tests/test_risk_statistics.py).
    """
    dc = ORE.Actual365Fixed()
    disc_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(eval_date, disc_rate, dc))
    fwd_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(eval_date, fwd_rate, dc))
    disc = np.array([disc_curve.discount(eval_date + int(round(t * 365))) for t in maturities])
    fwd = np.array([fwd_curve.discount(eval_date + int(round(t * 365))) for t in maturities])
    cube = np.stack([disc, fwd], axis=-1)
    return jnp.asarray(cube[None, None, :, :], dtype=jnp.float64)

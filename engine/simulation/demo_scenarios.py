"""
Shared demo and test scenarios, and an ORE flat-curve helper. Depends only on
`engine.simulation.market_model` and ORE, so anything can import it.
"""
import jax.numpy as jnp
import numpy as np
import ORE

from engine.simulation.market_model import EquityConfig, RatesConfig, SimulationConfig, ZeroCurveConfig

# Evaluation date of every scenario here.
EVAL_DATE = ORE.Date(30, 7, 2026)

# Accrual and payment times of the 2Y demo swap (spot start, semi-annual), which the cube's
# pillars must include (see engine.instruments.swap's pillar-alignment requirement).
SWAP_DEMO_MATURITIES = [
    0.010958904109589041, 0.5150684931506849, 1.010958904109589,
    1.515068493150685, 2.0136986301369864,
]


def cross_asset_demo_config() -> SimulationConfig:
    """
    Two equities/FX (AAPL, EUR/USD) and two rate factors (USD, EUR), with four output
    maturities. Used by market_model's demo.
    """
    return SimulationConfig(
        time_grid=[0.0, 0.25, 0.50, 0.75, 1.0],
        scenarios=4096,
        equities=EquityConfig(
            initial_prices=[150.0, 1.10],       # AAPL, EUR/USD
            dividend_yields=[0.01, 0.00],
            rate_mapping=[
                [1.0, 0.0],                     # AAPL drifts at the USD rate (factor 0)
                [1.0, -1.0],                    # EUR/USD drifts at USD - EUR
            ],
        ),
        rates=RatesConfig(
            initial_rates=[0.03, 0.02],         # USD SOFR, EURIBOR
            theta=[0.03, 0.02],
            mean_reversion=[0.1, 0.15],
            maturities=[1.0, 2.0, 5.0, 10.0],   # Output curves out to 10Y
            # One curve per rate factor, each at that factor's initial_rates/theta level.
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
    One equity and two USD rate factors (0 = discounting, 1 = forwarding): the minimal
    multi-curve scenario of the swap and var_es demos and tests. Maturity pillars are
    SWAP_DEMO_MATURITIES.
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
            # Factor 0 discounts, factor 1 forwards; each curve is flat at its
            # initial_rates level.
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


def swaption_demo_config() -> SimulationConfig:
    """
    One USD rate factor at 3%, stepped to 5Y, without a yield-curve cube (the swaption
    pricers read the short-rate paths directly). The one equity is a placeholder the
    simulation requires. The 5Y grid puts steps both before and after the demo swaption's
    exercise date.
    """
    return SimulationConfig(
        time_grid=[0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
        scenarios=4096,
        equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
        rates=RatesConfig(
            initial_rates=[0.03],
            theta=[0.03],
            mean_reversion=[0.03],
        ),
        joint_covariance=[
            [0.0400, 0.0000],
            [0.0000, 0.0001],
        ],
    )


def flat_yield_curves(disc_rate: float, fwd_rate: float, maturities=SWAP_DEMO_MATURITIES,
                       eval_date: ORE.Date = EVAL_DATE):
    """
    `[1, 1, len(maturities), 2]` deterministic yield-curve cube from two flat
    `ORE.FlatForward` curves (discount, forward): today's market with no simulation, for
    t=0 baselines and checks against ORE (tests/test_swap.py, tests/test_var_es.py).
    """
    dc = ORE.Actual365Fixed()
    disc_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(eval_date, disc_rate, dc))
    fwd_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(eval_date, fwd_rate, dc))
    disc = np.array([disc_curve.discount(eval_date + int(round(t * 365))) for t in maturities])
    fwd = np.array([fwd_curve.discount(eval_date + int(round(t * 365))) for t in maturities])
    cube = np.stack([disc, fwd], axis=-1)
    return jnp.asarray(cube[None, None, :, :], dtype=jnp.float64)

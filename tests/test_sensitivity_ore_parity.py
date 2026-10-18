"""
ORE parity of the sensitivities (I-51): ORE's own sensitivity analytic, run in-process
(tests/support/ore_sensitivity_oracle.py), against the engine's on the shared portfolio's sloped
two-curve market (tests/support/portfolio.py), every trade, factor and tenor.

    engine: engine.risk.greeks.bump.portfolio_sensitivities (ORE's bump and revalue)
    ORE:    OREApp's SENSITIVITY analytic, recalibrating the option models under every shift,
            with Theta

Delta and Gamma per curve tenor, Vega per swaption quote and Theta, each trade in its own
currency (USD, the reporting currency). Measured agreement (2026-10-17), against which the
tolerance is set: every Delta, Gamma and Vega within 2e-14 of its trade's NPV, Theta within 1e-13
(ORE's calibration converges to its Levenberg-Marquardt tolerance and the engine's to rounding,
which moves no Greek beyond that), so `TOLERANCE` of the NPV is a test of the definitions, not of
rounding: a shift convention, a sensitivity market or a Theta date off ORE's moves a Greek by far
more (the Theta market of 2026-10-17 was off by 60-290% of ORE's Theta).

The AD Greeks (`engine.risk.greeks.ad`) are ORE's in the limit of a vanishing shift, checked here
against ORE's own report for the trades of smooth price (swaps, Europeans, the bond), and in
tests/test_greeks.py against the bump Greeks this file holds to ORE. A Bermudan's/American's
grid price is piecewise smooth in the rates, so its AD Greeks are the price's derivatives, not
ORE's 1bp secants (tests/test_greeks_bermudan.py, I-78).
"""
import numpy as np
import pytest

from engine.pricing.config import PricingConfig
from engine.risk.greeks.ad import portfolio_greeks
from engine.risk.greeks.bump import portfolio_sensitivities
from tests.support import portfolio as shared
from tests.support.greeks import assert_close
from engine.run import GreeksConfig
from tests.support.ore_parity_matrix import cells
from tests.support.ore_sensitivity_oracle import ore_sensitivity_run

PRICING = PricingConfig(bermudan=shared.ENGINE, american=shared.ENGINE)
TRADES = list(shared.trades().values())
SMOOTH = [cfg for cfg in TRADES if cfg.product in ("swap", "european_swaption", "bond")]
#: Of the trade's NPV, for every Greek (see the module docstring).
TOLERANCE = 1e-11


@pytest.fixture(scope="module")
def ore():
    return ore_sensitivity_run(shared.market(), TRADES, PRICING)


# One test per method, each over every trade (its name in every message): pytest-xdist would run
# the module's ORE fixture once per worker for a test per trade.
@pytest.mark.ore_parity(cells("sensitivities", TRADES, PRICING, greeks=GreeksConfig()), tolerance="1e-11 of the NPV")
def test_every_greek_of_every_trade_is_ores(ore):
    engine = portfolio_sensitivities(TRADES, shared.market(), "USD", PRICING)
    for index, cfg in enumerate(TRADES):
        theirs, mine = ore.greeks[cfg.trade_id], engine[index]
        assert mine.keys() == theirs.keys(), cfg.trade_id
        scale = abs(ore.base_npv[cfg.trade_id])
        for key in mine:
            np.testing.assert_allclose(mine[key], theirs[key], rtol=0.0, atol=TOLERANCE * scale,
                                       err_msg=f"{cfg.trade_id} {key}")


@pytest.mark.ore_parity(cells("sensitivities", SMOOTH, PRICING, greeks=GreeksConfig("AD")),
                        tolerance="Delta, Gamma 1e-4 of each curve's largest, as the shift vanishes; Theta 1e-11 of the NPV")
def test_the_ad_greeks_are_ores_as_the_shift_vanishes(ore):
    """For the trades of smooth price (swaps, Europeans, the bond): ORE's forward-difference
    Delta less half its Gamma is the AD Delta to O(shift^3), and ORE's Gamma the AD Gamma to
    O(shift^4), per factor and tenor, measured within 2e-5 of each curve's largest (2026-10-17);
    Theta is the bump method's. (Vega's forward difference has an O(shift) term;
    tests/test_greeks.py extrapolates it away.)"""
    ad = portfolio_greeks(SMOOTH, shared.market(), "USD", PRICING)
    for index, cfg in enumerate(SMOOTH):
        theirs, mine = ore.greeks[cfg.trade_id], ad[index]
        assert mine.keys() == theirs.keys(), cfg.trade_id
        for key in mine:
            if key.startswith("delta"):
                gamma = "gamma" + key[len("delta"):]
                assert_close(mine[key], theirs[key] - 0.5 * theirs[gamma], rtol=1e-4, err_msg=f"{cfg.trade_id} {key}")
                assert_close(mine[gamma], theirs[gamma], rtol=1e-4, err_msg=f"{cfg.trade_id} {gamma}")
        np.testing.assert_allclose(mine["theta"], theirs["theta"], rtol=0.0,
                                   atol=TOLERANCE * abs(ore.base_npv[cfg.trade_id]), err_msg=cfg.trade_id)

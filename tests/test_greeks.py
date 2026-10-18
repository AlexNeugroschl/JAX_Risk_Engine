"""
Greeks by automatic differentiation (`engine.risk.greeks.ad`, `GreeksConfig.method = "AD"`) for
swaps, Europeans (Bachelier and Jamshidian) and bonds: ORE's sensitivity analysis in the limit
of a vanishing shift (I-78). Bermudans/Americans, calibrated inside their price function, are
tests/test_greeks_bermudan.py; the bump Greeks against ORE's own sensitivity analytic are
tests/test_sensitivity_ore_parity.py.

  * The price differentiated is the trade's on ORE's sensitivity market
    (`trade_price_function(..., tenors=...)`), whose value at its base is the bump method's base.
  * Delta, Gamma and Vega are its derivatives in each tenor's zero rate and each volatility quote:
    central differences of the same price (Gamma: of its gradient, as the price's second
    difference drowns in cancellation).
  * Against the bump method on the shared portfolio's sloped market, per trade, factor and tenor:
    ORE's forward-difference Delta less half its Gamma is the AD Delta to O(shift^3), the Gammas
    agree to O(shift^4), and the bump Vega extrapolated from two shifts (Richardson) is the AD
    Vega to O(shift^2). Theta is the same function in both.

The Jamshidian engine's root x* has the implicit function theorem's derivative
(`engine.solvers.roots.implicit_root`, tests/test_root_solvers.py). The t=0 prices differentiated
here equal ORE's (tests/test_shared_portfolio.py).
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.pricing.config import PricingConfig
from engine.pricing.cube import value_on
from engine.risk.greeks.ad import portfolio_greeks, trade_greeks
from engine.risk.greeks.bump import SensitivityConfig, portfolio_sensitivities, sensitivity_context
from engine.risk.greeks.price_functions import trade_price_function
from engine.run import JamshidianEngineConfig
from tests.support import portfolio as shared
from tests.support.greeks import assert_close

ASOF = shared.ASOF
JAMSHIDIAN = PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(0.03, 0.01))
CONFIG = SensitivityConfig()
SHIFT = CONFIG.curve_shift


def swap(**fields) -> SwapConfig:
    base = dict(notional=1e6, fixed_rate=0.04, payer=True, swap_tenor="5Y", evaluation_date=ASOF, trade_id="swap")
    base.update(fields)
    return SwapConfig(**base)


def european(**fields) -> SwaptionConfig:
    base = dict(notional=2e6, fixed_rate=0.044, payer=True, swap_tenor="5Y", forward_start=ORE.Period(1, ORE.Years),
                evaluation_date=ASOF, trade_id="european")
    base.update(fields)
    return SwaptionConfig(**base)


CASES = {
    "swap": (swap(), PricingConfig()),
    "receiver-swap": (swap(payer=False, fixed_rate=0.03), PricingConfig()),
    "european-bachelier": (european(), PricingConfig()),
    "european-otm-receiver": (european(payer=False, fixed_rate=0.035, swap_tenor="4Y",
                                       forward_start=ORE.Period(2, ORE.Years)), PricingConfig()),
    "european-jamshidian": (european(), JAMSHIDIAN),
    "bond": (shared.trades()["bond"], PricingConfig()),
}


def sensitivity_price(cfg, pricing):
    """The trade's price function on ORE's sensitivity market, and its node values there."""
    fn = trade_price_function(cfg, shared.market(), pricing, tenors=CONFIG.curve_tenors)
    return fn, [jnp.asarray(r) for r in fn.rates]


def moved(rates, curve, k, h):
    out = list(rates)
    out[curve] = rates[curve].at[k].add(h)
    return out


class TestTheDifferentiatedPrice:
    @pytest.mark.parametrize("name", CASES)
    def test_its_base_value_is_the_bump_methods(self, name):
        cfg, pricing = CASES[name]
        fn, rates = sensitivity_price(cfg, pricing)
        expected = value_on(cfg, sensitivity_context(shared.market(), CONFIG), pricing)
        assert float(fn.price(*rates)) == pytest.approx(expected, rel=1e-12, abs=1e-9)

    @pytest.mark.parametrize("name", CASES)
    def test_delta_is_the_derivative_in_each_tenor(self, name):
        cfg, pricing = CASES[name]
        fn, rates = sensitivity_price(cfg, pricing)
        greeks = trade_greeks(cfg, shared.market(), pricing, CONFIG)
        h = 1e-6
        for curve, (kind, key) in enumerate(fn.curves):
            delta = greeks[f"delta:{kind}:{key}"]
            assert delta.shape == (len(CONFIG.curve_tenors),)
            central = np.array([float(fn.price(*moved(rates, curve, k, h)) - fn.price(*moved(rates, curve, k, -h)))
                                / (2 * h) * SHIFT for k in range(delta.size)])
            assert_close(delta, central, rtol=1e-6)

    @pytest.mark.parametrize("name", ["swap", "european-bachelier", "european-jamshidian", "bond"])
    def test_gamma_is_the_derivative_of_delta(self, name):
        cfg, pricing = CASES[name]
        fn, rates = sensitivity_price(cfg, pricing)
        greeks = trade_greeks(cfg, shared.market(), pricing, CONFIG)
        gradient = jax.grad(fn.price, argnums=0)
        h = 1e-5
        gamma = greeks["gamma:discount:USD"]
        central = np.array([float(gradient(*moved(rates, 0, k, h))[k] - gradient(*moved(rates, 0, k, -h))[k])
                            / (2 * h) * SHIFT ** 2 for k in range(gamma.size)])
        assert_close(gamma, central, rtol=1e-5)

    def test_a_tenor_beyond_the_trade_has_no_delta(self):
        """A 5Y swap reads nothing beyond the 7Y tenor: the 30Y tenor's Delta is exactly 0."""
        greeks = trade_greeks(swap(), shared.market(), PricingConfig(), CONFIG)
        assert greeks["delta:discount:USD"][-1] == 0.0

    def test_payer_and_receiver_greeks_are_negations(self):
        payer = trade_greeks(swap(), shared.market(), PricingConfig(), CONFIG)
        receiver = trade_greeks(swap(payer=False), shared.market(), PricingConfig(), CONFIG)
        for key in payer:
            np.testing.assert_allclose(receiver[key], -payer[key], rtol=1e-12, atol=1e-12)

    def test_a_bond_has_discount_greeks_only(self):
        greeks = trade_greeks(CASES["bond"][0], shared.market(), PricingConfig(), CONFIG)
        assert set(greeks) == {"delta:discount:USD", "gamma:discount:USD"}

    def test_the_jamshidian_delta_differs_from_the_bachelier_one(self):
        """The engine is the configured one, not the default."""
        cfg = european()
        bachelier = trade_greeks(cfg, shared.market(), PricingConfig(), CONFIG)
        jamshidian = trade_greeks(cfg, shared.market(), JAMSHIDIAN, CONFIG)
        assert not np.allclose(bachelier["delta:index:" + shared.INDEX], jamshidian["delta:index:" + shared.INDEX])


class TestVega:
    def test_a_european_vega_is_the_derivative_in_each_quote(self):
        cfg = european()
        fn, rates = sensitivity_price(cfg, PricingConfig())
        vega = trade_greeks(cfg, shared.market(), PricingConfig(), CONFIG)["vega:USD"]
        assert vega.shape == fn.quotes.shape
        h = 1e-6
        expected = np.zeros_like(vega)
        for i, j in np.ndindex(vega.shape):
            up, down = (jnp.asarray(fn.quotes).at[i, j].add(s) for s in (h, -h))
            expected[i, j] = float(fn.price(*rates, quotes=up) - fn.price(*rates, quotes=down)) / (2 * h) * CONFIG.vol_shift
        assert_close(vega, expected, rtol=1e-6)
        # A 1Yx5Y option reads the 1Y and 2Y rows of the 5Y column only.
        assert np.count_nonzero(vega) <= 4 and vega.sum() > 0

    def test_engines_that_read_no_volatility_have_no_vega(self):
        assert "vega:USD" not in trade_greeks(swap(), shared.market(), PricingConfig(), CONFIG)
        assert "vega:USD" not in trade_greeks(european(), shared.market(), JAMSHIDIAN, CONFIG)


#: The shared portfolio's trades of smooth price (the grid-priced options are
#: tests/test_greeks_bermudan.py's).
SMOOTH = ["swap-payer", "swap-receiver-seasoned-icma", "european-payer", "european-receiver-otm-cash", "bond"]


class TestAgainstTheBumpMethod:
    """Both methods on the shared portfolio's sloped market, per trade, factor and tenor (I-78).

    ORE's Delta is a forward difference, `up - base = Delta h + Gamma h^2 / 2 + O(h^3)`, and its
    Gamma `up - 2 base + down = Gamma h^2 + O(h^4)`: with the curvature taken out, the Deltas
    agree to O(h^3), measured within 2e-5 of each curve's largest Delta (an out-of-the-money
    European's index Delta; the swaps' and the bond's within 1e-7) and the Gammas within 4e-6 of
    the largest Gamma (2026-10-17). The bump Vega is a forward difference too; its Richardson
    extrapolation from the shifts h and h/2 removes the O(h) term."""

    @pytest.fixture(scope="class")
    def both(self):
        trades = [shared.trades()[name] for name in SMOOTH]
        market = shared.market()
        half = dataclasses.replace(CONFIG, vol_shift=CONFIG.vol_shift / 2)
        return (portfolio_greeks(trades, market, "USD"), portfolio_sensitivities(trades, market, "USD"),
                portfolio_sensitivities(trades, market, "USD", config=half))

    def test_the_same_keys(self, both):
        ad, bump, _ = both
        for i in ad:
            assert ad[i].keys() == bump[i].keys()

    def test_the_same_theta(self, both):
        ad, bump, _ = both
        for i in ad:
            assert float(ad[i]["theta"]) == float(bump[i]["theta"])

    @pytest.mark.parametrize("index", range(len(SMOOTH)), ids=SMOOTH)
    def test_deltas_and_gammas_agree_in_every_tenor(self, both, index):
        ad, bump, _ = both
        for key in ad[index]:
            if key.startswith("delta"):
                gamma = "gamma" + key[len("delta"):]
                forward = bump[index][key] - 0.5 * bump[index][gamma]
                assert_close(ad[index][key], forward, rtol=1e-4)
                assert_close(ad[index][gamma], bump[index][gamma], rtol=1e-4)

    @pytest.mark.parametrize("index", [2, 3], ids=SMOOTH[2:4])
    def test_vegas_agree_as_the_shift_halves(self, both, index):
        ad, bump, half = both
        h = CONFIG.vol_shift
        extrapolated = (2.0 * half[index]["vega:USD"] / (h / 2) - bump[index]["vega:USD"] / h) * h
        assert_close(ad[index]["vega:USD"], extrapolated, rtol=1e-5)
        # The forward difference alone is O(h) away: the extrapolation is what agrees.
        assert not np.allclose(ad[index]["vega:USD"], bump[index]["vega:USD"], rtol=1e-6)

    def test_the_shift_sizes_scale_the_ad_greeks(self):
        """The AD Greeks are per unit shift: twice the shift, twice the Delta, four times the
        Gamma."""
        trades, market = [swap()], shared.market()
        one = portfolio_greeks(trades, market, "USD", config=SensitivityConfig(curve_shift=1e-4))[0]
        two = portfolio_greeks(trades, market, "USD", config=SensitivityConfig(curve_shift=2e-4))[0]
        np.testing.assert_allclose(two["delta:discount:USD"], 2 * one["delta:discount:USD"], rtol=1e-14)
        np.testing.assert_allclose(two["gamma:discount:USD"], 4 * one["gamma:discount:USD"], rtol=1e-14)

    def test_a_eur_trade_is_reported_in_the_base_currency(self):
        """Every Greek of a EUR trade is converted at the spot, as the bump method's."""
        from demos.demo_scenarios import EVAL_DATE, demo_market
        cfg = SwapConfig(notional=1e6, fixed_rate=0.03, payer=True, swap_tenor="5Y", currency="EUR",
                         evaluation_date=EVAL_DATE, trade_id="eur")
        market = demo_market()
        in_eur = portfolio_greeks([cfg], market, "EUR")[0]
        in_usd = portfolio_greeks([cfg], market, "USD")[0]
        for key in in_eur:
            np.testing.assert_allclose(in_usd[key], 1.10 * np.asarray(in_eur[key]), rtol=1e-14)

"""
Greeks by automatic differentiation (`engine.risk.greeks.ad`, `GreeksConfig.method = "AD"`) for
swaps, Europeans (Bachelier and Jamshidian) and bonds; Bermudans/Americans are
tests/test_greeks_bermudan.py.

  * Delta/Gamma per pillar of each market curve equal central differences of the engine's own
    price in that pillar's zero rate (Gamma: of the AD Delta, as the price's second difference
    drowns in cancellation).
  * The AD and Bump methods agree to first order (parallel Delta, Vega) and exactly (Theta,
    the same function). Their Deltas sit on different axes: the market curve's pillars vs ORE's
    sensitivity tenors.
  * Vega is the derivative in each quote of the volatility surface.

The Jamshidian engine's root x* has the implicit function theorem's derivative
(`engine.solvers.roots.implicit_root`, tests/test_root_solvers.py). The t=0 prices differentiated here equal ORE's (tests/test_shared_portfolio.py).
"""
import dataclasses

import numpy as np
import ORE
import pytest

from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.market_data.market import CurrencyMarket, Market, ZeroCurveConfig
from engine.pricing.config import PricingConfig
from engine.pricing.cube import value_today
from engine.risk.greeks.ad import curve_greeks, portfolio_greeks, vega_greek
from engine.risk.greeks.bump import SensitivityConfig, portfolio_sensitivities
from engine.run import JamshidianEngineConfig
from tests.support import portfolio as shared
from tests.support.greeks import assert_close, bumped_market

ASOF = shared.ASOF
JAMSHIDIAN = PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(0.03, 0.01))
SHIFT = 1e-4


def swap(**fields) -> SwapConfig:
    base = dict(notional=1e6, fixed_rate=0.04, payer=True, swap_tenor="5Y", evaluation_date=ASOF, trade_id="swap")
    base.update(fields)
    return SwapConfig(**base)


def european(**fields) -> SwaptionConfig:
    base = dict(notional=2e6, fixed_rate=0.044, payer=True, swap_tenor="5Y", forward_start=ORE.Period(1, ORE.Years),
                evaluation_date=ASOF, trade_id="european")
    base.update(fields)
    return SwaptionConfig(**base)


def central_delta(cfg, kind: str, pricing=PricingConfig(), h=1e-6) -> np.ndarray:
    """d NPV / d z_k * SHIFT by central differences of `value_today`, per market pillar."""
    price = lambda m: value_today([cfg], m, "USD", pricing)[0]  # noqa: E731
    return np.array([(price(bumped_market(kind, k, h)) - price(bumped_market(kind, k, -h))) / (2 * h) * SHIFT
                     for k in range(len(shared.PILLARS))])


def central_gamma(cfg, kind: str, pricing=PricingConfig(), h=1e-5) -> np.ndarray:
    """d^2 NPV / d z_k^2 * SHIFT^2 by central differences of the AD Delta."""
    key = f"delta:{kind}:{'USD' if kind == 'discount' else shared.INDEX}"
    delta_at = lambda k, s: curve_greeks(cfg, bumped_market(kind, k, s), pricing, SHIFT)[key][k]  # noqa: E731
    return np.array([(delta_at(k, h) - delta_at(k, -h)) / (2 * h) * SHIFT for k in range(len(shared.PILLARS))])


CASES = {
    "swap": (swap(), PricingConfig()),
    "receiver-swap": (swap(payer=False, fixed_rate=0.03), PricingConfig()),
    "european-bachelier": (european(), PricingConfig()),
    "european-otm-receiver": (european(payer=False, fixed_rate=0.035, swap_tenor="4Y",
                                       forward_start=ORE.Period(2, ORE.Years)), PricingConfig()),
    "european-jamshidian": (european(), JAMSHIDIAN),
}


class TestCurveDeltaGamma:
    @pytest.mark.parametrize("name", CASES)
    @pytest.mark.parametrize("kind", ["discount", "index"])
    def test_delta_is_the_derivative_in_each_market_pillar(self, name, kind):
        cfg, pricing = CASES[name]
        greeks = curve_greeks(cfg, shared.market(), pricing, SHIFT)
        key = f"delta:{kind}:{'USD' if kind == 'discount' else shared.INDEX}"
        assert greeks[key].shape == (len(shared.PILLARS),)
        assert_close(greeks[key], central_delta(cfg, kind, pricing), rtol=1e-6)

    @pytest.mark.parametrize("name", ["swap", "european-bachelier", "european-jamshidian"])
    def test_gamma_is_the_derivative_of_delta(self, name):
        cfg, pricing = CASES[name]
        greeks = curve_greeks(cfg, shared.market(), pricing, SHIFT)
        assert_close(greeks["gamma:discount:USD"], central_gamma(cfg, "discount", pricing), rtol=1e-5)

    def test_a_pillar_beyond_the_trade_has_no_delta(self):
        """A 5Y swap reads nothing beyond the 10Y pillar's left neighbourhood: the 30Y
        pillar's Delta is exactly 0."""
        greeks = curve_greeks(swap(), shared.market(), PricingConfig(), SHIFT)
        assert greeks["delta:discount:USD"][-1] == 0.0

    def test_payer_and_receiver_deltas_are_negations(self):
        payer = curve_greeks(swap(), shared.market(), PricingConfig(), SHIFT)
        receiver = curve_greeks(swap(payer=False), shared.market(), PricingConfig(), SHIFT)
        for key in payer:
            np.testing.assert_allclose(receiver[key], -payer[key], rtol=1e-12, atol=1e-12)

    def test_a_bond_has_discount_greeks_only(self):
        bond = shared.trades()["bond"]
        greeks = curve_greeks(bond, shared.market(), PricingConfig(), SHIFT)
        assert set(greeks) == {"delta:discount:USD", "gamma:discount:USD"}
        assert_close(greeks["delta:discount:USD"], central_delta(bond, "discount"), rtol=1e-6)

    def test_the_jamshidian_delta_differs_from_the_bachelier_one(self):
        """The engine is the configured one, not the default."""
        cfg = european()
        bachelier = curve_greeks(cfg, shared.market(), PricingConfig(), SHIFT)
        jamshidian = curve_greeks(cfg, shared.market(), JAMSHIDIAN, SHIFT)
        assert not np.allclose(bachelier["delta:index:" + shared.INDEX], jamshidian["delta:index:" + shared.INDEX])


class TestVega:
    def test_a_european_vega_is_the_derivative_in_each_quote(self):
        cfg, market = european(), shared.market()
        vega = vega_greek(cfg, market, PricingConfig(), SHIFT)
        surface = market.swaption_vols("USD")
        assert vega.shape == (len(surface.option_tenors), len(surface.swap_tenors))
        h = 1e-6
        expected = np.zeros_like(vega)
        for i in range(vega.shape[0]):
            for j in range(vega.shape[1]):
                values = []
                for s in (h, -h):
                    vols = np.array(surface.vols)
                    vols[i, j] += s
                    moved = dataclasses.replace(surface, vols=tuple(map(tuple, vols)))
                    usd = dataclasses.replace(market.currency("USD"), swaption_vols=moved)
                    values.append(value_today([cfg], dataclasses.replace(market, currencies={"USD": usd}), "USD")[0])
                expected[i, j] = (values[0] - values[1]) / (2 * h) * SHIFT
        assert_close(vega, expected, rtol=1e-6)
        # A 1Yx5Y option reads the 1Y and 2Y rows of the 5Y column only.
        assert np.count_nonzero(vega) <= 4 and vega.sum() > 0

    def test_engines_that_read_no_volatility_have_no_vega(self):
        assert vega_greek(swap(), shared.market(), PricingConfig(), SHIFT) is None
        assert vega_greek(european(), shared.market(), JAMSHIDIAN, SHIFT) is None


def flat_market() -> Market:
    """The shared market with flat curves: there the bump method's sensitivity market (the
    curves resampled at ORE's tenors, log-linear in the discount factor) is the market's own
    curve (linear in the zero rate), which on a sloped curve it is only near the pillars."""
    flat = lambda r: ZeroCurveConfig(list(shared.PILLARS), [r] * len(shared.PILLARS))  # noqa: E731
    return Market(ASOF, {"USD": CurrencyMarket(flat(0.035), {shared.INDEX: flat(0.038)}, shared.VOLS)})


class TestAgainstTheBumpMethod:
    """Both methods on the same trades: the same keys, the same Theta, and Deltas and Vegas
    that agree to the forward difference's curvature term. On flat curves, where the two
    methods' curve representations coincide (`flat_market`); on the sloped market the near-par
    swap's small discount Delta differs by 2% and the European's Vega by 0.9% between the two
    representations, which is not a derivative error (the finite-difference tests above hold
    there to 1e-6)."""

    @pytest.fixture(scope="class")
    def both(self):
        trades = [CASES[n][0] for n in ("swap", "european-bachelier")]
        trades[1] = dataclasses.replace(trades[1], trade_id="european-2")
        market = flat_market()
        return (portfolio_greeks(trades, market, "USD"), portfolio_sensitivities(trades, market, "USD"))

    def test_the_same_keys(self, both):
        ad, bump = both
        for i in ad:
            assert ad[i].keys() == bump[i].keys()

    def test_the_same_theta(self, both):
        ad, bump = both
        for i in ad:
            assert float(ad[i]["theta"]) == float(bump[i]["theta"])

    def test_parallel_deltas_agree_to_third_order(self, both):
        """ORE's Delta is a forward difference, up - base = Delta h + Gamma h^2 / 2 + O(h^3),
        and its Gamma up - 2 base + down = Gamma h^2 + O(h^4): with the curvature taken out
        the parallel Deltas agree to O(h^3). (Uncorrected, the European's differs by 0.6%.)
        The Gammas are not compared this way: both are diagonal (no cross-gammas, as ORE's
        default), so their sum depends on the axis, 6 pillars here and 13 tenors there."""
        ad, bump = both
        for i in ad:
            for key in ad[i]:
                if key.startswith("delta"):
                    forward = bump[i][key] - 0.5 * bump[i]["gamma" + key[len("delta"):]]
                    assert ad[i][key].sum() == pytest.approx(forward.sum(), rel=1e-4), (i, key)

    def test_vegas_agree(self, both):
        ad, bump = both
        assert_close(ad[1]["vega:USD"], bump[1]["vega:USD"], rtol=1e-3)

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



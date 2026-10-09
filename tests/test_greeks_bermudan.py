"""
Bermudan/American AD Greeks (`engine.risk.greeks.ad`): the option priced by the LGM grid engine
with the model calibrated to today's market (`engine.risk.greeks.price_functions.
bermudan_price_function`).

  * Delta/Gamma differentiate the price in each market-curve pillar with the calibrated
    volatility held fixed (the bump method recalibrates under each bump instead; decision
    A-5, documented in the module): checked against central differences of the same function.
  * Vega differentiates through the calibration by the implicit function theorem: a quote
    moves the helpers' volatilities, which move every later bucket of the bootstrap. Checked
    against central differences of the full pricing, recalibration included.

Gamma is checked by differencing the AD Delta, not the price: with an NPV of O(1e4) the
price's second difference at small bumps is below float64 cancellation error.
"""
import dataclasses

import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.market_data.curves import ZeroCurve
from engine.pricing.config import LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.cube import value_today
from engine.risk.greeks.ad import curve_greeks, portfolio_greeks, vega_greek
from engine.risk.greeks.price_functions import bermudan_price_function
from tests.support import portfolio as shared
from tests.support.greeks import assert_close, bumped_market

ASOF = shared.ASOF
ENGINE = LgmSwaptionEngineConfig(n_per_std=16, std_devs=6.0)
PRICING = PricingConfig(bermudan=ENGINE, american=ENGINE)
SHIFT = 1e-4


def bermudan(**fields) -> BermudanSwaptionConfig:
    base = dict(notional=1e6, fixed_rate=0.042, payer=True, swap_tenor="5Y", evaluation_date=ASOF,
                exercise_dates=[ASOF + ORE.Period(m, ORE.Months) for m in (12, 24, 36)], trade_id="bermudan")
    base.update(fields)
    return BermudanSwaptionConfig(**base)


def american(**fields) -> AmericanSwaptionConfig:
    base = dict(notional=1e6, fixed_rate=0.042, payer=True, swap_tenor="5Y", evaluation_date=ASOF,
                first_exercise_date=ASOF + ORE.Period(1, ORE.Years), last_exercise_date=ASOF + ORE.Period(3, ORE.Years),
                trade_id="american")
    base.update(fields)
    return AmericanSwaptionConfig(**base)


def held_sigma_price(cfg, market):
    """The option's price on `market`'s curves with the volatility calibrated on the shared
    market held fixed: what the AD Delta differentiates."""
    option = bermudan_price_function(cfg, shared.market(), PRICING, jnp.float64)
    usd = market.currency("USD")
    curve = lambda c: ZeroCurve(jnp.asarray(c.times), jnp.asarray(c.rates))  # noqa: E731
    return float(option.price(curve(usd.discount_curve), curve(usd.index_curves[shared.INDEX])))


@pytest.fixture(scope="module")
def greeks():
    return curve_greeks(bermudan(), shared.market(), PRICING, SHIFT)


class TestBermudanDeltaGamma:
    def test_finite_delta_and_gamma_for_every_pillar(self, greeks):
        for key in ("delta:discount:USD", "gamma:discount:USD", f"delta:index:{shared.INDEX}"):
            assert greeks[key].shape == (len(shared.PILLARS),) and np.all(np.isfinite(greeks[key]))

    @pytest.mark.slow
    @pytest.mark.parametrize("kind", ["discount", "index"])
    def test_delta_is_the_derivative_with_the_volatility_held(self, greeks, kind):
        h = 1e-6
        expected = np.array([(held_sigma_price(bermudan(), bumped_market(kind, k, h))
                              - held_sigma_price(bermudan(), bumped_market(kind, k, -h))) / (2 * h) * SHIFT
                             for k in range(len(shared.PILLARS))])
        key = f"delta:{kind}:{'USD' if kind == 'discount' else shared.INDEX}"
        assert_close(greeks[key], expected, rtol=1e-5)

    def test_gamma_is_the_derivative_of_delta(self, greeks):
        """Central differences of the gradient of the held-volatility price (the same model
        as the AD Gamma: a bumped market would recalibrate, which moves Delta at first
        order)."""
        import jax
        option = bermudan_price_function(bermudan(), shared.market(), PRICING, jnp.float64)
        times = jnp.asarray(shared.PILLARS)
        index = ZeroCurve(times, jnp.asarray(shared.FORWARDING))
        gradient = jax.grad(lambda rates: option.price(ZeroCurve(times, rates), index))
        base, h = np.asarray(shared.DISCOUNT, dtype=np.float64), 1e-5
        expected = np.zeros(len(base))
        for k in range(len(base)):
            up, down = base.copy(), base.copy()
            up[k] += h
            down[k] -= h
            expected[k] = (float(gradient(jnp.asarray(up))[k]) - float(gradient(jnp.asarray(down))[k])) / (2 * h)
        assert_close(greeks["gamma:discount:USD"], expected * SHIFT ** 2, rtol=1e-5)

    def test_zero_notional_has_zero_delta_and_gamma(self):
        zero = curve_greeks(bermudan(notional=0.0), shared.market(), PRICING, SHIFT)
        for value in zero.values():
            np.testing.assert_array_equal(value, 0.0)

    def test_payer_and_receiver_deltas_differ_in_sign_on_the_index_curve(self, greeks):
        receiver = curve_greeks(bermudan(payer=False, fixed_rate=0.035), shared.market(), PRICING, SHIFT)
        assert greeks[f"delta:index:{shared.INDEX}"].sum() > 0 > receiver[f"delta:index:{shared.INDEX}"].sum()


class TestBermudanVega:
    @pytest.mark.slow
    def test_vega_is_the_derivative_through_the_recalibration(self):
        """Central differences of the full pricing, the bootstrap rerun on each moved quote,
        against the implicit-function-theorem Vega."""
        cfg, market = bermudan(), shared.market()
        vega = vega_greek(cfg, market, PRICING, SHIFT)
        surface = market.swaption_vols("USD")
        h = 1e-6
        expected = np.zeros_like(vega)
        for i in range(vega.shape[0]):
            for j in range(vega.shape[1]):
                values = []
                for s in (h, -h):
                    vols = np.array(surface.vols)
                    vols[i, j] += s
                    usd = dataclasses.replace(market.currency("USD"), swaption_vols=dataclasses.replace(
                        surface, vols=tuple(map(tuple, vols))))
                    values.append(value_today([cfg], dataclasses.replace(market, currencies={"USD": usd}), "USD",
                                              PRICING)[0])
                expected[i, j] = (values[0] - values[1]) / (2 * h) * SHIFT
        assert_close(vega, expected, rtol=1e-4)

    @pytest.mark.parametrize("payer", [True, False])
    def test_a_long_option_has_positive_vega_where_its_basket_reads(self, payer):
        vega = vega_greek(bermudan(payer=payer), shared.market(), PRICING, SHIFT)
        assert vega.sum() > 0
        assert np.all(vega >= -1e-9 * np.max(vega))

    def test_an_uncalibrated_option_has_no_vega(self):
        """With `calibration="None"` the fixed model volatility has no quote to be sensitive
        to: Vega is omitted, not zero-filled."""
        pricing = PricingConfig(bermudan=dataclasses.replace(ENGINE, calibration="None"))
        assert vega_greek(bermudan(), shared.market(), pricing, SHIFT) is None
        greeks = portfolio_greeks([bermudan()], shared.market(), "USD", pricing)[0]
        assert not any(key.startswith("vega") for key in greeks)
        assert "delta:discount:USD" in greeks and "theta" in greeks


class TestAmericanSharesTheBermudanPath:
    @pytest.mark.slow
    def test_delta_vega_theta_finite_and_vega_positive(self):
        greeks = portfolio_greeks([american()], shared.market(), "USD", PRICING)[0]
        assert all(np.all(np.isfinite(v)) for v in greeks.values())
        assert greeks["vega:USD"].sum() > 0

    @pytest.mark.slow
    def test_an_american_is_worth_at_least_its_bermudan_and_so_is_its_vega(self):
        """Exercisable on more dates than the Bermudan on the same dates: at least the value."""
        b = value_today([bermudan()], shared.market(), "USD", PRICING)[0]
        a = value_today([american()], shared.market(), "USD", PRICING)[0]
        assert a >= b - 1e-6

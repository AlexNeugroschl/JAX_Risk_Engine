"""
Bermudan/American AD Greeks (`engine.risk.greeks.ad`): the option on the LGM grid engine,
calibrated inside its price function (`engine.risk.greeks.price_functions`), so every derivative
moves the calibration with the market, as ORE's sensitivity analysis recalibrates under every
shift (I-78).

  * Delta and Gamma in each tenor of ORE's sensitivity market, Vega in each quote: the derivatives
    of the recalibrating price, against its central differences at a step small enough to stay on
    one smooth piece of the grid's price, 1e-7 (measured 2026-10-17: the Delta to 1e-9, the Gamma
    from the gradient to 5e-8; at 1e-6 one tenor's step crossed a kink, 7% off).
  * Against ORE's bump Greeks: the grid engine's price (ORE's, to 1e-11) is piecewise smooth in
    the rates, its exercise switching on the state grid's nodes, so at ORE's 1bp shift a forward
    difference is a secant across those kinks. The AD Delta is the derivative of the price at the
    market, within `GRID_GAP` of each curve's largest bump Delta less half its Gamma; the bump
    Gamma, a second difference across kinks, is not a derivative and is not compared (it is
    ORE's, tests/test_sensitivity_ore_parity.py). The result says so (I-78).

Gamma is checked by differencing the gradient, not the price: with an NPV of O(1e4) the price's
second difference at small steps is below float64 cancellation error.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import ORE
import pytest

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.pricing.config import LgmSwaptionEngineConfig, PricingConfig
from engine.pricing.cube import value_on, value_today
from engine.risk.greeks.ad import portfolio_greeks, trade_greeks
from engine.risk.greeks.bump import SensitivityConfig, portfolio_sensitivities, sensitivity_context
from engine.risk.greeks.price_functions import trade_price_function
from tests.support import portfolio as shared
from tests.support.greeks import assert_close

ASOF = shared.ASOF
ENGINE = LgmSwaptionEngineConfig(n_per_std=16, std_devs=6.0)
PRICING = PricingConfig(bermudan=ENGINE, american=ENGINE)
CONFIG = SensitivityConfig()
SHIFT = CONFIG.curve_shift
#: How far the AD Delta of a grid-priced option is from ORE's bump Delta less half its Gamma, of
#: the curve's largest: measured within 4e-3 on the shared portfolio's options (2026-10-17).
GRID_GAP = 1e-2


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


@pytest.fixture(scope="module")
def option():
    """The Bermudan's price function on ORE's sensitivity market, its node values, and its AD
    Greeks."""
    fn = trade_price_function(bermudan(), shared.market(), PRICING, tenors=CONFIG.curve_tenors)
    return fn, [jnp.asarray(r) for r in fn.rates], trade_greeks(bermudan(), shared.market(), PRICING, CONFIG)


def moved(rates, curve, k, h):
    out = list(rates)
    out[curve] = rates[curve].at[k].add(h)
    return out


class TestTheRecalibratingPrice:
    def test_its_base_value_is_the_bump_methods(self, option):
        fn, rates, _ = option
        expected = value_on(bermudan(), sensitivity_context(shared.market(), CONFIG), PRICING)
        assert float(fn.price(*rates)) == pytest.approx(expected, rel=1e-12)

    def test_a_shifted_tenor_recalibrates(self, option):
        """The price function on a shifted sensitivity market is the bump method's revaluation
        there, calibration included: its forward difference in each tenor is ORE's bump Delta."""
        fn, rates, _ = option
        bump = portfolio_sensitivities([bermudan()], shared.market(), "USD", PRICING, CONFIG)[0]
        base = float(fn.price(*rates))
        for curve, (kind, key) in enumerate(fn.curves):
            forward = [float(fn.price(*moved(rates, curve, k, SHIFT))) - base for k in range(len(CONFIG.curve_tenors))]
            assert_close(forward, bump[f"delta:{kind}:{key}"], rtol=1e-9)


class TestBermudanDeltaGamma:
    def test_finite_greeks_in_every_tenor(self, option):
        _, _, greeks = option
        for key in ("delta:discount:USD", "gamma:discount:USD", f"delta:index:{shared.INDEX}"):
            assert greeks[key].shape == (len(CONFIG.curve_tenors),) and np.all(np.isfinite(greeks[key]))

    @pytest.mark.parametrize("curve", [0, 1], ids=["discount", "index"])
    def test_delta_is_the_derivative_through_the_recalibration(self, option, curve):
        fn, rates, greeks = option
        kind, key = fn.curves[curve]
        delta = greeks[f"delta:{kind}:{key}"]
        h = 1e-7
        central = np.array([float(fn.price(*moved(rates, curve, k, h)) - fn.price(*moved(rates, curve, k, -h)))
                            / (2 * h) * SHIFT for k in range(delta.size)])
        assert_close(delta, central, rtol=1e-6)

    def test_gamma_is_the_derivative_of_delta(self, option):
        """Central differences of the recalibrating price's gradient (reverse mode, another route
        than the AD Gamma's forward-over-forward one)."""
        fn, rates, greeks = option
        gradient = jax.jit(jax.grad(fn.price, argnums=0))
        gamma, h = greeks["gamma:discount:USD"], 1e-7
        central = np.array([float(gradient(*moved(rates, 0, k, h))[k] - gradient(*moved(rates, 0, k, -h))[k])
                            / (2 * h) * SHIFT ** 2 for k in range(gamma.size)])
        assert_close(gamma, central, rtol=1e-6)

    def test_the_delta_holds_the_calibrations_response(self, option):
        """The calibration's response to the curve is in the Delta: a model held at today's
        calibration (`calibration="None"` at its buckets) has another."""
        from engine.models.lgm import Sigma
        fn, _, greeks = option
        held_engine = dataclasses.replace(ENGINE, calibration="None")
        held_fn = trade_price_function(bermudan(), shared.market(), PricingConfig(bermudan=held_engine),
                                       tenors=CONFIG.curve_tenors)
        sigma = fn.terms.terms.prepared.sigma
        prepared = dataclasses.replace(held_fn.terms.terms.prepared, sigma=Sigma(sigma.times, sigma.values))
        held_terms = held_fn.terms._replace(terms=held_fn.terms.terms._replace(prepared=prepared))
        held = dataclasses.replace(held_fn, terms=held_terms)
        rates = [jnp.asarray(r) for r in held.rates]
        assert float(held.price(*rates)) == pytest.approx(float(fn.price(*rates)), rel=1e-12)
        held_delta = np.asarray(jax.grad(held.price, argnums=0)(*rates)) * SHIFT
        assert not np.allclose(greeks["delta:discount:USD"], held_delta, rtol=1e-3)

    def test_payer_and_receiver_deltas_differ_in_sign_on_the_index_curve(self, option):
        _, _, greeks = option
        receiver = trade_greeks(bermudan(payer=False, fixed_rate=0.035), shared.market(), PRICING, CONFIG)
        assert greeks[f"delta:index:{shared.INDEX}"].sum() > 0 > receiver[f"delta:index:{shared.INDEX}"].sum()


class TestBermudanVega:
    def test_vega_is_the_derivative_through_the_recalibration(self, option):
        """Central differences of the recalibrating price in each quote, against the Vega."""
        fn, rates, greeks = option
        vega, h = greeks["vega:USD"], 1e-7
        expected = np.zeros_like(vega)
        for i, j in np.ndindex(vega.shape):
            up, down = (jnp.asarray(fn.quotes).at[i, j].add(s) for s in (h, -h))
            expected[i, j] = float(fn.price(*rates, quotes=up) - fn.price(*rates, quotes=down)) / (2 * h) * CONFIG.vol_shift
        assert_close(vega, expected, rtol=1e-5)

    @pytest.mark.parametrize("payer", [True, False])
    def test_a_long_option_has_positive_vega_where_its_basket_reads(self, payer):
        vega = trade_greeks(bermudan(payer=payer), shared.market(), PRICING, CONFIG)["vega:USD"]
        assert vega.sum() > 0
        assert np.all(vega >= -1e-9 * np.max(vega))

    def test_an_uncalibrated_option_has_no_vega(self):
        """With `calibration="None"` the fixed model volatility has no quote to be sensitive
        to: Vega is omitted, not zero-filled."""
        pricing = PricingConfig(bermudan=dataclasses.replace(ENGINE, calibration="None"))
        assert "vega:USD" not in trade_greeks(bermudan(), shared.market(), pricing, CONFIG)
        greeks = portfolio_greeks([bermudan()], shared.market(), "USD", pricing)[0]
        assert not any(key.startswith("vega") for key in greeks)
        assert "delta:discount:USD" in greeks and "theta" in greeks


@pytest.mark.slow
class TestAgainstTheBumpMethod:
    """The shared portfolio's Bermudans and American, on ORE's default engine, against the bump
    Greeks (ORE's, tests/test_sensitivity_ore_parity.py): the same keys and Theta, and Deltas
    within the grid's gap (see the module docstring)."""

    NAMES = ["bermudan-payer-physical", "bermudan-receiver-cash", "american-payer"]

    @pytest.fixture(scope="class")
    def both(self):
        trades = [shared.trades()[name] for name in self.NAMES]
        pricing = PricingConfig(bermudan=shared.ENGINE, american=shared.ENGINE)
        return (portfolio_greeks(trades, shared.market(), "USD", pricing),
                portfolio_sensitivities(trades, shared.market(), "USD", pricing))

    def test_the_same_keys_and_theta(self, both):
        ad, bump = both
        for i in ad:
            assert ad[i].keys() == bump[i].keys()
            assert float(ad[i]["theta"]) == float(bump[i]["theta"])

    def test_deltas_are_within_the_grids_gap(self, both):
        ad, bump = both
        for i in ad:
            for key in ad[i]:
                if key.startswith("delta"):
                    forward = bump[i][key] - 0.5 * bump[i]["gamma" + key[len("delta"):]]
                    assert_close(ad[i][key], forward, rtol=GRID_GAP)


class TestAmericanSharesTheBermudanPath:
    @pytest.mark.slow
    def test_delta_vega_theta_finite_and_vega_positive(self):
        greeks = portfolio_greeks([american()], shared.market(), "USD", PRICING)[0]
        assert all(np.all(np.isfinite(v)) for v in greeks.values())
        assert greeks["vega:USD"].sum() > 0

    @pytest.mark.slow
    def test_an_american_is_worth_at_least_its_bermudan(self):
        """Exercisable on more dates than the Bermudan on the same dates: at least the value."""
        b = value_today([bermudan()], shared.market(), "USD", PRICING)[0]
        a = value_today([american()], shared.market(), "USD", PRICING)[0]
        assert a >= b - 1e-6

"""
Vanilla swaps at t=0 against ORE's `DiscountingSwapEngine`: the valuation pipeline's swap pricer
(`engine.valuation.legs`, ORE's at-par Ibor coupons) on flat discount and forwarding curves,
over directions, spreads, single-curve discounting, tenors with stubs, rate and notional
extremes, distinct curves and portfolios. The same pricer values every simulated path
(tests/test_valuation.py); `SwapConfig`'s own validation is at the end.
"""
import numpy as np
import ORE
import pytest

from engine.instruments.swap import SwapConfig
from engine.market import CurrencyMarket, Market, ZeroCurveConfig, index_name
from engine.valuation.portfolio import value_today

TODAY = ORE.Date(30, 7, 2026)
DISCOUNT_RATE = 0.030
FORWARD_RATE = 0.035
PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
DC = ORE.Actual365Fixed()
#: The legs pricer is ORE's arithmetic: measured agreement ~1e-15 relative.
RTOL = 1e-12


def _market(disc_rate=DISCOUNT_RATE, fwd_rate=FORWARD_RATE, index_tenor_months=6) -> Market:
    flat = lambda rate: ZeroCurveConfig(PILLARS, [rate] * len(PILLARS))  # noqa: E731
    return Market(TODAY, {"USD": CurrencyMarket(flat(disc_rate), {index_name("USD", index_tenor_months): flat(fwd_rate)})})


def _swap(notional=1_000_000.0, fixed_rate=0.03, payer=True, tenor="2Y", trade_id="swap", **fields) -> SwapConfig:
    return SwapConfig(notional=notional, fixed_rate=fixed_rate, payer=payer, swap_tenor=tenor, evaluation_date=TODAY,
                      trade_id=trade_id, **fields)


def _npv(cfg: SwapConfig, disc_rate=DISCOUNT_RATE, fwd_rate=FORWARD_RATE) -> float:
    return value_today([cfg], _market(disc_rate, fwd_rate, cfg.index_tenor_months), "USD")[0]


def _ore_swap(payer=True, notional=1_000_000.0, fixed_rate=0.03, tenor="2Y", disc_rate=DISCOUNT_RATE,
              fwd_rate=FORWARD_RATE, index_tenor_months=6, floating_spread=0.0) -> ORE.VanillaSwap:
    """The same swap in ORE (`MakeVanillaSwap`, ACT/365 legs, the `build_vanilla_swap` index)
    under `DiscountingSwapEngine` on flat curves."""
    ORE.Settings.instance().evaluationDate = TODAY
    fwd = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, fwd_rate, DC))
    disc = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, disc_rate, DC))
    index = ORE.IborIndex("SimIndex", ORE.Period(index_tenor_months, ORE.Months), 2, ORE.USDCurrency(), ORE.TARGET(),
                          ORE.ModifiedFollowing, False, DC, fwd)
    swap = ORE.MakeVanillaSwap(ORE.Period(tenor), index, fixed_rate, nominal=notional,
                               swapType=ORE.VanillaSwap.Payer if payer else ORE.VanillaSwap.Receiver,
                               floatingLegSpread=floating_spread, discountingTermStructure=disc,
                               fixedLegDayCount=DC, floatingLegDayCount=DC)
    swap.setPricingEngine(ORE.DiscountingSwapEngine(disc))
    return swap


class TestAgainstORE:
    @pytest.mark.parametrize("payer", [True, False])
    def test_matches_ore(self, payer):
        assert _npv(_swap(payer=payer)) == pytest.approx(_ore_swap(payer=payer).NPV(), rel=RTOL)

    def test_par_rate_prices_to_zero(self):
        fair = _ore_swap().fairRate()
        assert _npv(_swap(fixed_rate=fair)) == pytest.approx(0.0, abs=1e-8)

    def test_payer_and_receiver_are_negations(self):
        assert _npv(_swap(payer=True)) == pytest.approx(-_npv(_swap(payer=False)), rel=1e-14)

    def test_floating_spread(self):
        assert _npv(_swap(floating_spread=0.005)) == pytest.approx(_ore_swap(floating_spread=0.005).NPV(), rel=RTOL)

    def test_single_curve(self):
        """Discounting and forwarding on one curve."""
        assert _npv(_swap(), DISCOUNT_RATE, DISCOUNT_RATE) == pytest.approx(
            _ore_swap(fwd_rate=DISCOUNT_RATE).NPV(), rel=RTOL)

    @pytest.mark.parametrize("fixed_rate", [0.15, 0.50, -0.10])
    def test_far_off_market_fixed_rates(self, fixed_rate):
        assert _npv(_swap(fixed_rate=fixed_rate)) == pytest.approx(_ore_swap(fixed_rate=fixed_rate).NPV(), rel=RTOL)

    @pytest.mark.parametrize("tenor", ["6M", "15M", "30Y"])
    def test_tenors(self, tenor):
        """One cashflow per leg (6M), a stub period (15M), dozens of cashflows (30Y)."""
        assert _npv(_swap(tenor=tenor, fixed_rate=0.04)) == pytest.approx(
            _ore_swap(tenor=tenor, fixed_rate=0.04).NPV(), rel=RTOL)

    def test_three_month_index(self):
        cfg = _swap(index_tenor_months=3)
        assert _npv(cfg) == pytest.approx(_ore_swap(index_tenor_months=3).NPV(), rel=RTOL)

    def test_negative_notional_is_the_negation(self):
        negative = _npv(_swap(notional=-1_000_000.0))
        assert negative == pytest.approx(-_npv(_swap()), rel=1e-14)
        assert negative == pytest.approx(_ore_swap(notional=-1_000_000.0).NPV(), rel=RTOL)

    @pytest.mark.parametrize("fixed_rate", [0.03, 5.0])
    def test_zero_notional_prices_to_zero(self, fixed_rate):
        assert _npv(_swap(notional=0.0, fixed_rate=fixed_rate)) == 0.0

    @pytest.mark.parametrize("disc_rate, fwd_rate", [(0.010, 0.025), (0.025, 0.010), (0.040, 0.060), (0.060, 0.040),
                                                    (0.010, 0.060), (0.040, 0.010)])
    def test_distinct_discount_and_forwarding_curves(self, disc_rate, fwd_rate):
        assert _npv(_swap(), disc_rate, fwd_rate) == pytest.approx(
            _ore_swap(disc_rate=disc_rate, fwd_rate=fwd_rate).NPV(), rel=RTOL)


class TestNegativeRates:
    """A negative curve (discount factors above 1) prices finitely, without flooring the
    forward, and matches ORE."""
    NEG_DISC, NEG_FWD = -0.005, -0.002

    @pytest.mark.parametrize("payer", [True, False])
    def test_negative_curves(self, payer):
        cfg = _swap(fixed_rate=-0.003, payer=payer)
        assert _npv(cfg, self.NEG_DISC, self.NEG_FWD) == pytest.approx(
            _ore_swap(payer=payer, fixed_rate=-0.003, disc_rate=self.NEG_DISC, fwd_rate=self.NEG_FWD).NPV(), rel=RTOL)

    def test_a_negative_forward_is_not_floored(self):
        """A 0% payer isolates the floating leg: a negative forward gives a negative NPV."""
        mine = _npv(_swap(fixed_rate=0.0), self.NEG_DISC, self.NEG_FWD)
        assert mine < 0.0
        assert mine == pytest.approx(
            _ore_swap(fixed_rate=0.0, disc_rate=self.NEG_DISC, fwd_rate=self.NEG_FWD).NPV(), rel=RTOL)

    def test_near_zero_rates(self):
        assert _npv(_swap(fixed_rate=1e-6), 1e-7, -1e-7) == pytest.approx(
            _ore_swap(fixed_rate=1e-6, disc_rate=1e-7, fwd_rate=-1e-7).NPV(), rel=1e-9, abs=1e-9)


class TestPortfolios:
    def test_forty_trade_portfolio_matches_ore_trade_by_trade(self):
        """Random rates, notionals, directions and spreads in one call, each trade against ORE,
        in request order."""
        rng = np.random.default_rng(20260730)
        n = 40
        notionals, rates = rng.uniform(-5e6, 5e6, n), rng.uniform(-0.02, 0.20, n)
        payers, spreads = rng.integers(0, 2, n).astype(bool), rng.uniform(-0.01, 0.01, n)
        trades = [_swap(notional=float(notionals[i]), fixed_rate=float(rates[i]), payer=bool(payers[i]),
                        floating_spread=float(spreads[i]), trade_id=f"swap-{i}") for i in range(n)]
        ours = np.asarray(value_today(trades, _market(), "USD"))
        ore = np.array([_ore_swap(payer=bool(payers[i]), notional=float(notionals[i]), fixed_rate=float(rates[i]),
                                  floating_spread=float(spreads[i])).NPV() for i in range(n)])
        np.testing.assert_allclose(ours, ore, rtol=RTOL, atol=1e-8)
        assert np.any(ours > 0) and np.any(ours < 0)

    def test_a_portfolio_prices_each_trade_as_alone(self):
        trades = [_swap(fixed_rate=r, payer=p, trade_id=f"swap-{k}")
                  for k, (r, p) in enumerate([(0.02, True), (0.05, False), (0.035, True)])]
        together = value_today(trades, _market(), "USD")
        assert together == [value_today([t], _market(), "USD")[0] for t in trades]


class TestSwapConfigValidation:
    """`SwapConfig.__post_init__` rejects non-finite notional/fixed_rate and unparseable
    tenors before any ORE call. Zero and negative values are valid. Identity and the
    evaluation date are tests/test_trade_configs.py."""

    @pytest.mark.parametrize("field, value", [("notional", float("nan")), ("notional", float("inf")),
                                              ("fixed_rate", float("nan")), ("fixed_rate", float("-inf"))])
    def test_non_finite_values_rejected(self, field, value):
        with pytest.raises(ValueError, match=field):
            _swap(**{field: value})

    def test_unparseable_swap_tenor_rejected(self):
        with pytest.raises(ValueError, match="swap_tenor"):
            _swap(tenor="not-a-tenor")

    def test_zero_and_negative_notional_still_valid(self):
        _swap(notional=0.0)
        _swap(notional=-1_000_000.0)

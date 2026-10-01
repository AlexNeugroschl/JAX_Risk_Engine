"""
Bonds through `price_portfolio` itself (the pricer alone is tests/test_treasury_instrument.py).
A pricer passing its own tests says nothing about whether the portfolio path reaches it (the
lesson of I-01), so everything here goes through the public entry point, on a `Market`.
"""
import dataclasses
from dataclasses import replace

import numpy as np
import ORE
import pytest

from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market import CurrencyMarket, Market, ZeroCurveConfig, index_name
from engine.portfolio import GreeksConfig, HullWhiteConfig, PortfolioRequest, RunConfig, price_portfolio
from engine.simulation.config import CamConfig
from engine.valuation.portfolio import value_today

VALUATION = ORE.Date(2, 6, 2025)
FLAT_3PCT = ZeroCurveConfig(times=(0.0, 1.0, 2.0, 5.0, 10.0, 30.0), rates=(0.03,) * 6)
DISCOUNT = "delta:discount:USD"


def _date(iso: str) -> ORE.Date:
    year, month, day = (int(p) for p in iso.split("-"))
    return ORE.Date(day, month, year)


def make_market(curve: ZeroCurveConfig = FLAT_3PCT, asof: ORE.Date = VALUATION) -> Market:
    """USD with `curve` as both the discount and the 6M forwarding curve (swaps need the
    latter; bonds ignore it)."""
    return Market(asof, {"USD": CurrencyMarket(curve, {index_name("USD", 6): curve})})


def price_alone(cfg: BondConfig) -> float:
    """The bond priced by itself on the same market (`value_today`)."""
    return value_today([cfg], make_market(asof=cfg.evaluation_date), "USD")[0]


def make_bill(face: float = 100_000.0, maturity: str = "2025-12-15", trade_id: str = "bill") -> BondConfig:
    return BondConfig(face_amount=face, maturity_date=_date(maturity), evaluation_date=VALUATION, trade_id=trade_id)


NOTE_SCHEDULE = (
    CouponPeriod(_date("2024-12-15"), _date("2025-06-15"), _date("2025-06-15")),
    CouponPeriod(_date("2025-06-15"), _date("2025-12-15"), _date("2025-12-15")),
    CouponPeriod(_date("2025-12-15"), _date("2026-06-15"), _date("2026-06-15")),
)


def make_note(face: float = 100_000.0, trade_id: str = "note") -> BondConfig:
    return BondConfig(face_amount=face, maturity_date=_date("2026-06-15"), evaluation_date=VALUATION,
                      trade_id=trade_id, coupon_rate=0.04, coupon_schedule=NOTE_SCHEDULE,
                      accrual_day_count="ACT/ACT (ICMA)")


def make_swap(notional: float = 1_000_000.0) -> SwapConfig:
    return SwapConfig(notional=notional, fixed_rate=0.03, payer=True, swap_tenor="5Y",
                      evaluation_date=VALUATION, trade_id="swap")


SIMULATION = CamConfig(dates=tuple(VALUATION + ORE.Period(m, ORE.Months) for m in (3, 6, 12)),
                       base_currency="USD", ir={"USD": HullWhiteConfig(0.03, 0.01)}, samples=256, seed=7)


def bond_request(trades, compute_greeks: bool = False, scenario_risk: bool = False,
                 market: Market = None) -> PortfolioRequest:
    return PortfolioRequest(market=market or make_market(), trades=list(trades), compute_greeks=compute_greeks,
                            scenario_risk=scenario_risk,
                            config=RunConfig(simulation=SIMULATION if scenario_risk else None))


def delta(greeks) -> float:
    """The parallel Delta: the per-tenor discount Deltas summed."""
    return float(np.sum(greeks[DISCOUNT]))


def gamma(greeks) -> float:
    return float(np.sum(greeks["gamma:discount:USD"]))


class TestBondsReachBaseNpv:
    """A bond's t=0 value reaches the portfolio total and the breakdown."""

    def test_a_bill_prices_through_price_portfolio(self):
        result = price_portfolio(bond_request([make_bill()]))
        assert result.base_npv == pytest.approx(price_alone(make_bill()), rel=1e-14)

    def test_base_npv_equals_the_sum_of_the_breakdown(self):
        result = price_portfolio(bond_request([make_bill(), make_note()]))
        assert result.base_npv == pytest.approx(sum(result.base_npv_per_trade))

    def test_per_trade_values_are_in_request_order(self):
        bill, note = make_bill(), make_note()
        result = price_portfolio(bond_request([bill, note]))
        assert result.base_npv_per_trade[0] == pytest.approx(price_alone(bill), rel=1e-14)
        assert result.base_npv_per_trade[1] == pytest.approx(price_alone(note), rel=1e-14)
        assert result.trade_ids == ["bill", "note"]

    def test_a_short_bond_reduces_the_portfolio_total(self):
        long_only = price_portfolio(bond_request([make_bill()])).base_npv
        hedged = price_portfolio(bond_request([make_bill(), make_bill(face=-40_000.0, trade_id="short")])).base_npv
        assert hedged < long_only

    def test_a_bond_mixed_with_a_swap_prices_both(self):
        result = price_portfolio(bond_request([make_swap(), make_bill()]))
        assert len(result.base_npv_per_trade) == 2
        assert result.base_npv_per_trade[1] == pytest.approx(price_alone(make_bill()), rel=1e-14)


class TestBondGreeksReachThePortfolioPath:
    """I-01 regression class (ledger I-26): a trade type without a branch in the Greeks is
    silently skipped. Each test fails with bonds dropped from `portfolio_sensitivities`."""

    def test_a_bond_has_a_greeks_entry_at_all(self):
        result = price_portfolio(bond_request([make_bill()], compute_greeks=True))
        assert result.greeks is not None
        assert 0 in result.greeks, "the bond has NO greeks entry -- I-01 reproduced for a new type"

    def test_every_bond_in_a_multi_bond_portfolio_gets_greeks(self):
        trades = [make_bill(), make_note(), make_bill(face=-50_000.0, trade_id="short")]
        assert set(price_portfolio(bond_request(trades, compute_greeks=True)).greeks) == {0, 1, 2}

    def test_delta_gamma_and_theta_are_all_present(self):
        greeks = price_portfolio(bond_request([make_bill()], compute_greeks=True)).greeks[0]
        assert {DISCOUNT, "gamma:discount:USD", "theta"} <= set(greeks)

    def test_a_bond_reads_no_index_curve(self):
        """A bond discounts only: no index-curve Delta."""
        greeks = price_portfolio(bond_request([make_bill()], compute_greeks=True)).greeks[0]
        assert not any(k.startswith("delta:index") for k in greeks)

    def test_a_bond_is_not_skipped_when_mixed_with_a_swap(self):
        result = price_portfolio(bond_request([make_swap(), make_bill()], compute_greeks=True))
        assert 1 in result.greeks, "the bond was skipped while the swap was not"
        assert {DISCOUNT, "theta"} <= set(result.greeks[1])

    def test_delta_is_negative_for_a_long_bond(self):
        assert delta(price_portfolio(bond_request([make_bill()], compute_greeks=True)).greeks[0]) < 0

    def test_delta_flips_sign_for_a_short_bond(self):
        result = price_portfolio(bond_request([make_bill(face=-100_000.0)], compute_greeks=True))
        assert delta(result.greeks[0]) > 0

    def test_gamma_is_positive_for_a_long_bond(self):
        """Convexity: a long bond's Gamma is positive."""
        assert gamma(price_portfolio(bond_request([make_note()], compute_greeks=True)).greeks[0]) > 0

    def test_delta_matches_a_parallel_reprice(self):
        """The per-tenor Deltas sum to a 1bp parallel shift of the zero curve, to curvature."""
        bill = make_bill()
        up = replace(FLAT_3PCT, rates=tuple(r + 1e-4 for r in FLAT_3PCT.rates))
        expected = value_today([bill], make_market(up), "USD")[0] - price_alone(bill)
        got = delta(price_portfolio(bond_request([bill], compute_greeks=True)).greeks[0])
        assert got == pytest.approx(expected, rel=1e-3)

    def test_a_longer_bond_has_a_larger_delta(self):
        result = price_portfolio(bond_request([make_bill(maturity="2025-12-15"),
                                               make_bill(maturity="2032-12-15", trade_id="long")],
                                              compute_greeks=True))
        assert abs(delta(result.greeks[1])) > abs(delta(result.greeks[0]))

    def test_theta_is_the_dates_fixed_roll(self):
        """ORE's Theta market keeps each discount factor fixed in dates (not re-anchored at the
        Theta date), so on a flat curve a bill's value does not move: Theta 0, not the "pull
        to par" of a re-anchored curve. On a sloped curve it is not 0."""
        assert float(price_portfolio(bond_request([make_bill()], compute_greeks=True)).greeks[0]["theta"]) == 0.0
        sloped = ZeroCurveConfig(times=FLAT_3PCT.times, rates=(0.02, 0.02, 0.03, 0.04, 0.045, 0.05))
        result = price_portfolio(bond_request([make_note()], compute_greeks=True, market=make_market(sloped)))
        assert float(result.greeks[0]["theta"]) != 0.0

    def test_theta_adds_back_a_coupon_paid_the_next_day(self):
        """I-39: ORE's Theta adds back the flows paid in (t, t + 1 day]. Valued the day before
        the 2025-12-15 coupon, a one-day reprice drops the $2,000 coupon; Theta must not."""
        note = replace(make_note(), evaluation_date=_date("2025-12-14"))
        market = make_market(asof=note.evaluation_date)
        theta = float(price_portfolio(bond_request([note], compute_greeks=True, market=market)).greeks[0]["theta"])
        assert 0.0 < theta < 50.0

    def test_a_bond_maturing_tomorrow_does_not_crash_the_greeks(self):
        """Regression: Theta's one-day move lands on maturity, and pricing the bond there raised
        `BondPricingError` (a date the caller never gave). A matured bond is worth 0 and its
        redemption, paid on the Theta date, is a paid flow: Theta = redemption - NPV."""
        bill = make_bill(maturity="2025-06-03")
        greeks = price_portfolio(bond_request([bill], compute_greeks=True)).greeks[0]
        assert DISCOUNT in greeks
        assert float(greeks["theta"]) == pytest.approx(bill.face_amount - price_alone(bill), rel=1e-12)

    def test_no_vega_is_reported(self):
        """A fixed-coupon bond reads no volatility: Vega is omitted, not 0.0."""
        greeks = price_portfolio(bond_request([make_note()], compute_greeks=True)).greeks[0]
        assert not any(k.startswith("vega") for k in greeks)

    def test_ad_greeks_reach_bonds_too(self):
        """The AD method reports the same keys (Delta per market pillar rather than per
        sensitivity tenor), the bump's parallel Delta to first order and the same Theta."""
        request = bond_request([make_note()], compute_greeks=True)
        ad = replace(request, config=replace(request.config, greeks=GreeksConfig(method="AD")))
        bump, exact = price_portfolio(request).greeks[0], price_portfolio(ad).greeks[0]
        assert set(bump) == set(exact)
        assert delta(exact) == pytest.approx(delta(bump), rel=1e-3)
        assert float(exact["theta"]) == pytest.approx(float(bump["theta"]), rel=1e-12)


class TestBondsArePricedOnEveryPath:
    """Bonds enter the scenario cube like any other trade (I-24): no refusal, and no constant
    column (which gave VaR 0.00 and ES NaN)."""

    def test_a_bond_with_scenario_risk_on_prices(self):
        result = price_portfolio(bond_request([make_bill()], scenario_risk=True))
        assert result.scenario_risk_available is True
        cube = np.asarray(result.npv_cube)
        assert cube.shape[-1] == 1 and np.all(np.isfinite(cube))

    def test_the_bond_column_varies_across_paths(self):
        """Discounted on each path's curve, the bond's value varies with the rate."""
        cube = np.asarray(price_portfolio(bond_request([make_note()], scenario_risk=True)).npv_cube)
        assert np.std(cube[:, 1, 0]) > 0.0

    def test_a_bond_among_swaps_prices_on_the_paths(self):
        result = price_portfolio(bond_request([make_swap(), make_bill()], scenario_risk=True))
        assert np.asarray(result.npv_cube).shape[-1] == 2
        assert result.exposure is not None


class TestScenarioRiskAbsentNotZero:
    """With `scenario_risk=False`, risk is reported absent, never zero."""

    def test_exposure_is_absent_not_zero_filled(self):
        result = price_portfolio(bond_request([make_bill()]))
        assert result.exposure is None and result.trade_exposures == []

    def test_the_result_says_scenario_risk_was_unavailable(self):
        assert price_portfolio(bond_request([make_bill()])).scenario_risk_available is False

    def test_npv_cube_is_empty(self):
        assert np.asarray(price_portfolio(bond_request([make_bill()])).npv_cube).size == 0

    def test_scenario_risk_defaults_to_true(self):
        assert PortfolioRequest(market=make_market(), trades=[]).scenario_risk is True

    def test_scenario_risk_without_a_simulation_is_refused(self):
        request = dataclasses.replace(bond_request([make_bill()]), scenario_risk=True)
        with pytest.raises(ValueError, match="simulation"):
            price_portfolio(request)

"""
Bonds through `price_portfolio` itself (the pricer alone is tests/test_treasury_instrument.py).
A pricer passing its own tests says nothing about whether the portfolio path reaches it (the
lesson of I-01), so everything here goes through the public entry point.
"""
import numpy as np
import ORE
import pytest

from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import (
    BondConfig, CouponPeriod, ScenarioPricingNotSupported, price_bond_base,
)
from engine.portfolio import HULL_WHITE_CONFIG
from engine.portfolio.request import (
    DETERMINISTIC_ONLY_TYPES,
    PortfolioRequest,
    _compute_all_greeks,
    derive_maturity_pillars,
    price_portfolio,
)
from engine.simulation.market_model import (
    EquityConfig, RatesConfig, SimulationConfig, ZeroCurveConfig,
)

VALUATION = ORE.Date(2, 6, 2025)
FLAT_3PCT = ZeroCurveConfig(times=(0.0, 1.0, 2.0, 5.0, 10.0, 30.0), rates=(0.03,) * 6)


def _date(iso: str) -> ORE.Date:
    year, month, day = (int(p) for p in iso.split("-"))
    return ORE.Date(day, month, year)


def make_bill(face: float = 100_000.0, maturity: str = "2025-12-15") -> BondConfig:
    return BondConfig(
        face_amount=face, maturity_date=_date(maturity),
        evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
    )


NOTE_SCHEDULE = (
    CouponPeriod(_date("2024-12-15"), _date("2025-06-15"), _date("2025-06-15")),
    CouponPeriod(_date("2025-06-15"), _date("2025-12-15"), _date("2025-12-15")),
    CouponPeriod(_date("2025-12-15"), _date("2026-06-15"), _date("2026-06-15")),
)


def make_note(face: float = 100_000.0) -> BondConfig:
    return BondConfig(
        face_amount=face, maturity_date=_date("2026-06-15"),
        evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
        coupon_rate=0.04, coupon_schedule=NOTE_SCHEDULE,
        accrual_day_count="ACT/ACT (ICMA)",
    )


def make_market(num_scenarios: int = 64, trades=None) -> SimulationConfig:
    """A one-rate-factor `SimulationConfig`, shaped like the one in
    tests/test_portfolio_gap_fixes.py so bonds and swaps see the same market."""
    pillars = derive_maturity_pillars(list(trades or []), VALUATION)
    return SimulationConfig(
        time_grid=[0.0, 0.5, 1.0],
        scenarios=num_scenarios,
        equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0],
                              rate_mapping=[[0.0]]),
        rates=RatesConfig(
            initial_rates=[0.03], theta=[0.03], mean_reversion=[0.03],
            maturities=pillars, initial_zero_curves=[FLAT_3PCT],
        ),
        joint_covariance=[[0.04, 0.0], [0.0, 0.01 ** 2]],
    )


def bond_request(trades, compute_greeks: bool = False) -> PortfolioRequest:
    """A request containing bonds (so `scenario_risk=False`)."""
    trades = list(trades)
    return PortfolioRequest(
        market=make_market(trades=trades), trades=trades,
        compute_greeks=compute_greeks, scenario_risk=False, config=HULL_WHITE_CONFIG,
    )


def make_swap(notional: float = 1_000_000.0) -> SwapConfig:
    return SwapConfig(
        notional=notional, fixed_rate=0.03, payer=True,
        discount_curve_index=0, forward_curve_index=0,
        swap_tenor="5Y", evaluation_date=VALUATION,
    )


def swap_request(trades, **kwargs) -> PortfolioRequest:
    trades = list(trades)
    return PortfolioRequest(market=make_market(trades=trades), trades=trades, **kwargs, config=HULL_WHITE_CONFIG)


class TestBondsReachBaseNpv:
    """A bond's t=0 value reaches the portfolio total and the breakdown."""

    def test_a_bill_prices_through_price_portfolio(self):
        result = price_portfolio(bond_request([make_bill()]))
        assert result.base_npv == pytest.approx(price_bond_base(make_bill()))

    def test_base_npv_equals_the_sum_of_the_breakdown(self):
        """`base_npv` is the sum of the breakdown, with a bond in it."""
        result = price_portfolio(bond_request([make_bill(), make_note()]))
        assert result.base_npv == pytest.approx(sum(result.base_npv_per_trade))

    def test_per_trade_values_are_in_request_order(self):
        bill, note = make_bill(), make_note()
        result = price_portfolio(bond_request([bill, note]))
        assert result.base_npv_per_trade[0] == pytest.approx(price_bond_base(bill))
        assert result.base_npv_per_trade[1] == pytest.approx(price_bond_base(note))

    def test_a_short_bond_reduces_the_portfolio_total(self):
        long_only = price_portfolio(bond_request([make_bill()])).base_npv
        hedged = price_portfolio(
            bond_request([make_bill(), make_bill(face=-40_000.0)])
        ).base_npv
        assert hedged < long_only

    def test_a_bond_mixed_with_a_swap_prices_both(self):
        """A bond and a swap in one portfolio both price."""
        swap = make_swap()
        result = price_portfolio(bond_request([swap, make_bill()]))
        assert len(result.base_npv_per_trade) == 2
        # The bond's value does not depend on the swap.
        assert result.base_npv_per_trade[1] == pytest.approx(price_bond_base(make_bill()))


class TestBondGreeksReachThePortfolioPath:
    """I-01 regression class: a trade type without a branch in `_greeks_for_one_trade` is
    silently skipped. Each test fails with the `BondConfig` branch deleted."""

    def test_a_bond_has_a_greeks_entry_at_all(self):
        """A bond has a Greeks entry."""
        result = price_portfolio(bond_request([make_bill()], compute_greeks=True))
        assert result.greeks is not None
        assert 0 in result.greeks, (
            "the bond has NO greeks entry -- this is I-01 reproduced for a new "
            "instrument type: silently skipped, no error"
        )

    def test_every_bond_in_a_multi_bond_portfolio_gets_greeks(self):
        result = price_portfolio(
            bond_request([make_bill(), make_note(), make_bill(face=-50_000.0)],
                         compute_greeks=True)
        )
        assert set(result.greeks) == {0, 1, 2}

    def test_delta_gamma_and_theta_are_all_present(self):
        result = price_portfolio(bond_request([make_bill()], compute_greeks=True))
        assert {"delta", "gamma", "theta"} <= set(result.greeks[0])

    def test_a_bond_is_not_skipped_when_mixed_with_a_swap(self):
        """A bond mixed with swaps still gets Greeks (where a routing gap would hide)."""
        swap = make_swap()
        result = price_portfolio(bond_request([swap, make_bill()], compute_greeks=True))
        assert 1 in result.greeks, "the bond was skipped while the swap was not"
        assert {"delta", "gamma", "theta"} <= set(result.greeks[1])

    def test_delta_is_negative_for_a_long_bond(self):
        """A long bond's Delta is negative (a present but meaningless value would fail)."""
        result = price_portfolio(bond_request([make_bill()], compute_greeks=True))
        assert float(result.greeks[0]["delta"]) < 0

    def test_delta_flips_sign_for_a_short_bond(self):
        result = price_portfolio(
            bond_request([make_bill(face=-100_000.0)], compute_greeks=True)
        )
        assert float(result.greeks[0]["delta"]) > 0

    def test_gamma_is_positive_for_a_long_bond(self):
        """Convexity: a long bond's Gamma is positive."""
        result = price_portfolio(bond_request([make_note()], compute_greeks=True))
        assert float(result.greeks[0]["gamma"]) > 0

    def test_delta_is_not_silently_zero(self):
        """Delta is not a zero placeholder."""
        result = price_portfolio(bond_request([make_bill()], compute_greeks=True))
        assert abs(float(result.greeks[0]["delta"])) > 1e-6

    def test_theta_is_not_a_placeholder_zero(self):
        """Theta is not a zero placeholder (an implementation returning 0.0 once passed
        every other test): a bond one day nearer maturity discounts over a shorter time."""
        result = price_portfolio(bond_request([make_bill()], compute_greeks=True))
        assert abs(float(result.greeks[0]["theta"])) > 1e-9

    def test_theta_is_positive_for_a_discount_bond(self):
        """Pull to par: a discount bond's Theta is positive."""
        result = price_portfolio(bond_request([make_bill()], compute_greeks=True))
        assert float(result.greeks[0]["theta"]) > 0

    def test_theta_adds_back_a_coupon_paid_the_next_day(self):
        """I-39: ORE's Theta adds back the flows paid in (t, t + 1 day]. Valued the day before
        the 2025-12-15 coupon, the one-day reprice drops the $2,000 coupon; Theta must not
        (it was -1,988.83 at this setting in the register's measurement; red first: fails
        against the pre-fix code)."""
        from dataclasses import replace

        note = replace(make_note(), evaluation_date=_date("2025-12-14"))
        theta = float(price_portfolio(bond_request([note], compute_greeks=True)).greeks[0]["theta"])
        reprice = price_bond_base(replace(note, evaluation_date=note.evaluation_date + 1)) - price_bond_base(note)
        coupon = note.face_amount * note.coupon_rate * 0.5
        assert theta == pytest.approx(reprice + coupon, abs=1e-9)
        assert 0.0 < theta < 50.0

    def test_theta_matches_a_one_day_reprice(self):
        """Theta equals an independent one-day reprice when nothing pays in between."""
        from dataclasses import replace

        bill = make_bill()
        expected = price_bond_base(
            replace(bill, evaluation_date=bill.evaluation_date + 1)
        ) - price_bond_base(bill)
        result = price_portfolio(bond_request([bill], compute_greeks=True))
        assert float(result.greeks[0]["theta"]) == pytest.approx(expected, abs=1e-9)

    def test_gamma_matches_a_second_difference(self):
        """Gamma equals its second difference."""
        from engine.instruments.treasury import RATE_BUMP

        bill = make_bill()
        base = price_bond_base(bill)
        expected = (
            price_bond_base(bill, rate_shift=RATE_BUMP)
            - 2.0 * base
            + price_bond_base(bill, rate_shift=-RATE_BUMP)
        )
        result = price_portfolio(bond_request([bill], compute_greeks=True))
        assert float(result.greeks[0]["gamma"]) == pytest.approx(expected, abs=1e-12)

    def test_delta_matches_a_central_difference(self):
        """Delta equals its central difference."""
        from engine.instruments.treasury import RATE_BUMP

        bill = make_bill()
        expected = (
            price_bond_base(bill, rate_shift=RATE_BUMP)
            - price_bond_base(bill, rate_shift=-RATE_BUMP)
        ) / 2.0
        result = price_portfolio(bond_request([bill], compute_greeks=True))
        assert float(result.greeks[0]["delta"]) == pytest.approx(expected, abs=1e-12)

    def test_a_longer_bond_has_a_larger_delta(self):
        """A longer bond has a larger Delta: each trade gets its own value."""
        result = price_portfolio(
            bond_request([make_bill(maturity="2025-12-15"),
                          make_bill(maturity="2032-12-15")], compute_greeks=True)
        )
        short_delta = abs(float(result.greeks[0]["delta"]))
        long_delta = abs(float(result.greeks[1]["delta"]))
        assert long_delta > short_delta

    def test_delta_is_central_and_differs_slightly_from_the_one_sided_bump(self):
        """`_bond_greeks` uses a central difference; `treasury.rate_sensitivity` and the
        integration boundary's `rateSensitivity` use the one-sided bump agreed with TraderX.
        They agree within the curvature term and are not exactly equal."""
        from engine.instruments.treasury import rate_sensitivity

        bill = make_bill()
        central = float(price_portfolio(
            bond_request([bill], compute_greeks=True)
        ).greeks[0]["delta"])
        one_sided = rate_sensitivity(bill)

        assert central == pytest.approx(one_sided, abs=1e-3)
        assert central != one_sided

    def test_a_bond_maturing_tomorrow_does_not_crash_the_greeks(self):
        """Regression: for a bond maturing tomorrow, Theta's one-day move lands on maturity,
        which `BondConfig` refuses, and the whole Greeks call failed with a date the caller
        never gave."""
        tomorrow = BondConfig(
            face_amount=100_000.0, maturity_date=_date("2025-06-03"),
            evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
        )
        result = price_portfolio(bond_request([tomorrow], compute_greeks=True))
        assert 0 in result.greeks
        assert "delta" in result.greeks[0]
        assert "gamma" in result.greeks[0]

    def test_theta_is_omitted_not_zeroed_at_the_maturity_boundary(self):
        """At that boundary Theta is omitted (undefined), not reported as 0.0."""
        tomorrow = BondConfig(
            face_amount=100_000.0, maturity_date=_date("2025-06-03"),
            evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
        )
        result = price_portfolio(bond_request([tomorrow], compute_greeks=True))
        assert "theta" not in result.greeks[0]

    def test_theta_is_present_one_day_the_other_side_of_the_boundary(self):
        """A bond maturing in two days still has Theta, so the omission is exactly at the
        boundary."""
        two_days = BondConfig(
            face_amount=100_000.0, maturity_date=_date("2025-06-04"),
            evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
        )
        result = price_portfolio(bond_request([two_days], compute_greeks=True))
        assert "theta" in result.greeks[0]

    def test_no_vega_is_reported(self):
        """No Vega is reported (a fixed-coupon bond on a deterministic curve has no volatility
        input); omitted, not 0.0."""
        result = price_portfolio(bond_request([make_note()], compute_greeks=True))
        assert "vega" not in result.greeks[0]

    def test_greeks_survive_the_compute_all_greeks_helper_directly(self):
        """The helper directly, as well as through `price_portfolio`."""
        greeks = _compute_all_greeks([make_bill()], make_market())
        assert 0 in greeks and "delta" in greeks[0]


class TestScenarioRiskIsRefusedForBonds:
    """A bond cannot enter `npv_cube`; the refusal is explicit (a constant column would give
    VaR 0.00 and ES NaN; tests/test_treasury_instrument.py::TestScenarioPricingIsRefused)."""

    def test_a_bond_with_scenario_risk_on_is_refused(self):
        request = PortfolioRequest(
            market=make_market(), trades=[make_bill()], scenario_risk=True, config=HULL_WHITE_CONFIG,
        )
        with pytest.raises(ScenarioPricingNotSupported):
            price_portfolio(request)

    def test_the_refusal_names_the_offending_trade(self):
        """The refusal names the offending trade."""
        request = PortfolioRequest(
            market=make_market(),
            trades=[make_bill(), make_note(face=-7_777.0)],
            scenario_risk=True, config=HULL_WHITE_CONFIG,
        )
        with pytest.raises(ScenarioPricingNotSupported) as exc:
            price_portfolio(request)
        message = str(exc.value)
        assert "trade[0]" in message and "trade[1]" in message
        assert "-7777" in message.replace(",", "")

    def test_a_bond_hidden_among_swaps_is_still_refused(self):
        """One bond among swaps is still refused (not an opaque KeyError, not a fabricated
        column)."""
        swap = make_swap()
        request = PortfolioRequest(
            market=make_market(), trades=[swap, swap, make_bill()], scenario_risk=True, config=HULL_WHITE_CONFIG,
        )
        with pytest.raises(ScenarioPricingNotSupported):
            price_portfolio(request)

    def test_the_refusal_is_not_a_bare_key_error(self):
        """The failure is the named refusal, not the old bare `KeyError: <class BondConfig>`."""
        request = PortfolioRequest(
            market=make_market(), trades=[make_bill()], scenario_risk=True, config=HULL_WHITE_CONFIG,
        )
        with pytest.raises(ScenarioPricingNotSupported):
            price_portfolio(request)
        # Not a KeyError.
        try:
            price_portfolio(request)
        except KeyError:  # pragma: no cover - fails loudly if it regresses
            pytest.fail("refusal regressed to a bare KeyError")
        except ScenarioPricingNotSupported:
            pass

    def test_bond_config_is_registered_as_deterministic_only(self):
        assert BondConfig in DETERMINISTIC_ONLY_TYPES


class TestScenarioRiskAbsentNotZero:
    """With `scenario_risk=False`, risk is reported absent, never zero."""

    def test_exposure_is_absent_not_zero_filled(self):
        result = price_portfolio(bond_request([make_bill()]))
        assert result.exposure is None and result.trade_exposures == [], (
            "an absent exposure asserts nothing; a zero exposure would assert "
            "a measured absence of risk"
        )

    def test_the_result_says_scenario_risk_was_unavailable(self):
        result = price_portfolio(bond_request([make_bill()]))
        assert result.scenario_risk_available is False

    def test_npv_cube_is_empty(self):
        result = price_portfolio(bond_request([make_bill()]))
        assert np.asarray(result.npv_cube).size == 0

    def test_a_normal_run_still_reports_risk_available(self):
        """The flag distinguishes the two cases."""
        swap = make_swap()
        result = price_portfolio(swap_request([swap]))
        assert result.scenario_risk_available is True
        assert result.exposure is not None


class TestExistingBehaviourUnchanged:
    """Portfolios without bonds behave as before."""

    def test_a_swap_only_portfolio_still_produces_exposure(self):
        swap = make_swap()
        result = price_portfolio(swap_request([swap]))
        assert "PFE_95" in result.exposure.pfe
        assert np.asarray(result.npv_cube).shape[-1] == 1

    def test_scenario_risk_defaults_to_true(self):
        """`scenario_risk` defaults to True."""
        assert PortfolioRequest(market=make_market(), trades=[], config=HULL_WHITE_CONFIG).scenario_risk is True

    def test_swap_greeks_are_unaffected(self):
        """Swap Greeks are unchanged. A swap reports per-pillar vectors per curve
        (`discount_delta`/`forward_delta`), unlike a bond's scalar `delta`."""
        swap = make_swap()
        result = price_portfolio(swap_request([swap], compute_greeks=True))
        assert 0 in result.greeks
        assert "discount_delta" in result.greeks[0]
        assert "theta" in result.greeks[0]

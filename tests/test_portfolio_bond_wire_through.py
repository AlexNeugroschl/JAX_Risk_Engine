"""
W1.5 -- the bond wire-through into `engine.portfolio.request`.

Separate from `tests/test_treasury_instrument.py` (which tests the pricer in
isolation) because these tests drive **`price_portfolio` itself**. That
distinction is the whole lesson of I-01: `engine.risk.greeks.
swap_delta_gamma` was fully implemented and unit-tested while the portfolio
path silently returned no Greeks for any swap. A pricer passing its own
tests proves nothing about whether the orchestration reaches it.

So every test here goes through the public entry point, never through a
helper.
"""
import numpy as np
import ORE
import pytest

from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import (
    BondConfig, CouponPeriod, ScenarioPricingNotSupported, price_bond_base,
)
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
    """A one-rate-factor `SimulationConfig`.

    Shaped to match `tests/test_portfolio_gap_fixes.py::_sim_config` -- the
    same fixture the I-01 tests use, so a bond is exercised against exactly
    the market a swap is.
    """
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
    """A bond-containing request, which MUST be `scenario_risk=False`."""
    trades = list(trades)
    return PortfolioRequest(
        market=make_market(trades=trades), trades=trades,
        compute_greeks=compute_greeks, scenario_risk=False,
    )


def make_swap(notional: float = 1_000_000.0) -> SwapConfig:
    return SwapConfig(
        notional=notional, fixed_rate=0.03, payer=True,
        discount_curve_index=0, forward_curve_index=0,
        swap_tenor="5Y", evaluation_date=VALUATION,
    )


def swap_request(trades, **kwargs) -> PortfolioRequest:
    trades = list(trades)
    return PortfolioRequest(market=make_market(trades=trades), trades=trades, **kwargs)


class TestBondsReachBaseNpv:
    """A bond's t=0 value reaches the portfolio total and the breakdown."""

    def test_a_bill_prices_through_price_portfolio(self):
        result = price_portfolio(bond_request([make_bill()]))
        assert result.base_npv == pytest.approx(price_bond_base(make_bill()))

    def test_base_npv_equals_the_sum_of_the_breakdown(self):
        """The invariant `PortfolioResult` promises, now with a bond in it."""
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
        """The mixed-portfolio case: routing must not drop either type."""
        swap = make_swap()
        result = price_portfolio(bond_request([swap, make_bill()]))
        assert len(result.base_npv_per_trade) == 2
        # The bond's own value is unchanged by the swap's presence.
        assert result.base_npv_per_trade[1] == pytest.approx(price_bond_base(make_bill()))


class TestBondGreeksReachThePortfolioPath:
    """**The I-01 regression class** -- plan §W1.5's explicit warning.

    A new instrument type reaching `_compute_all_greeks` without its branch
    is silently skipped: no Greeks, no error. That is exactly how swaps lost
    theirs, and the test that pinned the bug even called the skip "by
    design".

    Every test in this class was verified to FAIL with the `BondConfig`
    branch of `_greeks_for_one_trade` deleted (working rule 3).
    """

    def test_a_bond_has_a_greeks_entry_at_all(self):
        """The bare I-01 assertion: present, not silently absent."""
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
        """The mixed case is where a routing gap actually hides.

        A bond-only portfolio failing is obvious; a bond quietly missing
        from a portfolio that also has swaps is not.
        """
        swap = make_swap()
        result = price_portfolio(bond_request([swap, make_bill()], compute_greeks=True))
        assert 1 in result.greeks, "the bond was skipped while the swap was not"
        assert {"delta", "gamma", "theta"} <= set(result.greeks[1])

    def test_delta_is_negative_for_a_long_bond(self):
        """Not just present -- directionally correct.

        A key that exists but carries a meaningless number would pass a
        presence check while being useless, so the sign is asserted too.
        """
        result = price_portfolio(bond_request([make_bill()], compute_greeks=True))
        assert float(result.greeks[0]["delta"]) < 0

    def test_delta_flips_sign_for_a_short_bond(self):
        result = price_portfolio(
            bond_request([make_bill(face=-100_000.0)], compute_greeks=True)
        )
        assert float(result.greeks[0]["delta"]) > 0

    def test_gamma_is_positive_for_a_long_bond(self):
        """Convexity: a bond's price/yield curve bends upward."""
        result = price_portfolio(bond_request([make_note()], compute_greeks=True))
        assert float(result.greeks[0]["gamma"]) > 0

    def test_delta_is_not_silently_zero(self):
        """A zero-filled placeholder would pass every presence check.

        This is the shape of the W1.6 'decorative field' bug: parsed,
        validated, and then never used. A Greek reported as exactly 0.0 for
        a $100k bond is not a measurement.
        """
        result = price_portfolio(bond_request([make_bill()], compute_greeks=True))
        assert abs(float(result.greeks[0]["delta"])) > 1e-6

    def test_theta_is_not_a_placeholder_zero(self):
        """**Found by a wrong implementation, not by design.**

        A first version of this suite asserted only that `theta` was
        *present*. An implementation returning `theta = 0.0` -- a literal
        placeholder -- passed 27 of 28 tests, which is the same bug of
        omission as W1.6's decorative `fractionDecimals`: the field is
        there, validated, and never computed.

        A bond one day nearer maturity discounts over a shorter year
        fraction, so a real theta is strictly non-zero. Verified to fail
        against the placeholder implementation.
        """
        result = price_portfolio(bond_request([make_bill()], compute_greeks=True))
        assert abs(float(result.greeks[0]["theta"])) > 1e-9

    def test_theta_is_positive_for_a_discount_bond(self):
        """Pull to par: a discount bond gains value as maturity approaches.

        Directional, so a placeholder that returned some arbitrary non-zero
        constant would still fail.
        """
        result = price_portfolio(bond_request([make_bill()], compute_greeks=True))
        assert float(result.greeks[0]["theta"]) > 0

    def test_theta_matches_a_one_day_reprice(self):
        """The strongest form: theta must equal what it claims to measure.

        Pins theta to an INDEPENDENT recomputation rather than to a sign or
        a magnitude, so no placeholder and no drifted formula can satisfy
        it.
        """
        from dataclasses import replace

        bill = make_bill()
        expected = price_bond_base(
            replace(bill, evaluation_date=bill.evaluation_date + 1)
        ) - price_bond_base(bill)
        result = price_portfolio(bond_request([bill], compute_greeks=True))
        assert float(result.greeks[0]["theta"]) == pytest.approx(expected, abs=1e-9)

    def test_gamma_matches_a_second_difference(self):
        """Same treatment for gamma: pinned to its own definition."""
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
        """And delta, so all three Greeks are pinned to a definition."""
        from engine.instruments.treasury import RATE_BUMP

        bill = make_bill()
        expected = (
            price_bond_base(bill, rate_shift=RATE_BUMP)
            - price_bond_base(bill, rate_shift=-RATE_BUMP)
        ) / 2.0
        result = price_portfolio(bond_request([bill], compute_greeks=True))
        assert float(result.greeks[0]["delta"]) == pytest.approx(expected, abs=1e-12)

    def test_a_longer_bond_has_a_larger_delta(self):
        """Duration ordering, through the portfolio path.

        Pins that each trade gets its OWN delta rather than one shared
        number broadcast across the portfolio.
        """
        result = price_portfolio(
            bond_request([make_bill(maturity="2025-12-15"),
                          make_bill(maturity="2032-12-15")], compute_greeks=True)
        )
        short_delta = abs(float(result.greeks[0]["delta"]))
        long_delta = abs(float(result.greeks[1]["delta"]))
        assert long_delta > short_delta

    def test_delta_is_central_and_differs_slightly_from_the_one_sided_bump(self):
        """The two sensitivity estimators are **deliberately** not identical.

        `_bond_greeks` uses a central difference (second-order accurate,
        symmetric); `engine.instruments.treasury.rate_sensitivity` and the
        integration boundary's `rateSensitivity` use the one-sided bump
        TraderX agreed to reconcile against.

        Pinned in both directions: they must agree to within the curvature
        term (so neither has drifted into being wrong), and they must NOT be
        exactly equal (so nobody has quietly made this one-sided to force an
        appearance of agreement).
        """
        from engine.instruments.treasury import rate_sensitivity

        bill = make_bill()
        central = float(price_portfolio(
            bond_request([bill], compute_greeks=True)
        ).greeks[0]["delta"])
        one_sided = rate_sensitivity(bill)

        assert central == pytest.approx(one_sided, abs=1e-3)
        assert central != one_sided

    def test_a_bond_maturing_tomorrow_does_not_crash_the_greeks(self):
        """**A real edge-case bug, found by checking rather than assuming.**

        Theta advances the evaluation date by one day. For a bond maturing
        *tomorrow* that lands exactly on maturity -- a state `BondConfig`
        refuses to construct. The whole Greeks call therefore died with a
        `BondPricingError` saying "matured bond ... 2025-06-03 is not after
        2025-06-03", naming a date the caller never supplied, for a bond
        that prices perfectly well today.

        Verified to fail against the pre-fix `_bond_greeks`.
        """
        tomorrow = BondConfig(
            face_amount=100_000.0, maturity_date=_date("2025-06-03"),
            evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
        )
        result = price_portfolio(bond_request([tomorrow], compute_greeks=True))
        assert 0 in result.greeks
        assert "delta" in result.greeks[0]
        assert "gamma" in result.greeks[0]

    def test_theta_is_omitted_not_zeroed_at_the_maturity_boundary(self):
        """Omitted, not 0.0.

        There is no next day on which this instrument still exists, so
        theta is undefined rather than nil. Reporting 0.0 would assert a
        measured absence of time decay on the one bond that decays fastest.
        """
        tomorrow = BondConfig(
            face_amount=100_000.0, maturity_date=_date("2025-06-03"),
            evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
        )
        result = price_portfolio(bond_request([tomorrow], compute_greeks=True))
        assert "theta" not in result.greeks[0]

    def test_theta_is_present_one_day_the_other_side_of_the_boundary(self):
        """The boundary must be exactly where it claims to be.

        A bond maturing in TWO days still has a next day to be repriced on,
        so theta is defined. Without this, omitting theta unconditionally
        would also pass the test above.
        """
        two_days = BondConfig(
            face_amount=100_000.0, maturity_date=_date("2025-06-04"),
            evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
        )
        result = price_portfolio(bond_request([two_days], compute_greeks=True))
        assert "theta" in result.greeks[0]

    def test_no_vega_is_reported(self):
        """Omitted, not zero.

        A fixed-coupon bond off a deterministic curve has no volatility
        input. Reporting `vega: 0.0` would assert 'measured, and found to be
        nil' about a quantity that is not defined here.
        """
        result = price_portfolio(bond_request([make_note()], compute_greeks=True))
        assert "vega" not in result.greeks[0]

    def test_greeks_survive_the_compute_all_greeks_helper_directly(self):
        """Belt-and-braces: the helper itself, not only via price_portfolio."""
        greeks = _compute_all_greeks([make_bill()], make_market())
        assert 0 in greeks and "delta" in greeks[0]


class TestScenarioRiskIsRefusedForBonds:
    """A bond cannot enter `npv_cube`, and the refusal is explicit.

    The alternative -- broadcasting a constant column -- produces VaR 0.00
    and ES NaN, measured in
    `tests/test_treasury_instrument.py::TestScenarioPricingIsRefused`.
    """

    def test_a_bond_with_scenario_risk_on_is_refused(self):
        request = PortfolioRequest(
            market=make_market(), trades=[make_bill()], scenario_risk=True,
        )
        with pytest.raises(ScenarioPricingNotSupported):
            price_portfolio(request)

    def test_the_refusal_names_the_offending_trade(self):
        """So a 200-trade portfolio says WHICH row, not just 'a bond'."""
        request = PortfolioRequest(
            market=make_market(),
            trades=[make_bill(), make_note(face=-7_777.0)],
            scenario_risk=True,
        )
        with pytest.raises(ScenarioPricingNotSupported) as exc:
            price_portfolio(request)
        message = str(exc.value)
        assert "trade[0]" in message and "trade[1]" in message
        assert "-7777" in message.replace(",", "")

    def test_a_bond_hidden_among_swaps_is_still_refused(self):
        """The dangerous case: one bond in an otherwise priceable portfolio.

        Without the explicit guard this would either raise an opaque
        KeyError deep in `_price_by_type`, or -- far worse -- be given a
        fabricated column.
        """
        swap = make_swap()
        request = PortfolioRequest(
            market=make_market(), trades=[swap, swap, make_bill()], scenario_risk=True,
        )
        with pytest.raises(ScenarioPricingNotSupported):
            price_portfolio(request)

    def test_the_refusal_is_not_a_bare_key_error(self):
        """Regression guard on the failure MODE, not just the failure.

        Before the explicit check, an unrouted type died on
        `groups[type(cfg)]` -- a `KeyError: <class BondConfig>` that names
        no trade and suggests no fix.
        """
        request = PortfolioRequest(
            market=make_market(), trades=[make_bill()], scenario_risk=True,
        )
        with pytest.raises(ScenarioPricingNotSupported):
            price_portfolio(request)
        # And specifically NOT a KeyError.
        try:
            price_portfolio(request)
        except KeyError:  # pragma: no cover - fails loudly if it regresses
            pytest.fail("refusal regressed to a bare KeyError")
        except ScenarioPricingNotSupported:
            pass

    def test_bond_config_is_registered_as_deterministic_only(self):
        assert BondConfig in DETERMINISTIC_ONLY_TYPES


class TestScenarioRiskAbsentNotZero:
    """`scenario_risk=False` reports risk as ABSENT, never as zero."""

    def test_risk_is_empty_not_zero_filled(self):
        result = price_portfolio(bond_request([make_bill()]))
        assert result.risk == {}, (
            "an empty dict asserts nothing; a VaR of 0.00 would assert a "
            "measured absence of risk"
        )

    def test_the_result_says_scenario_risk_was_unavailable(self):
        result = price_portfolio(bond_request([make_bill()]))
        assert result.scenario_risk_available is False

    def test_npv_cube_is_empty(self):
        result = price_portfolio(bond_request([make_bill()]))
        assert np.asarray(result.npv_cube).size == 0

    def test_a_normal_run_still_reports_risk_available(self):
        """The flag must distinguish the two cases, not always be False."""
        swap = make_swap()
        result = price_portfolio(swap_request([swap]))
        assert result.scenario_risk_available is True
        assert result.risk != {}


class TestExistingBehaviourUnchanged:
    """W1.5 is additive. A portfolio with no bond behaves exactly as before.

    The user's instruction was to be careful changing working systems;
    these are the tests that check the change was in fact additive rather
    than merely intended to be.
    """

    def test_a_swap_only_portfolio_still_produces_risk(self):
        swap = make_swap()
        result = price_portfolio(swap_request([swap]))
        assert "VaR_95" in result.risk
        assert np.asarray(result.npv_cube).shape[-1] == 1

    def test_scenario_risk_defaults_to_true(self):
        """The new flag must not change any existing caller's behaviour."""
        assert PortfolioRequest(market=make_market(), trades=[]).scenario_risk is True

    def test_swap_greeks_are_unaffected(self):
        """I-01's own fix must survive W1.5.

        A swap's Greeks are named per-curve and are per-pillar VECTORS
        (`discount_delta`/`forward_delta`), unlike a bond's scalar `delta`
        off its single own curve. Asserting the swap's own key names here
        rather than a shared one is deliberate: the two instrument types
        genuinely report different shapes, and a test that papered over
        that would be pinning a convention neither side implements.
        """
        swap = make_swap()
        result = price_portfolio(swap_request([swap], compute_greeks=True))
        assert 0 in result.greeks
        assert "discount_delta" in result.greeks[0]
        assert "theta" in result.greeks[0]

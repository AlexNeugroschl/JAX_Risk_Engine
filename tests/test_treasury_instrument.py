"""
`engine.instruments.treasury` (bills and notes) in isolation; the portfolio path is
tests/test_portfolio_bond_wire_through.py.

`TestScenarioPricingIsRefused` pins the scenario refusal and measures the alternative: a
constant `npv_cube` column gives VaR 0.00 and ES NaN, a position reported as risk-measured
whose risk was never modelled.
"""
import math

import numpy as np
import ORE
import pytest
import jax.numpy as jnp

from engine.instruments.treasury import (
    RATE_BUMP,
    BondConfig,
    BondPricingError,
    CouponPeriod,
    ScenarioPricingNotSupported,
    accrued_interest,
    clean_npv_of,
    price_bond_base,
    price_bond_scenarios,
    rate_sensitivity,
)
from engine.simulation.market_model import ZeroCurveConfig

VALUATION = ORE.Date(2, 6, 2025)
FLAT_3PCT = ZeroCurveConfig(times=(0.0, 1.0, 2.0, 5.0, 10.0, 30.0), rates=(0.03,) * 6)


def _date(iso: str) -> ORE.Date:
    year, month, day = (int(p) for p in iso.split("-"))
    return ORE.Date(day, month, year)


def make_bill(face: float = 100_000.0, maturity: str = "2025-12-15",
              curve: ZeroCurveConfig = FLAT_3PCT) -> BondConfig:
    return BondConfig(
        face_amount=face, maturity_date=_date(maturity),
        evaluation_date=VALUATION, initial_zero_curve=curve,
    )


#: The three-period semiannual schedule of the note fixture.
NOTE_SCHEDULE = (
    CouponPeriod(_date("2024-12-15"), _date("2025-06-15"), _date("2025-06-15")),
    CouponPeriod(_date("2025-06-15"), _date("2025-12-15"), _date("2025-12-15")),
    CouponPeriod(_date("2025-12-15"), _date("2026-06-15"), _date("2026-06-15")),
)


def make_note(face: float = 100_000.0, curve: ZeroCurveConfig = FLAT_3PCT) -> BondConfig:
    return BondConfig(
        face_amount=face, maturity_date=_date("2026-06-15"),
        evaluation_date=VALUATION, initial_zero_curve=curve,
        coupon_rate=0.04, coupon_schedule=NOTE_SCHEDULE,
        accrual_day_count="ACT/ACT (ICMA)",
    )


class TestBillPricing:
    """The zero-coupon single-cashflow case."""

    def test_prices_to_the_closed_form_value(self):
        bill = make_bill()
        # NPV = face x exp(-r x t), ACT/365 from 2025-06-02 to 2025-12-15.
        t = ORE.Actual365Fixed().yearFraction(VALUATION, _date("2025-12-15"))
        assert price_bond_base(bill) == pytest.approx(100_000.0 * math.exp(-0.03 * t))

    def test_is_identified_as_a_bill(self):
        assert make_bill().is_bill is True
        assert make_note().is_bill is False

    def test_a_bill_has_no_accrued_interest(self):
        # Structural zero: no coupons to accrue.
        assert accrued_interest(make_bill()) == 0.0

    def test_discounts_below_par(self):
        assert price_bond_base(make_bill()) < 100_000.0

    def test_a_short_position_is_negative(self):
        """A negative face gives a negative NPV directly (no second sign factor)."""
        long_npv = price_bond_base(make_bill(face=100_000.0))
        short_npv = price_bond_base(make_bill(face=-100_000.0))
        assert short_npv < 0
        assert short_npv == pytest.approx(-long_npv)

    def test_redemption_fraction_scales_the_value(self):
        par = price_bond_base(make_bill())
        half = price_bond_base(BondConfig(
            face_amount=100_000.0, maturity_date=_date("2025-12-15"),
            evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
            redemption_fraction=0.5,
        ))
        assert half == pytest.approx(par * 0.5)


class TestNotePricing:
    """The coupon-bearing scheduled case."""

    def test_prices_above_a_comparable_bill(self):
        # A 4% note is worth more than a zero-coupon claim on the same face and maturity.
        note = make_note()
        bill_same_maturity = make_bill(maturity="2026-06-15")
        assert price_bond_base(note) > price_bond_base(bill_same_maturity)

    def test_accrued_interest_uses_act_act_icma(self):
        """ACT/ACT (ICMA): the 2024-12-15 -> 2025-06-15 coupon accrues exactly 0.5 however
        many days it has, so accrual to 2025-06-02 is 169/182 of it:

            0.04 x (169/182) x 0.5 x 100,000 = 1,857.142857
        """
        assert accrued_interest(make_note()) == pytest.approx(1857.142857, abs=1e-5)

    def test_accrued_is_position_signed(self):
        assert accrued_interest(make_note(face=-100_000.0)) == pytest.approx(-1857.142857, abs=1e-5)

    def test_clean_is_dirty_minus_accrued(self):
        note = make_note()
        assert clean_npv_of(note) == pytest.approx(
            price_bond_base(note) - accrued_interest(note)
        )

    def test_price_bond_base_returns_the_DIRTY_value(self):
        """`price_bond_base` is dirty: the sum of discounted cashflows with no accrued
        deduction, as `engine.integration.note.NotePrice.npv`. (A clean-returning version
        once passed all but one test.)"""
        note = make_note()
        # Independently: coupons plus redemption, discounted; no accrued term.
        day_count = ORE.ActualActual(ORE.ActualActual.ISMA)
        expected_per_unit = 0.0
        for period in NOTE_SCHEDULE:
            payment = period.payment()
            if payment <= VALUATION:
                continue
            accrual = day_count.yearFraction(
                period.start_date, period.end_date, period.start_date, period.end_date,
            )
            t = ORE.Actual365Fixed().yearFraction(VALUATION, payment)
            expected_per_unit += 0.04 * accrual * math.exp(-0.03 * t)
        t_mat = ORE.Actual365Fixed().yearFraction(VALUATION, _date("2026-06-15"))
        expected_per_unit += 1.0 * math.exp(-0.03 * t_mat)

        assert price_bond_base(note) == pytest.approx(100_000.0 * expected_per_unit, abs=1e-6)

    def test_dirty_exceeds_clean_by_exactly_the_accrued(self):
        """Dirty exceeds clean by exactly the accrued (a clean `price_bond_base` would make
        them differ by twice that)."""
        note = make_note()
        assert price_bond_base(note) - clean_npv_of(note) == pytest.approx(
            accrued_interest(note), abs=1e-9
        )
        assert price_bond_base(note) > clean_npv_of(note)

    def test_clean_and_dirty_differ_by_a_material_amount(self):
        """Clean and dirty differ materially (~$1,857 on $100k), so a silently clean NPV
        would be caught."""
        note = make_note()
        assert abs(price_bond_base(note) - clean_npv_of(note)) > 1000.0

    def test_a_coupon_already_paid_is_excluded(self):
        """A coupon paid before the valuation date is excluded: the 2025-06-15 payment is
        included at 2025-06-02 and dropped after it."""
        before = price_bond_base(make_note())
        after_cfg = BondConfig(
            face_amount=100_000.0, maturity_date=_date("2026-06-15"),
            evaluation_date=_date("2025-06-20"), initial_zero_curve=FLAT_3PCT,
            coupon_rate=0.04, coupon_schedule=NOTE_SCHEDULE,
            accrual_day_count="ACT/ACT (ICMA)",
        )
        # Two fewer weeks of discounting, but the ~2,000 coupon is gone.
        assert price_bond_base(after_cfg) < before


class TestAgreesWithTheIntegrationPricers:
    """The instrument pricer and the integration pricers agree to the cent on the same
    instrument, so their duplicated discounting (integration sits above instruments and is
    not imported here) cannot drift apart unnoticed."""

    def _profile_and_curve(self):
        from engine.integration.market_inputs import ASSUMED_PROFILES
        profile = ASSUMED_PROFILES["flat-3pct-v1"]
        return profile, ZeroCurveConfig(times=profile.times, rates=profile.rates())

    def test_bill_matches_the_integration_bill_pricer(self):
        from engine.integration.bill import price_bill
        from engine.integration.terms import TermsEntry

        profile, curve = self._profile_and_curve()
        entry = TermsEntry(
            instrument_type="BILL",
            terms={"couponFrequency": "NONE", "maturityDate": "2025-12-15",
                   "redemptionFraction": 1.0},
            missing_terms=(), provenance={}, identity={},
        )
        integration_npv = price_bill(entry, 100_000.0, VALUATION, profile).npv
        instrument_npv = price_bond_base(make_bill(curve=curve))
        assert instrument_npv == pytest.approx(integration_npv, abs=1e-6)

    def test_note_matches_the_integration_note_pricer(self):
        from engine.integration.note import price_note
        from engine.integration.terms import TermsEntry

        profile, curve = self._profile_and_curve()
        schedule = [
            {"startDate": "2024-12-15", "endDate": "2025-06-15", "paymentDate": "2025-06-15"},
            {"startDate": "2025-06-15", "endDate": "2025-12-15", "paymentDate": "2025-12-15"},
            {"startDate": "2025-12-15", "endDate": "2026-06-15", "paymentDate": "2026-06-15"},
        ]
        entry = TermsEntry(
            instrument_type="NOTE",
            terms={"couponFrequency": "SEMIANNUAL", "maturityDate": "2026-06-15",
                   "couponRatePercent": 4.0, "redemptionFraction": 1.0,
                   "dayCount": "ACT/ACT (ICMA)", "settlementDays": 0,
                   "schedule": schedule},
            missing_terms=(), provenance={}, identity={},
        )
        integration = price_note(entry, 100_000.0, VALUATION, profile)
        instrument_npv = price_bond_base(make_note(curve=curve))
        assert instrument_npv == pytest.approx(integration.dirty_npv, abs=1e-6)

    def test_accrued_matches_the_integration_recomputed_path(self):
        """ACT/ACT (ICMA) accrual against the integration pricer's recomputed path (a direct
        caller has no exported accrued figure)."""
        from engine.integration.note import price_note
        from engine.integration.terms import TermsEntry

        profile, curve = self._profile_and_curve()
        schedule = [
            {"startDate": "2024-12-15", "endDate": "2025-06-15", "paymentDate": "2025-06-15"},
            {"startDate": "2025-06-15", "endDate": "2025-12-15", "paymentDate": "2025-12-15"},
            {"startDate": "2025-12-15", "endDate": "2026-06-15", "paymentDate": "2026-06-15"},
        ]
        entry = TermsEntry(
            instrument_type="NOTE",
            terms={"couponFrequency": "SEMIANNUAL", "maturityDate": "2026-06-15",
                   "couponRatePercent": 4.0, "redemptionFraction": 1.0,
                   "dayCount": "ACT/ACT (ICMA)", "settlementDays": 0,
                   "schedule": schedule},
            missing_terms=(), provenance={}, identity={},
        )
        integration = price_note(entry, 100_000.0, VALUATION, profile)
        expected = integration.accrual.recomputed_fraction * 100_000.0
        assert accrued_interest(make_note(curve=curve)) == pytest.approx(expected, abs=1e-6)


class TestRefusals:
    """What a bond refuses rather than approximates."""

    def test_a_matured_bond_is_refused(self):
        with pytest.raises(BondPricingError, match="not after"):
            make_bill(maturity="2025-01-01")

    def test_a_bond_maturing_on_the_valuation_date_is_refused(self):
        with pytest.raises(BondPricingError, match="not after"):
            BondConfig(face_amount=1.0, maturity_date=VALUATION,
                       evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT)

    def test_a_schedule_with_a_zero_coupon_rate_is_refused(self):
        """A schedule with a zero coupon rate is a contradiction, refused rather than priced
        as a bill."""
        with pytest.raises(BondPricingError, match="contradiction"):
            BondConfig(
                face_amount=100_000.0, maturity_date=_date("2026-06-15"),
                evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
                coupon_rate=0.0, coupon_schedule=NOTE_SCHEDULE,
            )

    def test_a_coupon_rate_with_no_schedule_is_refused(self):
        with pytest.raises(BondPricingError, match="no coupon_schedule"):
            BondConfig(
                face_amount=100_000.0, maturity_date=_date("2026-06-15"),
                evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
                coupon_rate=0.04,
            )

    def test_an_unsupported_accrual_day_count_is_refused(self):
        """An unsupported accrual day count (ACT/360) is refused, never defaulted."""
        with pytest.raises(BondPricingError):
            BondConfig(
                face_amount=100_000.0, maturity_date=_date("2026-06-15"),
                evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
                coupon_rate=0.04, coupon_schedule=NOTE_SCHEDULE,
                accrual_day_count="ACT/360",
            )

    def test_a_gapped_schedule_is_refused(self):
        gapped = (
            CouponPeriod(_date("2024-12-15"), _date("2025-06-15")),
            # Gap: starts a month after the previous period ended.
            CouponPeriod(_date("2025-07-15"), _date("2026-06-15")),
        )
        with pytest.raises(BondPricingError, match="gap or overlap"):
            BondConfig(
                face_amount=100_000.0, maturity_date=_date("2026-06-15"),
                evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
                coupon_rate=0.04, coupon_schedule=gapped,
            )

    def test_a_schedule_not_ending_at_maturity_is_refused(self):
        with pytest.raises(BondPricingError, match="schedule ends"):
            BondConfig(
                face_amount=100_000.0, maturity_date=_date("2026-12-15"),
                evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT,
                coupon_rate=0.04, coupon_schedule=NOTE_SCHEDULE,
            )

    def test_a_zero_length_period_is_refused(self):
        bad = (CouponPeriod(_date("2025-06-15"), _date("2025-06-15")),)
        with pytest.raises(BondPricingError, match="no length"):
            BondConfig(
                face_amount=100_000.0, maturity_date=_date("2025-06-15"),
                evaluation_date=_date("2025-01-01"), initial_zero_curve=FLAT_3PCT,
                coupon_rate=0.04, coupon_schedule=bad,
            )


class TestRateSensitivity:
    """The 1bp bumped revaluation."""

    def test_is_negative_for_a_long_position(self):
        # Price falls as rates rise.
        assert rate_sensitivity(make_bill()) < 0

    def test_is_positive_for_a_short_position(self):
        assert rate_sensitivity(make_bill(face=-100_000.0)) > 0

    def test_a_longer_bond_is_more_sensitive(self):
        """A longer bond moves more for the same bump."""
        short = abs(rate_sensitivity(make_bill(maturity="2025-12-15")))
        long_ = abs(rate_sensitivity(make_bill(maturity="2030-12-15")))
        assert long_ > short

    def test_matches_a_manual_reprice(self):
        bill = make_bill()
        expected = price_bond_base(bill, rate_shift=RATE_BUMP) - price_bond_base(bill)
        assert rate_sensitivity(bill) == pytest.approx(expected)


class TestCurveInterpolation:
    """A bond is priced off its own curve, including non-flat ones."""

    def test_a_non_flat_curve_is_interpolated(self):
        steep = ZeroCurveConfig(times=(0.0, 1.0, 10.0), rates=(0.01, 0.02, 0.05))
        flat = ZeroCurveConfig(times=(0.0, 1.0, 10.0), rates=(0.02,) * 3)
        # At ~0.5y the steep curve is near 1.5%, below the flat 2%, so the bill is worth more.
        assert price_bond_base(make_bill(curve=steep)) > price_bond_base(make_bill(curve=flat))

    def test_a_higher_curve_gives_a_lower_price(self):
        low = ZeroCurveConfig(times=(0.0, 30.0), rates=(0.01, 0.01))
        high = ZeroCurveConfig(times=(0.0, 30.0), rates=(0.06, 0.06))
        assert price_bond_base(make_bill(curve=high)) < price_bond_base(make_bill(curve=low))

    def test_beyond_the_last_pillar_holds_flat(self):
        """Beyond the last pillar the rate is held flat."""
        curve = ZeroCurveConfig(times=(0.0, 1.0), rates=(0.03, 0.04))
        far = make_bill(maturity="2035-12-15", curve=curve)
        t = ORE.Actual365Fixed().yearFraction(VALUATION, _date("2035-12-15"))
        # Held at the last pillar's 4%.
        assert price_bond_base(far) == pytest.approx(100_000.0 * math.exp(-0.04 * t))

    def test_a_negative_rate_curve_prices_above_par(self):
        """At a negative zero rate the discount factor exceeds 1 and a bill is worth more
        than face (pinned so a zero floor would fail)."""
        negative = ZeroCurveConfig(times=(0.0, 1.0, 30.0), rates=(-0.005,) * 3)
        assert price_bond_base(make_bill(curve=negative)) > 100_000.0

    def test_a_long_dated_bill_discounts_heavily(self):
        """A 30-year zero is worth well under half its face at 3%."""
        far = make_bill(maturity="2055-06-02")
        assert 0.0 < price_bond_base(far) < 50_000.0

    def test_a_zero_face_position_is_worth_zero(self):
        """A zero-face position is worth zero and does not raise."""
        assert price_bond_base(make_bill(face=0.0)) == 0.0

    def test_an_empty_curve_is_refused(self):
        empty = ZeroCurveConfig(times=(), rates=())
        with pytest.raises(BondPricingError, match="no pillars"):
            price_bond_base(make_bill(curve=empty))


class TestScenarioPricingIsRefused:
    """A bond has no scenario NPV and says so."""

    def test_price_bond_scenarios_always_raises(self):
        with pytest.raises(ScenarioPricingNotSupported):
            price_bond_scenarios(make_bill())

    def test_the_refusal_is_a_bond_pricing_error(self):
        # A caller catching the general pricing error catches this too.
        assert issubclass(ScenarioPricingNotSupported, BondPricingError)

    def test_the_refusal_names_the_alternative(self):
        """The refusal names what to use instead."""
        with pytest.raises(ScenarioPricingNotSupported) as exc:
            price_bond_scenarios(make_bill())
        message = str(exc.value)
        assert "base_npv" in message
        assert "I-24" in message

    def test_a_broadcast_constant_column_produces_var_zero_and_es_nan(self):
        """What a broadcast constant column would give: VaR exactly 0.00 and ES NaN. If a
        constant column ever produces something sensible, the refusal should be
        revisited."""
        from engine.risk.var_es import compute_risk_metrics

        constant_column = jnp.full((500, 3, 1), 98_401.95)
        risk = compute_risk_metrics(constant_column, 98_401.95, percentiles=(0.95, 0.99))

        assert float(risk["VaR_95"][0]) == 0.0
        assert math.isnan(float(risk["ES_95"][0]))

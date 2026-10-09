"""
`engine.instruments.treasury` (bills and notes): the trade, its cashflows and accrued interest,
priced by the valuation pipeline on its currency's discount curve (`value_today`, ORE's
`DiscountingRiskyBondEngine` without credit). The portfolio path, and the bond on every
simulated path under either model, are tests/test_portfolio_bond_wire_through.py and
tests/test_hull_white_model.py (I-24).
"""
import math

import ORE
import pytest

from engine.instruments.treasury import BondConfig, BondPricingError, CouponPeriod, accrued_interest
from engine.market_data.market import CurrencyMarket, Market, ZeroCurveConfig
from engine.pricing.cube import value_today

VALUATION = ORE.Date(2, 6, 2025)
FLAT_3PCT = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6)


def _date(iso: str) -> ORE.Date:
    year, month, day = (int(p) for p in iso.split("-"))
    return ORE.Date(day, month, year)


def price(cfg: BondConfig, curve: ZeroCurveConfig = FLAT_3PCT) -> float:
    """The bond's dirty NPV with `curve` as its currency's discount curve."""
    return value_today([cfg], Market(cfg.evaluation_date, {"USD": CurrencyMarket(curve)}), "USD")[0]


def shifted(curve: ZeroCurveConfig, shift: float) -> ZeroCurveConfig:
    return ZeroCurveConfig(times=list(curve.times), rates=[r + shift for r in curve.rates])


def make_bill(face: float = 100_000.0, maturity: str = "2025-12-15", **fields) -> BondConfig:
    return BondConfig(face_amount=face, maturity_date=_date(maturity), evaluation_date=VALUATION, trade_id="bill",
                      **fields)


#: The three-period semiannual schedule of the note fixture.
NOTE_SCHEDULE = (
    CouponPeriod(_date("2024-12-15"), _date("2025-06-15"), _date("2025-06-15")),
    CouponPeriod(_date("2025-06-15"), _date("2025-12-15"), _date("2025-12-15")),
    CouponPeriod(_date("2025-12-15"), _date("2026-06-15"), _date("2026-06-15")),
)


def make_note(face: float = 100_000.0, evaluation_date: ORE.Date = VALUATION) -> BondConfig:
    return BondConfig(face_amount=face, maturity_date=_date("2026-06-15"), evaluation_date=evaluation_date,
                      trade_id="note", coupon_rate=0.04, coupon_schedule=NOTE_SCHEDULE,
                      accrual_day_count="ACT/ACT (ICMA)")


class TestBillPricing:
    """The zero-coupon single-cashflow case."""

    def test_prices_to_the_closed_form_value(self):
        # NPV = face x exp(-r x t), ACT/365 from 2025-06-02 to 2025-12-15.
        t = ORE.Actual365Fixed().yearFraction(VALUATION, _date("2025-12-15"))
        assert price(make_bill()) == pytest.approx(100_000.0 * math.exp(-0.03 * t), rel=1e-14)

    def test_is_identified_as_a_bill(self):
        assert make_bill().is_bill is True
        assert make_note().is_bill is False

    def test_a_bill_has_no_accrued_interest(self):
        assert accrued_interest(make_bill()) == 0.0

    def test_discounts_below_par(self):
        assert price(make_bill()) < 100_000.0

    def test_a_short_position_is_negative(self):
        """A negative face gives a negative NPV directly (no second sign factor)."""
        long_npv, short_npv = price(make_bill(face=100_000.0)), price(make_bill(face=-100_000.0))
        assert short_npv < 0
        assert short_npv == pytest.approx(-long_npv)

    def test_redemption_fraction_scales_the_value(self):
        assert price(make_bill(redemption_fraction=0.5)) == pytest.approx(price(make_bill()) * 0.5)


class TestNotePricing:
    """The coupon-bearing scheduled case."""

    def test_prices_above_a_comparable_bill(self):
        assert price(make_note()) > price(make_bill(maturity="2026-06-15"))

    def test_accrued_interest_uses_act_act_icma(self):
        """ACT/ACT (ICMA): the 2024-12-15 -> 2025-06-15 coupon accrues exactly 0.5 however
        many days it has, so accrual to 2025-06-02 is 169/182 of it:

            0.04 x (169/182) x 0.5 x 100,000 = 1,857.142857
        """
        assert accrued_interest(make_note()) == pytest.approx(1857.142857, abs=1e-5)

    def test_accrued_is_position_signed(self):
        assert accrued_interest(make_note(face=-100_000.0)) == pytest.approx(-1857.142857, abs=1e-5)

    def test_the_npv_is_the_DIRTY_value(self):
        """Dirty: every remaining discounted cashflow, no accrued deduction, as
        `engine.traderx.note.NotePrice.npv`. (A clean-returning version once passed all
        but one test.)"""
        day_count = ORE.ActualActual(ORE.ActualActual.ISMA)
        expected_per_unit = 0.0
        for period in NOTE_SCHEDULE:
            if period.payment() <= VALUATION:
                continue
            accrual = day_count.yearFraction(period.start_date, period.end_date, period.start_date, period.end_date)
            t = ORE.Actual365Fixed().yearFraction(VALUATION, period.payment())
            expected_per_unit += 0.04 * accrual * math.exp(-0.03 * t)
        t_mat = ORE.Actual365Fixed().yearFraction(VALUATION, _date("2026-06-15"))
        expected_per_unit += math.exp(-0.03 * t_mat)
        assert price(make_note()) == pytest.approx(100_000.0 * expected_per_unit, abs=1e-6)

    def test_clean_and_dirty_differ_by_a_material_amount(self):
        """Clean (dirty - accrued) and dirty differ materially (~$1,857 on $100k), so a
        silently clean NPV would be caught."""
        assert accrued_interest(make_note()) > 1000.0

    def test_a_coupon_already_paid_is_excluded(self):
        """The 2025-06-15 coupon is included on 2025-06-02 and dropped after it: two fewer
        weeks of discounting, but the ~2,000 coupon is gone."""
        assert price(make_note(evaluation_date=_date("2025-06-20"))) < price(make_note())


class TestAgreesWithTheIntegrationPricers:
    """The pipeline's bond pricer and the integration boundary's pricers agree to the cent on
    the same instrument, so the boundary's own discounting (integration sits above the engine
    and is not imported by it) cannot drift apart unnoticed."""

    def _profile_and_curve(self):
        from engine.traderx.market_inputs import ASSUMED_PROFILES
        profile = ASSUMED_PROFILES["flat-3pct-v1"]
        return profile, ZeroCurveConfig(times=list(profile.times), rates=list(profile.rates()))

    def _note_entry(self):
        from engine.traderx.terms import TermsEntry
        schedule = [
            {"startDate": "2024-12-15", "endDate": "2025-06-15", "paymentDate": "2025-06-15"},
            {"startDate": "2025-06-15", "endDate": "2025-12-15", "paymentDate": "2025-12-15"},
            {"startDate": "2025-12-15", "endDate": "2026-06-15", "paymentDate": "2026-06-15"},
        ]
        return TermsEntry(
            instrument_type="NOTE",
            terms={"couponFrequency": "SEMIANNUAL", "maturityDate": "2026-06-15", "couponRatePercent": 4.0,
                   "redemptionFraction": 1.0, "dayCount": "ACT/ACT (ICMA)", "settlementDays": 0,
                   "schedule": schedule},
            missing_terms=(), provenance={}, identity={})

    def test_bill_matches_the_integration_bill_pricer(self):
        from engine.traderx.bill import price_bill
        from engine.traderx.terms import TermsEntry

        profile, curve = self._profile_and_curve()
        entry = TermsEntry(instrument_type="BILL", terms={"couponFrequency": "NONE", "maturityDate": "2025-12-15",
                                                          "redemptionFraction": 1.0},
                           missing_terms=(), provenance={}, identity={})
        assert price(make_bill(), curve) == pytest.approx(price_bill(entry, 100_000.0, VALUATION, profile).npv, abs=1e-6)

    def test_note_matches_the_integration_note_pricer(self):
        from engine.traderx.note import price_note

        profile, curve = self._profile_and_curve()
        integration = price_note(self._note_entry(), 100_000.0, VALUATION, profile)
        assert price(make_note(), curve) == pytest.approx(integration.dirty_npv, abs=1e-6)

    def test_accrued_matches_the_integration_recomputed_path(self):
        """ACT/ACT (ICMA) accrual against the integration pricer's recomputed path (a direct
        caller has no exported accrued figure)."""
        from engine.traderx.note import price_note

        profile, _ = self._profile_and_curve()
        integration = price_note(self._note_entry(), 100_000.0, VALUATION, profile)
        assert accrued_interest(make_note()) == pytest.approx(integration.accrual.recomputed_fraction * 100_000.0,
                                                              abs=1e-6)


class TestRefusals:
    """What a bond refuses rather than approximates."""

    def test_a_matured_bond_is_refused(self):
        with pytest.raises(BondPricingError, match="not after"):
            make_bill(maturity="2025-01-01")

    def test_a_bond_maturing_on_the_valuation_date_is_refused(self):
        with pytest.raises(BondPricingError, match="not after"):
            BondConfig(face_amount=1.0, maturity_date=VALUATION, evaluation_date=VALUATION, trade_id="bill")

    def test_a_schedule_with_a_zero_coupon_rate_is_refused(self):
        """A schedule with a zero coupon rate is a contradiction, refused rather than priced
        as a bill."""
        with pytest.raises(BondPricingError, match="contradiction"):
            make_bill(maturity="2026-06-15", coupon_rate=0.0, coupon_schedule=NOTE_SCHEDULE)

    def test_a_coupon_rate_with_no_schedule_is_refused(self):
        with pytest.raises(BondPricingError, match="no coupon_schedule"):
            make_bill(maturity="2026-06-15", coupon_rate=0.04)

    def test_an_unsupported_accrual_day_count_is_refused(self):
        """An unsupported accrual day count (ACT/360) is refused, never defaulted."""
        with pytest.raises(BondPricingError):
            make_bill(maturity="2026-06-15", coupon_rate=0.04, coupon_schedule=NOTE_SCHEDULE,
                      accrual_day_count="ACT/360")

    def test_a_gapped_schedule_is_refused(self):
        gapped = (CouponPeriod(_date("2024-12-15"), _date("2025-06-15")),
                  CouponPeriod(_date("2025-07-15"), _date("2026-06-15")))
        with pytest.raises(BondPricingError, match="gap or overlap"):
            make_bill(maturity="2026-06-15", coupon_rate=0.04, coupon_schedule=gapped)

    def test_a_schedule_not_ending_at_maturity_is_refused(self):
        with pytest.raises(BondPricingError, match="schedule ends"):
            make_bill(maturity="2026-12-15", coupon_rate=0.04, coupon_schedule=NOTE_SCHEDULE)

    def test_a_zero_length_period_is_refused(self):
        bad = (CouponPeriod(_date("2025-06-15"), _date("2025-06-15")),)
        with pytest.raises(BondPricingError, match="no length"):
            BondConfig(face_amount=100_000.0, maturity_date=_date("2025-06-15"), evaluation_date=_date("2025-01-01"),
                       trade_id="note", coupon_rate=0.04, coupon_schedule=bad)


class TestParallelShift:
    """A 1bp parallel shift of the curve."""

    def _move(self, cfg, bp=1.0):
        return price(cfg, shifted(FLAT_3PCT, bp * 1e-4)) - price(cfg)

    def test_is_negative_for_a_long_position(self):
        assert self._move(make_bill()) < 0

    def test_is_positive_for_a_short_position(self):
        assert self._move(make_bill(face=-100_000.0)) > 0

    def test_a_longer_bond_is_more_sensitive(self):
        assert abs(self._move(make_bill(maturity="2030-12-15"))) > abs(self._move(make_bill(maturity="2025-12-15")))


class TestCurveInterpolation:
    """A bond is priced off its currency's curve, including non-flat ones."""

    def test_a_non_flat_curve_is_interpolated(self):
        steep = ZeroCurveConfig(times=[0.0, 1.0, 10.0], rates=[0.01, 0.02, 0.05])
        flat = ZeroCurveConfig(times=[0.0, 1.0, 10.0], rates=[0.02] * 3)
        # At ~0.5y the steep curve is near 1.5%, below the flat 2%, so the bill is worth more.
        assert price(make_bill(), steep) > price(make_bill(), flat)

    def test_a_higher_curve_gives_a_lower_price(self):
        low = ZeroCurveConfig(times=[0.0, 30.0], rates=[0.01, 0.01])
        high = ZeroCurveConfig(times=[0.0, 30.0], rates=[0.06, 0.06])
        assert price(make_bill(), high) < price(make_bill(), low)

    def test_beyond_the_last_pillar_extrapolates_as_ore(self):
        """Beyond the last pillar the instantaneous forward is held flat, as ORE's
        `ZeroCurve` (QuantLib's ContinuousForward extrapolation, I-48); holding the zero
        rate flat instead is ~1.7% off here."""
        curve = ZeroCurveConfig(times=[0.0, 1.0], rates=[0.03, 0.04])
        ORE.Settings.instance().evaluationDate = VALUATION
        ore_curve = ORE.ZeroCurve([VALUATION, VALUATION + 365], [0.03, 0.04], ORE.Actual365Fixed())
        ore_curve.enableExtrapolation()
        expected = 100_000.0 * ore_curve.discount(_date("2035-12-15"))
        assert price(make_bill(maturity="2035-12-15"), curve) == pytest.approx(expected, rel=1e-12)

    def test_a_negative_rate_curve_prices_above_par(self):
        """At a negative zero rate the discount factor exceeds 1 and a bill is worth more
        than face (pinned so a zero floor would fail)."""
        negative = ZeroCurveConfig(times=[0.0, 1.0, 30.0], rates=[-0.005] * 3)
        assert price(make_bill(), negative) > 100_000.0

    def test_a_long_dated_bill_discounts_heavily(self):
        assert 0.0 < price(make_bill(maturity="2055-06-02")) < 50_000.0

    def test_a_zero_face_position_is_worth_zero(self):
        assert price(make_bill(face=0.0)) == 0.0

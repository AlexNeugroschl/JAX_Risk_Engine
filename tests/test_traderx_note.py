"""
Treasury note pricer (`engine.traderx.note`).

ORE parity is against `ORE.FixedRateBond` with the fixture's schedule and
`ORE.DiscountingBondEngine`, not a re-derived sum, which could share a bug with the
implementation. Also: accrued interest reconciles to TraderX's exported `0.018571`,
clean/dirty, long/short, and the rate sensitivity.

The assumed profile is a flat constant, so the sensitivity is a parallel shift, labelled
`zero-curve-parallel`; a per-pillar sensitivity needs a real curve (I-16).
"""
import json
import math
from pathlib import Path

import ORE
import pytest

from engine.traderx import price_bundle
from engine.traderx.capabilities import (
    PRICED_CALCULATIONS,
    PRICED_CALCULATIONS_BY_SHAPE,
    capabilities,
)
from engine.traderx.market_inputs import ASSUMED_PROFILES
from engine.traderx.note import (
    ACCRUAL_EXPORTED,
    ACCRUAL_MISMATCH,
    ACCRUAL_RECOMPUTED,
    DAY_COUNT_NOT_SUPPORTED,
    INSTRUMENT_MATURED,
    NOT_A_NOTE,
    RATE_BUMP,
    SCHEDULE_INCONSISTENT,
    SETTLEMENT_CONVENTION_NOT_SUPPORTED,
    TERMS_INCOMPLETE,
    NotePricingError,
    accrual_mismatch_tolerance,
    is_note,
    price_note,
    rate_sensitivity,
)
from engine.traderx.terms import TermsEntry

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"

#: The delivered note fixture's own terms: 4% semiannual, session date
#: 2025-06-02, maturing 2026-12-15, redeemed at par.
VALUATION = ORE.Date(2, 6, 2025)
MATURITY = ORE.Date(15, 12, 2026)
PROFILE = ASSUMED_PROFILES["flat-3pct-v1"]
FACE = 100_000.0

#: TraderX's exported accrued interest for this position, as a fraction of
#: par, HALF_EVEN-rounded at 6 decimals by their exporter. This exact
#: number is the plan's acceptance check.
EXPORTED_ACCRUED = 0.018571

#: The unrounded ACT/ACT (ICMA) value: 169/182 x 4%/2. The exported figure
#: above is this, rounded.
EXACT_ACCRUED = 169.0 / 182.0 * 0.04 / 2.0

#: The fixture's schedule, as ORE dates, used to build the independent
#: reference bond. Written out rather than read from the terms so the
#: reference does not inherit a bug in this engine's schedule parsing.
SCHEDULE_DATES = [
    ORE.Date(15, 12, 2024), ORE.Date(15, 6, 2025), ORE.Date(15, 12, 2025),
    ORE.Date(15, 6, 2026), ORE.Date(15, 12, 2026),
]

MARKET = {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}


def _fixture_terms() -> dict:
    """The delivered fixture's terms, read from the hashed artifact (not transcribed)."""
    doc = json.loads((FIXTURES / "note" / "v2" / "instrument-terms.json").read_text())
    return doc["entries"][0]["terms"]


def _terms(**overrides) -> TermsEntry:
    """A note terms entry matching the delivered fixture, with overrides."""
    terms = dict(_fixture_terms())
    terms.update(overrides)
    return TermsEntry(
        instrument_type="TREASURY", terms=terms,
        missing_terms=(), provenance={"origin": "synthetic"}, identity={},
    )


def _reference_bond(coupon: float = 0.04, day_count=None) -> ORE.FixedRateBond:
    """An independent ORE bond (`FixedRateBond`, `Schedule`, `DiscountingBondEngine`) over
    the fixture's schedule."""
    ORE.Settings.instance().evaluationDate = VALUATION
    schedule = ORE.Schedule(ORE.DateVector(SCHEDULE_DATES))
    return ORE.FixedRateBond(
        0, 100.0, schedule, ORE.DoubleVector([coupon]),
        day_count or ORE.ActualActual(ORE.ActualActual.ISMA),
        ORE.Unadjusted, 100.0, SCHEDULE_DATES[0],
    )


def _priced_reference(face: float = FACE, rate: float = None) -> ORE.FixedRateBond:
    """The reference bond on a flat curve.

    `NPV()` is the present value at the evaluation date; `dirtyPrice()` is that value
    forward to the settlement date. With `settlementDays: 0` they coincide at `VALUATION`,
    but not in general (before the issue date ORE clamps settlement to the issue date and
    they differ by ~$117 per $100k). `price_note` computes the former
    (`TestPresentValueIsAsOfTheValuationDate`).
    """
    bond = _reference_bond()
    curve = ORE.FlatForward(
        VALUATION,
        ORE.QuoteHandle(ORE.SimpleQuote(PROFILE.flat_rate if rate is None else rate)),
        ORE.Actual365Fixed(), ORE.Continuous, ORE.Annual,
    )
    bond.setPricingEngine(ORE.DiscountingBondEngine(ORE.YieldTermStructureHandle(curve)))
    return bond


class TestPresentValueIsAsOfTheValuationDate:
    """`price_note` is a present value at the valuation date, not a settlement-forward
    price. They coincide on the fixture, which is why the distinction is pinned here."""

    def test_npv_equals_ores_npv_not_merely_its_dirty_price(self):
        """Parity against `NPV()`, the quantity `price_note` computes."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.dirty_npv == pytest.approx(
            _priced_reference().NPV() / 100.0 * FACE, rel=1e-12
        )

    def test_the_two_ore_quantities_coincide_here_and_the_test_knows_why(self):
        """Zero settlement lag is why `dirtyPrice()` is a valid reference at this date."""
        bond = _priced_reference()
        assert bond.settlementDate() == VALUATION
        assert bond.NPV() == pytest.approx(bond.dirtyPrice(), rel=1e-12)

    def test_present_value_is_measured_from_the_valuation_date(self):
        """Before the first period, `price_note` still discounts to the valuation date,
        matching ORE's `NPV()`, not its `dirtyPrice()` (which differ here by ~$117 per
        $100k)."""
        pre_issue = ORE.Date(1, 12, 2024)
        priced = price_note(_terms(), FACE, pre_issue, PROFILE, None)

        ORE.Settings.instance().evaluationDate = pre_issue
        bond = _reference_bond()
        curve = ORE.FlatForward(
            pre_issue, ORE.QuoteHandle(ORE.SimpleQuote(PROFILE.flat_rate)),
            ORE.Actual365Fixed(), ORE.Continuous, ORE.Annual,
        )
        bond.setPricingEngine(
            ORE.DiscountingBondEngine(ORE.YieldTermStructureHandle(curve))
        )
        try:
            assert priced.dirty_npv == pytest.approx(bond.NPV() / 100.0 * FACE, rel=1e-12)
            # The two ORE quantities do differ here.
            assert abs(bond.NPV() - bond.dirtyPrice()) > 0.1
        finally:
            # ORE's evaluation date is global; restore it for later tests.
            ORE.Settings.instance().evaluationDate = VALUATION

    def test_nothing_has_accrued_before_the_first_period(self):
        """Nothing has accrued before the first period (a structural zero)."""
        priced = price_note(_terms(), FACE, ORE.Date(1, 12, 2024), PROFILE, None)
        assert priced.accrued_interest == 0.0
        assert len(priced.coupons) == 4


class TestOreParity:
    """Engine NPV equals an independent ORE valuation on the same terms."""

    def test_long_dirty_npv_matches_ore(self):
        """Long position. `dirtyPrice` is per 100 face."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        expected = _priced_reference().dirtyPrice() / 100.0 * FACE
        assert priced.dirty_npv == pytest.approx(expected, rel=1e-12)

    def test_short_dirty_npv_matches_ore(self):
        priced = price_note(_terms(), -FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        expected = -_priced_reference().dirtyPrice() / 100.0 * FACE
        assert priced.dirty_npv == pytest.approx(expected, rel=1e-12)

    def test_parity_is_exact_not_merely_close(self):
        """Parity to machine precision: both sides do the same arithmetic on the same dates,
        so any larger gap is a real error."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        expected = _priced_reference().dirtyPrice() / 100.0 * FACE
        assert abs(priced.dirty_npv - expected) < 1e-7

    def test_every_coupon_matches_ores_own_cashflows(self):
        """Coupon by coupon, so offsetting errors cannot cancel in the total."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        bond = _priced_reference()

        ore_coupons = [
            (cf.date(), cf.amount()) for cf in bond.cashflows()
            if not cf.hasOccurred(VALUATION) and cf.amount() != 100.0
        ]
        assert len(priced.coupons) == len(ore_coupons)

        for mine, (date, amount) in zip(priced.coupons, ore_coupons):
            iso = f"{date.year():04d}-{date.month():02d}-{date.dayOfMonth():02d}"
            assert mine.payment_date == iso
            # ORE's amount is per 100 face; mine is per unit face.
            assert mine.amount_per_unit_face * 100.0 == pytest.approx(amount, rel=1e-12)

    def test_accrued_matches_ores_own_accrual(self):
        """Recomputed accrued equals ORE's `accruedAmount` (ACT/ACT ICMA)."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        ore_accrued = _reference_bond().accruedAmount(VALUATION) / 100.0
        assert priced.accrual.recomputed_fraction == pytest.approx(ore_accrued, rel=1e-12)

    def test_discount_factors_match_ores_curve(self):
        """Discount factors equal ORE's flat curve's."""
        curve = ORE.FlatForward(
            VALUATION, ORE.QuoteHandle(ORE.SimpleQuote(PROFILE.flat_rate)),
            ORE.Actual365Fixed(), ORE.Continuous, ORE.Annual,
        )
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        for coupon in priced.coupons:
            year, month, day = (int(p) for p in coupon.payment_date.split("-"))
            expected = curve.discount(ORE.Date(day, month, year))
            assert coupon.discount_factor == pytest.approx(expected, rel=1e-12)
        assert priced.redemption_discount_factor == pytest.approx(
            curve.discount(MATURITY), rel=1e-12
        )


class TestAccruedReconcilesToTraderX:
    """Accrued interest reconciles to TraderX's exported `0.018571`."""

    def test_recomputed_fraction_equals_traderx_exported_value(self):
        """The unrounded recomputation, rounded HALF_EVEN to 6 decimals, is `0.018571`."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert round(priced.accrual.recomputed_fraction, 6) == EXPORTED_ACCRUED

    def test_recomputed_fraction_is_the_icma_ratio(self):
        """169/182 x 4%/2, independent of ORE and of the pricer."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.accrual.recomputed_fraction == pytest.approx(EXACT_ACCRUED, rel=1e-15)

    def test_reported_value_is_the_export_not_the_recomputation(self):
        """The exported value is reported, not the recomputation (they differ by $0.04
        here)."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.accrual.source == ACCRUAL_EXPORTED
        assert priced.accrued_interest == pytest.approx(EXPORTED_ACCRUED * FACE, rel=1e-12)
        # Not the recomputed value.
        assert priced.accrued_interest != pytest.approx(EXACT_ACCRUED * FACE, rel=1e-12)

    def test_both_paths_travel_in_the_payload(self):
        """Both values, their difference and the tolerance are in the payload."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        block = priced.to_payload()["accrualReconciliation"]
        assert block["exportedFraction"] == EXPORTED_ACCRUED
        assert block["recomputedFraction"] == pytest.approx(EXACT_ACCRUED)
        assert block["accrualSource"] == ACCRUAL_EXPORTED
        assert block["difference"] == pytest.approx(
            (EXPORTED_ACCRUED - EXACT_ACCRUED) * FACE
        )
        assert block["tolerance"] > 0

    def test_the_two_paths_are_not_equal_and_the_test_knows_it(self):
        """The two differ (so the mismatch tolerance is actually exercised) and are within
        tolerance."""
        assert EXPORTED_ACCRUED != EXACT_ACCRUED
        difference = abs(EXPORTED_ACCRUED - EXACT_ACCRUED) * FACE
        assert difference > 0.0
        assert difference < accrual_mismatch_tolerance(FACE)

    def test_short_position_accrues_negatively(self):
        """A short's accrued is negative, consistent with its NPV."""
        priced = price_note(_terms(), -FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.accrued_interest == pytest.approx(-EXPORTED_ACCRUED * FACE, rel=1e-12)

    def test_blank_export_falls_back_to_recomputation_under_its_own_label(self):
        """A blank export reports the recomputed value, labelled as recomputed."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, None)
        assert priced.accrual.source == ACCRUAL_RECOMPUTED
        assert priced.accrual.exported_fraction is None
        assert priced.accrued_interest == pytest.approx(EXACT_ACCRUED * FACE, rel=1e-12)


class TestToleranceIsDerivedNotConstant:
    """Tolerance `0.5 x 10^-decimals x |face| + 0.01`, derived from the stated rounding."""

    def test_matches_the_agreed_formula(self):
        assert accrual_mismatch_tolerance(FACE, 6) == pytest.approx(
            0.5 * 1e-6 * FACE + 0.01
        )

    def test_the_rounding_bound_is_not_itself_rounded(self):
        """Regression (TraderX v4/v5): the bound was once rounded to cents, truncating the
        quantity it bounds. At 124,000 face the rounding error is 0.062, so the tolerance is
        0.072, but rounding gave 0.07 and could refuse a legitimate reconciliation. The
        $100,000 fixture gives 0.06 either way, which is why it went unnoticed."""
        assert accrual_mismatch_tolerance(124_000.0, 6) == pytest.approx(0.072)
        # The rounded version gave this.
        assert accrual_mismatch_tolerance(124_000.0, 6) != pytest.approx(0.07)

    def test_the_fixture_face_is_unchanged_by_the_fix(self):
        """The fix leaves the delivered case unchanged."""
        assert accrual_mismatch_tolerance(FACE, 6) == pytest.approx(0.06)

    @pytest.mark.parametrize("face", [124_000.0, 3_000.0, 17_500.0, 999_999.0])
    def test_never_narrower_than_the_exporters_rounding_error(self, face):
        """For any face, the tolerance is at least the worst-case rounding plus a cent."""
        assert accrual_mismatch_tolerance(face, 6) >= 0.5e-6 * face + 0.01

    def test_scales_with_face(self):
        """A constant tolerance would be vacuous on a large position."""
        small = accrual_mismatch_tolerance(1_000.0)
        large = accrual_mismatch_tolerance(100_000_000.0)
        assert large > small * 100

    def test_is_sign_independent(self):
        assert accrual_mismatch_tolerance(FACE) == accrual_mismatch_tolerance(-FACE)

    def test_tighter_rounding_gives_a_tighter_tolerance(self):
        """The tolerance follows the declared decimals."""
        assert accrual_mismatch_tolerance(FACE, 9) < accrual_mismatch_tolerance(FACE, 6)

    @pytest.mark.parametrize("face", [
        1e2, 1e3, 1e5, 1e6, 1e8, 1e9,
    ])
    def test_covers_the_exporters_worst_case_rounding_at_every_size(self, face):
        """The refusal must not fire on legitimately rounded data at any size: the tolerance
        covers the worst-case rounding `0.5e-6 x face`."""
        worst_case_rounding = 0.5e-6 * face
        assert accrual_mismatch_tolerance(face) >= worst_case_rounding

    def test_is_still_tight_enough_to_catch_a_wrong_day_count(self):
        """Tight enough to catch a wrong day count: ACT/365 vs ICMA is ~$5.09 on $100k
        against a $0.06 bound."""
        act365_error = abs(0.018520547945205478 - EXACT_ACCRUED) * FACE
        assert act365_error > accrual_mismatch_tolerance(FACE) * 50


class TestCleanDirtyReconciliation:
    """Clean vs dirty reconciliation."""

    def test_clean_is_dirty_less_accrued(self):
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.clean_npv == pytest.approx(
            priced.dirty_npv - priced.accrued_interest, rel=1e-15
        )

    def test_npv_reports_the_dirty_value(self):
        """`npv` is the dirty value; a clean NPV would be off by the accrued (~$1,857 here)."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.npv == priced.dirty_npv
        assert priced.npv != pytest.approx(priced.clean_npv, rel=1e-6)
        assert priced.to_payload()["priceType"] == "dirty"

    def test_clean_npv_matches_ore_when_both_use_the_recomputed_accrual(self):
        """ORE recomputes accrued from the schedule, so its clean price matches the clean
        value on the recomputed path, not the exported one (which differs by $0.04)."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, None)
        expected = _priced_reference().cleanPrice() / 100.0 * FACE
        assert priced.clean_npv == pytest.approx(expected, rel=1e-12)

    def test_clean_on_the_exported_path_differs_by_exactly_the_rounding(self):
        """The clean values on the two paths differ by exactly the exporter's rounding."""
        exported = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        recomputed = price_note(_terms(), FACE, VALUATION, PROFILE, None)
        assert exported.clean_npv - recomputed.clean_npv == pytest.approx(
            (EXACT_ACCRUED - EXPORTED_ACCRUED) * FACE, rel=1e-9
        )


class TestLongShort:
    """Long and short positions."""

    def test_long_and_short_are_exact_mirrors(self):
        long_note = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        short = price_note(_terms(), -FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert long_note.dirty_npv + short.dirty_npv == pytest.approx(0.0, abs=1e-9)
        assert long_note.clean_npv + short.clean_npv == pytest.approx(0.0, abs=1e-9)
        assert long_note.accrued_interest + short.accrued_interest == pytest.approx(0.0, abs=1e-9)

    def test_short_npv_is_negative(self):
        """A short's values are negative: the sign comes from the signed face, once."""
        short = price_note(_terms(), -FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert short.dirty_npv < 0
        assert short.clean_npv < 0
        assert short.accrued_interest < 0

    def test_scaling_is_linear_in_face(self):
        one = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        ten = price_note(_terms(), FACE * 10, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert ten.dirty_npv == pytest.approx(one.dirty_npv * 10, rel=1e-12)


class TestRateSensitivity:
    """Rate sensitivity: a parallel shift (the profile is flat; see the module docstring)."""

    def test_is_non_zero(self):
        assert rate_sensitivity(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED) != 0.0

    def test_long_bond_loses_value_when_rates_rise(self):
        """A long bond loses value when rates rise."""
        assert rate_sensitivity(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED) < 0

    def test_short_bond_gains_when_rates_rise(self):
        assert rate_sensitivity(_terms(), -FACE, VALUATION, PROFILE, EXPORTED_ACCRUED) > 0

    def test_matches_a_direct_reprice_at_the_bumped_rate(self):
        """Equals a direct reprice at the bumped rate."""
        from engine.traderx.market_inputs import AssumedProfile

        bumped = AssumedProfile(
            profile_id=PROFILE.profile_id, description=PROFILE.description,
            flat_rate=PROFILE.flat_rate + RATE_BUMP, times=PROFILE.times,
        )
        base = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED).dirty_npv
        shifted = price_note(_terms(), FACE, VALUATION, bumped, EXPORTED_ACCRUED).dirty_npv
        assert rate_sensitivity(
            _terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED
        ) == pytest.approx(shifted - base, rel=1e-12)

    def test_magnitude_is_in_the_right_ballpark_for_the_duration(self):
        """Order of magnitude: a ~1.5y-duration bond on $100k moves roughly $15 per bp."""
        sensitivity = rate_sensitivity(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert 5.0 < abs(sensitivity) < 40.0

    def test_longer_maturity_is_more_sensitive(self):
        """A shorter schedule (first two periods only) is less sensitive."""
        short_terms = _terms(
            maturityDate="2025-12-15",
            schedule=_fixture_terms()["schedule"][:2],
        )
        short_sensitivity = rate_sensitivity(
            short_terms, FACE, VALUATION, PROFILE, EXPORTED_ACCRUED
        )
        full = rate_sensitivity(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert abs(full) > abs(short_sensitivity)

    def test_matches_ore_bps_within_convexity(self):
        """Equals the same one-sided 1bp bump repriced through ORE's bond."""
        base = _priced_reference().dirtyPrice() / 100.0 * FACE
        bumped = _priced_reference(rate=PROFILE.flat_rate + RATE_BUMP)
        shifted = bumped.dirtyPrice() / 100.0 * FACE
        mine = rate_sensitivity(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert mine == pytest.approx(shifted - base, rel=1e-4)


class TestWrongDayCountIsCaught:
    """ACT/ACT (ICMA) vs ACT/365: pricing this note on ACT/365 would give a plausible,
    wrong accrual."""

    def test_act365_accrual_would_differ_materially(self):
        """ACT/365 would be off by ~$5.09 on $100k, against a $0.06 tolerance."""
        act365 = ORE.Actual365Fixed()
        wrong = 0.04 * act365.yearFraction(ORE.Date(15, 12, 2024), VALUATION)
        error = abs(wrong - EXACT_ACCRUED) * FACE
        assert error > 5.0
        assert error > accrual_mismatch_tolerance(FACE) * 50

    def test_the_accrual_check_would_refuse_an_act365_note(self):
        """Terms claiming ACT/365 against an export accrued on ICMA fail reconciliation and
        are refused."""
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(dayCount="ACT/365"), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert excinfo.value.reason == ACCRUAL_MISMATCH

    def test_icma_period_is_exactly_half_whatever_its_length(self):
        """ICMA periods accrue exactly 0.5 whether 182 or 183 days long."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert [c.accrual_fraction for c in priced.coupons] == [0.5, 0.5, 0.5, 0.5]

    def test_unsupported_day_count_is_refused_not_defaulted(self):
        """An unsupported day count is refused, not defaulted."""
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(dayCount="30/360"), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert excinfo.value.reason == DAY_COUNT_NOT_SUPPORTED

    def test_absent_day_count_is_refused_not_defaulted(self):
        terms = dict(_fixture_terms())
        del terms["dayCount"]
        entry = TermsEntry(
            instrument_type="TREASURY", terms=terms, missing_terms=(),
            provenance={}, identity={},
        )
        with pytest.raises(NotePricingError) as excinfo:
            price_note(entry, FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert excinfo.value.reason == TERMS_INCOMPLETE

    def test_the_day_count_used_is_echoed_in_the_payload(self):
        """The day counts used are echoed in the payload."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        payload = priced.to_payload()
        assert payload["accrualDayCount"] == "ACT/ACT (ICMA)"
        assert payload["discountDayCount"] == "ACT/365 (Fixed)"


class TestAccrualMismatchIsRefused:
    """Accrued disagreement beyond tolerance is refused: the priced schedule is not the one
    the export accrued against."""

    def test_large_disagreement_is_refused(self):
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(), FACE, VALUATION, PROFILE, 0.025)
        assert excinfo.value.reason == ACCRUAL_MISMATCH

    def test_refusal_names_both_values_and_the_tolerance(self):
        """The refusal states both values and the tolerance."""
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(), FACE, VALUATION, PROFILE, 0.025)
        detail = excinfo.value.detail
        assert "0.0250000000" in detail
        assert "0.0185714286" in detail
        assert "tolerance" in detail.lower()

    def test_disagreement_within_tolerance_prices_normally(self):
        """The delivered fixture (rounded, not wrong) prices."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.accrual.agrees
        assert priced.dirty_npv > 0

    def test_tolerance_scales_so_a_large_position_is_not_falsely_refused(self):
        """6-decimal rounding on $1bn is $500; the derived tolerance admits it."""
        face = 1_000_000_000.0
        priced = price_note(_terms(), face, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.accrual.agrees

    def test_a_refusal_is_not_a_warning(self):
        """Nothing is returned on mismatch (not a price with a warning)."""
        with pytest.raises(NotePricingError):
            price_note(_terms(), FACE, VALUATION, PROFILE, 0.030)


class TestScheduleIsUsedNotRegenerated:
    """The exporter's schedule is used as given, never regenerated."""

    def test_uses_every_unpaid_period_from_the_terms(self):
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert len(priced.coupons) == 4
        assert [c.payment_date for c in priced.coupons] == [
            "2025-06-15", "2025-12-15", "2026-06-15", "2026-12-15",
        ]

    def test_already_paid_coupons_are_excluded(self):
        """A day after the first coupon pays, three coupons remain."""
        after_first = ORE.Date(16, 6, 2025)
        priced = price_note(_terms(), FACE, after_first, PROFILE, None)
        assert len(priced.coupons) == 3
        assert priced.coupons[0].payment_date == "2025-12-15"

    def test_a_coupon_paying_on_the_valuation_date_is_excluded(self):
        """A coupon paid on the valuation date is excluded."""
        on_coupon = ORE.Date(15, 6, 2025)
        priced = price_note(_terms(), FACE, on_coupon, PROFILE, None)
        assert len(priced.coupons) == 3
        assert all(c.payment_date != "2025-06-15" for c in priced.coupons)

    def test_a_gapped_schedule_is_refused_not_bridged(self):
        schedule = [dict(p) for p in _fixture_terms()["schedule"]]
        schedule[2]["startDate"] = "2025-12-20"  # gap after period 1
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(schedule=schedule), FACE, VALUATION, PROFILE, None)
        assert excinfo.value.reason == SCHEDULE_INCONSISTENT

    def test_an_overlapping_schedule_is_refused(self):
        schedule = [dict(p) for p in _fixture_terms()["schedule"]]
        schedule[2]["startDate"] = "2025-11-15"
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(schedule=schedule), FACE, VALUATION, PROFILE, None)
        assert excinfo.value.reason == SCHEDULE_INCONSISTENT

    def test_a_schedule_disagreeing_with_maturity_is_refused(self):
        """The final coupon and redemption must fall on the same date."""
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(maturityDate="2027-06-15"), FACE, VALUATION, PROFILE, None)
        assert excinfo.value.reason == SCHEDULE_INCONSISTENT

    def test_a_zero_length_period_is_refused(self):
        schedule = [dict(p) for p in _fixture_terms()["schedule"]]
        schedule[0]["endDate"] = schedule[0]["startDate"]
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(schedule=schedule), FACE, VALUATION, PROFILE, None)
        assert excinfo.value.reason == SCHEDULE_INCONSISTENT

    def test_a_payment_before_its_accrual_end_is_refused(self):
        schedule = [dict(p) for p in _fixture_terms()["schedule"]]
        schedule[1]["paymentDate"] = "2025-10-01"
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(schedule=schedule), FACE, VALUATION, PROFILE, None)
        assert excinfo.value.reason == SCHEDULE_INCONSISTENT

    def test_a_missing_schedule_is_refused_not_derived_from_frequency(self):
        """A schedule is not generated from `couponFrequency` and maturity."""
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(schedule=[]), FACE, VALUATION, PROFILE, None)
        # An empty schedule fails `is_note` first, so the reason is NOT_A_NOTE.
        assert excinfo.value.reason == NOT_A_NOTE


class TestRefusals:
    """Conditions the note pricer refuses rather than approximates."""

    def test_a_bill_is_not_priced_as_a_note(self):
        """A bill is not priced with the note model."""
        bill_terms = TermsEntry(
            instrument_type="TREASURY",
            terms={"couponFrequency": "NONE", "schedule": [],
                   "maturityDate": "2025-12-02", "dayCount": "NOT_APPLICABLE"},
            missing_terms=(), provenance={}, identity={},
        )
        with pytest.raises(NotePricingError) as excinfo:
            price_note(bill_terms, FACE, VALUATION, PROFILE, None)
        assert excinfo.value.reason == NOT_A_NOTE

    def test_a_matured_note_is_refused_not_priced_at_par(self):
        past = ORE.Date(2, 6, 2027)
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(), FACE, past, PROFILE, None)
        assert excinfo.value.reason == INSTRUMENT_MATURED

    def test_maturity_on_the_valuation_date_is_refused(self):
        """A note maturing on the valuation date is refused."""
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(), FACE, MATURITY, PROFILE, None)
        assert excinfo.value.reason == INSTRUMENT_MATURED

    def test_the_day_before_maturity_still_prices(self):
        """The day before maturity still prices (so the refusal above is not blanket)."""
        priced = price_note(_terms(), FACE, ORE.Date(14, 12, 2026), PROFILE, None)
        assert priced.dirty_npv > 0

    def test_a_settlement_lag_is_refused_not_ignored(self):
        """A settlement lag is refused, not ignored."""
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(settlementDays=1), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert excinfo.value.reason == SETTLEMENT_CONVENTION_NOT_SUPPORTED

    def test_zero_settlement_days_is_accepted(self):
        """Zero settlement days is accepted."""
        assert price_note(_terms(settlementDays=0), FACE, VALUATION, PROFILE,
                          EXPORTED_ACCRUED).dirty_npv > 0

    def test_an_absent_coupon_rate_is_refused(self):
        terms = dict(_fixture_terms())
        del terms["couponRatePercent"]
        entry = TermsEntry(instrument_type="TREASURY", terms=terms,
                           missing_terms=(), provenance={}, identity={})
        with pytest.raises(NotePricingError) as excinfo:
            price_note(entry, FACE, VALUATION, PROFILE, None)
        assert excinfo.value.reason == TERMS_INCOMPLETE

    def test_an_unparseable_coupon_rate_is_refused(self):
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(couponRatePercent="four"), FACE, VALUATION, PROFILE, None)
        assert excinfo.value.reason == TERMS_INCOMPLETE

    def test_an_absent_redemption_fraction_defaults_to_par(self):
        """An absent redemption fraction means par; an unparseable one is refused."""
        terms = dict(_fixture_terms())
        del terms["redemptionFraction"]
        entry = TermsEntry(instrument_type="TREASURY", terms=terms,
                           missing_terms=(), provenance={}, identity={})
        priced = price_note(entry, FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.redemption_fraction == 1.0

    def test_an_unparseable_redemption_fraction_is_refused(self):
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(redemptionFraction="par"), FACE, VALUATION, PROFILE, None)
        assert excinfo.value.reason == TERMS_INCOMPLETE

    def test_a_malformed_schedule_entry_is_refused(self):
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(schedule=["2025-06-15"]), FACE, VALUATION, PROFILE, None)
        assert excinfo.value.reason == TERMS_INCOMPLETE


class TestRefusalsAreNotePricingErrors:
    """Regression: a note's refusal is a `NotePricingError`, never a `BillPricingError`.

    `note.py` once reused `bill._parse_date`, so a malformed date in a note raised
    `BillPricingError`, which the pipeline's note handler does not catch; one bad row then
    failed the whole bundle (and the message talked about a bill).
    """

    @staticmethod
    def _bad_date_terms():
        schedule = [dict(p) for p in _fixture_terms()["schedule"]]
        schedule[1]["endDate"] = "not-a-date"
        return _terms(schedule=schedule)

    def test_a_malformed_schedule_date_raises_note_not_bill(self):
        from engine.traderx.bill import BillPricingError

        with pytest.raises(NotePricingError) as excinfo:
            price_note(self._bad_date_terms(), FACE, VALUATION, PROFILE, None)
        assert not isinstance(excinfo.value, BillPricingError)
        assert excinfo.value.reason == TERMS_INCOMPLETE

    def test_an_absent_maturity_raises_note_not_bill(self):
        from engine.traderx.bill import BillPricingError

        terms = dict(_fixture_terms())
        del terms["maturityDate"]
        entry = TermsEntry(instrument_type="TREASURY", terms=terms,
                           missing_terms=(), provenance={}, identity={})
        with pytest.raises(NotePricingError) as excinfo:
            price_note(entry, FACE, VALUATION, PROFILE, None)
        assert not isinstance(excinfo.value, BillPricingError)

    def test_no_refusal_message_mentions_a_bill(self):
        """No note refusal message mentions a bill."""
        terms = dict(_fixture_terms())
        del terms["maturityDate"]
        entry = TermsEntry(instrument_type="TREASURY", terms=terms,
                           missing_terms=(), provenance={}, identity={})
        for bad in (entry, self._bad_date_terms()):
            with pytest.raises(NotePricingError) as excinfo:
                price_note(bad, FACE, VALUATION, PROFILE, None)
            assert "bill" not in excinfo.value.detail.lower()

    def test_one_malformed_note_does_not_fail_the_whole_bundle(self):
        """Through the pipeline: a malformed note comes back as a refused item rather than
        raising out of the bundle."""
        from engine.traderx.market_inputs import MarketInputs
        from engine.traderx.pipeline import _note_outcomes
        from engine.traderx.terms import JoinedRow

        joined = JoinedRow(
            source="positions",
            row={"currency": "USD", "accruedInterestFraction": "0.018571"},
            entry=self._bad_date_terms(),
        )
        market = MarketInputs(
            mode="assumed-profile", profile=ASSUMED_PROFILES["flat-3pct-v1"],
        )
        # Before the fix this raised BillPricingError.
        outcomes = _note_outcomes(joined, FACE, market, "2025-06-02")
        assert outcomes["npv"].status == "unsupported"
        assert outcomes["npv"].reason == TERMS_INCOMPLETE
        assert outcomes["rateSensitivity"].status == "unsupported"


class TestIsNote:
    """`is_note` keys on the terms, never on the CSV's coupon column."""

    def test_the_delivered_note_is_a_note(self):
        assert is_note(_terms())

    def test_none_is_not_a_note(self):
        assert not is_note(None)

    def test_zero_coupon_frequency_is_not_a_note(self):
        assert not is_note(_terms(couponFrequency="NONE"))

    def test_absent_coupon_frequency_is_not_a_note(self):
        terms = dict(_fixture_terms())
        del terms["couponFrequency"]
        assert not is_note(TermsEntry(
            instrument_type="TREASURY", terms=terms, missing_terms=(),
            provenance={}, identity={},
        ))

    def test_a_frequency_with_no_schedule_is_not_a_note(self):
        """A frequency without a schedule is not a note (no schedule is generated)."""
        assert not is_note(_terms(schedule=[]))

    def test_bill_and_note_predicates_are_mutually_exclusive(self):
        """`is_bill` and `is_note` never both hold, so dispatch order cannot matter."""
        from engine.traderx.bill import is_bill

        note = _terms()
        assert is_note(note) and not is_bill(note)

        bill = TermsEntry(
            instrument_type="TREASURY",
            terms={"couponFrequency": "NONE", "schedule": [],
                   "maturityDate": "2025-12-02"},
            missing_terms=(), provenance={}, identity={},
        )
        assert is_bill(bill) and not is_note(bill)


class TestImpossibleCalendarDates:
    """Regression (TraderX v5): an impossible date such as 2025-02-30 aborted the whole
    bundle.

    `ORE.Date(30, 2, 2025)` raises `RuntimeError` (SWIG's form of QuantLib's
    `std::runtime_error`), which the parsers did not catch. One bad row must not cost the
    others their results. Tested through `price_bundle`, since the escape was between the
    pricer and the pipeline.
    """

    @staticmethod
    def _bundle_with_date(tmp_path, bad_date, case="note"):
        """A copy of a real bundle with `bad_date` in the terms and the manifest hash
        re-pinned, so the date (not the integrity check) is under test."""
        import hashlib
        import shutil

        root = tmp_path / f"{case}-{abs(hash(bad_date))}"
        shutil.copytree(FIXTURES / case / "v2", root)

        terms_path = root / "instrument-terms.json"
        terms = json.loads(terms_path.read_bytes())
        terms["entries"][0]["terms"]["maturityDate"] = bad_date
        raw = json.dumps(terms, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        terms_path.write_bytes(raw)

        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["artifacts"]["instrumentTerms"]["sha256"] = hashlib.sha256(raw).hexdigest()
        manifest_path.write_bytes(
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        )
        return root

    #: `not-a-date` fails in `int()` (ValueError); the rest parse as integers and fail in
    #: ORE with RuntimeError.
    BAD_DATES = ("2025-02-30", "2025-13-01", "2025-00-10", "2025-02-29", "not-a-date")

    @pytest.mark.parametrize("bad_date", BAD_DATES)
    def test_the_bundle_still_returns(self, tmp_path, bad_date):
        """Before the fix, 2025-02-30 raised out of `price_bundle`."""
        result = price_bundle(self._bundle_with_date(tmp_path, bad_date), MARKET)
        assert result.items

    @pytest.mark.parametrize("bad_date", BAD_DATES)
    def test_every_row_survives(self, tmp_path, bad_date):
        """Both positions are still reported."""
        result = price_bundle(self._bundle_with_date(tmp_path, bad_date), MARKET)
        assert len(result.items) == 2

    @pytest.mark.parametrize("bad_date", BAD_DATES)
    def test_it_is_an_item_level_refusal(self, tmp_path, bad_date):
        result = price_bundle(self._bundle_with_date(tmp_path, bad_date), MARKET)
        for item in result.items:
            npv = item.calculations["npv"]
            assert npv.status == "unsupported"
            assert npv.reason == TERMS_INCOMPLETE

    @pytest.mark.parametrize("bad_date", BAD_DATES)
    def test_the_refusal_is_identified(self, tmp_path, bad_date):
        """The refused items still carry their identity."""
        result = price_bundle(self._bundle_with_date(tmp_path, bad_date), MARKET)
        for item in result.items:
            assert item.item_id
            assert item.identity.account_id
            assert item.identity.security == "UST-NOTE-20261215"

    @pytest.mark.parametrize("bad_date", BAD_DATES)
    def test_the_detail_names_the_offending_value(self, tmp_path, bad_date):
        """The detail names the field and the bad value."""
        result = price_bundle(self._bundle_with_date(tmp_path, bad_date), MARKET)
        detail = result.items[0].calculations["npv"].detail
        assert "maturityDate" in detail
        assert bad_date in detail

    def test_coverage_still_sums(self, tmp_path):
        """Coverage stays consistent when every row refuses."""
        result = price_bundle(self._bundle_with_date(tmp_path, "2025-02-30"), MARKET)
        assert result.coverage.all_outcomes_accounted_for
        assert result.coverage.all_applicable_computed is False

    @pytest.mark.parametrize("bad_date", ("2025-02-30", "not-a-date"))
    def test_the_bill_path_is_hardened_too(self, tmp_path, bad_date):
        """The bill parser had the same gap and is fixed too."""
        result = price_bundle(self._bundle_with_date(tmp_path, bad_date, case="bill"), MARKET)
        assert len(result.items) == 2
        for item in result.items:
            assert item.calculations["npv"].status == "unsupported"

    def test_a_real_leap_day_is_still_accepted(self):
        """Real leap days parse (2028-02-29) and impossible ones refuse (2025-02-29).
        Tested on the parser: changing a fixture's maturity year would also break its
        schedule and refuse for an unrelated reason."""
        from engine.traderx.note import _parse_date

        assert _parse_date("2028-02-29", "maturityDate") == ORE.Date(29, 2, 2028)

        with pytest.raises(NotePricingError) as excinfo:
            _parse_date("2025-02-29", "maturityDate")
        assert excinfo.value.reason == TERMS_INCOMPLETE

    def test_the_unchanged_fixture_still_prices(self):
        """The delivered fixture still prices."""
        result = price_bundle(FIXTURES / "note" / "v2", MARKET)
        for item in result.items:
            assert item.calculations["npv"].status == "ok"


class TestPipelineEndToEnd:
    """The note through the whole path: bundle -> hash -> join -> normalize ->
    conventions -> price -> identified result."""

    @staticmethod
    def _result():
        return price_bundle(FIXTURES / "note" / "v2", MARKET)

    def test_both_positions_price(self):
        result = self._result().to_dict()
        assert len(result["items"]) == 2
        for item in result["items"]:
            assert item["calculations"]["npv"]["status"] == "ok"

    def test_long_and_short_sum_to_zero(self):
        """Long and short sum to zero, end to end."""
        values = [i["calculations"]["npv"]["value"] for i in self._result().to_dict()["items"]]
        assert sum(values) == pytest.approx(0.0, abs=1e-9)

    def test_the_npv_matches_the_independent_ore_valuation(self):
        """Parity with ORE through the public entry point."""
        expected = _priced_reference().dirtyPrice() / 100.0 * FACE
        values = [i["calculations"]["npv"]["value"] for i in self._result().to_dict()["items"]]
        assert max(values) == pytest.approx(expected, rel=1e-12)

    def test_rate_sensitivity_is_present_and_labelled(self):
        """The note's sensitivity is present and labelled (a missing branch would drop it
        silently, as in I-01); the bill's absence of one is tested in
        test_traderx_pipeline.py."""
        item = self._result().to_dict()["items"][0]
        sensitivity = item["calculations"]["rateSensitivity"]
        assert sensitivity["status"] == "ok"
        assert sensitivity["value"] != 0.0
        assert sensitivity["method"] == "bumped-revaluation"
        assert sensitivity["bump"] == RATE_BUMP
        assert sensitivity["shockedFactor"] == "zero-curve-parallel"
        assert sensitivity["currency"] == "USD"

    def test_accrued_interest_reports_the_exported_value(self):
        item = self._result().to_dict()["items"][0]
        accrued = item["calculations"]["accruedInterest"]
        assert accrued["status"] == "ok"
        assert accrued["value"] == pytest.approx(EXPORTED_ACCRUED * FACE, rel=1e-12)

    def test_the_npv_payload_carries_the_accrual_reconciliation(self):
        """The accrual reconciliation reaches the published result."""
        item = self._result().to_dict()["items"][0]
        block = item["calculations"]["npv"]["accrualReconciliation"]
        assert block["accrualSource"] == ACCRUAL_EXPORTED
        assert block["exportedFraction"] == EXPORTED_ACCRUED
        assert block["recomputedFraction"] == pytest.approx(EXACT_ACCRUED)

    def test_gamma_and_theta_are_still_unsupported(self):
        """`rateGamma` and `theta` are still unsupported."""
        item = self._result().to_dict()["items"][0]
        for name in ("rateGamma", "theta"):
            assert item["calculations"][name]["status"] == "unsupported"
            assert item["calculations"][name]["reason"] == "NO_PRICER_AT_THIS_STAGE"

    def test_vega_is_not_applicable_not_unsupported(self):
        """A Treasury has no optionality: vega is not-applicable, not a gap."""
        item = self._result().to_dict()["items"][0]
        assert item["calculations"]["vega"]["status"] == "not-applicable"

    def test_market_provenance_is_assumed(self):
        assert self._result().to_dict()["marketProvenance"] == "assumed"

    def test_every_item_is_identified(self):
        for item in self._result().to_dict()["items"]:
            assert item["itemId"]
            assert item["sourceIdentity"]["security"] == "UST-NOTE-20261215"
            assert item["sourceIdentity"]["accountId"]

    def test_coverage_counts_the_priced_calculations(self):
        coverage = self._result().to_dict()["coverage"]
        assert coverage["byCalculation"]["npv"]["ok"] == 2
        assert coverage["byCalculation"]["rateSensitivity"]["ok"] == 2
        assert coverage["allOutcomesAccountedFor"] is True

    def test_a_v1_bundle_prices_nothing(self):
        """Without terms the engine cannot tell the row is a note, so nothing prices."""
        result = price_bundle(FIXTURES / "note" / "v1", MARKET).to_dict()
        for item in result["items"]:
            assert item["calculations"]["npv"]["status"] == "unsupported"
            assert item["calculations"]["rateSensitivity"]["status"] == "unsupported"

    def test_no_market_inputs_prices_nothing(self):
        """Without market inputs nothing prices and no curve is substituted."""
        result = price_bundle(FIXTURES / "note" / "v2").to_dict()
        for item in result["items"]:
            assert item["calculations"]["npv"]["status"] == "unsupported"
        assert result["marketProvenance"] is None


class TestBillIsUnchangedByW13:
    """Adding the note pricer left the bill and SOFR results unchanged."""

    def test_the_bill_still_prices_to_its_delivered_value(self):
        """The bill still prices to +/-98,507.15."""
        result = price_bundle(FIXTURES / "bill" / "v2", MARKET).to_dict()
        values = sorted(i["calculations"]["npv"]["value"] for i in result["items"])
        assert values[1] == pytest.approx(98_507.15, abs=0.01)
        assert values[0] == pytest.approx(-98_507.15, abs=0.01)

    def test_the_bill_still_has_no_rate_sensitivity(self):
        """The bill still has no rate sensitivity."""
        result = price_bundle(FIXTURES / "bill" / "v2", MARKET).to_dict()
        for item in result["items"]:
            assert item["calculations"]["rateSensitivity"]["status"] == "unsupported"

    def test_the_sofr_refusal_is_unchanged(self):
        """The SOFR refusal is unchanged."""
        result = price_bundle(FIXTURES / "sofr" / "v2", MARKET).to_dict()
        refused = [i for i in result["items"] if i.get("refusal")]
        assert refused
        assert refused[0]["refusal"]["reason"] == "CONVENTION_NOT_SUPPORTED"
        assert len(refused[0]["refusal"]["missingTerms"]) == 13


class TestCapabilitiesAdvertiseW13:
    """The capability document advertises the note pricer."""

    def test_stage_is_at_least_w13(self):
        """The stage is at least W1.3 (a floor, since exact stage strings broke twice as the
        stage advanced)."""
        stage = capabilities()["deliveryStage"]
        assert stage.startswith("W1.")
        assert float(stage[1:]) >= 1.3

    def test_treasury_advertises_both_calculations(self):
        treasury = capabilities()["products"]["TREASURY"]
        assert treasury["priced"] is True
        assert set(treasury["calculations"]) == {"npv", "rateSensitivity"}

    def test_the_per_shape_split_is_advertised(self):
        """Per-shape: a bill has no sensitivity, a note does."""
        by_shape = capabilities()["products"]["TREASURY"]["byShape"]
        assert by_shape["zero-coupon"] == ["npv"]
        assert set(by_shape["coupon-bearing"]) == {"npv", "rateSensitivity"}

    def test_the_union_is_derived_from_the_per_shape_table(self):
        """The union is derived from the per-shape table."""
        for instrument_type, shapes in PRICED_CALCULATIONS_BY_SHAPE.items():
            expected = {calc for calcs in shapes.values() for calc in calcs}
            assert set(PRICED_CALCULATIONS[instrument_type]) == expected

    def test_gamma_and_theta_are_advertised_nowhere(self):
        for product in capabilities()["products"].values():
            assert "rateGamma" not in product["calculations"]
            assert "theta" not in product["calculations"]


class TestSensitivityUsesTheDeclaredFractionDecimals:
    """I-40: an exported fraction 1e-5 above the exact one is $1.00 on 100,000 face. With
    `fractionDecimals: 4` the tolerance is $5.01, so the NPV prices; the sensitivity used to
    reprice at the default 6 decimals ($0.06), refuse with ACCRUAL_MISMATCH, and take the
    NPV down with it in the pipeline."""

    OFF_BY_ONE_DOLLAR = EXACT_ACCRUED + 1e-5

    def test_sensitivity_reconciles_at_the_same_precision_as_the_npv(self):
        price_note(_terms(), FACE, VALUATION, PROFILE, self.OFF_BY_ONE_DOLLAR, fraction_decimals=4)
        declared = rate_sensitivity(_terms(), FACE, VALUATION, PROFILE, self.OFF_BY_ONE_DOLLAR,
                                    fraction_decimals=4)
        exact = rate_sensitivity(_terms(), FACE, VALUATION, PROFILE, EXACT_ACCRUED)
        assert declared == pytest.approx(exact, rel=1e-12)

    def test_the_default_precision_still_refuses_it(self):
        with pytest.raises(NotePricingError) as excinfo:
            rate_sensitivity(_terms(), FACE, VALUATION, PROFILE, self.OFF_BY_ONE_DOLLAR)
        assert excinfo.value.reason == ACCRUAL_MISMATCH

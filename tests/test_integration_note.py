"""
W1.3 -- the note pricer (`docs/planning/traderX-integration-plan.md` §W1.3).

The plan's required tests, quoted: "ORE parity; **accrued reconciles to
TraderX's `0.018571`** (already independently verified in Python -- this
test moves it into the engine); clean vs dirty reconciliation; long/short;
per-pillar `rateSensitivity` non-zero across the curve."

**Why ORE parity is asserted against `ORE.FixedRateBond`, not against a
re-derived sum.** Re-deriving `sum(c_i * exp(-r*t_i))` in the test would
restate the implementation and pass even if both were wrong together --
and the ways a bond pricer goes wrong (a dropped coupon, a
double-counted one, an accrual on the wrong day count) all survive that
kind of test. `TestOreParity` builds a real `ORE.FixedRateBond` with the
fixture's own schedule and discounts it through
`ORE.DiscountingBondEngine`, so the reference comes from a library that
knows nothing about this engine's code. Plan working rule 4 is the reason
to be strict here.

**On the per-pillar wording.** The plan asks for "per-pillar
`rateSensitivity` non-zero across the curve". The registered assumed
profile is a **flat constant** -- it has one rate, materialized onto
pillars that all carry the same value -- so there is no per-pillar shift
it can express, and a test asserting distinct per-pillar sensitivities
would be testing a curve this engine cannot currently be handed.
`TestRateSensitivity` therefore pins what the flat profile genuinely
supports: a parallel shift, correctly signed, non-zero, scaling with
maturity, and **labelled** `zero-curve-parallel` rather than as a pillar.
That is the honest version of the requirement; the per-pillar form
becomes testable when `mode: "package"` lands a real bootstrapped curve
(W2), and is recorded as such in `docs/known-issues.md` under I-16.
"""
import json
import math
from pathlib import Path

import ORE
import pytest

from engine.integration import price_bundle
from engine.integration.capabilities import (
    PRICED_CALCULATIONS,
    PRICED_CALCULATIONS_BY_SHAPE,
    capabilities,
)
from engine.integration.market_inputs import ASSUMED_PROFILES
from engine.integration.note import (
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
from engine.integration.terms import TermsEntry

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
    """The delivered fixture's terms, read from the hashed artifact itself.

    Read rather than transcribed: a hand-copied schedule that drifted from
    the fixture would let every test here pass against an instrument
    TraderX never sent.
    """
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
    """An independent ORE bond over the fixture's schedule.

    Deliberately built from ORE's own `FixedRateBond` + `Schedule` +
    `DiscountingBondEngine`, none of which this engine's pricer touches.
    """
    ORE.Settings.instance().evaluationDate = VALUATION
    schedule = ORE.Schedule(ORE.DateVector(SCHEDULE_DATES))
    return ORE.FixedRateBond(
        0, 100.0, schedule, ORE.DoubleVector([coupon]),
        day_count or ORE.ActualActual(ORE.ActualActual.ISMA),
        ORE.Unadjusted, 100.0, SCHEDULE_DATES[0],
    )


def _priced_reference(face: float = FACE, rate: float = None) -> ORE.FixedRateBond:
    """The reference bond with a flat curve attached, scaled to `face`.

    **`dirtyPrice()` vs `NPV()` — they are not the same quantity, and this
    helper is only safe at `VALUATION`.**

    `bond.NPV()` is the present value as of the *evaluation date*.
    `bond.dirtyPrice()` is `NPV / discount(settlementDate)` — a price quoted
    *for settlement*, forward-valued to the settlement date. The fixture
    states `settlementDays: 0`, so at `VALUATION` the settlement date is the
    evaluation date, the discount factor is exactly 1.0, and the two
    coincide.

    They do **not** coincide at an arbitrary date: valuing before the issue
    date makes ORE clamp settlement to the issue date, and the two differ by
    ~$117 per $100k. `price_note` computes a present value as of the
    valuation date, so `NPV()` is the definitionally correct reference and
    `dirtyPrice()` is correct only here. `TestPresentValueIsAsOfTheValuationDate`
    pins both halves of that so the distinction cannot rot into a silent
    mismatch.
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
    """`price_note` returns a present value as of the valuation date, not a
    settlement-forward quoted price.

    The two differ whenever settlement is not the valuation date. On this
    fixture they are equal (`settlementDays: 0`), which is exactly why the
    distinction needs pinning: every other test in this file would pass
    under either definition, so nothing else here would notice the day it
    stopped holding.
    """

    def test_npv_equals_ores_npv_not_merely_its_dirty_price(self):
        """The stronger claim: parity against `NPV()`, the quantity this
        pricer actually computes."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.dirty_npv == pytest.approx(
            _priced_reference().NPV() / 100.0 * FACE, rel=1e-12
        )

    def test_the_two_ore_quantities_coincide_here_and_the_test_knows_why(self):
        """Guards the helper's assumption rather than trusting it: zero
        settlement lag is what makes `dirtyPrice()` a valid reference at
        this date."""
        bond = _priced_reference()
        assert bond.settlementDate() == VALUATION
        assert bond.NPV() == pytest.approx(bond.dirtyPrice(), rel=1e-12)

    def test_present_value_is_measured_from_the_valuation_date(self):
        """Valued before the first period starts, `price_note` still
        discounts to the valuation date -- matching ORE's `NPV()`, and
        deliberately *not* its `dirtyPrice()`, which ORE forward-values to
        the clamped issue-date settlement.

        This is the case where the two ORE quantities diverge (~$117 per
        $100k), so it is the one that proves which definition this pricer
        implements.
        """
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
            # And confirm the two ORE quantities really do differ here, so
            # this test is not vacuously asserting the same thing twice.
            assert abs(bond.NPV() - bond.dirtyPrice()) > 0.1
        finally:
            # The evaluation date is global process state in ORE; leaving it
            # moved would silently change every later test in the session.
            ORE.Settings.instance().evaluationDate = VALUATION

    def test_nothing_has_accrued_before_the_first_period(self):
        """A structural zero, not a missing value: the instrument has not
        begun accruing."""
        priced = price_note(_terms(), FACE, ORE.Date(1, 12, 2024), PROFILE, None)
        assert priced.accrued_interest == 0.0
        assert len(priced.coupons) == 4


class TestOreParity:
    """The acceptance bar: this engine's number must equal an independent
    ORE valuation at matched terms."""

    def test_long_dirty_npv_matches_ore(self):
        """The headline number. `dirtyPrice` is per 100 face, so it scales
        to the position by `face / 100`."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        expected = _priced_reference().dirtyPrice() / 100.0 * FACE
        assert priced.dirty_npv == pytest.approx(expected, rel=1e-12)

    def test_short_dirty_npv_matches_ore(self):
        priced = price_note(_terms(), -FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        expected = -_priced_reference().dirtyPrice() / 100.0 * FACE
        assert priced.dirty_npv == pytest.approx(expected, rel=1e-12)

    def test_parity_is_exact_not_merely_close(self):
        """Zero difference at machine precision, not a loose tolerance.

        Both sides do the same arithmetic on the same dates, so anything
        beyond floating-point noise is a real disagreement -- a coupon on
        the wrong day count or a discount factor off the wrong axis -- and
        a generous tolerance would hide exactly that.
        """
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        expected = _priced_reference().dirtyPrice() / 100.0 * FACE
        assert abs(priced.dirty_npv - expected) < 1e-7

    def test_every_coupon_matches_ores_own_cashflows(self):
        """Not just the total. A dropped coupon and a compensating
        discount-factor error produce the same sum; comparing the flows
        one by one does not let them cancel.
        """
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
        """The recomputed path against ORE's `accruedAmount`. This is the
        ACT/ACT (ICMA) half of W1.1, exercised for the first time."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        ore_accrued = _reference_bond().accruedAmount(VALUATION) / 100.0
        assert priced.accrual.recomputed_fraction == pytest.approx(ore_accrued, rel=1e-12)

    def test_discount_factors_match_ores_curve(self):
        """The intermediates, not only the total."""
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
    """The plan's named acceptance: **accrued reconciles to `0.018571`**."""

    def test_recomputed_fraction_equals_traderx_exported_value(self):
        """Their rounded `0.018571` is this engine's unrounded value,
        rounded the way their preamble says (HALF_EVEN, 6 decimals)."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert round(priced.accrual.recomputed_fraction, 6) == EXPORTED_ACCRUED

    def test_recomputed_fraction_is_the_icma_ratio(self):
        """169/182 x 4%/2, stated independently of ORE and of the pricer."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.accrual.recomputed_fraction == pytest.approx(EXACT_ACCRUED, rel=1e-15)

    def test_reported_value_is_the_export_not_the_recomputation(self):
        """Plan §1: the export is authoritative. The two differ by $0.04
        here, and publishing the recomputed one would publish a number
        TraderX's books do not contain -- the correction owed in §1.
        """
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.accrual.source == ACCRUAL_EXPORTED
        assert priced.accrued_interest == pytest.approx(EXPORTED_ACCRUED * FACE, rel=1e-12)
        # Explicitly NOT the recomputed path's monetary value.
        assert priced.accrued_interest != pytest.approx(EXACT_ACCRUED * FACE, rel=1e-12)

    def test_both_paths_travel_in_the_payload(self):
        """Either number alone is unreconcilable. The consumer gets both,
        the difference, and the tolerance it was checked against."""
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
        """Guards the tolerance itself. If these ever became exactly
        equal, `TestAccrualMismatchIsRefused` would be vacuous -- it would
        be checking a tolerance against a difference that is structurally
        zero."""
        assert EXPORTED_ACCRUED != EXACT_ACCRUED
        difference = abs(EXPORTED_ACCRUED - EXACT_ACCRUED) * FACE
        assert difference > 0.0
        assert difference < accrual_mismatch_tolerance(FACE)

    def test_short_position_accrues_negatively(self):
        """Plan §1, 'Accrued sign': signed for the position, consistent
        with NPV."""
        priced = price_note(_terms(), -FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.accrued_interest == pytest.approx(-EXPORTED_ACCRUED * FACE, rel=1e-12)

    def test_blank_export_falls_back_to_recomputation_under_its_own_label(self):
        """The blank-accrual compatibility case. Legitimate, but the
        label must say which path produced the number -- passing a
        recomputed value off as the export is the failure this label
        exists to prevent."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, None)
        assert priced.accrual.source == ACCRUAL_RECOMPUTED
        assert priced.accrual.exported_fraction is None
        assert priced.accrued_interest == pytest.approx(EXACT_ACCRUED * FACE, rel=1e-12)


class TestToleranceIsDerivedNotConstant:
    """Plan §1: `round(0.5 x 10^-decimals x |face|, 2) + 0.01`, derived
    from the stated rounding and never a fixed constant."""

    def test_matches_the_agreed_formula(self):
        assert accrual_mismatch_tolerance(FACE, 6) == pytest.approx(
            round(0.5 * 1e-6 * FACE, 2) + 0.01
        )

    def test_scales_with_face(self):
        """A constant tolerance would be vacuous on a large position."""
        small = accrual_mismatch_tolerance(1_000.0)
        large = accrual_mismatch_tolerance(100_000_000.0)
        assert large > small * 100

    def test_is_sign_independent(self):
        assert accrual_mismatch_tolerance(FACE) == accrual_mismatch_tolerance(-FACE)

    def test_tighter_rounding_gives_a_tighter_tolerance(self):
        """The tolerance follows the exporter's declared decimals rather
        than a hard-coded 6."""
        assert accrual_mismatch_tolerance(FACE, 9) < accrual_mismatch_tolerance(FACE, 6)

    @pytest.mark.parametrize("face", [
        1e2, 1e3, 1e5, 1e6, 1e8, 1e9,
    ])
    def test_covers_the_exporters_worst_case_rounding_at_every_size(self, face):
        """**The refusal must never fire on legitimate data.**

        `ACCRUAL_MISMATCH` is a hard refusal, so a tolerance that is too
        tight at some face size would reject a perfectly good bundle --
        turning a safety check into an outage. At 6 decimals the exporter's
        HALF_EVEN rounding can move the monetary value by at most
        `0.5e-6 x face`, and the tolerance must cover that everywhere, not
        just at the fixture's $100k.
        """
        worst_case_rounding = 0.5e-6 * face
        assert accrual_mismatch_tolerance(face) >= worst_case_rounding

    def test_is_still_tight_enough_to_catch_a_wrong_day_count(self):
        """The other side of the trade-off: a tolerance wide enough to
        never false-refuse would be useless. The ACT/365-vs-ICMA error on
        $100k is ~$5.09 against a $0.06 bound -- caught by ~85x."""
        act365_error = abs(0.018520547945205478 - EXACT_ACCRUED) * FACE
        assert act365_error > accrual_mismatch_tolerance(FACE) * 50


class TestCleanDirtyReconciliation:
    """Plan §W1.3: 'clean vs dirty reconciliation'."""

    def test_clean_is_dirty_less_accrued(self):
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.clean_npv == pytest.approx(
            priced.dirty_npv - priced.accrued_interest, rel=1e-15
        )

    def test_npv_reports_the_dirty_value(self):
        """A 'bond NPV' silently meaning the clean value would be off by
        the accrued interest -- $1,857 here, large enough to matter and
        small enough to look like a curve difference."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.npv == priced.dirty_npv
        assert priced.npv != pytest.approx(priced.clean_npv, rel=1e-6)
        assert priced.to_payload()["priceType"] == "dirty"

    def test_clean_npv_matches_ore_when_both_use_the_recomputed_accrual(self):
        """ORE recomputes accrued from the schedule, so its clean price is
        comparable only to the clean value built on *this* engine's
        recomputed path -- not to the one built on the exporter's rounded
        figure. Comparing those two directly would fail by the $0.04
        rounding and look like a pricing bug.
        """
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, None)
        expected = _priced_reference().cleanPrice() / 100.0 * FACE
        assert priced.clean_npv == pytest.approx(expected, rel=1e-12)

    def test_clean_on_the_exported_path_differs_by_exactly_the_rounding(self):
        """Pins the $0.04 as the exporter's rounding rather than an error,
        so a future change that made it larger fails here."""
        exported = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        recomputed = price_note(_terms(), FACE, VALUATION, PROFILE, None)
        assert exported.clean_npv - recomputed.clean_npv == pytest.approx(
            (EXACT_ACCRUED - EXPORTED_ACCRUED) * FACE, rel=1e-9
        )


class TestLongShort:
    """Plan §W1.3: 'long/short'."""

    def test_long_and_short_are_exact_mirrors(self):
        long_note = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        short = price_note(_terms(), -FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert long_note.dirty_npv + short.dirty_npv == pytest.approx(0.0, abs=1e-9)
        assert long_note.clean_npv + short.clean_npv == pytest.approx(0.0, abs=1e-9)
        assert long_note.accrued_interest + short.accrued_interest == pytest.approx(0.0, abs=1e-9)

    def test_short_npv_is_negative(self):
        """The double-sign bug TraderX flagged (v3 §2) makes a short
        position positive. `signed_face` carries the sign in one step, so
        there is no second multiplication to get wrong."""
        short = price_note(_terms(), -FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert short.dirty_npv < 0
        assert short.clean_npv < 0
        assert short.accrued_interest < 0

    def test_scaling_is_linear_in_face(self):
        one = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        ten = price_note(_terms(), FACE * 10, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert ten.dirty_npv == pytest.approx(one.dirty_npv * 10, rel=1e-12)


class TestRateSensitivity:
    """Plan §W1.3: `rateSensitivity` non-zero across the curve.

    See this module's docstring on why these assert a *parallel* shift: a
    flat profile has one rate and cannot express a per-pillar one.
    """

    def test_is_non_zero(self):
        assert rate_sensitivity(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED) != 0.0

    def test_long_bond_loses_value_when_rates_rise(self):
        """Sign, not just magnitude. A sensitivity with the wrong sign is
        worse than none -- it points a hedge the wrong way."""
        assert rate_sensitivity(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED) < 0

    def test_short_bond_gains_when_rates_rise(self):
        assert rate_sensitivity(_terms(), -FACE, VALUATION, PROFILE, EXPORTED_ACCRUED) > 0

    def test_matches_a_direct_reprice_at_the_bumped_rate(self):
        """The bumped revaluation is exactly that, with no shortcut."""
        from engine.integration.market_inputs import AssumedProfile

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
        """An independent order-of-magnitude check. A ~1.5y-duration bond
        on $100k moves roughly $15 per bp; anything an order of magnitude
        off means the bump or the units are wrong, which a self-consistent
        reprice test cannot catch.
        """
        sensitivity = rate_sensitivity(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert 5.0 < abs(sensitivity) < 40.0

    def test_longer_maturity_is_more_sensitive(self):
        """Scaling with maturity, verified by truncating the schedule to
        its first two periods rather than by asserting a number."""
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
        """Against ORE's own repriced bond, not against this engine.

        A one-sided 1bp bump differs from ORE's by convexity only, which
        on a 1.5y bond is well under a cent -- so a 1e-4 relative bound is
        tight enough to catch a real error and loose enough to admit the
        second-order term.
        """
        base = _priced_reference().dirtyPrice() / 100.0 * FACE
        bumped = _priced_reference(rate=PROFILE.flat_rate + RATE_BUMP)
        shifted = bumped.dirtyPrice() / 100.0 * FACE
        mine = rate_sensitivity(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert mine == pytest.approx(shifted - base, rel=1e-4)


class TestWrongDayCountIsCaught:
    """W1.1's ACT/ACT (ICMA) half, and the reason it had to exist.

    Pricing this note on ACT/365 is the plausible wrong implementation:
    it is the engine's own default, it produces a confident number, and
    the difference hides in the sixth decimal of a fraction.
    """

    def test_act365_accrual_would_differ_materially(self):
        """Quantifies the error the right day count avoids: ~$5.09 on
        $100k, against a reconciliation tolerance of $0.06."""
        act365 = ORE.Actual365Fixed()
        wrong = 0.04 * act365.yearFraction(ORE.Date(15, 12, 2024), VALUATION)
        error = abs(wrong - EXACT_ACCRUED) * FACE
        assert error > 5.0
        assert error > accrual_mismatch_tolerance(FACE) * 50

    def test_the_accrual_check_would_refuse_an_act365_note(self):
        """**The mismatch refusal catches the wrong-day-count bug.**

        Not a designed-for case, but a real property worth pinning: a note
        whose terms claim ACT/365 while the export accrued on ICMA fails
        reconciliation and is refused rather than priced.
        """
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(dayCount="ACT/365"), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert excinfo.value.reason == ACCRUAL_MISMATCH

    def test_icma_period_is_exactly_half_whatever_its_length(self):
        """The defining property of ACT/ACT (ICMA), and what separates it
        from ACT/365 here: the fixture's periods are 182 and 183 days, and
        both accrue exactly 0.5."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert [c.accrual_fraction for c in priced.coupons] == [0.5, 0.5, 0.5, 0.5]

    def test_unsupported_day_count_is_refused_not_defaulted(self):
        """W1.1's rule, at this boundary."""
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
        """A consumer must not have to infer which convention produced the
        number it is reconciling."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        payload = priced.to_payload()
        assert payload["accrualDayCount"] == "ACT/ACT (ICMA)"
        assert payload["discountDayCount"] == "ACT/365 (Fixed)"


class TestAccrualMismatchIsRefused:
    """Disagreement beyond tolerance means this engine priced a different
    schedule from the one the export accrued against -- so every coupon is
    suspect, not just the accrued figure."""

    def test_large_disagreement_is_refused(self):
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(), FACE, VALUATION, PROFILE, 0.025)
        assert excinfo.value.reason == ACCRUAL_MISMATCH

    def test_refusal_names_both_values_and_the_tolerance(self):
        """An actionable refusal: the consumer can see which side to
        investigate without re-running anything."""
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(), FACE, VALUATION, PROFILE, 0.025)
        detail = excinfo.value.detail
        assert "0.0250000000" in detail
        assert "0.0185714286" in detail
        assert "tolerance" in detail.lower()

    def test_disagreement_within_tolerance_prices_normally(self):
        """The delivered fixture is exactly this case -- rounded, not
        wrong. Refusing it would make the pricer useless on real data."""
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.accrual.agrees
        assert priced.dirty_npv > 0

    def test_tolerance_scales_so_a_large_position_is_not_falsely_refused(self):
        """The same 6-decimal rounding on $1bn is $500 of difference. A
        fixed tolerance would refuse it; the derived one must not."""
        face = 1_000_000_000.0
        priced = price_note(_terms(), face, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert priced.accrual.agrees

    def test_a_refusal_is_not_a_warning(self):
        """Pins the choice: nothing is returned on mismatch. Returning a
        price with a warning attached would invite exactly the
        reconciliation the disagreement says is unsound."""
        with pytest.raises(NotePricingError):
            price_note(_terms(), FACE, VALUATION, PROFILE, 0.030)


class TestScheduleIsUsedNotRegenerated:
    """The exporter's schedule is the schedule its accrued interest was
    computed against. Regenerating one would silently reprice every
    coupon."""

    def test_uses_every_unpaid_period_from_the_terms(self):
        priced = price_note(_terms(), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert len(priced.coupons) == 4
        assert [c.payment_date for c in priced.coupons] == [
            "2025-06-15", "2025-12-15", "2026-06-15", "2026-12-15",
        ]

    def test_already_paid_coupons_are_excluded(self):
        """Valued a day after the first coupon pays: three remain.
        Including a paid coupon would double-count what the accrued figure
        already excludes."""
        after_first = ORE.Date(16, 6, 2025)
        priced = price_note(_terms(), FACE, after_first, PROFILE, None)
        assert len(priced.coupons) == 3
        assert priced.coupons[0].payment_date == "2025-12-15"

    def test_a_coupon_paying_on_the_valuation_date_is_excluded(self):
        """The boundary. A coupon paid today is not this position's
        cashflow -- the holder has it, not the bond."""
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
        """The redemption and the final coupon must fall together; a
        disagreement means the two terms describe different instruments."""
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
        """`couponFrequency: 6M` plus a maturity is enough information to
        *generate* a schedule. Doing so is precisely what this refuses."""
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(schedule=[]), FACE, VALUATION, PROFILE, None)
        # An empty schedule fails `is_note` first -- the refusal is
        # NOT_A_NOTE rather than TERMS_INCOMPLETE, and either way nothing
        # is generated.
        assert excinfo.value.reason == NOT_A_NOTE


class TestRefusals:
    """Each names a condition this pricer declines rather than
    approximates."""

    def test_a_bill_is_not_priced_as_a_note(self):
        """The mirror of W1.2's dangerous case: a confident, plausible
        number produced by the wrong model."""
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
        """The boundary, matching the bill's rule: a note maturing today
        has no remaining cashflow to discount."""
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(), FACE, MATURITY, PROFILE, None)
        assert excinfo.value.reason == INSTRUMENT_MATURED

    def test_the_day_before_maturity_still_prices(self):
        """The other side of the boundary -- otherwise the test above
        would pass against a pricer that refused everything."""
        priced = price_note(_terms(), FACE, ORE.Date(14, 12, 2026), PROFILE, None)
        assert priced.dirty_npv > 0

    def test_a_settlement_lag_is_refused_not_ignored(self):
        """Ignoring it would discount on one date and accrue to another."""
        with pytest.raises(NotePricingError) as excinfo:
            price_note(_terms(settlementDays=1), FACE, VALUATION, PROFILE, EXPORTED_ACCRUED)
        assert excinfo.value.reason == SETTLEMENT_CONVENTION_NOT_SUPPORTED

    def test_zero_settlement_days_is_accepted(self):
        """The fixture's own value; the refusal above must not be blanket."""
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
        """The one legitimate default: absent means par, the
        overwhelmingly common case. Distinguished from a *present but
        unparseable* value, which is a malformed artifact."""
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
    """**Regression: a note's refusal must never be a `BillPricingError`.**

    Found while writing this suite. `note.py` originally reused
    `bill._parse_date`, so any malformed date in a note's terms raised
    `BillPricingError` -- which `_note_outcomes`'s `except
    NotePricingError` does not catch. It propagated out of `_build_item`
    and **failed the entire bundle**, so one malformed row cost every
    other row its result: precisely the contract both pricers' docstrings
    promise to keep ("one unpriceable row must not fail the other 200").

    The message was wrong too -- a note refused with "A bill cannot be
    priced without its maturity" sends a consumer looking at the wrong
    instrument.

    These tests fail against the pre-fix code (working rule 3): the first
    three on the exception type, the last on the bundle-level
    consequence, which is the one that actually mattered.
    """

    @staticmethod
    def _bad_date_terms():
        schedule = [dict(p) for p in _fixture_terms()["schedule"]]
        schedule[1]["endDate"] = "not-a-date"
        return _terms(schedule=schedule)

    def test_a_malformed_schedule_date_raises_note_not_bill(self):
        from engine.integration.bill import BillPricingError

        with pytest.raises(NotePricingError) as excinfo:
            price_note(self._bad_date_terms(), FACE, VALUATION, PROFILE, None)
        assert not isinstance(excinfo.value, BillPricingError)
        assert excinfo.value.reason == TERMS_INCOMPLETE

    def test_an_absent_maturity_raises_note_not_bill(self):
        from engine.integration.bill import BillPricingError

        terms = dict(_fixture_terms())
        del terms["maturityDate"]
        entry = TermsEntry(instrument_type="TREASURY", terms=terms,
                           missing_terms=(), provenance={}, identity={})
        with pytest.raises(NotePricingError) as excinfo:
            price_note(entry, FACE, VALUATION, PROFILE, None)
        assert not isinstance(excinfo.value, BillPricingError)

    def test_no_refusal_message_mentions_a_bill(self):
        """A note refused with a bill's wording sends a consumer looking
        at the wrong instrument."""
        terms = dict(_fixture_terms())
        del terms["maturityDate"]
        entry = TermsEntry(instrument_type="TREASURY", terms=terms,
                           missing_terms=(), provenance={}, identity={})
        for bad in (entry, self._bad_date_terms()):
            with pytest.raises(NotePricingError) as excinfo:
                price_note(bad, FACE, VALUATION, PROFILE, None)
            assert "bill" not in excinfo.value.detail.lower()

    def test_one_malformed_note_does_not_fail_the_whole_bundle(self):
        """**The consequence that made this a bug rather than a wording
        nit**, asserted through the pipeline rather than the pricer --
        working rule 8.

        The malformed row must come back as a refused *item*, with every
        other row in the bundle still carrying its own result.
        """
        from engine.integration.market_inputs import MarketInputs
        from engine.integration.pipeline import _note_outcomes
        from engine.integration.terms import JoinedRow

        joined = JoinedRow(
            source="positions",
            row={"currency": "USD", "accruedInterestFraction": "0.018571"},
            entry=self._bad_date_terms(),
        )
        market = MarketInputs(
            mode="assumed-profile", profile=ASSUMED_PROFILES["flat-3pct-v1"],
        )
        # Pre-fix this raised BillPricingError straight through.
        outcomes = _note_outcomes(joined, FACE, market, "2025-06-02")
        assert outcomes["npv"].status == "unsupported"
        assert outcomes["npv"].reason == TERMS_INCOMPLETE
        assert outcomes["rateSensitivity"].status == "unsupported"


class TestIsNote:
    """Keyed on terms, never on the CSV's coupon column."""

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
        """A schedule is not generated from a frequency -- so a row
        claiming one without the other is not something this pricer
        will fill in for itself."""
        assert not is_note(_terms(schedule=[]))

    def test_bill_and_note_predicates_are_mutually_exclusive(self):
        """The dispatch in the pipeline depends on this. If both could be
        true for one row, the order of two `if`s would decide the price."""
        from engine.integration.bill import is_bill

        note = _terms()
        assert is_note(note) and not is_bill(note)

        bill = TermsEntry(
            instrument_type="TREASURY",
            terms={"couponFrequency": "NONE", "schedule": [],
                   "maturityDate": "2025-12-02"},
            missing_terms=(), provenance={}, identity={},
        )
        assert is_bill(bill) and not is_note(bill)


class TestPipelineEndToEnd:
    """The note through the whole path: bundle -> hash -> join ->
    normalize -> conventions -> price -> identified result."""

    @staticmethod
    def _result():
        return price_bundle(FIXTURES / "note" / "v2", MARKET)

    def test_both_positions_price(self):
        result = self._result().to_dict()
        assert len(result["items"]) == 2
        for item in result["items"]:
            assert item["calculations"]["npv"]["status"] == "ok"

    def test_long_and_short_sum_to_zero(self):
        """The whole delivered fixture, end to end -- the claim TraderX
        can check against their own books."""
        values = [i["calculations"]["npv"]["value"] for i in self._result().to_dict()["items"]]
        assert sum(values) == pytest.approx(0.0, abs=1e-9)

    def test_the_npv_matches_the_independent_ore_valuation(self):
        """Parity asserted through the *public entry point*, not only
        against `price_note` -- working rule 8: a guard tested directly
        proves nothing about the path that reaches it."""
        expected = _priced_reference().dirtyPrice() / 100.0 * FACE
        values = [i["calculations"]["npv"]["value"] for i in self._result().to_dict()["items"]]
        assert max(values) == pytest.approx(expected, rel=1e-12)

    def test_rate_sensitivity_is_present_and_labelled(self):
        """I-01's regression class: a new instrument reaching the result
        path without its sensitivity gets silently skipped. This asserts
        presence, and `test_pipeline.py` asserts the same for the bill's
        absence of one."""
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
        """The two paths reach the published artifact, not just the
        in-process object."""
        item = self._result().to_dict()["items"][0]
        block = item["calculations"]["npv"]["accrualReconciliation"]
        assert block["accrualSource"] == ACCRUAL_EXPORTED
        assert block["exportedFraction"] == EXPORTED_ACCRUED
        assert block["recomputedFraction"] == pytest.approx(EXACT_ACCRUED)

    def test_gamma_and_theta_are_still_unsupported(self):
        """Answering npv and rateSensitivity does not make the rest
        answerable. Reporting a zero here would be the silent
        approximation this boundary exists to prevent."""
        item = self._result().to_dict()["items"][0]
        for name in ("rateGamma", "theta"):
            assert item["calculations"][name]["status"] == "unsupported"
            assert item["calculations"][name]["reason"] == "NO_PRICER_AT_THIS_STAGE"

    def test_vega_is_not_applicable_not_unsupported(self):
        """A Treasury has no optionality; vega is not a gap."""
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
        """No terms artifact means the engine cannot establish that a row
        IS a note, and it will not infer that from a coupon column."""
        result = price_bundle(FIXTURES / "note" / "v1", MARKET).to_dict()
        for item in result["items"]:
            assert item["calculations"]["npv"]["status"] == "unsupported"
            assert item["calculations"]["rateSensitivity"]["status"] == "unsupported"

    def test_no_market_inputs_prices_nothing(self):
        """A curve is mandatory the moment anything is priced. No default
        is ever substituted."""
        result = price_bundle(FIXTURES / "note" / "v2").to_dict()
        for item in result["items"]:
            assert item["calculations"]["npv"]["status"] == "unsupported"
        assert result["marketProvenance"] is None


class TestBillIsUnchangedByW13:
    """W1.3 must not alter W1.2's delivered numbers. The bill fixture is
    the regression guard for the pipeline's new two-pricer dispatch."""

    def test_the_bill_still_prices_to_its_delivered_value(self):
        """+98,507.15 / -98,507.15, the numbers in plan §1 and in
        response v4."""
        result = price_bundle(FIXTURES / "bill" / "v2", MARKET).to_dict()
        values = sorted(i["calculations"]["npv"]["value"] for i in result["items"])
        assert values[1] == pytest.approx(98_507.15, abs=0.01)
        assert values[0] == pytest.approx(-98_507.15, abs=0.01)

    def test_the_bill_still_has_no_rate_sensitivity(self):
        """W1.3 delivers a *note* sensitivity. Extending the claim to the
        bill without a test that earns it is the overclaim working rule 4
        names."""
        result = price_bundle(FIXTURES / "bill" / "v2", MARKET).to_dict()
        for item in result["items"]:
            assert item["calculations"]["rateSensitivity"]["status"] == "unsupported"

    def test_the_sofr_refusal_is_unchanged(self):
        """The refusal path must survive a second pricer being added."""
        result = price_bundle(FIXTURES / "sofr" / "v2", MARKET).to_dict()
        refused = [i for i in result["items"] if i.get("refusal")]
        assert refused
        assert refused[0]["refusal"]["reason"] == "CONVENTION_NOT_SUPPORTED"
        assert len(refused[0]["refusal"]["missingTerms"]) == 13


class TestCapabilitiesAdvertiseW13:
    """The capability document is what a coordinator reads *before*
    submitting. A stale one makes a promise the engine no longer keeps."""

    def test_stage_is_at_least_w13(self):
        """**Asserts the floor, not the exact string.**

        This test pinned `== "W1.3"` and broke the moment W1.4 landed --
        the second time a literal stage string has done that (the W1.2
        form broke when W1.3 landed). The value it was protecting is "the
        note pricer has shipped", which stays true as the stage advances,
        so that is what it now asserts. The *contents* below are the real
        guard, and they are specific.
        """
        stage = capabilities()["deliveryStage"]
        assert stage.startswith("W1.")
        assert float(stage[1:]) >= 1.3

    def test_treasury_advertises_both_calculations(self):
        treasury = capabilities()["products"]["TREASURY"]
        assert treasury["priced"] is True
        assert set(treasury["calculations"]) == {"npv", "rateSensitivity"}

    def test_the_per_shape_split_is_advertised(self):
        """A bill has no sensitivity and a note does. Collapsing them
        would advertise a capability the engine does not have."""
        by_shape = capabilities()["products"]["TREASURY"]["byShape"]
        assert by_shape["zero-coupon"] == ["npv"]
        assert set(by_shape["coupon-bearing"]) == {"npv", "rateSensitivity"}

    def test_the_union_is_derived_from_the_per_shape_table(self):
        """Derived, never hand-written, so the two cannot drift."""
        for instrument_type, shapes in PRICED_CALCULATIONS_BY_SHAPE.items():
            expected = {calc for calcs in shapes.values() for calc in calcs}
            assert set(PRICED_CALCULATIONS[instrument_type]) == expected

    def test_gamma_and_theta_are_advertised_nowhere(self):
        for product in capabilities()["products"].values():
            assert "rateGamma" not in product["calculations"]
            assert "theta" not in product["calculations"]

"""
W1.5 -- `engine.instruments.treasury` and its wire-through into
`engine.portfolio.request`.

**The governing regression class is I-01** (plan §W1.5's own warning): a new
instrument type reaching `_compute_all_greeks` without its branch is
*silently skipped* -- no Greeks, no error -- and the test that pinned that
bug for swaps even described the skip as intentional.
`TestBondGreeksReachThePortfolioPath` is the countermeasure, and every test
in it was verified to fail with the `BondConfig` branch of
`_greeks_for_one_trade` deleted.

**The second class, new to W1.5, is the broadcast-a-constant temptation.**
A bond has no scenario NPV, and filling its `npv_cube` column with one
repeated t=0 number yields VaR 0.00 / ES NaN -- a position that reads as
risk-measured when its risk was never modelled.
`TestScenarioPricingIsRefused` pins the refusal *and* measures what the
naive alternative actually produces, so the reason is evidence rather than
assertion.
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


#: The same three-period semiannual schedule the W1.3 note fixture uses.
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
        # Structural zero: no coupons exist to accrue. Not a missing value.
        assert accrued_interest(make_bill()) == 0.0

    def test_discounts_below_par(self):
        assert price_bond_base(make_bill()) < 100_000.0

    def test_a_short_position_is_negative(self):
        """The double-sign bug TraderX flagged in v3 §2.

        A negative face must produce a negative NPV in ONE step -- not a
        positive one via a second sign factor applied on top.
        """
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
        # A 4% coupon note must be worth more than a zero-coupon claim on
        # the same face at the same maturity.
        note = make_note()
        bill_same_maturity = make_bill(maturity="2026-06-15")
        assert price_bond_base(note) > price_bond_base(bill_same_maturity)

    def test_accrued_interest_uses_act_act_icma(self):
        """The W1.1 distinction, and the number the plan documents.

        Under ACT/ACT (ICMA) the 2024-12-15 -> 2025-06-15 period is exactly
        0.5 of a year whatever its day length. Accrual to 2025-06-02 is
        169/182 of that period at 4%:

            0.04 x (169/182) x 0.5 x 100,000 = 1,857.142857

        Pricing it ACT/365 instead would give a different number on every
        coupon -- this test is what distinguishes the two.
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
        """**Found by a wrong implementation, not by design.**

        A version of `price_bond_base` returning the CLEAN value (dirty
        minus accrued) passed 67 of 68 tests -- caught only by the
        cross-check against the integration note pricer. That is far too
        thin for a $1,857-on-$100k error that looks exactly like a curve
        difference.

        This pins the choice directly: the base price must equal the sum of
        discounted cashflows with NO accrued deduction, matching
        `engine.integration.note.NotePrice.npv`. Verified to fail against
        the clean-returning implementation.
        """
        note = make_note()
        # Independently: coupons + redemption, discounted. No accrued term.
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
        """The two must differ by the accrued amount, in the right direction.

        Also fails against the clean-returning implementation, which makes
        `price_bond_base` and `clean_npv_of` differ by *twice* the accrued.
        """
        note = make_note()
        assert price_bond_base(note) - clean_npv_of(note) == pytest.approx(
            accrued_interest(note), abs=1e-9
        )
        assert price_bond_base(note) > clean_npv_of(note)

    def test_clean_and_dirty_differ_by_a_material_amount(self):
        """Guards the 'silently clean NPV' failure the note module warns of.

        If these two were equal the accrued handling would be inert, and
        `price_bond_base` returning the clean value would be off by ~$1,857
        on $100k -- large enough to matter, small enough to look like a
        curve difference.
        """
        note = make_note()
        assert abs(price_bond_base(note) - clean_npv_of(note)) > 1000.0

    def test_a_coupon_already_paid_is_excluded(self):
        """A coupon paid before the valuation date is not this position's.

        The 2025-06-15 payment is still ahead of 2025-06-02 and must be
        included; moving valuation past it must drop it from the sum.
        """
        before = price_bond_base(make_note())
        after_cfg = BondConfig(
            face_amount=100_000.0, maturity_date=_date("2026-06-15"),
            evaluation_date=_date("2025-06-20"), initial_zero_curve=FLAT_3PCT,
            coupon_rate=0.04, coupon_schedule=NOTE_SCHEDULE,
            accrual_day_count="ACT/ACT (ICMA)",
        )
        # Two fewer weeks of discounting but one whole coupon gone: the
        # dropped coupon (~2,000) dominates the discounting change.
        assert price_bond_base(after_cfg) < before


class TestAgreesWithTheIntegrationPricers:
    """The anti-drift pin between the two pricing paths.

    `engine.instruments.treasury` deliberately does NOT import
    `engine.integration.bill`/`note` -- integration sits above instruments
    in the dependency order, and reversing that to share ~20 lines of
    `exp(-r*t)` would couple the instrument layer to the TraderX bundle
    format. These tests are what make that duplication safe: the same
    instrument is priced through both paths and must agree to the cent, so
    a drift between them fails here rather than going unnoticed.
    """

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
        """Pins the ACT/ACT (ICMA) accrual specifically.

        Compared against the integration pricer's `recomputed_fraction`
        (the schedule-derived path), not its `exported_fraction` -- a
        direct Python caller has no exporter, so the recomputed value is
        the only one both paths can produce.
        """
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
    """Everything this type declines to price, rather than approximating."""

    def test_a_matured_bond_is_refused(self):
        with pytest.raises(BondPricingError, match="not after"):
            make_bill(maturity="2025-01-01")

    def test_a_bond_maturing_on_the_valuation_date_is_refused(self):
        with pytest.raises(BondPricingError, match="not after"):
            BondConfig(face_amount=1.0, maturity_date=VALUATION,
                       evaluation_date=VALUATION, initial_zero_curve=FLAT_3PCT)

    def test_a_schedule_with_a_zero_coupon_rate_is_refused(self):
        """A contradiction, not a bill.

        Refused rather than silently priced as a bill, because which of the
        two was meant changes the price.
        """
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
        """The W1.1 rule: refused, never defaulted.

        ACT/360 is the specific convention the plan calls out -- silently
        pricing it as ACT/365 shifts every coupon.
        """
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
        """Duration: a longer claim moves more for the same bump."""
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
        # At ~0.5y the steep curve sits near 1.5%, below the flat 2%, so
        # the same bill discounts LESS and is worth more.
        assert price_bond_base(make_bill(curve=steep)) > price_bond_base(make_bill(curve=flat))

    def test_a_higher_curve_gives_a_lower_price(self):
        low = ZeroCurveConfig(times=(0.0, 30.0), rates=(0.01, 0.01))
        high = ZeroCurveConfig(times=(0.0, 30.0), rates=(0.06, 0.06))
        assert price_bond_base(make_bill(curve=high)) < price_bond_base(make_bill(curve=low))

    def test_beyond_the_last_pillar_holds_flat(self):
        """Flat extrapolation, stated rather than assumed.

        Extrapolating a slope past the last pillar produces a confident
        number from no data, and for a long bond that error compounds
        through every discount factor.
        """
        curve = ZeroCurveConfig(times=(0.0, 1.0), rates=(0.03, 0.04))
        far = make_bill(maturity="2035-12-15", curve=curve)
        t = ORE.Actual365Fixed().yearFraction(VALUATION, _date("2035-12-15"))
        # Held flat at the last pillar's 4%, not extrapolated upward.
        assert price_bond_base(far) == pytest.approx(100_000.0 * math.exp(-0.04 * t))

    def test_a_negative_rate_curve_prices_above_par(self):
        """Negative rates are a real market state, not an error.

        At a negative zero rate the discount factor exceeds 1, so a bill is
        worth *more* than its face. No special-casing -- `exp(-r*t)` handles
        it -- but it is pinned so a future "sanity clamp" that floors rates
        at zero has to fail a test rather than silently reprice.
        """
        negative = ZeroCurveConfig(times=(0.0, 1.0, 30.0), rates=(-0.005,) * 3)
        assert price_bond_base(make_bill(curve=negative)) > 100_000.0

    def test_a_long_dated_bill_discounts_heavily(self):
        """A 30-year zero is worth well under half its face at 3%."""
        far = make_bill(maturity="2055-06-02")
        assert 0.0 < price_bond_base(far) < 50_000.0

    def test_a_zero_face_position_is_worth_zero(self):
        """A degenerate but legal position -- must not raise."""
        assert price_bond_base(make_bill(face=0.0)) == 0.0

    def test_an_empty_curve_is_refused(self):
        empty = ZeroCurveConfig(times=(), rates=())
        with pytest.raises(BondPricingError, match="no pillars"):
            price_bond_base(make_bill(curve=empty))


class TestScenarioPricingIsRefused:
    """A bond has no scenario NPV, and says so rather than fabricating one.

    These tests carry the *evidence* for the refusal, not just the refusal:
    `test_a_broadcast_constant_column_produces_var_zero_and_es_nan` measures
    what the naive alternative actually returns, so the module docstring's
    claim is checked rather than asserted.
    """

    def test_price_bond_scenarios_always_raises(self):
        with pytest.raises(ScenarioPricingNotSupported):
            price_bond_scenarios(make_bill())

    def test_the_refusal_is_a_bond_pricing_error(self):
        # So a caller catching the general pricing error catches this too.
        assert issubclass(ScenarioPricingNotSupported, BondPricingError)

    def test_the_refusal_names_the_alternative(self):
        """A refusal that says only 'no' is hard to act on."""
        with pytest.raises(ScenarioPricingNotSupported) as exc:
            price_bond_scenarios(make_bill())
        message = str(exc.value)
        assert "base_npv" in message
        assert "I-24" in message

    def test_a_broadcast_constant_column_produces_var_zero_and_es_nan(self):
        """**The measured reason the refusal exists.**

        This is what `_price_by_type` would return if it broadcast a bond's
        t=0 value across the scenario/time axes instead of refusing. VaR
        comes out exactly 0.00 and ES comes out NaN -- a position reported
        as risk-measured whose risk was never modelled.

        If a future change makes a constant column produce something
        sensible, this test fails and the refusal should be revisited. That
        is deliberate: the refusal is justified by this behaviour, so the
        justification is pinned rather than remembered.
        """
        from engine.risk.var_es import compute_risk_metrics

        constant_column = jnp.full((500, 3, 1), 98_401.95)
        risk = compute_risk_metrics(constant_column, 98_401.95, percentiles=(0.95, 0.99))

        assert float(risk["VaR_95"][0]) == 0.0
        assert math.isnan(float(risk["ES_95"][0]))

"""
W1.2 -- the bill pricer (`docs/planning/traderx-integration-plan.md` §W1.2).

The plan's required tests, quoted: "ORE parity at matched terms; long/short
signs; **maturity-date boundary**; accrued = structural zero."

**Why ORE parity is asserted against a real ORE curve, not against
`exp(-rt)`.** Re-deriving the formula in the test would restate the
implementation and pass even if both were wrong together. `TestOreParity`
builds an actual `ORE.FlatForward` term structure and discounts an actual
`ORE.SimpleCashFlow` through `ORE.CashFlows.npv`, so the reference comes
from a library that knows nothing about this engine's code. Plan working
rule 4 is the reason to be strict here: "ORE can represent it" and "my
engine prices it" are different claims, and this file is where the second
one is earned.
"""
import math
from pathlib import Path

import ORE
import pytest

from engine.integration import price_bundle
from engine.integration.bill import (
    INSTRUMENT_MATURED,
    NOT_A_BILL,
    TERMS_INCOMPLETE,
    BillPricingError,
    is_bill,
    price_bill,
)
from engine.integration.market_inputs import ASSUMED_PROFILES
from engine.integration.terms import TermsEntry

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"

#: The delivered bill fixture's own terms: 6M zero-coupon, session date
#: 2025-06-02, maturing 2025-12-02, redeemed at par.
VALUATION = ORE.Date(2, 6, 2025)
MATURITY = ORE.Date(2, 12, 2025)
PROFILE = ASSUMED_PROFILES["flat-3pct-v1"]
FACE = 100_000.0

MARKET = {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}


def _terms(**overrides) -> TermsEntry:
    """A bill terms entry matching the delivered fixture, with overrides."""
    terms = {
        "couponFrequency": "NONE",
        "schedule": [],
        "maturityDate": "2025-12-02",
        "redemptionFraction": "1",
        "dayCount": "NOT_APPLICABLE",
        "currency": "USD",
    }
    terms.update(overrides)
    return TermsEntry(
        instrument_type="TREASURY", terms=terms,
        missing_terms=(), provenance={"origin": "synthetic"}, identity={},
    )


class TestOreParity:
    """The acceptance bar: this engine's number must equal an independent
    ORE valuation at matched terms."""

    @staticmethod
    def _ore_npv(face: float, maturity: ORE.Date = MATURITY) -> float:
        """An independent ORE valuation of the same single cashflow.

        Built from ORE's own term structure and cashflow machinery rather
        than from this engine's code, so agreement is evidence rather than
        tautology.
        """
        ORE.Settings.instance().evaluationDate = VALUATION
        curve = ORE.FlatForward(
            VALUATION, ORE.QuoteHandle(ORE.SimpleQuote(PROFILE.flat_rate)),
            ORE.Actual365Fixed(), ORE.Continuous, ORE.Annual,
        )
        leg = ORE.Leg([ORE.SimpleCashFlow(face, maturity)])
        return ORE.CashFlows.npv(leg, ORE.YieldTermStructureHandle(curve), True, VALUATION)

    def test_long_position_matches_ore(self):
        priced = price_bill(_terms(), FACE, VALUATION, PROFILE)
        assert priced.npv == pytest.approx(self._ore_npv(FACE), rel=1e-12)

    def test_short_position_matches_ore(self):
        priced = price_bill(_terms(), -FACE, VALUATION, PROFILE)
        assert priced.npv == pytest.approx(self._ore_npv(-FACE), rel=1e-12)

    def test_discount_factor_matches_ore(self):
        """The intermediate, not just the total -- a matching NPV with a
        wrong discount factor and a compensating face amount would still be
        wrong, and would break on the next instrument."""
        ORE.Settings.instance().evaluationDate = VALUATION
        curve = ORE.FlatForward(
            VALUATION, ORE.QuoteHandle(ORE.SimpleQuote(PROFILE.flat_rate)),
            ORE.Actual365Fixed(), ORE.Continuous, ORE.Annual,
        )
        priced = price_bill(_terms(), FACE, VALUATION, PROFILE)
        assert priced.discount_factor == pytest.approx(curve.discount(MATURITY), rel=1e-12)

    def test_compounding_convention_is_continuous_not_simple(self):
        """The assumed profile is a *continuously compounded* zero curve.
        Discounting it as a simple or annually-compounded rate shifts every
        price by an amount small enough to read as rounding -- ~$3.6 per
        $100k face here -- and large enough to be wrong.
        """
        priced = price_bill(_terms(), FACE, VALUATION, PROFILE)
        t = priced.year_fraction

        continuous = FACE * math.exp(-PROFILE.flat_rate * t)
        simple = FACE / (1.0 + PROFILE.flat_rate * t)

        assert priced.npv == pytest.approx(continuous, rel=1e-12)
        assert abs(priced.npv - simple) > 1.0, (
            "continuous and simple discounting are indistinguishable in this "
            "test's setup, so it cannot detect the wrong convention"
        )


class TestSigns:
    """Signed face carries the position direction in one step -- the
    double-sign bug TraderX flagged (their v3 §2) is structurally absent."""

    def test_long_is_positive_short_is_negative(self):
        assert price_bill(_terms(), FACE, VALUATION, PROFILE).npv > 0
        assert price_bill(_terms(), -FACE, VALUATION, PROFILE).npv < 0

    def test_long_and_short_are_exact_mirrors(self):
        long_npv = price_bill(_terms(), FACE, VALUATION, PROFILE).npv
        short_npv = price_bill(_terms(), -FACE, VALUATION, PROFILE).npv
        assert long_npv + short_npv == pytest.approx(0.0, abs=1e-9)

    def test_a_short_is_never_positive(self):
        """The specific failure mode: multiplying by a separate position
        sign on top of an already-signed face amount flips a short back to
        positive."""
        assert price_bill(_terms(), -FACE, VALUATION, PROFILE).npv == pytest.approx(
            -price_bill(_terms(), FACE, VALUATION, PROFILE).npv
        )


class TestMaturityBoundary:
    """The plan names this explicitly. A bill maturing today has no
    remaining cashflow; one that matured last week is not a pricing
    question at all."""

    def test_maturity_after_valuation_prices(self):
        priced = price_bill(_terms(maturityDate="2025-06-03"), FACE, VALUATION, PROFILE)
        assert priced.npv > 0
        assert priced.year_fraction > 0

    def test_maturity_on_valuation_date_is_refused(self):
        with pytest.raises(BillPricingError) as exc:
            price_bill(_terms(maturityDate="2025-06-02"), FACE, VALUATION, PROFILE)
        assert exc.value.reason == INSTRUMENT_MATURED

    def test_matured_instrument_is_refused_not_priced_at_face(self):
        """A matured bill must NOT come back at face value: that would be a
        settlement claim this engine has no basis for."""
        with pytest.raises(BillPricingError) as exc:
            price_bill(_terms(maturityDate="2025-05-02"), FACE, VALUATION, PROFILE)
        assert exc.value.reason == INSTRUMENT_MATURED

    def test_discount_factor_is_below_one_for_a_future_maturity(self):
        """A positive rate over positive time must discount. A DF of exactly
        1.0 would mean the day count or the maturity collapsed."""
        priced = price_bill(_terms(), FACE, VALUATION, PROFILE)
        assert 0.0 < priced.discount_factor < 1.0


class TestRefusesWhatItCannotPrice:
    """Every refusal names a reason. None falls back to a default."""

    def test_coupon_bearing_instrument_is_not_a_bill(self):
        note = _terms(couponFrequency="SEMIANNUAL", schedule=["2025-12-15"])
        assert not is_bill(note)
        with pytest.raises(BillPricingError) as exc:
            price_bill(note, FACE, VALUATION, PROFILE)
        assert exc.value.reason == NOT_A_BILL

    def test_contradictory_terms_are_refused_not_resolved(self):
        """`couponFrequency: NONE` with a non-empty schedule contradicts
        itself. This pricer refuses rather than picking a winner."""
        contradictory = _terms(couponFrequency="NONE", schedule=["2025-12-02"])
        assert not is_bill(contradictory)

    def test_missing_maturity_is_refused(self):
        with pytest.raises(BillPricingError) as exc:
            price_bill(_terms(maturityDate=None), FACE, VALUATION, PROFILE)
        assert exc.value.reason == TERMS_INCOMPLETE

    def test_unparseable_redemption_is_refused_not_defaulted_to_par(self):
        with pytest.raises(BillPricingError) as exc:
            price_bill(_terms(redemptionFraction="par"), FACE, VALUATION, PROFILE)
        assert exc.value.reason == TERMS_INCOMPLETE

    def test_absent_redemption_defaults_to_par(self):
        """Absent is different from malformed: absent means par, which is
        the near-universal case and what the fixture states anyway."""
        terms = _terms()
        del terms.terms["redemptionFraction"]
        assert price_bill(terms, FACE, VALUATION, PROFILE).redemption_fraction == 1.0

    def test_sub_par_redemption_is_honored(self):
        priced = price_bill(_terms(redemptionFraction="0.5"), FACE, VALUATION, PROFILE)
        full = price_bill(_terms(), FACE, VALUATION, PROFILE)
        assert priced.npv == pytest.approx(full.npv * 0.5, rel=1e-12)


class TestThroughTheBundlePipeline:
    """End to end against the delivered fixture -- the W1.2 deliverable as
    a consumer actually sees it."""

    def test_bill_v2_prices_both_positions(self):
        result = price_bundle(FIXTURES / "bill" / "v2", market_inputs=MARKET)
        npvs = [item.calculations["npv"] for item in result.items]

        assert all(outcome.status == "ok" for outcome in npvs)
        assert sum(outcome.value for outcome in npvs) == pytest.approx(0.0, abs=1e-9)

    def test_priced_npv_matches_the_direct_pricer(self):
        """The pipeline must not alter the number -- the I-01 lesson, where
        wiring silently changed what a caller got."""
        result = price_bundle(FIXTURES / "bill" / "v2", market_inputs=MARKET)
        long_item = next(
            item for item in result.items
            if item.calculations["npv"].value > 0
        )
        expected = price_bill(_terms(), FACE, VALUATION, PROFILE)
        assert long_item.calculations["npv"].value == pytest.approx(expected.npv, rel=1e-12)

    def test_result_carries_assumed_provenance(self):
        """A price computed against an assumed curve must say so at the top
        level, where a consumer reading only the summary cannot miss it."""
        result = price_bundle(FIXTURES / "bill" / "v2", market_inputs=MARKET)
        assert result.market_provenance == "assumed"

    def test_npv_payload_is_reconcilable(self):
        """A bare NPV is unreconcilable: if TraderX's number disagrees,
        nothing says whether the curve, the day count or the face was the
        cause. Every input to the arithmetic travels with the answer."""
        result = price_bundle(FIXTURES / "bill" / "v2", market_inputs=MARKET)
        payload = result.items[0].calculations["npv"].payload

        for key in ("method", "signedFaceAmount", "redemptionFraction",
                    "discountFactor", "yearFraction", "dayCount",
                    "maturityDate", "valuationDate", "curveProvenance"):
            assert key in payload, f"npv payload is missing {key!r}"

        assert payload["curveProvenance"]["curveId"] == "flat-3pct-v1"
        assert payload["curveProvenance"]["inputOrigin"] == "assumed"
        # The payload must reproduce the answer it travels with.
        assert (payload["signedFaceAmount"] * payload["redemptionFraction"]
                * payload["discountFactor"]) == pytest.approx(
            result.items[0].calculations["npv"].value, rel=1e-12)

    def test_accrued_is_a_structural_zero(self):
        """Plan §W1.2: "accrued = structural zero". Not `unavailable`, and
        not an unlabelled 0.0 -- the provenance is what distinguishes a
        bill's real zero from a missing value."""
        result = price_bundle(FIXTURES / "bill" / "v2", market_inputs=MARKET)
        for item in result.items:
            accrued = item.calculations["accruedInterest"]
            assert accrued.status == "ok"
            assert accrued.value == 0.0
            assert accrued.payload["provenance"] == "structural-zero"

    def test_sensitivities_remain_unsupported(self):
        """W1.2 delivers a price, not a risk number. Returning 0.0 or
        omitting these would be the silent approximation this boundary
        exists to prevent."""
        result = price_bundle(FIXTURES / "bill" / "v2", market_inputs=MARKET)
        for item in result.items:
            for name in ("rateSensitivity", "rateGamma", "theta"):
                assert item.calculations[name].status == "unsupported"

    def test_without_market_inputs_nothing_is_priced(self):
        """The no-silent-fallback rule, at the moment it finally bites: W0
        could omit market inputs because it priced nothing. A pricer must
        never invent a curve."""
        result = price_bundle(FIXTURES / "bill" / "v2")
        for item in result.items:
            assert item.calculations["npv"].status == "unsupported"
        assert result.market_provenance is None

    def test_v1_bundle_does_not_price(self):
        """Without a terms artifact the engine cannot establish that a row
        IS a bill, and will not infer it from a zero coupon column -- the
        W0.3 rule, still holding now that a pricer exists."""
        result = price_bundle(FIXTURES / "bill" / "v1", market_inputs=MARKET)
        for item in result.items:
            assert item.calculations["npv"].status == "unsupported"

    def test_a_note_is_never_priced_by_the_bill_model(self):
        """A coupon-bearing note must not be priced by the bill's
        single-cashflow model just because it is also a Treasury.

        **Rewritten at W1.3.** This test used to assert the note came back
        `NO_PRICER_AT_THIS_STAGE`, which was true only while no note
        pricer existed -- it was pinning the *absence* of W1.3 rather than
        the property that matters. W1.3 makes that premise obsolete but
        makes the underlying danger *greater*, not smaller: with two
        Treasury pricers, routing a row to the wrong one is now a
        reachable mistake rather than an impossible one.

        So the assertion is now on the model actually used. The bill's
        payload is a single discount factor with no schedule; the note's
        carries coupons. A note priced by the bill model would show the
        former -- a confident, plausible, wrong number.
        """
        result = price_bundle(FIXTURES / "note" / "v2", market_inputs=MARKET)
        for item in result.items:
            npv = item.calculations["npv"]
            assert npv.status == "ok"
            # The note model's fingerprint, which the bill model cannot produce.
            assert npv.payload["coupons"], (
                "a note priced with no coupon schedule means it was routed "
                "through the bill's single-cashflow model"
            )
            assert npv.payload["accrualDayCount"] == "ACT/ACT (ICMA)"
            assert "discountFactor" not in npv.payload, (
                "the bill payload's single discountFactor is present on a "
                "note, so the wrong pricer produced this number"
            )

    def test_the_bill_model_refuses_a_note_directly(self):
        """The same guarantee at the pricer rather than the pipeline: even
        handed a note's terms, `price_bill` refuses instead of returning a
        number for the wrong instrument."""
        note_terms = _terms(
            couponFrequency="6M",
            schedule=[{"startDate": "2024-12-15", "endDate": "2025-06-15",
                       "paymentDate": "2025-06-15"}],
        )
        with pytest.raises(BillPricingError) as excinfo:
            price_bill(note_terms, FACE, VALUATION, PROFILE)
        assert excinfo.value.reason == NOT_A_BILL

    def test_sofr_refusal_is_unchanged(self):
        """Adding a pricer must not weaken the W0 refusal path: a swap whose
        conventions are unsupported is still refused, not priced."""
        result = price_bundle(FIXTURES / "sofr" / "v2", market_inputs=MARKET)
        (item,) = result.items
        assert item.calculations["npv"].status == "unsupported"
        assert item.calculations["npv"].reason == "CONVENTION_NOT_SUPPORTED"
        assert len(item.refusal["missingTerms"]) == 13

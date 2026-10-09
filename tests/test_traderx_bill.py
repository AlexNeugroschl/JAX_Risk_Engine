"""
Treasury bill pricer (`engine.traderx.bill`): ORE parity, long/short signs, the
maturity-date boundary, and accrued as a structural zero.

ORE parity uses an `ORE.FlatForward` curve and `ORE.CashFlows.npv` on an
`ORE.SimpleCashFlow`, not a re-derived `exp(-rt)`, which would share any error with the
implementation.
"""
import math
from pathlib import Path

import ORE
import pytest

from engine.traderx import price_bundle
from engine.traderx.bill import (
    INSTRUMENT_MATURED,
    NOT_A_BILL,
    TERMS_INCOMPLETE,
    BillPricingError,
    is_bill,
    price_bill,
)
from engine.traderx.market_inputs import ASSUMED_PROFILES
from engine.traderx.terms import TermsEntry

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"

#: The bill fixture's terms: 6M zero-coupon, session date 2025-06-02, maturing 2025-12-02,
#: redeemed at par.
VALUATION = ORE.Date(2, 6, 2025)
MATURITY = ORE.Date(2, 12, 2025)
PROFILE = ASSUMED_PROFILES["flat-3pct-v1"]
FACE = 100_000.0

MARKET = {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}


def _terms(**overrides) -> TermsEntry:
    """A bill terms entry matching the fixture, with overrides."""
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
    """Engine NPV equals an independent ORE valuation on the same terms."""

    @staticmethod
    def _ore_npv(face: float, maturity: ORE.Date = MATURITY) -> float:
        """ORE's valuation of the same single cashflow, from its own curve and cashflow
        classes."""
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
        """The discount factor itself matches (not only the total)."""
        ORE.Settings.instance().evaluationDate = VALUATION
        curve = ORE.FlatForward(
            VALUATION, ORE.QuoteHandle(ORE.SimpleQuote(PROFILE.flat_rate)),
            ORE.Actual365Fixed(), ORE.Continuous, ORE.Annual,
        )
        priced = price_bill(_terms(), FACE, VALUATION, PROFILE)
        assert priced.discount_factor == pytest.approx(curve.discount(MATURITY), rel=1e-12)

    def test_compounding_convention_is_continuous_not_simple(self):
        """The profile is continuously compounded; simple compounding would be ~$11 higher
        on $100k here."""
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
    """The signed face carries direction in one step (no double sign)."""

    def test_long_is_positive_short_is_negative(self):
        assert price_bill(_terms(), FACE, VALUATION, PROFILE).npv > 0
        assert price_bill(_terms(), -FACE, VALUATION, PROFILE).npv < 0

    def test_long_and_short_are_exact_mirrors(self):
        long_npv = price_bill(_terms(), FACE, VALUATION, PROFILE).npv
        short_npv = price_bill(_terms(), -FACE, VALUATION, PROFILE).npv
        assert long_npv + short_npv == pytest.approx(0.0, abs=1e-9)

    def test_a_short_is_never_positive(self):
        """A short is never positive (a second sign factor would flip it back)."""
        assert price_bill(_terms(), -FACE, VALUATION, PROFILE).npv == pytest.approx(
            -price_bill(_terms(), FACE, VALUATION, PROFILE).npv
        )


class TestMaturityBoundary:
    """Maturity boundary: a bill maturing on or before the valuation date is refused."""

    def test_maturity_after_valuation_prices(self):
        priced = price_bill(_terms(maturityDate="2025-06-03"), FACE, VALUATION, PROFILE)
        assert priced.npv > 0
        assert priced.year_fraction > 0

    def test_maturity_on_valuation_date_is_refused(self):
        with pytest.raises(BillPricingError) as exc:
            price_bill(_terms(maturityDate="2025-06-02"), FACE, VALUATION, PROFILE)
        assert exc.value.reason == INSTRUMENT_MATURED

    def test_matured_instrument_is_refused_not_priced_at_face(self):
        """A matured bill is refused, not priced at face."""
        with pytest.raises(BillPricingError) as exc:
            price_bill(_terms(maturityDate="2025-05-02"), FACE, VALUATION, PROFILE)
        assert exc.value.reason == INSTRUMENT_MATURED

    def test_discount_factor_is_below_one_for_a_future_maturity(self):
        """A future maturity discounts (DF < 1)."""
        priced = price_bill(_terms(), FACE, VALUATION, PROFILE)
        assert 0.0 < priced.discount_factor < 1.0


class TestRefusesWhatItCannotPrice:
    """Every refusal names a reason; nothing falls back to a default."""

    def test_coupon_bearing_instrument_is_not_a_bill(self):
        note = _terms(couponFrequency="SEMIANNUAL", schedule=["2025-12-15"])
        assert not is_bill(note)
        with pytest.raises(BillPricingError) as exc:
            price_bill(note, FACE, VALUATION, PROFILE)
        assert exc.value.reason == NOT_A_BILL

    def test_contradictory_terms_are_refused_not_resolved(self):
        """`couponFrequency: NONE` with a schedule is contradictory and refused."""
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
        """An absent redemption means par; an unparseable one is refused."""
        terms = _terms()
        del terms.terms["redemptionFraction"]
        assert price_bill(terms, FACE, VALUATION, PROFILE).redemption_fraction == 1.0

    def test_sub_par_redemption_is_honored(self):
        priced = price_bill(_terms(redemptionFraction="0.5"), FACE, VALUATION, PROFILE)
        full = price_bill(_terms(), FACE, VALUATION, PROFILE)
        assert priced.npv == pytest.approx(full.npv * 0.5, rel=1e-12)


class TestThroughTheBundlePipeline:
    """End to end on the delivered fixture."""

    def test_bill_v2_prices_both_positions(self):
        result = price_bundle(FIXTURES / "bill" / "v2", market_inputs=MARKET)
        npvs = [item.calculations["npv"] for item in result.items]

        assert all(outcome.status == "ok" for outcome in npvs)
        assert sum(outcome.value for outcome in npvs) == pytest.approx(0.0, abs=1e-9)

    def test_priced_npv_matches_the_direct_pricer(self):
        """The pipeline returns the direct pricer's number."""
        result = price_bundle(FIXTURES / "bill" / "v2", market_inputs=MARKET)
        long_item = next(
            item for item in result.items
            if item.calculations["npv"].value > 0
        )
        expected = price_bill(_terms(), FACE, VALUATION, PROFILE)
        assert long_item.calculations["npv"].value == pytest.approx(expected.npv, rel=1e-12)

    def test_result_carries_assumed_provenance(self):
        """A result on an assumed curve says so at the top level."""
        result = price_bundle(FIXTURES / "bill" / "v2", market_inputs=MARKET)
        assert result.market_provenance == "assumed"

    def test_npv_payload_is_reconcilable(self):
        """Every input to the arithmetic travels with the NPV, so a disagreement can be
        traced to curve, day count or face."""
        result = price_bundle(FIXTURES / "bill" / "v2", market_inputs=MARKET)
        payload = result.items[0].calculations["npv"].payload

        for key in ("method", "signedFaceAmount", "redemptionFraction",
                    "discountFactor", "yearFraction", "dayCount",
                    "maturityDate", "valuationDate", "curveProvenance"):
            assert key in payload, f"npv payload is missing {key!r}"

        assert payload["curveProvenance"]["curveId"] == "flat-3pct-v1"
        assert payload["curveProvenance"]["inputOrigin"] == "assumed"
        # The payload reproduces the NPV.
        assert (payload["signedFaceAmount"] * payload["redemptionFraction"]
                * payload["discountFactor"]) == pytest.approx(
            result.items[0].calculations["npv"].value, rel=1e-12)

    def test_accrued_is_a_structural_zero(self):
        """Accrued is `ok` at a labelled structural zero, not `unavailable` or a bare 0.0."""
        result = price_bundle(FIXTURES / "bill" / "v2", market_inputs=MARKET)
        for item in result.items:
            accrued = item.calculations["accruedInterest"]
            assert accrued.status == "ok"
            assert accrued.value == 0.0
            assert accrued.payload["provenance"] == "structural-zero"

    def test_sensitivities_remain_unsupported(self):
        """Sensitivities stay unsupported for a bill (not 0.0 or omitted)."""
        result = price_bundle(FIXTURES / "bill" / "v2", market_inputs=MARKET)
        for item in result.items:
            for name in ("rateSensitivity", "rateGamma", "theta"):
                assert item.calculations[name].status == "unsupported"

    def test_without_market_inputs_nothing_is_priced(self):
        """Without market inputs nothing is priced and no curve is invented."""
        result = price_bundle(FIXTURES / "bill" / "v2")
        for item in result.items:
            assert item.calculations["npv"].status == "unsupported"
        assert result.market_provenance is None

    def test_v1_bundle_does_not_price(self):
        """Without terms the row cannot be established as a bill, so nothing prices."""
        result = price_bundle(FIXTURES / "bill" / "v1", market_inputs=MARKET)
        for item in result.items:
            assert item.calculations["npv"].status == "unsupported"

    def test_a_note_is_never_priced_by_the_bill_model(self):
        """A note is never priced by the bill model: its payload carries coupons, which the
        bill model's single discount factor cannot produce."""
        result = price_bundle(FIXTURES / "note" / "v2", market_inputs=MARKET)
        for item in result.items:
            npv = item.calculations["npv"]
            assert npv.status == "ok"
            # The note model's signature, which the bill model cannot produce.
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
        """`price_bill` refuses a note's terms directly."""
        note_terms = _terms(
            couponFrequency="6M",
            schedule=[{"startDate": "2024-12-15", "endDate": "2025-06-15",
                       "paymentDate": "2025-06-15"}],
        )
        with pytest.raises(BillPricingError) as excinfo:
            price_bill(note_terms, FACE, VALUATION, PROFILE)
        assert excinfo.value.reason == NOT_A_BILL

    def test_sofr_refusal_is_unchanged(self):
        """The SOFR swap is still refused."""
        result = price_bundle(FIXTURES / "sofr" / "v2", market_inputs=MARKET)
        (item,) = result.items
        assert item.calculations["npv"].status == "unsupported"
        assert item.calculations["npv"].reason == "CONVENTION_NOT_SUPPORTED"
        assert len(item.refusal["missingTerms"]) == 13

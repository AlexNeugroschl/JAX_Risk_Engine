"""
Cash equity positions (`engine.traderx.equity`), which are refused: there is no spot or
FX source at this boundary (I-18). The engine must still read the position correctly:

  - long/short: the sign reaches `multipliedQuantity`;
  - the multiplier applied exactly once (tested with a multiplier other than 1);
  - a non-USD position refuses differently (a spot alone would not price it).

`closingMark` is never used: quantity x mark x multiplier reproduces the exporter's own
`marketValue`, so it would hand TraderX its own number back as a valuation
(`TestDoesNotEchoTheExportedMark`).
"""
import json
from pathlib import Path

import pytest

from engine.traderx import price_bundle
from engine.traderx.capabilities import BLOCKED_ON_MARKET_INPUT, capabilities
from engine.traderx.equity import (
    EQUITY,
    FX_SOURCE_NOT_SUPPLIED,
    NOT_AN_EQUITY,
    REPORTING_CURRENCY,
    SPOT_SOURCE_NOT_SUPPLIED,
    TERMS_INCOMPLETE,
    EquityPricingError,
    is_equity,
    price_equity,
    read_position,
)
from engine.traderx.terms import TermsEntry

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"

MARKET = {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}

#: The v2 fixture's numbers.
QUANTITY = 1000.0
CLOSING_MARK = 10.0
#: quantity x closingMark x contractMultiplier, the exporter's `marketValue`. Never an npv.
EXPORTED_MARKET_VALUE = 10_000.0


def _terms(**overrides) -> TermsEntry:
    """An equity terms entry matching the fixture."""
    terms = {
        "contractMultiplier": "1",
        "currency": "USD",
        "priceBasis": "price-per-share",
        "quantityUnit": "signed-shares",
        "settlementDays": 0,
    }
    terms.update(overrides)
    return TermsEntry(
        instrument_type=overrides.pop("_type", "EQUITY"), terms=terms,
        missing_terms=(), provenance={"origin": "synthetic"}, identity={},
    )


def _row(**overrides) -> dict:
    row = {
        "accountId": "22214",
        "security": "SYNTH-EQ",
        "instrumentType": "EQUITY",
        "quantity": str(QUANTITY),
        "contractMultiplier": "1",
        "closingMark": str(CLOSING_MARK),
        "marketValue": str(EXPORTED_MARKET_VALUE),
        "currency": "USD",
    }
    row.update(overrides)
    return row


class TestRefusesRatherThanPrices:
    """No spot source, so no number."""

    def test_price_equity_always_raises(self):
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(_terms(), _row())
        assert excinfo.value.reason == SPOT_SOURCE_NOT_SUPPLIED

    def test_the_refusal_names_the_missing_input(self):
        """The refusal names what to send."""
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(_terms(), _row())
        assert excinfo.value.payload["missingInputs"] == ["spot"]

    def test_the_refusal_explains_why_the_mark_is_not_used(self):
        """The refusal says why the mark is not used."""
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(_terms(), _row())
        detail = excinfo.value.detail.lower()
        assert "closingmark" in detail
        assert "observation" in detail

    def test_no_market_input_makes_it_priceable(self):
        """Requesting a rate curve does not change the outcome (a curve is not a spot)."""
        with_curve = price_bundle(FIXTURES / "equity" / "v2", MARKET).to_dict()
        without = price_bundle(FIXTURES / "equity" / "v2").to_dict()
        for item in with_curve["items"] + without["items"]:
            assert item["calculations"]["npv"]["status"] == "unsupported"


class TestDoesNotEchoTheExportedMark:
    """The mark-derived value reproduces the exporter's `marketValue` exactly, so it would
    reconcile perfectly while proving nothing."""

    def test_the_exported_market_value_never_appears_as_an_npv(self):
        result = price_bundle(FIXTURES / "equity" / "v2", MARKET).to_dict()
        for item in result["items"]:
            npv = item["calculations"]["npv"]
            assert npv["status"] != "ok"
            assert npv.get("value") is None

    def test_no_payload_field_carries_the_mark_derived_value(self):
        """Nor anywhere in the payload."""
        result = price_bundle(FIXTURES / "equity" / "v2", MARKET).to_dict()
        for item in result["items"]:
            payload = item["calculations"]["npv"]
            flat = json.dumps(payload)
            assert str(EXPORTED_MARKET_VALUE) not in flat
            assert "closingMark" not in payload

    def test_the_fixture_would_actually_expose_the_bug(self):
        """The fixture really contains a mark that could be echoed (so the tests above are
        not vacuous)."""
        rows = (FIXTURES / "equity" / "v2" / "positions.csv").read_text().splitlines()
        data = [r for r in rows if r and not r.startswith("#")][1:]
        assert any(f",{CLOSING_MARK:.6f}," in r for r in data)


class TestLongShort:
    """The sign survives into the refusal."""

    def test_long_quantity_is_positive(self):
        assert read_position(_terms(), _row()).multiplied_quantity == QUANTITY

    def test_short_quantity_is_negative(self):
        position = read_position(_terms(), _row(quantity=str(-QUANTITY)))
        assert position.multiplied_quantity == -QUANTITY

    def test_long_and_short_are_exact_mirrors(self):
        long_pos = read_position(_terms(), _row())
        short = read_position(_terms(), _row(quantity=str(-QUANTITY)))
        assert long_pos.multiplied_quantity + short.multiplied_quantity == 0.0

    def test_the_sign_reaches_the_published_refusal(self):
        """Through the pipeline, not just the reader."""
        result = price_bundle(FIXTURES / "equity" / "v2", MARKET).to_dict()
        quantities = [
            item["calculations"]["npv"].get("multipliedQuantity")
            for item in result["items"]
            if item["sourceIdentity"]["security"] == "SYNTH-EQ"
        ]
        assert sorted(quantities) == [-QUANTITY, QUANTITY]


class TestMultiplierAppliedExactlyOnce:
    """Every test uses a multiplier other than 1 (with 1, applied twice, once or never look
    the same)."""

    def test_multiplier_is_applied(self):
        position = read_position(_terms(contractMultiplier="100"), _row())
        assert position.multiplied_quantity == QUANTITY * 100

    def test_not_applied_twice(self):
        position = read_position(_terms(contractMultiplier="100"), _row())
        assert position.multiplied_quantity != QUANTITY * 100 * 100

    def test_not_omitted(self):
        position = read_position(_terms(contractMultiplier="100"), _row())
        assert position.multiplied_quantity != QUANTITY

    def test_the_raw_quantity_stays_unmultiplied(self):
        """Raw and multiplied quantities are both reported and differ."""
        position = read_position(_terms(contractMultiplier="100"), _row())
        assert position.signed_quantity == QUANTITY
        assert position.contract_multiplier == 100.0
        assert position.multiplied_quantity == QUANTITY * 100

    def test_a_fractional_multiplier_is_honoured(self):
        """A fractional multiplier is not rounded."""
        position = read_position(_terms(contractMultiplier="0.5"), _row())
        assert position.multiplied_quantity == QUANTITY * 0.5

    def test_absent_multiplier_defaults_to_one(self):
        """An absent multiplier defaults to 1."""
        terms = TermsEntry(
            instrument_type="EQUITY", terms={"currency": "USD"},
            missing_terms=(), provenance={}, identity={},
        )
        row = _row()
        del row["contractMultiplier"]
        assert read_position(terms, row).contract_multiplier == 1.0

    def test_an_unparseable_multiplier_is_refused_not_defaulted(self):
        """An unparseable multiplier is refused, not defaulted."""
        with pytest.raises(EquityPricingError) as excinfo:
            read_position(_terms(contractMultiplier="x100"), _row())
        assert excinfo.value.reason == TERMS_INCOMPLETE

    def test_terms_multiplier_wins_over_the_csv(self):
        """The terms' multiplier wins over the CSV's."""
        position = read_position(
            _terms(contractMultiplier="100"), _row(contractMultiplier="1"),
        )
        assert position.contract_multiplier == 100.0

    def test_the_csv_multiplier_is_used_when_terms_omit_it(self):
        terms = TermsEntry(
            instrument_type="EQUITY", terms={"currency": "USD"},
            missing_terms=(), provenance={}, identity={},
        )
        position = read_position(terms, _row(contractMultiplier="50"))
        assert position.contract_multiplier == 50.0


class TestCurrencyAndFx:
    """Currency and FX."""

    def test_a_usd_position_refuses_for_the_spot_only(self):
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(_terms(), _row())
        assert excinfo.value.reason == SPOT_SOURCE_NOT_SUPPLIED
        assert excinfo.value.payload["missingInputs"] == ["spot"]

    def test_a_non_usd_position_refuses_differently(self):
        """A EUR position refuses with FX_SOURCE_NOT_SUPPLIED ("send a spot" would be wrong)."""
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(_terms(currency="EUR"), _row(currency="EUR"))
        assert excinfo.value.reason == FX_SOURCE_NOT_SUPPLIED
        assert excinfo.value.payload["missingInputs"] == ["spot", "fx"]

    def test_the_two_refusals_are_distinct_codes(self):
        assert SPOT_SOURCE_NOT_SUPPLIED != FX_SOURCE_NOT_SUPPLIED

    def test_currency_is_case_insensitive(self):
        """`usd` is USD."""
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(_terms(currency="usd"), _row(currency="usd"))
        assert excinfo.value.reason == SPOT_SOURCE_NOT_SUPPLIED

    def test_an_absent_currency_is_treated_as_foreign_not_assumed_usd(self):
        """A blank currency is treated as foreign, not assumed USD."""
        terms = TermsEntry(
            instrument_type="EQUITY", terms={"contractMultiplier": "1"},
            missing_terms=(), provenance={}, identity={},
        )
        row = _row()
        del row["currency"]
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(terms, row)
        assert excinfo.value.reason == FX_SOURCE_NOT_SUPPLIED

    def test_the_reporting_currency_is_stated_not_implied(self):
        assert REPORTING_CURRENCY == "USD"

    def test_both_refusals_reach_the_published_result(self):
        """End to end: the fixture's USD pair and EUR row each get their own reason."""
        result = price_bundle(FIXTURES / "equity" / "v2", MARKET).to_dict()
        reasons = {
            item["sourceIdentity"]["security"]: item["calculations"]["npv"]["reason"]
            for item in result["items"]
        }
        assert reasons["SYNTH-EQ"] == SPOT_SOURCE_NOT_SUPPLIED
        assert reasons["SYNTH-EQ-EUR"] == FX_SOURCE_NOT_SUPPLIED


class TestIsEquity:
    """`is_equity` keys on `instrumentType`, never on blank columns."""

    def test_an_equity_entry_is_an_equity(self):
        assert is_equity(_terms())

    def test_none_is_not_an_equity(self):
        assert not is_equity(None)

    def test_a_treasury_is_not_an_equity(self):
        entry = TermsEntry(
            instrument_type="TREASURY", terms={}, missing_terms=(),
            provenance={}, identity={},
        )
        assert not is_equity(entry)

    def test_case_insensitive(self):
        entry = TermsEntry(
            instrument_type="equity", terms={}, missing_terms=(),
            provenance={}, identity={},
        )
        assert is_equity(entry)

    def test_blank_bond_columns_do_not_make_a_row_an_equity(self):
        """Blank bond columns (which a bill also has) do not make a row an equity."""
        bill = TermsEntry(
            instrument_type="TREASURY",
            terms={"couponFrequency": "NONE", "schedule": []},
            missing_terms=(), provenance={}, identity={},
        )
        assert not is_equity(bill)

    def test_price_equity_refuses_a_non_equity(self):
        treasury = TermsEntry(
            instrument_type="TREASURY", terms={}, missing_terms=(),
            provenance={}, identity={},
        )
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(treasury, _row())
        assert excinfo.value.reason == NOT_AN_EQUITY


class TestMalformedRows:
    """A broken row refuses differently from a missing market input."""

    def test_an_unparseable_quantity_is_terms_incomplete(self):
        with pytest.raises(EquityPricingError) as excinfo:
            read_position(_terms(), _row(quantity="ten"))
        assert excinfo.value.reason == TERMS_INCOMPLETE

    def test_an_absent_quantity_is_refused(self):
        row = _row()
        del row["quantity"]
        with pytest.raises(EquityPricingError) as excinfo:
            read_position(_terms(), row)
        assert excinfo.value.reason == TERMS_INCOMPLETE

    def test_a_malformed_row_is_not_reported_as_a_missing_spot(self):
        """A malformed row is not reported as a missing spot (different fixes)."""
        with pytest.raises(EquityPricingError) as excinfo:
            read_position(_terms(), _row(quantity="ten"))
        assert excinfo.value.reason != SPOT_SOURCE_NOT_SUPPLIED

    def test_a_zero_quantity_is_read_not_refused(self):
        """A zero quantity (closed-out position) is read, not refused."""
        assert read_position(_terms(), _row(quantity="0")).multiplied_quantity == 0.0


class TestRefusalsAreEquityPricingErrors:
    """I-17 regression class: equity refusals are their own exception type, so the
    pipeline's handler catches them and one bad row cannot fail the bundle."""

    def test_refusals_are_not_bill_or_note_errors(self):
        from engine.traderx.bill import BillPricingError
        from engine.traderx.note import NotePricingError

        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(_terms(), _row())
        assert not isinstance(excinfo.value, (BillPricingError, NotePricingError))

    def test_no_refusal_message_mentions_a_bill_or_a_note(self):
        for bad in (_row(quantity="ten"), _row()):
            with pytest.raises(EquityPricingError) as excinfo:
                price_equity(_terms(), bad)
            detail = excinfo.value.detail.lower()
            assert "a bill cannot" not in detail
            assert "a matured note" not in detail

    def test_one_malformed_equity_does_not_fail_the_whole_bundle(self):
        """One malformed equity does not fail the bundle."""
        from engine.traderx.market_inputs import ASSUMED_PROFILES, MarketInputs
        from engine.traderx.pipeline import _equity_outcomes
        from engine.traderx.terms import JoinedRow

        joined = JoinedRow(source="positions", row=_row(quantity="ten"), entry=_terms())
        outcomes = _equity_outcomes(joined)
        assert outcomes["npv"].status == "unsupported"
        assert outcomes["npv"].reason == TERMS_INCOMPLETE


class TestPipelineEndToEnd:
    """The equity through the whole path, in both bundle versions."""

    @staticmethod
    def _v2():
        return price_bundle(FIXTURES / "equity" / "v2", MARKET).to_dict()

    def test_every_row_is_identified_despite_refusing(self):
        """Refused rows still carry their identity."""
        for item in self._v2()["items"]:
            assert item["itemId"]
            assert item["sourceIdentity"]["accountId"]
            assert item["sourceIdentity"]["security"]

    def test_the_refusal_carries_the_validated_inputs(self):
        """The refusal carries the validated inputs."""
        item = self._v2()["items"][0]
        npv = item["calculations"]["npv"]
        assert npv["signedQuantity"] == QUANTITY
        assert npv["contractMultiplier"] == 1.0
        assert npv["multipliedQuantity"] == QUANTITY
        assert npv["intendedMethod"] == "spot-revaluation"

    def test_vega_is_not_applicable_for_an_equity(self):
        """Vega is not-applicable for an equity (not a coverage gap)."""
        for item in self._v2()["items"]:
            assert item["calculations"]["vega"]["status"] == "not-applicable"

    def test_accrued_interest_is_not_claimed_for_an_equity(self):
        """No accrued interest is claimed for an equity."""
        for item in self._v2()["items"]:
            assert item["calculations"]["accruedInterest"]["status"] != "ok"

    def test_coverage_accounts_for_every_outcome(self):
        coverage = self._v2()["coverage"]
        assert coverage["allOutcomesAccountedFor"] is True
        assert coverage["byCalculation"]["npv"]["unsupported"] == 3

    def test_the_result_is_json_serializable(self):
        payload = self._v2()
        assert json.loads(json.dumps(payload)) == payload

    def test_the_v1_bundle_refuses_for_want_of_terms(self):
        """A v1 bundle has no terms, so the row cannot be established as an equity; that
        refusal takes precedence."""
        result = price_bundle(FIXTURES / "equity" / "v1", MARKET).to_dict()
        for item in result["items"]:
            npv = item["calculations"]["npv"]
            assert npv["status"] == "unsupported"
            assert npv["reason"] == "TERMS_NOT_SUPPLIED"

    def test_the_v1_bundle_is_the_real_traderx_fixture(self):
        """The v1 bundle is TraderX's own `golden-v1/basic`, LF-exact, verifying against
        their published hashes."""
        from engine.traderx import load_bundle

        bundle = load_bundle(FIXTURES / "equity" / "v1")
        assert bundle.bundle_id == (
            "9a0cb99550dd2e18c15ea92a1b05b03d1035f0bb44eb1717a296312dec7c5d36"
        )
        assert bundle.cluster_epoch == "synthetic-golden"
        assert not bundle.has_terms

    def test_the_v1_swap_row_still_refuses_for_its_conventions(self):
        """The golden-v1 bundle's USD-SOFR swap still refuses for its conventions."""
        result = price_bundle(FIXTURES / "equity" / "v1", MARKET).to_dict()
        swap = [i for i in result["items"] if i["sourceIdentity"].get("contractId")]
        assert swap
        assert swap[0]["calculations"]["npv"]["status"] == "unsupported"


class TestTreasuriesAreUnchangedByW14:
    """The equity branch (dispatched first) does not change Treasury results."""

    def test_the_bill_still_prices(self):
        result = price_bundle(FIXTURES / "bill" / "v2", MARKET).to_dict()
        values = sorted(i["calculations"]["npv"]["value"] for i in result["items"])
        assert values[1] == pytest.approx(98_507.15, abs=0.01)
        assert values[0] == pytest.approx(-98_507.15, abs=0.01)

    def test_the_note_still_prices(self):
        result = price_bundle(FIXTURES / "note" / "v2", MARKET).to_dict()
        values = sorted(i["calculations"]["npv"]["value"] for i in result["items"])
        assert values[1] == pytest.approx(103_308.33, abs=0.01)
        assert values[0] == pytest.approx(-103_308.33, abs=0.01)

    def test_the_note_still_reports_a_rate_sensitivity(self):
        result = price_bundle(FIXTURES / "note" / "v2", MARKET).to_dict()
        for item in result["items"]:
            assert item["calculations"]["rateSensitivity"]["status"] == "ok"

    def test_the_sofr_refusal_is_unchanged(self):
        result = price_bundle(FIXTURES / "sofr" / "v2", MARKET).to_dict()
        refused = [i for i in result["items"] if i.get("refusal")]
        assert refused
        assert len(refused[0]["refusal"]["missingTerms"]) == 13


class TestCapabilitiesAdvertiseW14:
    """The capability document distinguishes "wait for a release" from "send a spot"."""

    def test_stage_is_at_least_w14(self):
        """The stage has reached W1.4 (a floor, not an exact string)."""
        stage = capabilities()["deliveryStage"]
        assert stage.startswith("W1.")
        major, minor = stage.removeprefix("W").split(".")[:2]
        assert (int(major), int(minor)) >= (1, 4), (
            f"deliveryStage {stage!r} is earlier than W1.4, which is when the "
            f"equity refusal this class describes was delivered"
        )

    def test_equity_is_a_known_type(self):
        assert EQUITY in capabilities()["products"]

    def test_equity_is_not_advertised_as_priced(self):
        equity = capabilities()["products"][EQUITY]
        assert equity["priced"] is False
        assert equity["calculations"] == []

    def test_equity_npv_is_advertised_as_blocked_on_a_market_input(self):
        """Equity npv is advertised as blocked on a market input (the spot), not as lacking
        a pricer."""
        blocked = capabilities()["products"][EQUITY]["blockedOnMarketInput"]
        assert blocked["npv"]["reason"] == SPOT_SOURCE_NOT_SUPPLIED
        assert "spot" in blocked["npv"]["requires"]

    def test_treasuries_are_not_blocked_on_a_market_input(self):
        """Treasuries are not listed as blocked."""
        assert capabilities()["products"]["TREASURY"]["blockedOnMarketInput"] == {}

    def test_the_advertised_block_matches_the_module_constant(self):
        """The advertised block matches the module constant."""
        advertised = capabilities()["products"][EQUITY]["blockedOnMarketInput"]
        assert advertised == BLOCKED_ON_MARKET_INPUT[EQUITY]

    def test_i18_is_advertised_as_a_known_limitation(self):
        ids = {lim["id"] for lim in capabilities()["knownLimitations"]}
        assert "I-18" in ids

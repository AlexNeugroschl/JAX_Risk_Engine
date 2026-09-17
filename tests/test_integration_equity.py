"""
W1.4 -- the equity position pricer (`docs/planning/traderX-integration-plan.md` §W1.4).

The plan's required tests, quoted: "long/short; multiplier applied
**exactly once**; currency/FX handling."

**Why this file tests a refusal rather than a price.** W1.4's formula is
`signedQuantity x multiplier x spot x fx`, and this boundary has a source
for neither `spot` nor `fx`: `marketInputs` registers flat interest-rate
profiles only, and `SimulationConfig.equities` drives simulated paths
rather than valuing a position (the plan says so outright, and so does
I-07). So the deliverable is an honest, *diagnosable* refusal -- and the
plan's three checks still apply to it, because the engine must read the
position correctly in order to refuse it correctly:

  - **long/short** -- the sign reaches `multipliedQuantity` intact, so the
    day a spot arrives the sign is already right;
  - **multiplier exactly once** -- asserted against a fixture whose
    multiplier is deliberately *not* 1, which is the only way the test can
    fail;
  - **currency/FX** -- a non-USD position refuses *differently*, because
    fixing only the spot would still not price it.

**The tempting wrong implementation is `closingMark`.** The CSV carries
one, and `quantity x closingMark x contractMultiplier` reproduces the
exporter's own `marketValue` column exactly. `TestDoesNotEchoTheExportedMark`
is the guard: that number must never appear as an `npv`, because it would
hand TraderX their own figure back as though this engine had valued it.
"""
import json
from pathlib import Path

import pytest

from engine.integration import price_bundle
from engine.integration.capabilities import BLOCKED_ON_MARKET_INPUT, capabilities
from engine.integration.equity import (
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
from engine.integration.terms import TermsEntry

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"

MARKET = {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}

#: The synthetic v2 fixture's own numbers.
QUANTITY = 1000.0
CLOSING_MARK = 10.0
#: `quantity x closingMark x contractMultiplier` -- the exporter's stated
#: `marketValue`. This number must never come back as an `npv`.
EXPORTED_MARKET_VALUE = 10_000.0


def _terms(**overrides) -> TermsEntry:
    """An equity terms entry matching the delivered fixture."""
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
    """The W1.4 decision: no spot source, so no number."""

    def test_price_equity_always_raises(self):
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(_terms(), _row())
        assert excinfo.value.reason == SPOT_SOURCE_NOT_SUPPLIED

    def test_the_refusal_names_the_missing_input(self):
        """Actionable: the coordinator must be able to tell what to send."""
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(_terms(), _row())
        assert excinfo.value.payload["missingInputs"] == ["spot"]

    def test_the_refusal_explains_why_the_mark_is_not_used(self):
        """The reasoning travels with the refusal, because 'you have a
        closingMark right there' is the obvious objection and the answer
        should not require reading the source."""
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(_terms(), _row())
        detail = excinfo.value.detail.lower()
        assert "closingmark" in detail
        assert "observation" in detail

    def test_no_market_input_makes_it_priceable(self):
        """A rate curve is not an equity spot. Requesting one must not
        change this row's outcome -- if it did, the refusal would be about
        the request rather than about the missing spot."""
        with_curve = price_bundle(FIXTURES / "equity" / "v2", MARKET).to_dict()
        without = price_bundle(FIXTURES / "equity" / "v2").to_dict()
        for item in with_curve["items"] + without["items"]:
            assert item["calculations"]["npv"]["status"] == "unsupported"


class TestDoesNotEchoTheExportedMark:
    """**The dangerous wrong implementation.**

    `quantity x closingMark x contractMultiplier` reproduces the
    exporter's own `marketValue` exactly. Returning it would look like a
    successful valuation and reconcile perfectly against TraderX -- while
    proving nothing, because it is their number.
    """

    def test_the_exported_market_value_never_appears_as_an_npv(self):
        result = price_bundle(FIXTURES / "equity" / "v2", MARKET).to_dict()
        for item in result["items"]:
            npv = item["calculations"]["npv"]
            assert npv["status"] != "ok"
            assert npv.get("value") is None

    def test_no_payload_field_carries_the_mark_derived_value(self):
        """Not merely absent from `value` -- absent from the payload
        entirely, so nothing downstream can pick it up as a price."""
        result = price_bundle(FIXTURES / "equity" / "v2", MARKET).to_dict()
        for item in result["items"]:
            payload = item["calculations"]["npv"]
            flat = json.dumps(payload)
            assert str(EXPORTED_MARKET_VALUE) not in flat
            assert "closingMark" not in payload

    def test_the_fixture_would_actually_expose_the_bug(self):
        """Guards the two tests above from being vacuous: the fixture must
        genuinely contain a mark that *could* have been echoed."""
        rows = (FIXTURES / "equity" / "v2" / "positions.csv").read_text().splitlines()
        data = [r for r in rows if r and not r.startswith("#")][1:]
        assert any(f",{CLOSING_MARK:.6f}," in r for r in data)


class TestLongShort:
    """Plan §W1.4: 'long/short'. The sign must survive into the refusal,
    so it is already correct the day a spot arrives."""

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
        """Through the pipeline, not just the reader -- working rule 8."""
        result = price_bundle(FIXTURES / "equity" / "v2", MARKET).to_dict()
        quantities = [
            item["calculations"]["npv"].get("multipliedQuantity")
            for item in result["items"]
            if item["sourceIdentity"]["security"] == "SYNTH-EQ"
        ]
        assert sorted(quantities) == [-QUANTITY, QUANTITY]


class TestMultiplierAppliedExactlyOnce:
    """Plan §W1.4: 'multiplier applied **exactly once**'.

    Every assertion here uses a multiplier that is **not 1**, because a
    multiplier of 1 makes applying it twice, once, or never
    indistinguishable -- the test would pass against all three.
    """

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
        """Both are reported, and they must not be the same number --
        a consumer reconciling needs the input and the product distinctly."""
        position = read_position(_terms(contractMultiplier="100"), _row())
        assert position.signed_quantity == QUANTITY
        assert position.contract_multiplier == 100.0
        assert position.multiplied_quantity == QUANTITY * 100

    def test_a_fractional_multiplier_is_honoured(self):
        """Not all multipliers are >= 1; nothing may round it to an int."""
        position = read_position(_terms(contractMultiplier="0.5"), _row())
        assert position.multiplied_quantity == QUANTITY * 0.5

    def test_absent_multiplier_defaults_to_one(self):
        """The legitimate default for a cash equity, and the fixture
        states it explicitly anyway."""
        terms = TermsEntry(
            instrument_type="EQUITY", terms={"currency": "USD"},
            missing_terms=(), provenance={}, identity={},
        )
        row = _row()
        del row["contractMultiplier"]
        assert read_position(terms, row).contract_multiplier == 1.0

    def test_an_unparseable_multiplier_is_refused_not_defaulted(self):
        """A malformed value is a broken artifact, not an omission."""
        with pytest.raises(EquityPricingError) as excinfo:
            read_position(_terms(contractMultiplier="x100"), _row())
        assert excinfo.value.reason == TERMS_INCOMPLETE

    def test_terms_multiplier_wins_over_the_csv(self):
        """The terms are the reference statement for a static property.
        Pinned because silently preferring the other source would change
        every equity's size."""
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
    """Plan §W1.4: 'currency/FX handling'."""

    def test_a_usd_position_refuses_for_the_spot_only(self):
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(_terms(), _row())
        assert excinfo.value.reason == SPOT_SOURCE_NOT_SUPPLIED
        assert excinfo.value.payload["missingInputs"] == ["spot"]

    def test_a_non_usd_position_refuses_differently(self):
        """**The distinction that matters.** Supplying a spot alone would
        still not price a EUR position -- so telling the coordinator
        'send a spot' would be wrong for this row."""
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(_terms(currency="EUR"), _row(currency="EUR"))
        assert excinfo.value.reason == FX_SOURCE_NOT_SUPPLIED
        assert excinfo.value.payload["missingInputs"] == ["spot", "fx"]

    def test_the_two_refusals_are_distinct_codes(self):
        assert SPOT_SOURCE_NOT_SUPPLIED != FX_SOURCE_NOT_SUPPLIED

    def test_currency_is_case_insensitive(self):
        """`usd` is USD. A lowercase currency must not be mistaken for a
        foreign one and refused for FX."""
        with pytest.raises(EquityPricingError) as excinfo:
            price_equity(_terms(currency="usd"), _row(currency="usd"))
        assert excinfo.value.reason == SPOT_SOURCE_NOT_SUPPLIED

    def test_an_absent_currency_is_treated_as_foreign_not_assumed_usd(self):
        """Refusing to infer, applied to currency: a blank is not USD.
        Assuming it would value a foreign position at parity."""
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
        """End to end: the fixture carries a USD pair and a EUR row, and
        each must come back under its own reason."""
        result = price_bundle(FIXTURES / "equity" / "v2", MARKET).to_dict()
        reasons = {
            item["sourceIdentity"]["security"]: item["calculations"]["npv"]["reason"]
            for item in result["items"]
        }
        assert reasons["SYNTH-EQ"] == SPOT_SOURCE_NOT_SUPPLIED
        assert reasons["SYNTH-EQ-EUR"] == FX_SOURCE_NOT_SUPPLIED


class TestIsEquity:
    """Keyed on `instrumentType`, never on which columns are blank."""

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
        """A bill also has a blank coupon schedule. Inferring the
        instrument from empty columns is the blank-reading mistake the
        whole boundary refuses."""
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
    """A broken row refuses distinctly from a missing market input --
    different problems, different fixes."""

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
        """The codes must not be conflated: one is fixed by re-exporting,
        the other by sending market data."""
        with pytest.raises(EquityPricingError) as excinfo:
            read_position(_terms(), _row(quantity="ten"))
        assert excinfo.value.reason != SPOT_SOURCE_NOT_SUPPLIED

    def test_a_zero_quantity_is_read_not_refused(self):
        """A closed-out position is a legitimate row, not an error."""
        assert read_position(_terms(), _row(quantity="0")).multiplied_quantity == 0.0


class TestRefusalsAreEquityPricingErrors:
    """**I-17's regression class, applied to the third pricer.**

    A refusal raised by one module and caught as another's escapes the
    pipeline's handler and fails the whole bundle. `equity.py` therefore
    raises its own exception type and shares no parser that raises a
    different one.
    """

    def test_refusals_are_not_bill_or_note_errors(self):
        from engine.integration.bill import BillPricingError
        from engine.integration.note import NotePricingError

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
        """The consequence that made I-17 a bug, asserted for equities."""
        from engine.integration.market_inputs import ASSUMED_PROFILES, MarketInputs
        from engine.integration.pipeline import _equity_outcomes
        from engine.integration.terms import JoinedRow

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
        """Plan §W0.7: identity travels on unsupported rows too. A refusal
        nobody can attribute to a position is useless."""
        for item in self._v2()["items"]:
            assert item["itemId"]
            assert item["sourceIdentity"]["accountId"]
            assert item["sourceIdentity"]["security"]

    def test_the_refusal_carries_the_validated_inputs(self):
        """Diagnosable, not merely negative: the consumer can confirm the
        engine read the row correctly."""
        item = self._v2()["items"][0]
        npv = item["calculations"]["npv"]
        assert npv["signedQuantity"] == QUANTITY
        assert npv["contractMultiplier"] == 1.0
        assert npv["multipliedQuantity"] == QUANTITY
        assert npv["intendedMethod"] == "spot-revaluation"

    def test_vega_is_not_applicable_for_an_equity(self):
        """A cash equity has no optionality, so vega is not a gap that
        should count against coverage."""
        for item in self._v2()["items"]:
            assert item["calculations"]["vega"]["status"] == "not-applicable"

    def test_accrued_interest_is_not_claimed_for_an_equity(self):
        """An equity has no accrual. Reporting `ok` at zero would assert a
        coupon structure it does not have."""
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
        """A v1 bundle has no terms artifact, so the engine cannot even
        establish the row IS an equity -- that refusal is more fundamental
        than the missing spot and wins."""
        result = price_bundle(FIXTURES / "equity" / "v1", MARKET).to_dict()
        for item in result["items"]:
            npv = item["calculations"]["npv"]
            assert npv["status"] == "unsupported"
            assert npv["reason"] == "TERMS_NOT_SUPPLIED"

    def test_the_v1_bundle_is_the_real_traderx_fixture(self):
        """Vendored from TraderX's own `golden-v1/basic`, LF-exact, and it
        verifies against their published hashes -- so the v1 path is
        exercised against real bytes rather than something we authored."""
        from engine.integration import load_bundle

        bundle = load_bundle(FIXTURES / "equity" / "v1")
        assert bundle.bundle_id == (
            "9a0cb99550dd2e18c15ea92a1b05b03d1035f0bb44eb1717a296312dec7c5d36"
        )
        assert bundle.cluster_epoch == "synthetic-golden"
        assert not bundle.has_terms

    def test_the_v1_swap_row_still_refuses_for_its_conventions(self):
        """The golden-v1 bundle also carries a USD-SOFR swap. Adding an
        equity branch must not disturb it."""
        result = price_bundle(FIXTURES / "equity" / "v1", MARKET).to_dict()
        swap = [i for i in result["items"] if i["sourceIdentity"].get("contractId")]
        assert swap
        assert swap[0]["calculations"]["npv"]["status"] == "unsupported"


class TestTreasuriesAreUnchangedByW14:
    """W1.4 must not alter W1.2/W1.3's delivered numbers. The equity
    branch is dispatched first, so this is the guard that it does not
    swallow a bond."""

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
    """A coordinator reads this *before* submitting. It must be able to
    tell 'wait for a release' from 'send me a spot'."""

    def test_stage_is_at_least_w14(self):
        """**Updated by W1.6**, which bumped the stage to `W1.6`.

        This previously pinned the literal `"W1.4"`, which made it fail on
        every future stage bump regardless of whether anything about the
        equity contract changed — a test that breaks for reasons unrelated
        to what it is named for. What W1.4 actually needs to hold is that
        the advertised stage has *reached* W1.4, so the equity capability
        below is the one being described. The equity-specific assertions in
        this class are what pin the contract itself.
        """
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
        """**The distinction W1.4 exists to publish.** Not 'no pricer' --
        'no spot', which the coordinator can fix."""
        blocked = capabilities()["products"][EQUITY]["blockedOnMarketInput"]
        assert blocked["npv"]["reason"] == SPOT_SOURCE_NOT_SUPPLIED
        assert "spot" in blocked["npv"]["requires"]

    def test_treasuries_are_not_blocked_on_a_market_input(self):
        """The block list must be specific, not a blanket disclaimer."""
        assert capabilities()["products"]["TREASURY"]["blockedOnMarketInput"] == {}

    def test_the_advertised_block_matches_the_module_constant(self):
        """Derived, never hand-written, so the document cannot drift from
        the code that produces the refusal."""
        advertised = capabilities()["products"][EQUITY]["blockedOnMarketInput"]
        assert advertised == BLOCKED_ON_MARKET_INPUT[EQUITY]

    def test_i18_is_advertised_as_a_known_limitation(self):
        ids = {lim["id"] for lim in capabilities()["knownLimitations"]}
        assert "I-18" in ids

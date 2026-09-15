"""
W0.4 -- the convention allowlist and the refusal path
(`docs/planning/traderx-integration-plan.md` §W0.4). Closes part of I-05.

The named tests from the plan:
  - SOFR fixture -> `unsupported` naming all **13** `missingTerms`
  - a booking with absent conventions -> `unsupported`, **not** generically priced
  - an allowlisted generic swap still prices (here: is still *accepted*; W0
    ships no pricer, so acceptance is what there is to assert)

`TestRefusesToInfer` is the one guarding the money: a booking with no stated
conventions is the case the generic builder would have swallowed, returning
a confident wrong number.
"""
from pathlib import Path

import pytest

from engine.integration.bundle import load_bundle
from engine.integration.conventions import (
    CONVENTION_NOT_SUPPORTED,
    INSTRUMENT_TYPE_NOT_SUPPORTED,
    REQUIRED_SWAP_CONVENTIONS,
    SUPPORTED_FLOAT_INDICES,
    SUPPORTED_OVERNIGHT_COMPOUNDING,
    SUPPORTED_SWAP_DAY_COUNTS,
    TERMS_NOT_SUPPLIED,
    check_conventions,
)
from engine.integration.terms import JoinedRow, TermsEntry, join_terms

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"

#: A booking whose conventions ARE on the allowlist -- the generic
#: term-IBOR/ACT-365 profile the engine actually implements.
ALLOWLISTED_SWAP_TERMS = {
    "currency": "USD",
    "payReceive": "PAY_FIXED",
    "notional": "1000000",
    "fixedRate": "0.040000",
    "floatIndex": "SimIndex6M",
    "effectiveDate": "2025-06-03",
    "maturityDate": "2030-06-03",
    "fixedPaymentFrequency": "6M",
    "fixedDayCount": "ACT/365",
    "floatingDayCount": "ACT/365",
}


def _swap_row(terms: dict, missing=()) -> JoinedRow:
    entry = TermsEntry(
        instrument_type="SWAP",
        terms=terms,
        missing_terms=tuple(missing),
        provenance={"origin": "synthetic"},
        identity={"source": "contracts", "contractId": "SW-1", "clusterEpoch": "e"},
    )
    return JoinedRow(
        source="contracts",
        row={"accountId": "1", "contractId": "SW-1"},
        entry=entry,
    )


class TestSofrFixtureIsRefused:
    """Plan §W0.4's first named test, and the W0 exit criterion."""

    def test_sofr_is_refused(self):
        join = join_terms(load_bundle(FIXTURES / "sofr" / "v2"))
        refusal = check_conventions(join.rows[0])

        assert refusal is not None
        assert refusal.reason == CONVENTION_NOT_SUPPORTED

    def test_refusal_names_all_thirteen_missing_terms(self):
        join = join_terms(load_bundle(FIXTURES / "sofr" / "v2"))
        refusal = check_conventions(join.rows[0])

        assert len(refusal.missing_terms) == 13
        assert set(refusal.missing_terms) == {
            "businessDayAdjustment", "calendar", "fixedPaymentLagBusinessDays",
            "fixedSchedule", "fixingCalendar", "fixingHistoryReference",
            "floatingDayCount", "floatingPaymentLagBusinessDays", "floatingSchedule",
            "lockoutBusinessDays", "lookbackBusinessDays", "observationShift",
            "overnightCompounding",
        }

    def test_missing_terms_reach_the_published_result(self):
        from engine.integration import price_bundle
        result = price_bundle(FIXTURES / "sofr" / "v2").to_dict()
        item = result["items"][0]

        assert item["refusal"]["reason"] == CONVENTION_NOT_SUPPORTED
        assert len(item["refusal"]["missingTerms"]) == 13
        # ...and on the calculation itself, not only in a summary block.
        assert len(item["calculations"]["npv"]["missingTerms"]) == 13

    def test_sofr_is_refused_even_with_complete_terms(self):
        """The deeper point: SOFR is refused for its CONVENTIONS, not only
        for its incomplete export. Supplying the 13 missing terms does not
        make USD-SOFR priceable -- the engine has no overnight-compounded
        index. This test fails against an implementation that only checks
        `missingTerms`."""
        terms = dict(ALLOWLISTED_SWAP_TERMS)
        terms.update({
            "floatIndex": "USD-SOFR",
            "fixedDayCount": "ACT/360",
            "floatingDayCount": "ACT/360",
            "overnightCompounding": "COMPOUNDED_IN_ARREARS",
        })
        refusal = check_conventions(_swap_row(terms, missing=()))

        assert refusal is not None
        assert refusal.reason == CONVENTION_NOT_SUPPORTED
        assert "floatIndex" in refusal.offending_fields


class TestRefusesToInfer:
    """Plan §W0.4 step 4 -- **the bug this prevents**.

    `build_vanilla_swap` produces SimIndex6M, ACT/365 both legs, a TARGET
    calendar and a tenor-derived schedule. A booking that states none of
    those would be given all of them. ACT/360-vs-ACT/365 alone is ~$1,906
    on a $1mm 5Y leg, about 46x a 1bp DV01 -- and no existing test catches
    it, because every test builds its inputs with that same builder.
    """

    def test_booking_with_no_conventions_is_refused(self):
        bare = {"currency": "USD", "notional": "1000000", "fixedRate": "0.04"}
        refusal = check_conventions(_swap_row(bare))

        assert refusal is not None
        assert refusal.reason == CONVENTION_NOT_SUPPORTED

    def test_refusal_names_every_absent_convention(self):
        bare = {"currency": "USD", "notional": "1000000", "fixedRate": "0.04"}
        refusal = check_conventions(_swap_row(bare))

        assert set(refusal.offending_fields) == set(REQUIRED_SWAP_CONVENTIONS)
        assert "never defaulted" in refusal.detail

    @pytest.mark.parametrize("dropped", REQUIRED_SWAP_CONVENTIONS)
    def test_any_single_absent_convention_is_refused(self, dropped):
        """Each required convention individually, so none is accidentally
        unchecked."""
        terms = dict(ALLOWLISTED_SWAP_TERMS)
        del terms[dropped]

        refusal = check_conventions(_swap_row(terms))
        assert refusal is not None
        assert dropped in refusal.offending_fields

    @pytest.mark.parametrize("blanked", REQUIRED_SWAP_CONVENTIONS)
    def test_blank_convention_is_treated_as_absent(self, blanked):
        """A present-but-empty field is not a stated convention. Treating
        `""` as 'supplied' would let an empty export through."""
        terms = dict(ALLOWLISTED_SWAP_TERMS)
        terms[blanked] = ""

        refusal = check_conventions(_swap_row(terms))
        assert refusal is not None
        assert blanked in refusal.offending_fields

    def test_absent_conventions_are_checked_before_supplied_ones(self):
        """A booking stating nothing must be refused for stating nothing,
        rather than passing vacuously because there was nothing to
        disallow."""
        refusal = check_conventions(_swap_row({}))
        assert refusal is not None
        assert "<absent>" in dict(refusal.offending).values()


class TestDayCountAllowlist:
    """ACT/360 is not 'not yet added'; it is a different instrument."""

    @pytest.mark.parametrize("field", ("fixedDayCount", "floatingDayCount"))
    def test_act360_leg_is_refused(self, field):
        terms = dict(ALLOWLISTED_SWAP_TERMS)
        terms[field] = "ACT/360"

        refusal = check_conventions(_swap_row(terms))
        assert refusal is not None
        assert field in refusal.offending_fields

    def test_the_offending_value_is_reported_not_just_the_field(self):
        """A consumer needs to know *what* was rejected to act on it."""
        terms = dict(ALLOWLISTED_SWAP_TERMS)
        terms["fixedDayCount"] = "30/360"

        refusal = check_conventions(_swap_row(terms))
        assert ("fixedDayCount", "30/360") in refusal.offending

    def test_only_act365_is_allowlisted_today(self):
        """Pins the allowlist's current contents. Widening it is a
        financial assertion and should fail this test deliberately."""
        assert SUPPORTED_SWAP_DAY_COUNTS == ("ACT/365",)


class TestFloatIndexAllowlist:
    def test_generic_sim_index_is_accepted(self):
        assert check_conventions(_swap_row(dict(ALLOWLISTED_SWAP_TERMS))) is None

    @pytest.mark.parametrize("index", ("USD-SOFR", "SOFR", "ESTR", "USD-LIBOR-3M"))
    def test_real_indices_are_refused(self, index):
        terms = dict(ALLOWLISTED_SWAP_TERMS)
        terms["floatIndex"] = index

        refusal = check_conventions(_swap_row(terms))
        assert refusal is not None
        assert "floatIndex" in refusal.offending_fields

    def test_sofr_is_absent_from_the_allowlist(self):
        """W2 is blocked on D03/D04. Until then USD-SOFR must not appear
        here -- this test is the tripwire on someone adding it early."""
        assert not any("SOFR" in i for i in SUPPORTED_FLOAT_INDICES)

    def test_overnight_compounding_is_not_implemented_at_all(self):
        assert SUPPORTED_OVERNIGHT_COMPOUNDING == ()

    def test_any_compounding_method_is_refused(self):
        terms = dict(ALLOWLISTED_SWAP_TERMS)
        terms["overnightCompounding"] = "COMPOUNDED_IN_ARREARS"

        refusal = check_conventions(_swap_row(terms))
        assert refusal is not None
        assert "overnightCompounding" in refusal.offending_fields


class TestAllowlistedSwapIsAccepted:
    """Plan §W0.4's third named test: 'an allowlisted generic swap still
    prices'. W0 ships no pricer, so what is asserted is that it is not
    REFUSED -- the refusal path must not become a blanket deny."""

    def test_generic_swap_passes_the_convention_check(self):
        assert check_conventions(_swap_row(dict(ALLOWLISTED_SWAP_TERMS))) is None

    def test_complete_treasury_passes_the_convention_check(self):
        join = join_terms(load_bundle(FIXTURES / "note" / "v2"))
        for row in join.rows:
            assert check_conventions(row) is None

    def test_accepted_is_not_the_same_as_priced(self):
        """Plan working rule 4: 'ORE can represent it' is not 'my engine
        prices it'. A row passing the convention check still reports no
        NPV at W0, with a *different* reason."""
        from engine.integration import price_bundle
        result = price_bundle(FIXTURES / "note" / "v2")
        npv = result.items[0].calculations["npv"]

        assert npv.status == "unsupported"
        assert npv.reason == "NO_PRICER_AT_THIS_STAGE"
        assert npv.reason != CONVENTION_NOT_SUPPORTED


class TestMissingTermsShortCircuits:
    def test_incomplete_export_is_refused_regardless_of_supplied_terms(self):
        """An incompletely specified instrument is refused full stop --
        the engine does not opine on whether the supplied subset would
        have been acceptable."""
        refusal = check_conventions(
            _swap_row(dict(ALLOWLISTED_SWAP_TERMS), missing=("calendar",))
        )
        assert refusal is not None
        assert refusal.missing_terms == ("calendar",)


class TestNoTermsAtAll:
    def test_row_without_terms_is_refused(self):
        row = JoinedRow(
            source="contracts", row={"accountId": "1", "contractId": "SW-1"},
            entry=None, unjoined_reason="NO_TERMS_ARTIFACT",
        )
        refusal = check_conventions(row)

        assert refusal is not None
        assert refusal.reason == TERMS_NOT_SUPPLIED
        assert "will not be inferred" in refusal.detail

    def test_unsupported_instrument_type_is_refused(self):
        entry = TermsEntry(
            instrument_type="CORPORATE", terms={}, missing_terms=(),
            provenance={}, identity={"source": "positions", "security": "X"},
        )
        row = JoinedRow(source="positions", row={"accountId": "1", "security": "X"}, entry=entry)

        refusal = check_conventions(row)
        assert refusal is not None
        assert refusal.reason == INSTRUMENT_TYPE_NOT_SUPPORTED


class TestRefusalIsSerializable:
    def test_refusal_dict_separates_missing_from_offending(self):
        """The two lists answer different questions: `missingTerms` is
        closed by a better export, `offendingFields` by a convention
        agreement and engine work."""
        terms = dict(ALLOWLISTED_SWAP_TERMS)
        terms["fixedDayCount"] = "ACT/360"
        payload = check_conventions(_swap_row(terms)).to_dict()

        assert payload["missingTerms"] == []
        assert payload["offendingFields"] == [{"field": "fixedDayCount", "value": "ACT/360"}]

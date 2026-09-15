"""
W0.5 -- result schema and coverage model
(`docs/planning/traderx-integration-plan.md` §W0.5).

The plan's named tests: statuses sum to item count; `complete: false` on a
partial aggregate; mixed-currency returns per-currency, not a blended
scalar.
"""
from pathlib import Path

import pytest

from engine.integration.identity import ItemIdentity
from engine.integration.result import (
    CALCULATIONS,
    STATUSES,
    CalculationOutcome,
    Coverage,
    CurrencyAggregate,
    ItemResult,
    RiskResult,
    aggregate_by_currency,
    compute_coverage,
    sensitivity_payload,
)

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"


def _identity(account: str = "1", security: str = "SEC") -> ItemIdentity:
    return ItemIdentity(
        kind="position", account_id=account, cluster_epoch="e", security=security,
    )


def _item(outcomes: dict, currency: str = "USD", account: str = "1", security: str = "SEC") -> ItemResult:
    """An item with `outcomes` for the named calculations and
    `unsupported` for the rest."""
    calculations = {
        name: outcomes.get(name, CalculationOutcome.unsupported("X"))
        for name in CALCULATIONS
    }
    return ItemResult(
        identity=_identity(account, security),
        calculations=calculations,
        currency=currency,
    )


class TestFrozenVocabulary:
    def test_calculation_names_are_frozen(self):
        assert CALCULATIONS == (
            "npv", "accruedInterest", "rateSensitivity", "rateGamma",
            "theta", "vega", "varEs",
        )

    def test_five_statuses(self):
        assert STATUSES == ("ok", "unsupported", "unavailable", "failed", "not-applicable")

    def test_unknown_calculation_name_is_rejected(self):
        with pytest.raises(ValueError, match="unknown calculation name"):
            ItemResult(
                identity=_identity(),
                calculations={**{n: CalculationOutcome.unsupported("X") for n in CALCULATIONS},
                              "cva": CalculationOutcome.unsupported("X")},
            )

    def test_unknown_status_is_rejected(self):
        with pytest.raises(ValueError, match="unknown status"):
            CalculationOutcome(status="maybe")

    def test_every_calculation_needs_an_explicit_outcome(self):
        """An omitted calculation is indistinguishable from a forgotten
        one -- exactly what the coverage model exists to prevent."""
        with pytest.raises(ValueError, match="every calculation needs an explicit outcome"):
            ItemResult(identity=_identity(), calculations={"npv": CalculationOutcome.ok(1.0)})


class TestNonOkOutcomesCarryNoValue:
    @pytest.mark.parametrize("status", ("unsupported", "unavailable", "failed", "not-applicable"))
    def test_value_on_a_non_ok_outcome_is_rejected(self, status):
        """A number attached to a refusal is a number someone will read."""
        with pytest.raises(ValueError, match="must not carry a value"):
            CalculationOutcome(status=status, value=0.0)

    def test_ok_outcome_carries_its_value(self):
        assert CalculationOutcome.ok(42.0).to_dict()["value"] == 42.0

    def test_refusal_dict_has_no_value_key(self):
        payload = CalculationOutcome.unsupported("R", "d").to_dict()
        assert "value" not in payload
        assert payload["reason"] == "R"


class TestCoverageSumsToItemCount:
    """Plan's named test: 'statuses sum to item count'."""

    def test_statuses_sum_to_item_count(self):
        items = [
            _item({"npv": CalculationOutcome.ok(1.0)}),
            _item({"npv": CalculationOutcome.unsupported("X")}),
            _item({"npv": CalculationOutcome.unavailable("Y")}),
            _item({"npv": CalculationOutcome.failed("Z")}),
            _item({"npv": CalculationOutcome.not_applicable()}),
        ]
        coverage = compute_coverage(items)

        assert coverage.item_count == 5
        assert sum(coverage.by_calculation["npv"].values()) == 5
        assert coverage.all_outcomes_accounted_for

    def test_every_status_key_is_present_even_at_zero(self):
        """A consumer reading `counts['failed']` must never have to handle
        a missing key differently from a zero."""
        coverage = compute_coverage([_item({"npv": CalculationOutcome.ok(1.0)})])
        assert set(coverage.by_calculation["npv"]) == {
            "ok", "unsupported", "unavailable", "failed", "notApplicable",
        }

    def test_accounted_for_is_true_even_when_everything_failed(self):
        """It is a consistency check on the document, not a quality
        judgement on the portfolio."""
        items = [_item({n: CalculationOutcome.failed("boom") for n in CALCULATIONS})]
        coverage = compute_coverage(items)

        assert coverage.all_outcomes_accounted_for is True
        assert coverage.all_applicable_computed is False

    def test_empty_result_is_accounted_for(self):
        coverage = compute_coverage([])
        assert coverage.item_count == 0
        assert coverage.all_outcomes_accounted_for


class TestNotApplicableNeverCountsAgainstCoverage:
    """Plan §W0.5: 'vega on a vanilla swap is not a gap'."""

    def test_all_applicable_computed_ignores_not_applicable(self):
        items = [_item({
            **{n: CalculationOutcome.ok(1.0) for n in CALCULATIONS},
            "vega": CalculationOutcome.not_applicable("no optionality"),
        })]
        coverage = compute_coverage(items)

        assert coverage.by_calculation["vega"]["notApplicable"] == 1
        assert coverage.all_applicable_computed is True

    @pytest.mark.parametrize("gap", ("unsupported", "unavailable", "failed"))
    def test_real_gaps_do_count(self, gap):
        outcome = {
            "unsupported": CalculationOutcome.unsupported("X"),
            "unavailable": CalculationOutcome.unavailable("X"),
            "failed": CalculationOutcome.failed("X"),
        }[gap]
        items = [_item({**{n: CalculationOutcome.ok(1.0) for n in CALCULATIONS}, "vega": outcome})]

        assert compute_coverage(items).all_applicable_computed is False

    def test_the_two_flags_are_independent(self):
        """A W0 result is exactly this shape: everything accounted for,
        nothing computed."""
        from engine.integration import price_bundle
        coverage = price_bundle(FIXTURES / "sofr" / "v2").coverage

        assert coverage.all_outcomes_accounted_for is True
        assert coverage.all_applicable_computed is False


class TestAggregatesAreCompleteOrSayOtherwise:
    """Plan's named test: `complete: false` on a partial aggregate."""

    def test_full_coverage_is_complete(self):
        items = [_item({"npv": CalculationOutcome.ok(10.0)}) for _ in range(3)]
        (aggregate,) = aggregate_by_currency(items, "npv")

        assert aggregate.value == pytest.approx(30.0)
        assert aggregate.complete is True
        assert aggregate.excluded_items == ()

    def test_partial_coverage_is_not_complete(self):
        items = [
            _item({"npv": CalculationOutcome.ok(10.0)}, security="A"),
            _item({"npv": CalculationOutcome.unsupported("X")}, security="B"),
        ]
        (aggregate,) = aggregate_by_currency(items, "npv")

        assert aggregate.complete is False
        assert aggregate.covered_item_count == 1
        assert aggregate.total_item_count == 2

    def test_excluded_items_are_named(self):
        """'A total over a subset, presented as a total' is only safe if
        the subset is identified."""
        excluded_item = _item({"npv": CalculationOutcome.unsupported("X")}, security="B")
        items = [_item({"npv": CalculationOutcome.ok(10.0)}, security="A"), excluded_item]
        (aggregate,) = aggregate_by_currency(items, "npv")

        assert aggregate.excluded_items == (excluded_item.item_id,)

    def test_missing_values_are_not_treated_as_zero(self):
        """Summing a refusal as 0.0 would make a partial total look
        plausible. It is excluded instead."""
        items = [
            _item({"npv": CalculationOutcome.ok(10.0)}, security="A"),
            _item({"npv": CalculationOutcome.unavailable("X")}, security="B"),
        ]
        (aggregate,) = aggregate_by_currency(items, "npv")

        assert aggregate.value == pytest.approx(10.0)
        assert aggregate.complete is False

    def test_complete_flag_is_in_the_serialized_form(self):
        items = [_item({"npv": CalculationOutcome.unsupported("X")})]
        payload = aggregate_by_currency(items, "npv")[0].to_dict()

        assert payload["complete"] is False
        assert payload["coveredItemCount"] == 0
        assert payload["totalItemCount"] == 1


class TestCrossCurrency:
    """Plan's named test: 'mixed-currency returns per-currency, not a
    blended scalar'."""

    def test_per_currency_aggregates(self):
        items = [
            _item({"npv": CalculationOutcome.ok(10.0)}, currency="USD", security="A"),
            _item({"npv": CalculationOutcome.ok(20.0)}, currency="EUR", security="B"),
        ]
        aggregates = aggregate_by_currency(items, "npv")

        assert len(aggregates) == 2
        assert {a.currency for a in aggregates} == {"USD", "EUR"}

    def test_values_are_not_blended(self):
        """30.0 would be the wrong answer -- it adds dollars to euros."""
        items = [
            _item({"npv": CalculationOutcome.ok(10.0)}, currency="USD", security="A"),
            _item({"npv": CalculationOutcome.ok(20.0)}, currency="EUR", security="B"),
        ]
        by_currency = {a.currency: a.value for a in aggregate_by_currency(items, "npv")}

        assert by_currency == {"USD": pytest.approx(10.0), "EUR": pytest.approx(20.0)}
        assert 30.0 not in by_currency.values()

    def test_conversion_is_explicitly_not_applied(self):
        """The engine is given no FX rates; inventing them would be the
        silent approximation this design refuses."""
        items = [_item({"npv": CalculationOutcome.ok(10.0)}, currency="USD")]
        assert aggregate_by_currency(items, "npv")[0].to_dict()[
            "reportingCurrencyConversion"] == "NOT_APPLIED"


class TestSensitivityPayload:
    """Plan §W0.5: 'never a method implied by a field name'."""

    def test_payload_carries_method_bump_and_factor(self):
        payload = sensitivity_payload(
            method="bumped-revaluation", derivative="dNPV/dz",
            shocked_factor="USD-OIS-2Y", bump=1e-4, value=123.4, currency="USD",
        )
        assert payload["method"] == "bumped-revaluation"
        assert payload["bump"] == 1e-4
        assert payload["shockedFactor"] == "USD-OIS-2Y"

    def test_bump_is_numeric_not_a_description(self):
        payload = sensitivity_payload(
            "ad-first-order", "dNPV/dz", "f", 0.0001, 1.0, "USD")
        assert isinstance(payload["bump"], float)

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="unknown sensitivity method"):
            sensitivity_payload("finite-difference-ish", "d", "f", 1e-4, 1.0, "USD")


class TestSerializedResult:
    def test_result_round_trips_to_a_dict(self):
        from engine.integration import price_bundle
        payload = price_bundle(FIXTURES / "note" / "v2").to_dict()

        assert payload["sessionDate"] == "2025-06-02"
        assert payload["mappingVersion"]
        assert payload["engineVersion"]
        assert len(payload["items"]) == 2
        assert payload["coverage"]["itemCount"] == 2

    def test_every_item_has_every_calculation(self):
        from engine.integration import price_bundle
        payload = price_bundle(FIXTURES / "note" / "v2").to_dict()

        for item in payload["items"]:
            assert set(item["calculations"]) == set(CALCULATIONS)

    def test_w0_never_reports_failed(self):
        """`failed` means 'attempted and errored'. W0 attempts nothing, so
        reporting it would misdescribe the stage and mislead a coordinator
        into retrying."""
        from engine.integration import price_bundle
        for case in ("bill", "note", "sofr"):
            for version in ("v1", "v2"):
                payload = price_bundle(FIXTURES / case / version).to_dict()
                for item in payload["items"]:
                    for name, outcome in item["calculations"].items():
                        assert outcome["status"] != "failed"

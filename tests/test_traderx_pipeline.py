"""
`price_bundle` end to end without market inputs (the W0 exit criterion of
docs/planning/details/traderx-integration.md), plus the capability
document.

Exit criterion: the SOFR case returns `CONVENTION_NOT_SUPPORTED` naming all 13 missing
terms, and bill/note return structurally valid results with `npv: unsupported`.
"""
import json
from pathlib import Path

import pytest

from engine.traderx import capabilities, price_bundle
from engine.traderx.bundle import BundleIntegrityError, load_bundle
from engine.traderx.result import CALCULATIONS

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"


class TestExitCriterion:
    """The W0 exit criterion."""

    def test_sofr_returns_convention_not_supported(self):
        result = price_bundle(FIXTURES / "sofr" / "v2")
        (item,) = result.items

        assert item.calculations["npv"].status == "unsupported"
        assert item.calculations["npv"].reason == "CONVENTION_NOT_SUPPORTED"

    def test_sofr_names_all_thirteen_missing_terms(self):
        result = price_bundle(FIXTURES / "sofr" / "v2")
        (item,) = result.items

        assert len(item.refusal["missingTerms"]) == 13

    def test_sofr_refusal_is_identified(self):
        """The refusal is attributable to its booking."""
        result = price_bundle(FIXTURES / "sofr" / "v2")
        (item,) = result.items

        assert item.identity.contract_id == "SW-3"
        assert item.identity.account_id == "22214"
        assert item.item_id

    @pytest.mark.parametrize("case", ("bill", "note"))
    def test_bill_and_note_return_npv_unsupported(self, case):
        result = price_bundle(FIXTURES / case / "v2")

        assert result.items
        for item in result.items:
            assert item.calculations["npv"].status == "unsupported"

    @pytest.mark.parametrize("case", ("bill", "note"))
    def test_bill_and_note_results_are_structurally_valid(self, case):
        """Every item carries every calculation and its identity, and coverage sums."""
        result = price_bundle(FIXTURES / case / "v2")
        payload = result.to_dict()

        assert result.coverage.all_outcomes_accounted_for
        for item in payload["items"]:
            assert set(item["calculations"]) == set(CALCULATIONS)
            assert item["itemId"]
            assert item["sourceIdentity"]
        assert payload["itemOrder"]["itemCount"] == len(payload["items"])

    def test_the_whole_result_is_json_serializable(self):
        """The result serializes to JSON."""
        for case in ("bill", "note", "sofr"):
            payload = price_bundle(FIXTURES / case / "v2").to_dict()
            assert json.loads(json.dumps(payload)) == payload


class TestNothingIsPricedAtW0:
    """Without market inputs, no model-driven calculation is `ok`."""

    @pytest.mark.parametrize("case", ("bill", "note", "sofr"))
    @pytest.mark.parametrize("version", ("v1", "v2"))
    def test_no_model_driven_calculation_is_ok(self, case, version):
        result = price_bundle(FIXTURES / case / version)
        model_driven = [c for c in CALCULATIONS if c != "accruedInterest"]

        for item in result.items:
            for name in model_driven:
                assert item.calculations[name].status != "ok", (
                    f"{case}/{version}: {name} was computed, but W0 ships no pricer"
                )

    def test_all_applicable_computed_is_false_everywhere(self):
        for case in ("bill", "note", "sofr"):
            assert price_bundle(FIXTURES / case / "v2").coverage.all_applicable_computed is False


class TestAccruedInterestIsAnsweredForReal:
    """Accrued interest is answered without a model (a unit conversion of the export)."""

    def test_note_accrued_is_ok(self):
        result = price_bundle(FIXTURES / "note" / "v2")
        for item in result.items:
            assert item.calculations["accruedInterest"].status == "ok"

    def test_bill_accrued_is_a_structural_zero(self):
        result = price_bundle(FIXTURES / "bill" / "v2")
        for item in result.items:
            outcome = item.calculations["accruedInterest"]
            assert outcome.status == "ok"
            assert outcome.value == 0.0
            assert outcome.payload["provenance"] == "structural-zero"

    def test_v1_bill_accrued_is_unavailable_not_zero(self):
        """Without terms a blank accrued is `unavailable` (fixed by resending with terms),
        not `unsupported`."""
        result = price_bundle(FIXTURES / "bill" / "v1")
        for item in result.items:
            outcome = item.calculations["accruedInterest"]
            assert outcome.status == "unavailable"
            assert outcome.reason == "NO_TERMS_ARTIFACT"
            assert outcome.value is None

    def test_accrued_survives_a_convention_refusal(self):
        """A convention refusal does not flatten the accrued verdict."""
        result = price_bundle(FIXTURES / "bill" / "v1")
        item = result.items[0]

        assert item.calculations["npv"].reason == "TERMS_NOT_SUPPLIED"
        assert item.calculations["accruedInterest"].status == "unavailable"


class TestNotApplicableIsUsedPrecisely:
    def test_vega_on_a_treasury_is_not_applicable(self):
        """Vega on a Treasury is not-applicable (no optionality), not a gap."""
        result = price_bundle(FIXTURES / "note" / "v2")
        for item in result.items:
            assert item.calculations["vega"].status == "not-applicable"

    def test_vega_on_a_swap_is_not_applicable(self):
        result = price_bundle(FIXTURES / "sofr" / "v2")
        assert result.items[0].calculations["vega"].status == "not-applicable"

    def test_var_es_is_not_applicable_per_item(self):
        """VaR/ES is portfolio-level: not-applicable per item, so coverage still sums."""
        result = price_bundle(FIXTURES / "note" / "v2")
        for item in result.items:
            assert item.calculations["varEs"].status == "not-applicable"

    def test_a_swaption_vega_is_a_real_gap_not_not_applicable(self):
        """A swaption has vega, so its absence is a real gap, not not-applicable (as a naive
        "contracts rows are swaps" fallback would say)."""
        from engine.traderx.pipeline import _build_item
        from engine.traderx.terms import JoinedRow

        swaption = JoinedRow(
            source="contracts", entry=None, unjoined_reason="NO_TERMS_ARTIFACT",
            row={"accountId": "1", "contractId": "SWPT-1",
                 "productType": "SWAPTION", "currency": "USD"},
        )
        swap = JoinedRow(
            source="contracts", entry=None, unjoined_reason="NO_TERMS_ARTIFACT",
            row={"accountId": "1", "contractId": "SW-1",
                 "productType": "SWAP", "currency": "USD"},
        )

        assert _build_item(swaption, "e").calculations["vega"].status == "unsupported"
        assert _build_item(swap, "e").calculations["vega"].status == "not-applicable"


class TestV1Warns:
    @pytest.mark.parametrize("case", ("bill", "note", "sofr"))
    def test_v1_carries_a_warning(self, case):
        result = price_bundle(FIXTURES / case / "v1")
        assert result.warnings
        assert any("no instrument-terms artifact" in w for w in result.warnings)

    @pytest.mark.parametrize("case", ("bill", "note", "sofr"))
    def test_v2_does_not(self, case):
        """A v2 bundle has no terms warning (it still warns that market inputs are
        NOT_SUPPLIED)."""
        warnings = price_bundle(FIXTURES / case / "v2").warnings
        assert not any("instrument-terms artifact" in w for w in warnings)


class TestPipelineAcceptsBundleOrPath:
    def test_accepts_a_path(self):
        assert price_bundle(FIXTURES / "note" / "v2").items

    def test_accepts_a_loaded_bundle(self):
        bundle = load_bundle(FIXTURES / "note" / "v2")
        assert price_bundle(bundle).items

    def test_both_produce_the_same_result(self):
        from_path = price_bundle(FIXTURES / "note" / "v2").to_dict()
        from_bundle = price_bundle(load_bundle(FIXTURES / "note" / "v2")).to_dict()
        assert from_path == from_bundle

    def test_an_unverifiable_bundle_raises_rather_than_refusing(self, tmp_path):
        """An unverifiable bundle raises: a bad input is not a refusable item."""
        import shutil
        root = tmp_path / "broken"
        shutil.copytree(FIXTURES / "note" / "v2", root)
        (root / "positions.csv").write_bytes(b"# tampered\n")

        with pytest.raises(BundleIntegrityError):
            price_bundle(root)


class TestMarketProvenance:
    def test_w0_claims_no_market_provenance(self):
        """No curve was used, so no market provenance is claimed."""
        assert price_bundle(FIXTURES / "note" / "v2").market_provenance is None


#: The engine modules the TraderX path may import besides its own: the accrual day counts, a
#: leaf importing only ORE.
TRADERX_MAY_IMPORT = ("engine.market_data.day_counts",)
TRADERX = Path(__file__).parents[1] / "engine" / "traderx"


class TestPackageImportsNoSimulationPricer:
    """`engine/traderx/` imports no simulation or model layer, so a booking with
    unsupported conventions cannot reach `build_vanilla_swap` (I-05). ORE itself is allowed
    (the bill and note need dates and day counts); of the engine, only `TRADERX_MAY_IMPORT`,
    an allowlist, so that a renamed package cannot slip past it."""

    def test_no_simulation_or_model_pricer_import(self):
        import ast

        sources = sorted(TRADERX.glob("*.py"))
        assert len(sources) > 10, f"no TraderX path at {TRADERX}"
        violations = []
        for source in sources:
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                else:
                    continue
                for module in modules:
                    if module.split(".")[0] == "jax" or (
                            module.startswith("engine.") and not module.startswith("engine.traderx")
                            and module not in TRADERX_MAY_IMPORT):
                        violations.append(f"{source.name} imports {module}")

        assert violations == [], (
            "engine/traderx/ must not import the simulation/model pricing "
            "layer -- an unsupported convention could then reach a pricing "
            "object despite W0.4's refusal (see I-05): " + "; ".join(violations)
        )

    def test_bill_pricer_does_not_reach_the_swap_builder(self):
        """The bill pricer does not construct a vanilla swap."""
        source = (TRADERX / "bill.py").read_text(encoding="utf-8")
        assert "build_vanilla_swap" not in source
        assert "schedules" not in source

    def test_note_pricer_does_not_reach_the_swap_builder(self):
        """Nor does the note pricer (its ACT/ACT day count moved to the leaf module
        `engine.market_data.day_counts` for this reason)."""
        source = (TRADERX / "note.py").read_text(encoding="utf-8")
        assert "build_vanilla_swap" not in source
        assert "schedules" not in source

    def test_importing_the_package_does_not_pull_in_the_model_layer(self):
        """The transitive closure: after importing the package in a clean interpreter, no
        model or simulation module is loaded. (The AST test above sees only direct imports;
        a leaf such as `engine.market_data.day_counts` importing `engine.models` would pass it.)"""
        import subprocess
        import sys

        allowed = ("engine", "engine.traderx", "engine.market_data", *TRADERX_MAY_IMPORT)
        probe = (
            "import sys; import engine.traderx; "
            "banned = [m for m in sys.modules "
            f"if (m.startswith('engine') and m not in {allowed!r} and not m.startswith('engine.traderx.')) "
            "or m.split('.')[0] == 'jax']; "
            "print(','.join(sorted(banned)))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parents[1]),
        )
        assert completed.returncode == 0, completed.stderr
        loaded = [m for m in completed.stdout.strip().split(",") if m]
        assert loaded == [], (
            "importing engine.traderx transitively loaded the "
            "simulation/model layer: " + ", ".join(loaded)
        )


class TestCapabilities:
    """The capability document: what is priceable, known before submitting."""

    def test_reports_the_delivery_stage_honestly(self):
        """The document says exactly what is priced: past refusal-only, per shape (a note
        has `rateSensitivity`, a bill does not)."""
        doc = capabilities()
        # A floor, not an exact stage string (exact pins broke on every increment).
        assert doc["deliveryStage"].startswith("W1.")
        assert float(doc["deliveryStage"][1:]) >= 1.3
        assert doc["calculations"]["mode"] == "partial"

        treasury = doc["products"]["TREASURY"]
        assert treasury["priced"] is True
        assert set(treasury["calculations"]) == {"npv", "rateSensitivity"}

        # A bill still has no sensitivity.
        assert treasury["byShape"]["zero-coupon"] == ["npv"]

    def test_does_not_advertise_unearned_calculations(self):
        """A priced NPV is not advertised as a priced risk number: `rateGamma`, `theta` and
        `vega` are unearned everywhere, and the bill's sensitivity too."""
        doc = capabilities()
        for product, entry in doc["products"].items():
            for banned in ("rateGamma", "theta", "vega"):
                assert banned not in entry["calculations"], (
                    f"{product} advertises {banned}, which no pricer computes"
                )
            for shape, calcs in entry.get("byShape", {}).items():
                for banned in ("rateGamma", "theta", "vega"):
                    assert banned not in calcs, (
                        f"{product}/{shape} advertises {banned}, which no "
                        f"pricer computes"
                    )

    def test_derived_from_the_allowlist_not_hand_maintained(self):
        """Derived from the allowlist, so adding a convention updates it."""
        from engine.traderx.conventions import (
            SUPPORTED_FLOAT_INDICES, SUPPORTED_SWAP_DAY_COUNTS,
        )
        doc = capabilities()
        assert doc["conventions"]["swap"]["floatIndex"] == list(SUPPORTED_FLOAT_INDICES)
        assert doc["conventions"]["swap"]["legDayCount"] == list(SUPPORTED_SWAP_DAY_COUNTS)

    def test_calculations_match_the_frozen_set(self):
        assert capabilities()["calculations"]["names"] == list(CALCULATIONS)

    def test_bundle_schemas_match_the_loader(self):
        from engine.traderx.bundle import SUPPORTED_BUNDLE_SCHEMAS
        assert capabilities()["bundleSchemas"] == list(SUPPORTED_BUNDLE_SCHEMAS)

    def test_advertises_no_market_input_fallback(self):
        """No market-input fallback is advertised."""
        assert capabilities()["marketInputs"]["fallbackOnMissingInputs"] is False

    def test_known_limitations_are_advertised(self):
        """Known limitations (e.g. I-04) are advertised."""
        ids = {limitation["id"] for limitation in capabilities()["knownLimitations"]}
        assert {"I-04", "I-05"} <= ids

    def test_sofr_limitation_says_it_is_refused(self):
        (i05,) = [l for l in capabilities()["knownLimitations"] if l["id"] == "I-05"]
        assert "REFUSED" in i05["summary"]
        assert "D03/D04" in i05["blockedOn"]

    def test_capabilities_is_json_serializable(self):
        doc = capabilities()
        assert json.loads(json.dumps(doc)) == doc

    def test_capabilities_predicts_the_sofr_refusal(self):
        """The matrix agrees with behaviour: USD-SOFR is not advertised and is refused."""
        advertised = capabilities()["conventions"]["swap"]["floatIndex"]
        assert not any("SOFR" in index for index in advertised)

        result = price_bundle(FIXTURES / "sofr" / "v2")
        assert result.items[0].calculations["npv"].status == "unsupported"

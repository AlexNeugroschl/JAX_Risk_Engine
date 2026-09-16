"""
The W0 exit criterion, end to end, plus W0.9 capabilities
(`docs/planning/traderx-integration-plan.md` §2 and §W0.9).

**The exit criterion, quoted:** "the SOFR case returns
`CONVENTION_NOT_SUPPORTED` naming all 13 missing terms, and bill/note return
structurally valid results with `npv: unsupported`."

`TestExitCriterion` asserts exactly that, and nothing else in this file is
allowed to make it pass vacuously.
"""
import json
from pathlib import Path

import pytest

from engine.integration import capabilities, price_bundle
from engine.integration.bundle import BundleIntegrityError, load_bundle
from engine.integration.result import CALCULATIONS

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"


class TestExitCriterion:
    """Plan §2: the W0 exit criterion, and §6's '★ first real result'."""

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
        """A refusal nobody can attribute to a booking is useless."""
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
        """'Structurally valid' means every item carries every calculation,
        identity, and a coverage block that sums."""
        result = price_bundle(FIXTURES / case / "v2")
        payload = result.to_dict()

        assert result.coverage.all_outcomes_accounted_for
        for item in payload["items"]:
            assert set(item["calculations"]) == set(CALCULATIONS)
            assert item["itemId"]
            assert item["sourceIdentity"]
        assert payload["itemOrder"]["itemCount"] == len(payload["items"])

    def test_the_whole_result_is_json_serializable(self):
        """It has to survive the wire, not just the type checker."""
        for case in ("bill", "note", "sofr"):
            payload = price_bundle(FIXTURES / case / "v2").to_dict()
            assert json.loads(json.dumps(payload)) == payload


class TestNothingIsPricedAtW0:
    """W0 is 'contract and refusal machinery (no pricing)'. A result with
    an `ok` NPV would mean a pricer crept in ahead of W1."""

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
    """The one calculation W0 can answer honestly: a unit conversion of an
    exported value, not a model output."""

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
        """Plan §W0.3 row 4, end to end: without terms the blank is
        uninterpretable. `unavailable` tells the coordinator a resend with
        terms fixes it; `unsupported` would wrongly imply engine work."""
        result = price_bundle(FIXTURES / "bill" / "v1")
        for item in result.items:
            outcome = item.calculations["accruedInterest"]
            assert outcome.status == "unavailable"
            assert outcome.reason == "NO_TERMS_ARTIFACT"
            assert outcome.value is None

    def test_accrued_survives_a_convention_refusal(self):
        """A convention refusal has no bearing on a unit conversion, so the
        more specific W0.3 verdict is not flattened by it."""
        result = price_bundle(FIXTURES / "bill" / "v1")
        item = result.items[0]

        assert item.calculations["npv"].reason == "TERMS_NOT_SUPPLIED"
        assert item.calculations["accruedInterest"].status == "unavailable"


class TestNotApplicableIsUsedPrecisely:
    def test_vega_on_a_treasury_is_not_applicable(self):
        """Not a gap: a Treasury has no optionality, so there is no vega to
        miss. Counting it would make a complete result look incomplete."""
        result = price_bundle(FIXTURES / "note" / "v2")
        for item in result.items:
            assert item.calculations["vega"].status == "not-applicable"

    def test_vega_on_a_swap_is_not_applicable(self):
        result = price_bundle(FIXTURES / "sofr" / "v2")
        assert result.items[0].calculations["vega"].status == "not-applicable"

    def test_var_es_is_not_applicable_per_item(self):
        """A portfolio-level statistic, reported per item rather than
        omitted, so coverage still sums to the item count."""
        result = price_bundle(FIXTURES / "note" / "v2")
        for item in result.items:
            assert item.calculations["varEs"].status == "not-applicable"

    def test_a_swaption_vega_is_a_real_gap_not_not_applicable(self):
        """The distinction `not-applicable` has to earn: a swaption DOES
        have vega, so its absence is a genuine gap that must count against
        coverage. Marking it `not-applicable` -- which a naive 'contracts
        rows are swaps' fallback would do -- would hide a missing number.
        """
        from engine.integration.pipeline import _build_item
        from engine.integration.terms import JoinedRow

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
        """A v2 bundle carries no *terms* warning.

        It does still warn about market inputs -- every delivered fixture
        declares `marketInputs.status: NOT_SUPPLIED`, and W0.6 surfaces
        that rather than passing over it. So this asserts the absence of
        the specific warning, not the absence of all warnings.
        """
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
        """A bad *input* is not a refusable item -- it is a failure of the
        submission. Returning a refusal would imply the bundle was read."""
        import shutil
        root = tmp_path / "broken"
        shutil.copytree(FIXTURES / "note" / "v2", root)
        (root / "positions.csv").write_bytes(b"# tampered\n")

        with pytest.raises(BundleIntegrityError):
            price_bundle(root)


class TestMarketProvenance:
    def test_w0_claims_no_market_provenance(self):
        """W0 uses no curve at all. Asserting 'assumed' for a computation
        that never happened would be its own small lie."""
        assert price_bundle(FIXTURES / "note" / "v2").market_provenance is None


class TestPackageImportsNoSimulationPricer:
    """The package's central architectural claim, narrowed by W1.2 rather
    than abandoned.

    **What changed.** At W0 this asserted that `engine/integration/`
    imported no pricer, no ORE and no JAX at all -- true because W0 priced
    nothing. W1.2 adds `bill.py`, which genuinely needs `ORE` for its date
    and day-count arithmetic, so a blanket ban is no longer the right
    statement.

    **What has NOT changed, and is what this test actually protects.** The
    bill pricer is a closed-form discounted cashflow: dates, a day count,
    one `exp()`. It does not touch the Monte Carlo simulation, the JAX
    pricing kernels, or `build_vanilla_swap` -- the last of which is the
    specific thing W0.4's refusal path exists to keep away from a booking
    whose conventions are unsupported (I-05). Importing *those* here would
    mean an unsupported convention could reach a pricing object after all,
    which is the ordering guarantee this test exists to enforce.

    So the ban is now on the simulation/model layer, and `ORE` alone is
    permitted.
    """

    def test_no_simulation_or_model_pricer_import(self):
        import ast
        from pathlib import Path

        # `ORE` is deliberately absent: bill.py needs ORE.Date/day counts.
        # `engine.models` stays banned -- it is where build_vanilla_swap
        # lives, the exact object W0.4 refuses before constructing.
        banned = (
            "jax",
            "engine.instruments", "engine.models", "engine.risk",
            "engine.simulation", "engine.portfolio",
        )
        package = Path(__file__).parents[1] / "engine" / "integration"

        violations = []
        for source in sorted(package.glob("*.py")):
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module]
                else:
                    continue
                for module in modules:
                    if any(module == b or module.startswith(b + ".") for b in banned):
                        violations.append(f"{source.name} imports {module}")

        assert violations == [], (
            "engine/integration/ must not import the simulation/model pricing "
            "layer -- an unsupported convention could then reach a pricing "
            "object despite W0.4's refusal (see I-05): " + "; ".join(violations)
        )

    def test_bill_pricer_does_not_reach_the_swap_builder(self):
        """The specific thing the ban above is about, stated directly: the
        bill pricer must not construct a generic vanilla swap, whatever
        else it imports."""
        from pathlib import Path

        source = (Path(__file__).parents[1] / "engine" / "integration" / "bill.py").read_text(encoding="utf-8")
        assert "build_vanilla_swap" not in source
        assert "ore_builders" not in source

    def test_note_pricer_does_not_reach_the_swap_builder(self):
        """The same, for W1.3's note pricer.

        The note needs the ACT/ACT (ICMA) day count, which lived in
        `engine.models.ore_builders` until W1.3 moved it to the leaf
        module `engine.day_count`. That move happened *because* of this
        ban rather than around it -- see `engine/day_count.py`.
        """
        from pathlib import Path

        source = (Path(__file__).parents[1] / "engine" / "integration" / "note.py").read_text(encoding="utf-8")
        assert "build_vanilla_swap" not in source
        assert "ore_builders" not in source

    def test_importing_the_package_does_not_pull_in_the_model_layer(self):
        """**Transitive closure, not just direct imports.**

        The AST test above reads each file's own import statements, so it
        catches `integration/x.py` importing `engine.models` directly --
        but not `integration/x.py` importing a leaf that imports it. That
        gap is real: W1.3's day-count extraction created exactly such a
        leaf (`engine.day_count`), and if it ever grew an import back into
        `engine.models`, the AST test would stay green while the pricing
        layer became reachable from the refusal boundary again.

        This asserts the property that actually matters -- after importing
        the integration package in a clean interpreter, the model and
        simulation modules are **not loaded**.
        """
        import subprocess
        import sys

        probe = (
            "import sys; import engine.integration; "
            "banned = [m for m in sys.modules "
            "if m.startswith(('engine.models', 'engine.instruments', "
            "'engine.simulation', 'engine.portfolio', 'engine.risk', 'jax'))]; "
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
            "importing engine.integration transitively loaded the "
            "simulation/model layer: " + ", ".join(loaded)
        )


class TestCapabilities:
    """W0.9 -- the supported matrix, so a coordinator can tell *before
    submitting* whether a bundle is priceable."""

    def test_reports_the_delivery_stage_honestly(self):
        """The document must say exactly what is priced -- neither still
        claiming `refusal-only` (false since W1.2) nor implying the whole
        product is priced (still false at W1.3).

        **Updated from the W1.2 form of this test.** It asserted
        `deliveryStage == "W1.2"` and `TREASURY == ["npv"]`, both of which
        W1.3 deliberately changes by adding the note's `rateSensitivity`.
        The narrowness it was protecting is preserved below, one level
        finer: per *shape* rather than per type.
        """
        doc = capabilities()
        # The floor, not the exact string: a literal stage pin has broken
        # on every W1 increment. What matters is that the document has
        # advanced past refusal-only, which the contents below verify
        # specifically.
        assert doc["deliveryStage"].startswith("W1.")
        assert float(doc["deliveryStage"][1:]) >= 1.3
        assert doc["calculations"]["mode"] == "partial"

        treasury = doc["products"]["TREASURY"]
        assert treasury["priced"] is True
        assert set(treasury["calculations"]) == {"npv", "rateSensitivity"}

        # The claim stays narrow where it must: a BILL still has no
        # sensitivity. W1.3 earned the note's with a parity test and
        # earned nothing for the bill.
        assert treasury["byShape"]["zero-coupon"] == ["npv"]

    def test_does_not_advertise_unearned_calculations(self):
        """The specific overclaim to avoid: a priced NPV must not be read
        as a priced risk number.

        `rateSensitivity` left this list for the *note* in W1.3, backed by
        an ORE parity test. `rateGamma`, `theta` and `vega` remain
        unearned for everything, and the bill's sensitivity remains
        unearned too -- asserted per shape.
        """
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
        """A stale capability document makes a promise the engine no longer
        keeps. Pinning the derivation means adding a convention updates this
        automatically."""
        from engine.integration.conventions import (
            SUPPORTED_FLOAT_INDICES, SUPPORTED_SWAP_DAY_COUNTS,
        )
        doc = capabilities()
        assert doc["conventions"]["swap"]["floatIndex"] == list(SUPPORTED_FLOAT_INDICES)
        assert doc["conventions"]["swap"]["legDayCount"] == list(SUPPORTED_SWAP_DAY_COUNTS)

    def test_calculations_match_the_frozen_set(self):
        assert capabilities()["calculations"]["names"] == list(CALCULATIONS)

    def test_bundle_schemas_match_the_loader(self):
        from engine.integration.bundle import SUPPORTED_BUNDLE_SCHEMAS
        assert capabilities()["bundleSchemas"] == list(SUPPORTED_BUNDLE_SCHEMAS)

    def test_advertises_no_market_input_fallback(self):
        """Plan §W0.6: missing market data fails the job, never falls back.
        Advertised so a coordinator knows before submitting."""
        assert capabilities()["marketInputs"]["fallbackOnMissingInputs"] is False

    def test_known_limitations_are_advertised(self):
        """The engine's real defects are part of its capability surface: a
        consumer weighing an exposure profile needs I-04 before submitting,
        not after reconciling."""
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
        """The whole point of W0.9: the matrix must agree with what the
        engine actually does. USD-SOFR is absent from the advertised
        indices, and submitting it is refused."""
        advertised = capabilities()["conventions"]["swap"]["floatIndex"]
        assert not any("SOFR" in index for index in advertised)

        result = price_bundle(FIXTURES / "sofr" / "v2")
        assert result.items[0].calculations["npv"].status == "unsupported"

"""
W1.6.2 -- published document schema versions and their JSON Schema
(`docs/planning/traderx-integration-plan.md` §W1.6).

**The load-bearing tests here are the strictness ones.** A schema that
accepts everything validates every document and proves nothing, so
`TestResultSchemaIsStrict` mutates a real priced result in ways a consumer
would need to catch and asserts each is rejected. Without those, the
"validates cleanly" tests below would pass against a schema of `{}`.

The other half is derivation: the schema is generated from
`engine.integration.result`'s frozen vocabulary rather than hand-written, so
a calculation added there appears in the published schema automatically. A
hand-maintained schema drifts, and a stale schema certifies documents that
no longer match it.
"""
import copy
from pathlib import Path

import pytest

jsonschema = pytest.importorskip(
    "jsonschema",
    reason="jsonschema is a dev-extra; it validates published documents in tests only",
)
from jsonschema import Draft202012Validator

from engine.integration import price_bundle
from engine.integration.capabilities import capabilities
from engine.integration.result import CALCULATIONS, STATUSES
from engine.integration.schema import (
    CAPABILITY_SCHEMA_VERSION,
    JSON_SCHEMA_DIALECT,
    RESULT_SCHEMA_VERSION,
    capability_schema,
    result_schema,
)
from engine.integration.schema_version import (
    CAPABILITY_SCHEMA_VERSION as LEAF_CAPABILITY_VERSION,
    RESULT_SCHEMA_VERSION as LEAF_RESULT_VERSION,
)

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"
MARKET = {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}


@pytest.fixture(scope="module")
def priced_note():
    return price_bundle(FIXTURES / "note" / "v2", MARKET).to_dict()


@pytest.fixture(scope="module")
def priced_bill():
    return price_bundle(FIXTURES / "bill" / "v2", MARKET).to_dict()


@pytest.fixture(scope="module")
def refused_sofr():
    return price_bundle(FIXTURES / "sofr" / "v2", MARKET).to_dict()


@pytest.fixture(scope="module")
def validator():
    return Draft202012Validator(result_schema())


class TestSchemasAreThemselvesValid:
    """A malformed schema silently validates nothing in some validators."""

    def test_result_schema_is_valid_draft_2020_12(self):
        Draft202012Validator.check_schema(result_schema())

    def test_capability_schema_is_valid_draft_2020_12(self):
        Draft202012Validator.check_schema(capability_schema())

    def test_dialect_is_declared_explicitly(self):
        """`additionalProperties` and `$defs` semantics differ across
        drafts; a validator guessing the dialect can apply different rules
        than the author intended."""
        assert result_schema()["$schema"] == JSON_SCHEMA_DIALECT
        assert capability_schema()["$schema"] == JSON_SCHEMA_DIALECT

    def test_each_call_returns_an_independent_copy(self):
        """A caller mutating the returned dict must not corrupt the next
        caller's schema."""
        first = result_schema()
        first["properties"].pop("bundleId")
        assert "bundleId" in result_schema()["properties"]


class TestRealDocumentsValidate:
    """Every document this engine actually publishes must conform."""

    def test_priced_note_validates(self, validator, priced_note):
        assert list(validator.iter_errors(priced_note)) == []

    def test_priced_bill_validates(self, validator, priced_bill):
        assert list(validator.iter_errors(priced_bill)) == []

    def test_refused_sofr_validates(self, validator, refused_sofr):
        """A refusal is a published result too -- and the one most likely to
        carry an unusual shape, since every calculation is non-ok."""
        assert list(validator.iter_errors(refused_sofr)) == []

    def test_v1_bundle_result_validates(self, validator):
        """A v1 bundle prices nothing; the document still conforms."""
        result = price_bundle(FIXTURES / "note" / "v1", MARKET).to_dict()
        assert list(validator.iter_errors(result)) == []

    def test_result_with_no_market_inputs_validates(self, validator):
        """`marketProvenance: null` is a legal state -- it means no curve
        was consulted, which is not the same as 'observed'."""
        result = price_bundle(FIXTURES / "note" / "v2").to_dict()
        assert result["marketProvenance"] is None
        assert list(validator.iter_errors(result)) == []

    def test_equity_refusal_validates(self, validator):
        result = price_bundle(FIXTURES / "equity" / "v2", MARKET).to_dict()
        assert list(validator.iter_errors(result)) == []

    def test_capabilities_validates(self):
        errors = list(Draft202012Validator(capability_schema()).iter_errors(capabilities()))
        assert errors == []


class TestResultSchemaIsStrict:
    """**The tests that give the ones above their meaning.**

    Each mutation is something a consumer would need to catch. A schema that
    missed any of these would still pass every `TestRealDocumentsValidate`
    case, which is why strictness is asserted separately and explicitly.
    """

    def _expect_rejected(self, validator, doc, mutate):
        bad = copy.deepcopy(doc)
        mutate(bad)
        errors = list(validator.iter_errors(bad))
        assert errors, "schema accepted a document it should have rejected"

    def test_missing_result_schema_version_is_rejected(self, validator, priced_note):
        self._expect_rejected(validator, priced_note, lambda d: d.pop("resultSchema"))

    def test_wrong_result_schema_version_is_rejected(self, validator, priced_note):
        """A document declaring a version the consumer has not validated
        against must not pass as one that has."""
        self._expect_rejected(
            validator, priced_note,
            lambda d: d.__setitem__("resultSchema", "jaxrisk.eod-result.v2"),
        )

    def test_unknown_top_level_field_is_rejected(self, validator, priced_note):
        """An unexpected field is a version mismatch, and saying so is the
        whole purpose of publishing a schema."""
        self._expect_rejected(
            validator, priced_note, lambda d: d.__setitem__("surpriseTotal", 1.0),
        )

    @pytest.mark.parametrize("calculation", CALCULATIONS)
    def test_omitting_any_calculation_is_rejected(self, validator, priced_note, calculation):
        """An omitted calculation is indistinguishable from a forgotten
        one. All seven are required."""
        self._expect_rejected(
            validator, priced_note,
            lambda d: d["items"][0]["calculations"].pop(calculation),
        )

    def test_unknown_calculation_name_is_rejected(self, validator, priced_note):
        self._expect_rejected(
            validator, priced_note,
            lambda d: d["items"][0]["calculations"].__setitem__(
                "sharpeRatio", {"status": "ok", "value": 1.0},
            ),
        )

    def test_invalid_status_is_rejected(self, validator, priced_note):
        self._expect_rejected(
            validator, priced_note,
            lambda d: d["items"][0]["calculations"]["npv"].__setitem__("status", "maybe"),
        )

    @pytest.mark.parametrize("key", ["ok", "unsupported", "unavailable", "failed", "notApplicable"])
    def test_coverage_missing_any_status_key_is_rejected(self, validator, priced_note, key):
        """A consumer reading `counts["failed"]` must never have to
        distinguish a missing key from a zero."""
        self._expect_rejected(
            validator, priced_note,
            lambda d: d["coverage"]["byCalculation"]["npv"].pop(key),
        )

    def test_negative_coverage_count_is_rejected(self, validator, priced_note):
        self._expect_rejected(
            validator, priced_note,
            lambda d: d["coverage"]["byCalculation"]["npv"].__setitem__("ok", -1),
        )

    def test_empty_item_id_is_rejected(self, validator, priced_note):
        self._expect_rejected(
            validator, priced_note, lambda d: d["items"][0].__setitem__("itemId", ""),
        )

    def test_missing_source_identity_is_rejected(self, validator, priced_note):
        """An unidentified row is useless even when it carries a number."""
        self._expect_rejected(
            validator, priced_note, lambda d: d["items"][0].pop("sourceIdentity"),
        )

    def test_missing_coverage_block_is_rejected(self, validator, priced_note):
        self._expect_rejected(validator, priced_note, lambda d: d.pop("coverage"))

    def test_missing_item_order_is_rejected(self, validator, priced_note):
        """Identity never rides on array position -- the ordering artifact
        is required, not advisory."""
        self._expect_rejected(validator, priced_note, lambda d: d.pop("itemOrder"))

    def test_item_count_of_wrong_type_is_rejected(self, validator, priced_note):
        self._expect_rejected(
            validator, priced_note,
            lambda d: d["coverage"].__setitem__("itemCount", "two"),
        )


class TestCapabilitySchemaIsStrict:
    def test_missing_capability_version_is_rejected(self):
        doc = capabilities()
        doc.pop("capabilitySchema")
        assert list(Draft202012Validator(capability_schema()).iter_errors(doc))

    def test_fallback_flag_cannot_be_true(self):
        """`fallbackOnMissingInputs` is pinned to `false` by `const`. A
        document claiming the engine falls back would contradict the single
        rule this whole boundary enforces."""
        doc = capabilities()
        doc["marketInputs"]["fallbackOnMissingInputs"] = True
        assert list(Draft202012Validator(capability_schema()).iter_errors(doc))


class TestSchemaIsDerivedNotHandWritten:
    """Pins the derivation, not the contents -- so adding a calculation
    updates the published schema automatically instead of silently leaving
    it stale."""

    def test_every_frozen_calculation_appears_in_the_schema(self):
        props = result_schema()["properties"]["items"]["items"]["properties"]
        assert set(props["calculations"]["properties"]) == set(CALCULATIONS)
        assert props["calculations"]["required"] == list(CALCULATIONS)

    def test_every_status_appears_in_the_outcome_enum(self):
        props = result_schema()["properties"]["items"]["items"]["properties"]
        enum = props["calculations"]["properties"]["npv"]["properties"]["status"]["enum"]
        assert set(enum) == set(STATUSES)

    def test_coverage_block_covers_every_calculation(self):
        coverage = result_schema()["properties"]["coverage"]["properties"]["byCalculation"]
        assert set(coverage["properties"]) == set(CALCULATIONS)


class TestVersionsAreEmittedAndSeparate:
    def test_result_document_carries_its_schema_version(self, priced_note):
        assert priced_note["resultSchema"] == RESULT_SCHEMA_VERSION

    def test_capability_document_carries_its_schema_version(self):
        assert capabilities()["capabilitySchema"] == CAPABILITY_SCHEMA_VERSION

    def test_the_two_versions_are_distinct(self):
        """They change for different reasons: a new pricer changes the
        capability document without touching the result's shape. Sharing
        one version would force a lockstep neither side wants."""
        assert RESULT_SCHEMA_VERSION != CAPABILITY_SCHEMA_VERSION

    def test_leaf_module_is_the_single_source_of_the_versions(self):
        """`schema.py` re-exports from `schema_version.py` rather than
        redefining -- otherwise the published version could drift from the
        schema that describes it."""
        assert RESULT_SCHEMA_VERSION is LEAF_RESULT_VERSION
        assert CAPABILITY_SCHEMA_VERSION is LEAF_CAPABILITY_VERSION

    def test_capabilities_advertises_both_schema_versions(self):
        """A coordinator pins its validator from this block, before
        submitting."""
        schemas = capabilities()["schemas"]
        assert schemas["resultSchema"] == RESULT_SCHEMA_VERSION
        assert schemas["capabilitySchema"] == CAPABILITY_SCHEMA_VERSION


class TestNoImportCycle:
    """`result` needs the version constant; `schema` derives itself from
    `result`. The leaf module exists to break that -- if it stops working,
    importing either one first must still succeed."""

    def test_importing_result_first_works(self):
        import subprocess, sys
        code = "import engine.integration.result, engine.integration.schema; print('ok')"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr

    def test_importing_schema_first_works(self):
        import subprocess, sys
        code = "import engine.integration.schema, engine.integration.result; print('ok')"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr

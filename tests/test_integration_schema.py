"""
Published schema versions and their JSON Schema (`engine.integration.schema`, W1.6.2).

`TestResultSchemaIsStrict` mutates a real result and asserts each mutation is rejected;
without it the "validates" tests would pass against a schema of `{}`. The schema is derived
from `engine.integration.result`'s vocabulary rather than hand-written, so it cannot drift.
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
    """The schema itself is well formed."""

    def test_result_schema_is_valid_draft_2020_12(self):
        Draft202012Validator.check_schema(result_schema())

    def test_capability_schema_is_valid_draft_2020_12(self):
        Draft202012Validator.check_schema(capability_schema())

    def test_dialect_is_declared_explicitly(self):
        """The draft is declared (`additionalProperties` and `$defs` differ across drafts)."""
        assert result_schema()["$schema"] == JSON_SCHEMA_DIALECT
        assert capability_schema()["$schema"] == JSON_SCHEMA_DIALECT

    def test_each_call_returns_an_independent_copy(self):
        """Mutating the returned dict does not affect the next caller."""
        first = result_schema()
        first["properties"].pop("bundleId")
        assert "bundleId" in result_schema()["properties"]


class TestRealDocumentsValidate:
    """Every document the engine publishes conforms."""

    def test_priced_note_validates(self, validator, priced_note):
        assert list(validator.iter_errors(priced_note)) == []

    def test_priced_bill_validates(self, validator, priced_bill):
        assert list(validator.iter_errors(priced_bill)) == []

    def test_refused_sofr_validates(self, validator, refused_sofr):
        """A refusal conforms (every calculation non-ok)."""
        assert list(validator.iter_errors(refused_sofr)) == []

    def test_v1_bundle_result_validates(self, validator):
        """A v1 bundle prices nothing; the document still conforms."""
        result = price_bundle(FIXTURES / "note" / "v1", MARKET).to_dict()
        assert list(validator.iter_errors(result)) == []

    def test_result_with_no_market_inputs_validates(self, validator):
        """`marketProvenance: null` (no curve consulted) is legal."""
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
    """Each mutation is something a consumer must catch, and would pass every
    `TestRealDocumentsValidate` case against a lax schema."""

    def _expect_rejected(self, validator, doc, mutate):
        bad = copy.deepcopy(doc)
        mutate(bad)
        errors = list(validator.iter_errors(bad))
        assert errors, "schema accepted a document it should have rejected"

    def test_missing_result_schema_version_is_rejected(self, validator, priced_note):
        self._expect_rejected(validator, priced_note, lambda d: d.pop("resultSchema"))

    def test_wrong_result_schema_version_is_rejected(self, validator, priced_note):
        """An unvalidated version is rejected."""
        self._expect_rejected(
            validator, priced_note,
            lambda d: d.__setitem__("resultSchema", "jaxrisk.eod-result.v2"),
        )

    def test_unknown_top_level_field_is_rejected(self, validator, priced_note):
        """An unexpected field is rejected."""
        self._expect_rejected(
            validator, priced_note, lambda d: d.__setitem__("surpriseTotal", 1.0),
        )

    @pytest.mark.parametrize("calculation", CALCULATIONS)
    def test_omitting_any_calculation_is_rejected(self, validator, priced_note, calculation):
        """All seven calculations are required."""
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
        """Every coverage key is required (a missing key is not a zero)."""
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
        """An unidentified row is rejected even with a number."""
        self._expect_rejected(
            validator, priced_note, lambda d: d["items"][0].pop("sourceIdentity"),
        )

    def test_missing_coverage_block_is_rejected(self, validator, priced_note):
        self._expect_rejected(validator, priced_note, lambda d: d.pop("coverage"))

    def test_missing_item_order_is_rejected(self, validator, priced_note):
        """The ordering artifact is required."""
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
        """`fallbackOnMissingInputs` is pinned to `false` by `const`."""
        doc = capabilities()
        doc["marketInputs"]["fallbackOnMissingInputs"] = True
        assert list(Draft202012Validator(capability_schema()).iter_errors(doc))


class TestSchemaIsDerivedNotHandWritten:
    """Pins the derivation (not the contents), so adding a calculation updates the schema."""

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
        """They change for different reasons (a new pricer changes capabilities, not the
        result shape), so each has its own version."""
        assert RESULT_SCHEMA_VERSION != CAPABILITY_SCHEMA_VERSION

    def test_leaf_module_is_the_single_source_of_the_versions(self):
        """`schema.py` re-exports the versions from `schema_version.py` rather than
        redefining them."""
        assert RESULT_SCHEMA_VERSION is LEAF_RESULT_VERSION
        assert CAPABILITY_SCHEMA_VERSION is LEAF_CAPABILITY_VERSION

    def test_capabilities_advertises_both_schema_versions(self):
        """Advertised so a coordinator can pin its validator before submitting."""
        schemas = capabilities()["schemas"]
        assert schemas["resultSchema"] == RESULT_SCHEMA_VERSION
        assert schemas["capabilitySchema"] == CAPABILITY_SCHEMA_VERSION


class TestNoImportCycle:
    """`result` needs the version constant and `schema` derives from `result`; the leaf
    module breaks the cycle, so importing either first works."""

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

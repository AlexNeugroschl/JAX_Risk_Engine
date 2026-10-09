"""
JSON Schemas (draft 2020-12) of the two published EOD documents, with their versions.

Every document carries a version so a consumer's validator can pin exactly what it has
validated against and refuse the rest. The result and capability documents are versioned
separately, since they change for different reasons.

The schemas are derived from the code (`CALCULATIONS`, `STATUSES` in
`engine.traderx.result`), not hand-written, and the tests pin the derivation.

Closed objects use `additionalProperties: false`, so an unknown field shows up as a version
mismatch. Per-calculation payloads are the exception: they differ by pricer and are left
open.
"""
from typing import Dict

from engine.traderx.result import CALCULATIONS, STATUSES
# Re-exported from the dependency-free leaf (see schema_version).
from engine.traderx.schema_version import (
    CAPABILITY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
)

#: JSON Schema dialect, stated explicitly (draft semantics differ).
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

#: Canonical `$id`s. The reserved `.invalid` TLD (RFC 2606) never resolves, so a validator
#: cannot dereference the id over the network; the schemas are served at `/eod/schemas/*`.
RESULT_SCHEMA_ID = f"https://jax-risk-engine.invalid/schemas/{RESULT_SCHEMA_VERSION}.json"
CAPABILITY_SCHEMA_ID = f"https://jax-risk-engine.invalid/schemas/{CAPABILITY_SCHEMA_VERSION}.json"


def _calculation_outcome_schema() -> Dict:
    """One calculation's outcome: `status` from the frozen statuses; `value` untyped
    (present only when ok). Open to extra properties, because
    `CalculationOutcome.to_dict` merges per-calculation payloads in at this level."""
    return {
        "type": "object",
        "required": ["status"],
        "properties": {
            "status": {
                "enum": list(STATUSES),
                "description": (
                    "ok = computed; unsupported = engine cannot faithfully price "
                    "this; unavailable = an input was not supplied; failed = "
                    "attempted and errored; not-applicable = meaningless for this "
                    "instrument, and never counted as a coverage gap."
                ),
            },
            "value": {
                "description": (
                    "Present only when status is 'ok'. A non-ok outcome carries no "
                    "value by construction -- attaching one would invite consumers "
                    "to read it."
                ),
            },
            "reason": {"type": "string"},
            "detail": {"type": "string"},
        },
        # Per-calculation payloads are merged in here.
        "additionalProperties": True,
    }


def _coverage_counts_schema() -> Dict:
    """Per-calculation status counts; all five keys required, so a missing key is never
    confused with zero."""
    keys = ["ok", "unsupported", "unavailable", "failed", "notApplicable"]
    return {
        "type": "object",
        "required": keys,
        "properties": {k: {"type": "integer", "minimum": 0} for k in keys},
        "additionalProperties": False,
    }


def _source_identity_schema() -> Dict:
    return {
        "type": "object",
        "required": ["kind", "clusterEpoch"],
        "properties": {
            "kind": {"type": "string"},
            "accountId": {"type": ["string", "null"]},
            "security": {"type": ["string", "null"]},
            "contractId": {"type": ["string", "null"]},
            "clusterEpoch": {"type": "string"},
        },
        "additionalProperties": True,
    }


def _item_schema() -> Dict:
    """One item: identity and every calculation's outcome. All calculations are required
    (as `ItemResult.__post_init__` also enforces), so none can be silently omitted."""
    return {
        "type": "object",
        "required": ["itemId", "sourceIdentity", "calculations"],
        "properties": {
            "itemId": {"type": "string", "minLength": 1},
            "sourceIdentity": _source_identity_schema(),
            "calculations": {
                "type": "object",
                "required": list(CALCULATIONS),
                "properties": {
                    name: _calculation_outcome_schema() for name in CALCULATIONS
                },
                "additionalProperties": False,
            },
            "currency": {"type": ["string", "null"]},
            "mappingVersion": {"type": ["string", "null"]},
            "refusal": {"type": ["object", "null"]},
        },
        "additionalProperties": False,
    }


def result_schema() -> Dict:
    """JSON Schema of the result document. A fresh dict on every call."""
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": RESULT_SCHEMA_ID,
        "title": "JAX Risk Engine EOD result",
        "description": (
            "One published end-of-day risk result: identified items, each with "
            "an explicit outcome for every calculation, plus the coverage block "
            "and the provenance needed to reproduce or distrust it."
        ),
        "type": "object",
        "required": [
            "resultSchema", "bundleId", "clusterEpoch", "sessionDate",
            "valuationTime", "mappingVersion", "engineVersion", "items",
            "itemOrder", "coverage",
        ],
        "properties": {
            "resultSchema": {
                "const": RESULT_SCHEMA_VERSION,
                "description": (
                    "The schema version this document conforms to. Pin against it; "
                    "a document declaring a version you have not validated against "
                    "should be refused rather than parsed optimistically."
                ),
            },
            "bundleId": {"type": "string"},
            "clusterEpoch": {"type": "string"},
            "sessionDate": {"type": "string"},
            "valuationTime": {"type": "string"},
            "mappingVersion": {"type": "string"},
            "engineVersion": {"type": "string"},
            "marketProvenance": {
                "type": ["string", "null"],
                "description": (
                    "'assumed' whenever any curve behind these numbers was not "
                    "observed. null means no curve was consulted at all -- which is "
                    "not the same as 'observed', and is never reported as such."
                ),
            },
            "marketInputs": {"type": ["object", "null"]},
            "measure": {
                "type": ["string", "null"],
                "description": (
                    "Which measure the risk figures are under. A tail statistic "
                    "without its measure is unactionable."
                ),
            },
            "items": {"type": "array", "items": _item_schema()},
            "itemOrder": {
                "type": "object",
                "description": (
                    "Item ordering published as its own hashed artifact. Identity "
                    "never rides on array position."
                ),
            },
            "coverage": {
                "type": "object",
                "required": [
                    "byCalculation", "itemCount",
                    "allOutcomesAccountedFor", "allApplicableComputed",
                ],
                "properties": {
                    "byCalculation": {
                        "type": "object",
                        "required": list(CALCULATIONS),
                        "properties": {
                            name: _coverage_counts_schema() for name in CALCULATIONS
                        },
                        "additionalProperties": False,
                    },
                    "itemCount": {"type": "integer", "minimum": 0},
                    "allOutcomesAccountedFor": {
                        "type": "boolean",
                        "description": (
                            "Did every item get some verdict? True even if "
                            "everything failed -- an internal-consistency check on "
                            "this document, not a quality judgement."
                        ),
                    },
                    "allApplicableComputed": {
                        "type": "boolean",
                        "description": (
                            "True only when unsupported + unavailable + failed == 0. "
                            "not-applicable is excluded by construction."
                        ),
                    },
                },
                "additionalProperties": False,
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "additionalProperties": False,
    }


def capability_schema() -> Dict:
    """JSON Schema of the capability document. Looser than the result schema, since the
    document is advisory and grows with each pricer; the version fields, calculation
    vocabulary and the no-fallback flag are pinned."""
    return {
        "$schema": JSON_SCHEMA_DIALECT,
        "$id": CAPABILITY_SCHEMA_ID,
        "title": "JAX Risk Engine EOD capabilities",
        "description": (
            "The supported (product x convention x calculation) matrix, so a "
            "coordinator can determine BEFORE submitting whether a bundle is "
            "priceable."
        ),
        "type": "object",
        "required": [
            "capabilitySchema", "engineVersion", "mappingVersion",
            "deliveryStage", "calculations", "products", "marketInputs",
        ],
        "properties": {
            "capabilitySchema": {"const": CAPABILITY_SCHEMA_VERSION},
            "engineVersion": {"type": "string"},
            "mappingVersion": {"type": "string"},
            "deliveryStage": {"type": "string"},
            "stageSummary": {"type": "string"},
            "bundleSchemas": {"type": "array", "items": {"type": "string"}},
            "termsSchemas": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Terms artifact schemas accepted. Independent of bundleSchemas: "
                    "a v2 bundle may carry either terms version."
                ),
            },
            "calculations": {
                "type": "object",
                "required": ["names", "statuses", "mode"],
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"enum": list(CALCULATIONS)},
                    },
                    "statuses": {
                        "type": "array",
                        "items": {"enum": list(STATUSES)},
                    },
                    "mode": {"type": "string"},
                },
                "additionalProperties": True,
            },
            "products": {"type": "object"},
            "conventions": {"type": "object"},
            "marketInputs": {
                "type": "object",
                "required": ["modes", "fallbackOnMissingInputs"],
                "properties": {
                    "modes": {"type": "array", "items": {"type": "string"}},
                    "fallbackOnMissingInputs": {
                        "const": False,
                        "description": (
                            "Always false. Missing market inputs fail the job; no "
                            "curve is ever substituted."
                        ),
                    },
                },
                "additionalProperties": True,
            },
            "riskMeasure": {"type": "object"},
            "knownLimitations": {"type": "array", "items": {"type": "object"}},
            "schemas": {"type": "object"},
        },
        "additionalProperties": True,
    }

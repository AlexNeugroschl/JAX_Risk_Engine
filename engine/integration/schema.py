"""
W1.6.2 -- document schema versions, and the JSON Schema that defines them.

**Why a version on every published document.** TraderX's validator has to
decide whether a result it receives is one it understands. Without a version
field it can only guess from the shape, which means a change this engine
makes is discoverable by them only as a parse failure -- or, worse, as a
silently-ignored new field. A version lets them pin: accept exactly what
they have validated against, and refuse the rest loudly.

**The version is emitted before intake is extended, deliberately** (plan
§W1.6.2: "emit the version *before* extending intake, so their validator can
pin it"). A consumer cannot pin a version that was never published, so
shipping the field first -- even while its value stays at `.v1` -- is what
makes the *next* change safe rather than breaking.

**Two separately versioned documents.** The result document and the
capability document change for different reasons and at different times: a
new pricer changes what `capabilities()` advertises without changing the
result's shape at all. Sharing one version would force a lockstep neither
side wants, and would make "did the result schema change?" unanswerable.

**The schemas are derived from the code they describe, not hand-written.**
`result_schema()` reads `CALCULATIONS` and `STATUSES` out of
`engine.integration.result`, exactly as `capabilities()` reads the
allowlist. A hand-maintained schema drifts from the documents it validates,
and a stale schema is worse than none: it certifies documents that no longer
match it. The test suite pins the *derivation*, so adding a calculation
updates the published schema automatically.

**Draft 2020-12**, declared explicitly via `$schema`. Stating the dialect
matters for `additionalProperties` and `$defs` semantics, which differ
across drafts -- a validator guessing the dialect can silently apply
different rules than the author intended.

**`additionalProperties: false` is deliberate on the closed objects.** A
result document carrying a field this schema does not know about is a
version mismatch, and saying so is the entire purpose of publishing a
schema. The exception is per-calculation payloads, which are genuinely
open -- a sensitivity carries `method`/`bump`/`shockedFactor` while an NPV
carries a cashflow breakdown, and enumerating every pricer's payload here
would couple this module to all of them.
"""
from typing import Dict

from engine.integration.result import CALCULATIONS, STATUSES
# Re-exported from the dependency-free leaf, so `result` can stamp its own
# version without importing this module back. See `schema_version`'s
# docstring for why the cycle is broken there rather than here.
from engine.integration.schema_version import (
    CAPABILITY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
)

#: JSON Schema dialect these documents are written against. Stated rather
#: than assumed -- see the module docstring.
JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

#: Canonical `$id`s, so a consumer can cache each schema by a stable name
#: rather than by the URL it happened to fetch it from.
#:
#: **`.invalid` is deliberate, not a placeholder.** It is the RFC 2606
#: reserved TLD, guaranteed never to resolve. A `$id` is an *identifier*,
#: and some validators will fetch one that looks fetchable -- pointing it at
#: a reserved name makes an accidental network dereference impossible, and
#: makes it obvious that the id is a name rather than a location. The
#: schemas are served over HTTP at `/eod/schemas/*`; that is where a
#: consumer fetches them.
RESULT_SCHEMA_ID = f"https://jax-risk-engine.invalid/schemas/{RESULT_SCHEMA_VERSION}.json"
CAPABILITY_SCHEMA_ID = f"https://jax-risk-engine.invalid/schemas/{CAPABILITY_SCHEMA_VERSION}.json"


def _calculation_outcome_schema() -> Dict:
    """One calculation's outcome.

    `status` is pinned to the five frozen statuses, read from
    `engine.integration.result` rather than restated. `value` is
    deliberately untyped beyond "present or null": it is a float for `npv`
    and a structured object for nothing today, but constraining it here
    would make adding a structured calculation a schema-breaking change
    for no consumer benefit.

    **Open to extra properties**, because per-calculation payloads are
    merged in at the top level of this object by
    `CalculationOutcome.to_dict` -- a sensitivity's `method`/`bump`, a
    note's cashflow breakdown. See the module docstring.
    """
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
        # Per-calculation payloads are merged in here. See the docstring.
        "additionalProperties": True,
    }


def _coverage_counts_schema() -> Dict:
    """Per-calculation status counts.

    Every one of the five keys is `required`: a consumer reading
    `counts["failed"]` must never have to distinguish a missing key from a
    zero. That asymmetry is how a wrong dashboard gets built.
    """
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
    """One item: identity plus every calculation's outcome.

    **All seven calculations are `required`.** An omitted calculation is
    indistinguishable from a forgotten one, which is exactly what the
    coverage model exists to prevent -- so the schema enforces what
    `ItemResult.__post_init__` already enforces in code. Two independent
    checks of the same invariant is the point: the code protects this
    engine, the schema protects the consumer.
    """
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
    """JSON Schema for the published EOD result document.

    Derived from `CALCULATIONS` and `STATUSES`, never restated -- see the
    module docstring. Returns a fresh dict on every call so a caller
    mutating the result cannot corrupt the next one's.
    """
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
    """JSON Schema for the capability document.

    Looser than the result schema on purpose. The capability document is
    advisory -- it tells a coordinator what to expect before submitting --
    and its nested blocks grow as pricers land. Pinning every nested shape
    would make each new pricer a schema-breaking change for a document whose
    whole job is to describe change.

    The *contract* parts are still pinned: the version, the engine and
    mapping versions, the calculation vocabulary, and the flag saying no
    fallback curve is ever substituted.
    """
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

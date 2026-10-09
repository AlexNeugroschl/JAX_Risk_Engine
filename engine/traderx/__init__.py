"""
The TraderX path: TraderX's end-of-day bundle in, identified risk result out
(docs/planning/details/traderx-integration.md).

The rule: nothing is silently approximated. An explicit `unsupported` is recoverable; a
plausible wrong number is not.

Pricers here are closed-form discounted cashflows (Treasury bills and notes), using ORE
only for dates and day counts. The package imports no FastAPI, Pydantic, JAX or simulation
pricer, so unsupported conventions are refused before any pricing object exists (I-05;
`tests/test_traderx_pipeline.py::TestPackageImportsNoSimulationPricer`). HTTP routes
live in `engine/api/traderx_routes.py`.

Modules, in dependency order:

  `bundle.py`          read and hash-verify a v1/v2 bundle
  `terms.py`           join `instrument-terms.json` onto rows
  `normalize.py`       source units -> engine units
  `conventions.py`     convention allowlist and refusal
  `result.py`          `RiskResult` and per-calculation coverage
  `market_inputs.py`   explicit market-input mode; no silent fallback
  `identity.py`        opaque `itemId` and source identity
  `capabilities.py`    the supported product x convention x calculation matrix
  `bill.py`            zero-coupon Treasury NPV
  `note.py`            coupon Treasury NPV and rate sensitivity
  `equity.py`          cash equity: a refusal naming the missing spot
  `schema_version.py`  document versions (dependency-free leaf)
  `schema.py`          JSON Schemas of the published documents
  `workload.py`        workload key and attempt store
  `publication.py`     durable result store
  `pipeline.py`        the composition, `price_bundle`
"""
from engine.traderx.bill import (
    BillPrice,
    BillPricingError,
    is_bill,
    price_bill,
)
from engine.traderx.bundle import (
    Bundle,
    BundleIntegrityError,
    load_bundle,
)
from engine.traderx.capabilities import capabilities
from engine.traderx.conventions import (
    ConventionRefusal,
    check_conventions,
)
from engine.traderx.equity import (
    EquityPosition,
    EquityPricingError,
    is_equity,
    price_equity,
    read_position,
)
from engine.traderx.identity import ItemIdentity, item_id
from engine.traderx.market_inputs import (
    ASSUMED_PROFILES,
    ENGINE_RISK_MEASURE,
    AssumedProfile,
    CurveProvenance,
    MarketInputs,
    MarketInputsNotSupplied,
    resolve_market_inputs,
)
from engine.traderx.normalize import (
    MAPPING_VERSION,
    NormalizedPosition,
    Quantity,
    normalize_position,
)
from engine.traderx.note import (
    AccruedReconciliation,
    CouponFlow,
    NotePrice,
    NotePricingError,
    accrual_mismatch_tolerance,
    is_note,
    price_note,
    rate_sensitivity,
)
from engine.traderx.pipeline import price_bundle
from engine.traderx.result import (
    CALCULATIONS,
    STATUSES,
    Coverage,
    ItemResult,
    RiskResult,
)
from engine.traderx.schema import (
    CAPABILITY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    capability_schema,
    result_schema,
)
from engine.traderx.terms import (
    SUPPORTED_ACCRUAL_BASIS_SCHEMAS,
    SUPPORTED_TERMS_SCHEMAS,
    AccrualBasis,
    TermsJoinError,
    TermsEntry,
    join_terms,
)
from engine.traderx.workload import (
    UNKNOWN_WORKLOAD,
    Attempt,
    AttemptStore,
    workload_key,
)

__all__ = [
    "BillPrice", "BillPricingError", "is_bill", "price_bill",
    "AccruedReconciliation", "CouponFlow", "NotePrice", "NotePricingError",
    "accrual_mismatch_tolerance", "is_note", "price_note", "rate_sensitivity",
    "EquityPosition", "EquityPricingError", "is_equity", "price_equity",
    "read_position",
    "Bundle", "BundleIntegrityError", "load_bundle",
    "AccrualBasis", "TermsEntry", "TermsJoinError", "join_terms",
    "SUPPORTED_ACCRUAL_BASIS_SCHEMAS", "SUPPORTED_TERMS_SCHEMAS",
    "CAPABILITY_SCHEMA_VERSION", "RESULT_SCHEMA_VERSION",
    "capability_schema", "result_schema",
    "UNKNOWN_WORKLOAD", "Attempt", "AttemptStore", "workload_key",
    "MAPPING_VERSION", "NormalizedPosition", "Quantity", "normalize_position",
    "ConventionRefusal", "check_conventions",
    "ItemIdentity", "item_id",
    "ASSUMED_PROFILES", "ENGINE_RISK_MEASURE", "AssumedProfile", "CurveProvenance",
    "MarketInputs", "MarketInputsNotSupplied", "resolve_market_inputs",
    "CALCULATIONS", "STATUSES", "Coverage", "ItemResult", "RiskResult",
    "capabilities",
    "price_bundle",
]

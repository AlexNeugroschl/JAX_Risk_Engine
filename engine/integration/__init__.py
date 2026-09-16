"""
TraderX EOD integration boundary -- bundle in, identified risk result out.

Implements W0 ("contract and refusal machinery") of
`docs/planning/traderx-integration-plan.md`: everything needed to return a
correct, *identified*, **refusing** result for every delivered fixture
without pricing anything.

**W0 deliberately contained no pricing math at all.** That ordering is the
plan's own (§2, "Why this ordering"): it proved the whole transport ->
identity -> coverage -> publication path while pricing was still out of
scope, so contract bugs and pricing bugs never got debugged
simultaneously.

**W1.2 added the first pricer (`bill.py`), and W1.3 the second
(`note.py`), behind that same boundary and without changing it.** A
zero-coupon Treasury in a v2 bundle returns a real NPV; a coupon-bearing
one returns an NPV and a rate sensitivity; everything else still returns
the refusal it returned before. Both pricers are closed-form discounted
cashflows -- they use `ORE` for date and day-count arithmetic and touch
neither the Monte Carlo simulation nor `build_vanilla_swap`, which is what
keeps W0.4's "refuse before constructing a pricing object" guarantee
intact (see I-05, and
`tests/test_integration_pipeline.py::TestPackageImportsNoSimulationPricer`).

Module map, in dependency order:

  `bundle.py`       W0.1  read + hash-verify a v1/v2 bundle
  `terms.py`        W0.2  join `instrument-terms.json` onto rows
  `normalize.py`    W0.3  source units -> engine units
  `conventions.py`  W0.4  convention allowlist and the refusal path
  `result.py`       W0.5  `RiskResult` + per-calculation coverage
  `market_inputs.py` W0.6 explicit market-input mode; no silent fallback
  `identity.py`     W0.7  opaque `itemId` + source identity
  `capabilities.py` W0.9  the supported (product x convention x calculation) matrix
  `bill.py`         W1.2  the first pricer: zero-coupon Treasury NPV
  `note.py`         W1.3  coupon-bearing Treasury NPV + rate sensitivity
  `equity.py`       W1.4  cash equity -- a refusal naming the missing spot
  `pipeline.py`     the composition of the above into one call

The single governing rule, from which most of this code follows: **nothing
is ever silently approximated.** An explicit `unsupported` is recoverable;
a plausible wrong number is not.
"""
from engine.integration.bill import (
    BillPrice,
    BillPricingError,
    is_bill,
    price_bill,
)
from engine.integration.bundle import (
    Bundle,
    BundleIntegrityError,
    load_bundle,
)
from engine.integration.capabilities import capabilities
from engine.integration.conventions import (
    ConventionRefusal,
    check_conventions,
)
from engine.integration.equity import (
    EquityPosition,
    EquityPricingError,
    is_equity,
    price_equity,
    read_position,
)
from engine.integration.identity import ItemIdentity, item_id
from engine.integration.market_inputs import (
    ASSUMED_PROFILES,
    ENGINE_RISK_MEASURE,
    AssumedProfile,
    CurveProvenance,
    MarketInputs,
    MarketInputsNotSupplied,
    resolve_market_inputs,
)
from engine.integration.normalize import (
    MAPPING_VERSION,
    NormalizedPosition,
    Quantity,
    normalize_position,
)
from engine.integration.note import (
    AccruedReconciliation,
    CouponFlow,
    NotePrice,
    NotePricingError,
    accrual_mismatch_tolerance,
    is_note,
    price_note,
    rate_sensitivity,
)
from engine.integration.pipeline import price_bundle
from engine.integration.result import (
    CALCULATIONS,
    STATUSES,
    Coverage,
    ItemResult,
    RiskResult,
)
from engine.integration.terms import TermsJoinError, TermsEntry, join_terms

__all__ = [
    "BillPrice", "BillPricingError", "is_bill", "price_bill",
    "AccruedReconciliation", "CouponFlow", "NotePrice", "NotePricingError",
    "accrual_mismatch_tolerance", "is_note", "price_note", "rate_sensitivity",
    "EquityPosition", "EquityPricingError", "is_equity", "price_equity",
    "read_position",
    "Bundle", "BundleIntegrityError", "load_bundle",
    "TermsEntry", "TermsJoinError", "join_terms",
    "MAPPING_VERSION", "NormalizedPosition", "Quantity", "normalize_position",
    "ConventionRefusal", "check_conventions",
    "ItemIdentity", "item_id",
    "ASSUMED_PROFILES", "ENGINE_RISK_MEASURE", "AssumedProfile", "CurveProvenance",
    "MarketInputs", "MarketInputsNotSupplied", "resolve_market_inputs",
    "CALCULATIONS", "STATUSES", "Coverage", "ItemResult", "RiskResult",
    "capabilities",
    "price_bundle",
]

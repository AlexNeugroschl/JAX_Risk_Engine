"""
TraderX EOD integration boundary -- bundle in, identified risk result out.

Implements W0 ("contract and refusal machinery") of
`docs/planning/traderx-integration-plan.md`: everything needed to return a
correct, *identified*, **refusing** result for every delivered fixture
without pricing anything.

**This package deliberately contains no pricing math and imports no
pricer.** That ordering is the plan's own (§2, "Why this ordering"): it
proves the whole transport -> identity -> coverage -> publication path
while pricing is still out of scope, so contract bugs and pricing bugs
never get debugged simultaneously. W1 adds the first real pricers behind
this same boundary; nothing here changes when it does.

Module map, in dependency order:

  `bundle.py`       W0.1  read + hash-verify a v1/v2 bundle
  `terms.py`        W0.2  join `instrument-terms.json` onto rows
  `normalize.py`    W0.3  source units -> engine units
  `conventions.py`  W0.4  convention allowlist and the refusal path
  `result.py`       W0.5  `RiskResult` + per-calculation coverage
  `market_inputs.py` W0.6 explicit market-input mode; no silent fallback
  `identity.py`     W0.7  opaque `itemId` + source identity
  `capabilities.py` W0.9  the supported (product x convention x calculation) matrix
  `pipeline.py`     the composition of the above into one call

The single governing rule, from which most of this code follows: **nothing
is ever silently approximated.** An explicit `unsupported` is recoverable;
a plausible wrong number is not.
"""
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

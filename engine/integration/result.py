"""
W0.5 -- the result schema and the per-calculation coverage model.

**Coverage is per calculation per item, not per item.** An item whose NPV
computed but whose vega did not is not "covered" and not "uncovered" -- it
is covered for one calculation and not the other. Collapsing that to a
single per-item flag is how a partial result comes to look complete.

**The five statuses, used precisely:**

| Status           | Meaning                                                           |
|------------------|-------------------------------------------------------------------|
| `ok`             | Computed. A number is present.                                    |
| `unsupported`    | The engine cannot faithfully price this. Refused, not attempted.  |
| `unavailable`    | An input the calculation needs was not supplied.                  |
| `failed`         | Attempted and errored.                                            |
| `not-applicable` | The calculation is meaningless for this instrument.               |

`unsupported` vs `unavailable` is the distinction that carries the most
weight downstream: the first is closed by engine work or a convention
agreement, the second by a better export. Conflating them tells the
coordinator to retry something that will never succeed, or to give up on
something a resend would fix.

**`not-applicable` never counts against coverage.** Vega on a vanilla swap
is not a gap -- a swap has no optionality, so there is no vega to miss.
Counting it as one would make a complete result look incomplete and train
consumers to ignore the coverage block. This is why `all_applicable_computed`
excludes it while `all_outcomes_accounted_for` includes it.

**The two flags answer different questions:**

  - `allOutcomesAccountedFor` -- did every item get *some* verdict? Sums the
    five statuses and compares to `itemCount`. **True even if everything
    failed.** It is an internal-consistency check on the result document
    itself: a false here means the engine lost track of an item, which is a
    bug in this module, not a fact about the portfolio.
  - `allApplicableComputed` -- did everything that *could* have a number get
    one? True only when `unsupported + unavailable + failed == 0`.

A result can have `allOutcomesAccountedFor: true` and
`allApplicableComputed: false`, and for every W0 result it does: every item
is accounted for, and nothing is computed.

**Aggregates and currency.** An aggregate carries `coveredItemCount`,
`totalItemCount`, `excludedItems[]`, and **`complete: false` whenever those
two differ**. A total over a subset, presented as a total, is the most
dangerous number in a risk report. Cross-currency portfolios get
per-currency aggregates with `reportingCurrencyConversion: "NOT_APPLIED"` --
never a blended scalar, because the engine is not given FX rates and
inventing them would be exactly the silent approximation this design
refuses.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from engine.integration.identity import ItemIdentity, item_order_artifact
# The bare version string, from the dependency-free leaf. `schema.py`
# derives the JSON Schema *from this module*, so importing it here would be
# a cycle -- see `engine.integration.schema_version`.
from engine.integration.schema_version import RESULT_SCHEMA_VERSION

#: The frozen calculation names (plan §W0.5). Frozen means a consumer can
#: switch on them exhaustively; adding one is a contract change.
CALCULATIONS = (
    "npv",
    "accruedInterest",
    "rateSensitivity",
    "rateGamma",
    "theta",
    "vega",
    "varEs",
)

#: The five per-calculation statuses. See this module's docstring.
STATUSES = ("ok", "unsupported", "unavailable", "failed", "not-applicable")

OK = "ok"
UNSUPPORTED = "unsupported"
UNAVAILABLE = "unavailable"
FAILED = "failed"
NOT_APPLICABLE = "not-applicable"

#: Status -> the key it is counted under in the coverage block. camelCase on
#: the wire; `not-applicable` becomes `notApplicable`.
_COVERAGE_KEYS = {
    OK: "ok",
    UNSUPPORTED: "unsupported",
    UNAVAILABLE: "unavailable",
    FAILED: "failed",
    NOT_APPLICABLE: "notApplicable",
}

#: Statuses that represent a gap a consumer might act on. `not-applicable`
#: is deliberately absent -- see this module's docstring.
_GAP_STATUSES = (UNSUPPORTED, UNAVAILABLE, FAILED)


@dataclass(frozen=True)
class CalculationOutcome:
    """One calculation's outcome for one item.

    `value` is meaningful only when `status == "ok"`. Every other status
    carries a `reason` code instead -- and `detail` for a human.
    """
    status: str
    value: Optional[object] = None
    reason: Optional[str] = None
    detail: Optional[str] = None
    #: Free-form per-calculation extras (e.g. a sensitivity's `method`,
    #: `bump` and `shockedFactor`). See `SensitivityPayload`.
    payload: Optional[Dict] = None

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"unknown status {self.status!r}; expected one of {list(STATUSES)}")
        if self.status != OK and self.value is not None:
            raise ValueError(
                f"a {self.status!r} outcome must not carry a value; got {self.value!r}. "
                "A non-ok status means there is no number -- attaching one invites "
                "consumers to read it."
            )

    def to_dict(self) -> Dict:
        out: Dict = {"status": self.status}
        if self.status == OK:
            out["value"] = self.value
        if self.reason is not None:
            out["reason"] = self.reason
        if self.detail is not None:
            out["detail"] = self.detail
        if self.payload is not None:
            out.update(self.payload)
        return out

    @classmethod
    def ok(cls, value, payload: Optional[Dict] = None) -> "CalculationOutcome":
        return cls(status=OK, value=value, payload=payload)

    @classmethod
    def unsupported(cls, reason: str, detail: str = None, payload: Dict = None) -> "CalculationOutcome":
        return cls(status=UNSUPPORTED, reason=reason, detail=detail, payload=payload)

    @classmethod
    def unavailable(cls, reason: str, detail: str = None) -> "CalculationOutcome":
        return cls(status=UNAVAILABLE, reason=reason, detail=detail)

    @classmethod
    def failed(cls, reason: str, detail: str = None) -> "CalculationOutcome":
        return cls(status=FAILED, reason=reason, detail=detail)

    @classmethod
    def not_applicable(cls, detail: str = None) -> "CalculationOutcome":
        return cls(status=NOT_APPLICABLE, detail=detail)


def sensitivity_payload(
    method: str, derivative: str, shocked_factor: str, bump: float,
    value, currency: str,
) -> Dict:
    """A sensitivity's self-describing payload (plan §W0.5).

    **Never a method implied by a field name.** A field called `dv01` tells
    a consumer nothing about whether it came from algorithmic
    differentiation or a bumped revaluation, what was bumped, or by how
    much -- and those change the number. `method` is one of
    `ad-first-order` | `bumped-revaluation`, and `bump` is an explicit
    numeric, not a description.
    """
    if method not in ("ad-first-order", "bumped-revaluation"):
        raise ValueError(
            f"unknown sensitivity method {method!r}; expected 'ad-first-order' "
            f"or 'bumped-revaluation'"
        )
    return {
        "method": method,
        "derivative": derivative,
        "shockedFactor": shocked_factor,
        "bump": bump,
        "value": value,
        "currency": currency,
    }


@dataclass(frozen=True)
class ItemResult:
    """One item's identity plus its per-calculation outcomes.

    Identity is mandatory and comes first, structurally: an
    `ItemResult` cannot be constructed without it, so an unidentified
    refusal is unrepresentable rather than merely discouraged (plan §W0.7
    step 3).
    """
    identity: ItemIdentity
    calculations: Dict[str, CalculationOutcome]
    currency: Optional[str] = None
    mapping_version: Optional[str] = None
    #: Present when this item was refused, carrying the exporter's
    #: `missingTerms` and/or this engine's offending fields verbatim.
    refusal: Optional[Dict] = None

    def __post_init__(self) -> None:
        unknown = set(self.calculations) - set(CALCULATIONS)
        if unknown:
            raise ValueError(
                f"unknown calculation name(s) {sorted(unknown)}; the frozen set is "
                f"{list(CALCULATIONS)}"
            )
        missing = set(CALCULATIONS) - set(self.calculations)
        if missing:
            raise ValueError(
                f"every calculation needs an explicit outcome; missing {sorted(missing)}. "
                "An omitted calculation would be indistinguishable from a forgotten "
                "one, which is what the coverage model exists to prevent."
            )

    @property
    def item_id(self) -> str:
        return self.identity.item_id

    def to_dict(self) -> Dict:
        out: Dict = {
            "itemId": self.item_id,
            "sourceIdentity": self.identity.to_dict(),
            "calculations": {
                name: self.calculations[name].to_dict() for name in CALCULATIONS
            },
        }
        if self.currency is not None:
            out["currency"] = self.currency
        if self.mapping_version is not None:
            out["mappingVersion"] = self.mapping_version
        if self.refusal is not None:
            out["refusal"] = self.refusal
        return out


@dataclass(frozen=True)
class Coverage:
    """Per-calculation status counts plus the two completeness flags."""
    by_calculation: Dict[str, Dict[str, int]]
    item_count: int

    @property
    def all_outcomes_accounted_for(self) -> bool:
        """Every calculation's statuses sum to `item_count`. True even if
        everything failed -- this is a consistency check on the document,
        not a quality judgement on the portfolio."""
        return all(
            sum(counts.values()) == self.item_count
            for counts in self.by_calculation.values()
        )

    @property
    def all_applicable_computed(self) -> bool:
        """No gaps anywhere. `not-applicable` is excluded by construction --
        `_GAP_STATUSES` does not contain it."""
        return all(
            counts[_COVERAGE_KEYS[status]] == 0
            for counts in self.by_calculation.values()
            for status in _GAP_STATUSES
        )

    def to_dict(self) -> Dict:
        return {
            "byCalculation": self.by_calculation,
            "itemCount": self.item_count,
            "allOutcomesAccountedFor": self.all_outcomes_accounted_for,
            "allApplicableComputed": self.all_applicable_computed,
        }


def compute_coverage(items: Sequence[ItemResult]) -> Coverage:
    """Tallies per-calculation statuses across every item.

    Every calculation gets an entry with all five status keys present, even
    at zero. A consumer reading `counts["failed"]` should never have to
    handle a missing key differently from a zero -- that is the kind of
    asymmetry that produces a wrong dashboard.
    """
    by_calculation: Dict[str, Dict[str, int]] = {
        name: {key: 0 for key in _COVERAGE_KEYS.values()} for name in CALCULATIONS
    }
    for item in items:
        for name in CALCULATIONS:
            status = item.calculations[name].status
            by_calculation[name][_COVERAGE_KEYS[status]] += 1
    return Coverage(by_calculation=by_calculation, item_count=len(items))


@dataclass(frozen=True)
class CurrencyAggregate:
    """An aggregate over the items of ONE currency.

    There is deliberately no cross-currency total. The engine is not
    supplied FX rates, so a blended scalar would require inventing them --
    see this module's docstring.
    """
    currency: str
    calculation: str
    value: float
    covered_item_count: int
    total_item_count: int
    excluded_items: Tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """False whenever the aggregate covers fewer items than exist."""
        return self.covered_item_count == self.total_item_count

    def to_dict(self) -> Dict:
        return {
            "currency": self.currency,
            "calculation": self.calculation,
            "value": self.value,
            "coveredItemCount": self.covered_item_count,
            "totalItemCount": self.total_item_count,
            "excludedItems": list(self.excluded_items),
            "complete": self.complete,
            "reportingCurrencyConversion": "NOT_APPLIED",
        }


def aggregate_by_currency(items: Sequence[ItemResult], calculation: str) -> List[CurrencyAggregate]:
    """Sums one calculation per currency, excluding every non-`ok` item and
    naming what was excluded.

    An item whose value is missing is not treated as zero. Summing over
    `ok` items only, and reporting the rest in `excluded_items`, is what
    makes `complete: false` meaningful rather than decorative.
    """
    if calculation not in CALCULATIONS:
        raise ValueError(f"unknown calculation {calculation!r}")

    currencies: Dict[str, List[ItemResult]] = {}
    for item in items:
        currencies.setdefault(item.currency or "UNKNOWN", []).append(item)

    aggregates = []
    for currency in sorted(currencies):
        group = currencies[currency]
        covered = [i for i in group if i.calculations[calculation].status == OK]
        excluded = tuple(
            i.item_id for i in group if i.calculations[calculation].status != OK
        )
        aggregates.append(CurrencyAggregate(
            currency=currency,
            calculation=calculation,
            value=float(sum(i.calculations[calculation].value for i in covered)),
            covered_item_count=len(covered),
            total_item_count=len(group),
            excluded_items=excluded,
        ))
    return aggregates


@dataclass(frozen=True)
class RiskResult:
    """The whole published result: identified items, coverage, aggregates,
    and the provenance needed to reproduce or distrust it."""
    bundle_id: str
    cluster_epoch: str
    session_date: str
    valuation_time: str
    items: Tuple[ItemResult, ...]
    mapping_version: str
    engine_version: str
    #: "assumed" whenever any curve used was not observed. Top-level, so it
    #: cannot be missed by a consumer reading only the summary (plan §W0.6).
    market_provenance: Optional[str] = None
    #: The resolved `marketInputs` block -- mode, assumed profile id, and
    #: that profile's curve provenance. Echoed verbatim so "what was this
    #: priced against?" is answerable from the published result alone,
    #: without re-deriving it from the request.
    market_inputs: Optional[Dict] = None
    #: Which measure the risk figures are under (`risk-neutral-pricing` |
    #: `historical-forecast` | `deterministic-stress`). A tail statistic
    #: without its measure is unactionable -- see
    #: `engine.risk.var_es`'s RISK MEASURE VOCABULARY block and I-11.
    measure: Optional[str] = None
    warnings: Tuple[str, ...] = ()

    @property
    def coverage(self) -> Coverage:
        return compute_coverage(self.items)

    def aggregates(self, calculation: str) -> List[CurrencyAggregate]:
        return aggregate_by_currency(self.items, calculation)

    @property
    def is_assumed(self) -> bool:
        """Whether any curve behind these numbers was not observed."""
        return self.market_provenance == "assumed"

    def to_dict(self) -> Dict:
        return {
            # W1.6.2: the schema version comes first, so a consumer can
            # decide whether it understands this document before reading
            # anything else in it. Emitted even while the value stays at
            # `.v1` -- a version that was never published cannot be pinned.
            "resultSchema": RESULT_SCHEMA_VERSION,
            "bundleId": self.bundle_id,
            "clusterEpoch": self.cluster_epoch,
            "sessionDate": self.session_date,
            "valuationTime": self.valuation_time,
            "mappingVersion": self.mapping_version,
            "engineVersion": self.engine_version,
            "marketProvenance": self.market_provenance,
            "marketInputs": self.market_inputs,
            "measure": self.measure,
            "items": [item.to_dict() for item in self.items],
            # Ordering published as its own hashed artifact, never inferred
            # from the `items` array's position (plan §W0.7 step 4).
            "itemOrder": item_order_artifact(item.item_id for item in self.items),
            "coverage": self.coverage.to_dict(),
            "warnings": list(self.warnings),
        }

"""
The EOD result document and its per-calculation coverage model.

Coverage is per calculation per item: an item can be covered for NPV and not for vega.

| Status           | Meaning                                                           |
|------------------|-------------------------------------------------------------------|
| `ok`             | Computed; a number is present.                                    |
| `unsupported`    | The engine cannot price this faithfully; refused, not attempted.  |
| `unavailable`    | An input the calculation needs was not supplied.                  |
| `failed`         | Attempted and errored.                                            |
| `not-applicable` | Meaningless for this instrument (vega on a swap); never a gap.    |

`unsupported` is closed by engine work or a convention agreement, `unavailable` by a better
export.

Coverage flags: `allOutcomesAccountedFor` checks every item got some verdict (true even if
all failed; false means an item was lost, a bug here). `allApplicableComputed` is true only
when unsupported + unavailable + failed == 0.

Aggregates are per currency, never blended (no FX rates are supplied), and carry
`coveredItemCount`, `totalItemCount`, `excludedItems` and `complete: false` whenever an item
was left out.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from engine.integration.identity import ItemIdentity, item_order_artifact
# The version string comes from the leaf `schema_version` (schema imports this module).
from engine.integration.schema_version import RESULT_SCHEMA_VERSION

#: Frozen calculation names: consumers may switch on them exhaustively, so adding one is a
#: contract change.
CALCULATIONS = (
    "npv",
    "accruedInterest",
    "rateSensitivity",
    "rateGamma",
    "theta",
    "vega",
    "varEs",
)

#: The five per-calculation statuses (see the module docstring).
STATUSES = ("ok", "unsupported", "unavailable", "failed", "not-applicable")

OK = "ok"
UNSUPPORTED = "unsupported"
UNAVAILABLE = "unavailable"
FAILED = "failed"
NOT_APPLICABLE = "not-applicable"

#: Status -> its camelCase key in the coverage block.
_COVERAGE_KEYS = {
    OK: "ok",
    UNSUPPORTED: "unsupported",
    UNAVAILABLE: "unavailable",
    FAILED: "failed",
    NOT_APPLICABLE: "notApplicable",
}

#: Statuses that count as gaps (`not-applicable` does not).
_GAP_STATUSES = (UNSUPPORTED, UNAVAILABLE, FAILED)


@dataclass(frozen=True)
class CalculationOutcome:
    """One calculation's outcome for one item. `value` only when `status == "ok"`;
    otherwise a `reason` code and a human `detail`."""
    status: str
    value: Optional[object] = None
    reason: Optional[str] = None
    detail: Optional[str] = None
    #: Per-calculation extras, merged into the output (e.g. a sensitivity's `method`,
    #: `bump`, `shockedFactor`).
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
    """A sensitivity's self-describing payload: `method` (`ad-first-order` or
    `bumped-revaluation`), what was shocked, and the numeric `bump`, never implied by a field
    name."""
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
    """One item's identity (required, so an unidentified refusal cannot be built) and an
    explicit outcome for every calculation."""
    identity: ItemIdentity
    calculations: Dict[str, CalculationOutcome]
    currency: Optional[str] = None
    mapping_version: Optional[str] = None
    #: For a refused item: the exporter's `missingTerms` and/or offending fields, verbatim.
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
        """Each calculation's statuses sum to `item_count` (a consistency check, true even
        if everything failed)."""
        return all(
            sum(counts.values()) == self.item_count
            for counts in self.by_calculation.values()
        )

    @property
    def all_applicable_computed(self) -> bool:
        """No unsupported, unavailable or failed outcome anywhere."""
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
    """Per-calculation status counts over all items, with all five keys present even at
    zero."""
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
    """An aggregate over the items of one currency. There is no cross-currency total (no FX
    rates are supplied)."""
    currency: str
    calculation: str
    value: float
    covered_item_count: int
    total_item_count: int
    excluded_items: Tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """False whenever an item was excluded."""
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
    """Sum one calculation per currency over `ok` items, naming the excluded ones. A
    missing value is excluded, not treated as zero."""
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
    """The published result: identified items, coverage, aggregates and provenance."""
    bundle_id: str
    cluster_epoch: str
    session_date: str
    valuation_time: str
    items: Tuple[ItemResult, ...]
    mapping_version: str
    engine_version: str
    #: "assumed" whenever any curve used was not observed; top-level so it cannot be missed.
    market_provenance: Optional[str] = None
    #: The resolved `marketInputs` block (mode, profile id, curve provenance), echoed so the
    #: pricing basis is visible in the result itself.
    market_inputs: Optional[Dict] = None
    #: The measure of the risk figures (see `engine.risk.var_es`; I-11).
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
            # Schema version first, so a consumer can check it understands the document.
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
            # Ordering as its own hashed artifact, not inferred from array position.
            "itemOrder": item_order_artifact(item.item_id for item in self.items),
            "coverage": self.coverage.to_dict(),
            "warnings": list(self.warnings),
        }

"""
Joins each bundle row to its `instrument-terms.json` entry, or records why it has none. The
artifact's hash is verified earlier, in `engine.integration.bundle`.

Join keys:
  * Securities join on the security alone: one terms entry serves every account holding it
    (terms describe the instrument; amounts and signs stay in the rows).
  * Contracts join on `contractId` + `clusterEpoch`, since a contract id is unique only
    within its epoch.

`missingTerms` is carried verbatim, in order, into the refusal reason; it is not judged or
filled in.

A v1 bundle has no terms artifact: every row is unjoined with `NO_TERMS_ARTIFACT` (valid,
but everything needing terms is refused downstream).

Terms schemas v1 and v2 are both accepted, independently of the bundle version. v2 adds an
optional per-entry `accrualBasis` (`traderx.accrual-basis.v1`), whose `fractionDecimals`
sets the accrued-interest reconciliation tolerance
(`engine.integration.note.accrual_mismatch_tolerance`). An absent block means the exporter
stated none; it is not defaulted.

Unrecognized `accrualBasis` enum values are refused rather than read with v1 meaning. This
is an assumption, not a confirmed contract (I-23): if TraderX adds values in place, a valid
bundle will be refused here. Check a `dateBasis`/`settlementAdjustment`/`rounding`
refusal against the pinned sets below before treating it as a bad export.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from engine.integration.bundle import Bundle

#: Reasons a row has no terms entry; carried into an identified refusal, not raised.
NO_TERMS_ARTIFACT = "NO_TERMS_ARTIFACT"
TERMS_ENTRY_NOT_FOUND = "TERMS_ENTRY_NOT_FOUND"

#: Accepted terms schemas, independent of the bundle schema. Only v2 may carry
#: `accrualBasis`.
TERMS_SCHEMA_V1 = "traderx.instrument-terms.v1"
TERMS_SCHEMA_V2 = "traderx.instrument-terms.v2"
SUPPORTED_TERMS_SCHEMAS = (TERMS_SCHEMA_V1, TERMS_SCHEMA_V2)

#: `accrualBasis` block schema, versioned separately from the terms document.
ACCRUAL_BASIS_SCHEMA_V1 = "traderx.accrual-basis.v1"
SUPPORTED_ACCRUAL_BASIS_SCHEMAS = (ACCRUAL_BASIS_SCHEMA_V1,)

#: Accepted values of each `accrualBasis` enum (an unconfirmed assumption, I-23). Widening
#: adds values to these tuples; the allowlist itself stays. `SESSION_DATE` (accrual to the
#: session date, no settlement offset) is why `engine.integration.note` supports a zero
#: settlement lag only.
SUPPORTED_DATE_BASES = ("SESSION_DATE",)
SUPPORTED_SETTLEMENT_ADJUSTMENTS = ("NONE",)
SUPPORTED_ACCRUAL_ROUNDING = ("HALF_EVEN",)

#: Bounds on `fractionDecimals`, which scales the reconciliation tolerance (0 would make it
#: a full unit of face; a huge value tighter than float64).
MIN_FRACTION_DECIMALS = 1
MAX_FRACTION_DECIMALS = 12


class TermsJoinError(Exception):
    """The terms artifact itself is unusable: a duplicate identity, a malformed entry, a
    contract entry from another epoch, or an entry matching no row. Unlike an unjoined row
    (a normal state, refused downstream), this means the artifact cannot be trusted."""


@dataclass(frozen=True)
class AccrualBasis:
    """The exporter's stated accrual convention, from a v2 entry: a record of how the
    exported accrued figure was produced, not a claim about market conventions.
    `fraction_decimals` sets the reconciliation tolerance."""
    date_basis: str
    settlement_adjustment: str
    rounding: str
    fraction_decimals: int
    schema: str = ACCRUAL_BASIS_SCHEMA_V1

    def to_dict(self) -> Dict:
        return {
            "schema": self.schema,
            "dateBasis": self.date_basis,
            "settlementAdjustment": self.settlement_adjustment,
            "rounding": self.rounding,
            "fractionDecimals": self.fraction_decimals,
        }


def _parse_accrual_basis(raw, index: int) -> Optional[AccrualBasis]:
    """Parse and validate one entry's `accrualBasis`: `None` if absent; otherwise raise
    `TermsJoinError` for an unknown schema, a missing key, an unrecognized enum value or an
    out-of-range `fractionDecimals`."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TermsJoinError(
            f"terms entry [{index}] accrualBasis must be an object; got {raw!r}"
        )

    schema = raw.get("schema")
    if schema not in SUPPORTED_ACCRUAL_BASIS_SCHEMAS:
        raise TermsJoinError(
            f"terms entry [{index}] accrualBasis has unsupported schema {schema!r}; "
            f"this consumer accepts {list(SUPPORTED_ACCRUAL_BASIS_SCHEMAS)}. A new "
            f"accrual-basis version may change how the exported fraction was "
            f"produced, so it is refused rather than parsed on v1 assumptions."
        )

    for key in ("dateBasis", "settlementAdjustment", "rounding", "fractionDecimals"):
        if key not in raw:
            raise TermsJoinError(
                f"terms entry [{index}] accrualBasis is missing required key {key!r}"
            )

    # Each enum must be in its accepted set (I-23).
    for key, value, accepted in (
        ("dateBasis", raw["dateBasis"], SUPPORTED_DATE_BASES),
        ("settlementAdjustment", raw["settlementAdjustment"], SUPPORTED_SETTLEMENT_ADJUSTMENTS),
        ("rounding", raw["rounding"], SUPPORTED_ACCRUAL_ROUNDING),
    ):
        if value not in accepted:
            raise TermsJoinError(
                f"terms entry [{index}] accrualBasis.{key}={value!r} is not recognized; "
                f"this consumer accepts {list(accepted)}. Refusing rather than "
                f"assuming the v1 meaning: an unrecognized basis may describe an "
                f"accrual this engine would reconcile against the wrong date."
            )

    decimals = raw["fractionDecimals"]
    # bool is an int subclass; exclude it.
    if isinstance(decimals, bool) or not isinstance(decimals, int):
        raise TermsJoinError(
            f"terms entry [{index}] accrualBasis.fractionDecimals must be an integer; "
            f"got {decimals!r}"
        )
    if not (MIN_FRACTION_DECIMALS <= decimals <= MAX_FRACTION_DECIMALS):
        raise TermsJoinError(
            f"terms entry [{index}] accrualBasis.fractionDecimals={decimals} is outside "
            f"the accepted range [{MIN_FRACTION_DECIMALS}, {MAX_FRACTION_DECIMALS}]. "
            f"This value scales the accrual reconciliation tolerance, so an "
            f"out-of-range one would silently widen or collapse that check."
        )

    return AccrualBasis(
        date_basis=raw["dateBasis"],
        settlement_adjustment=raw["settlementAdjustment"],
        rounding=raw["rounding"],
        fraction_decimals=decimals,
        schema=schema,
    )


@dataclass(frozen=True)
class TermsEntry:
    """One terms entry, parsed but not interpreted: units are handled by
    `engine.integration.normalize`, conventions by `engine.integration.conventions`."""
    instrument_type: str
    terms: Dict
    missing_terms: Tuple[str, ...]
    provenance: Dict
    identity: Dict
    #: The exporter's accrual convention (v2 only), or `None`.
    accrual_basis: Optional[AccrualBasis] = None
    #: The entry's terms schema, to tell "v1, no basis possible" from "v2, basis omitted".
    terms_schema: str = TERMS_SCHEMA_V1

    @property
    def is_complete(self) -> bool:
        """The exporter listed no missing terms. Not a claim that the instrument is
        priceable; `engine.integration.conventions` decides that."""
        return not self.missing_terms

    @property
    def origin(self) -> Optional[str]:
        return self.provenance.get("origin")


@dataclass(frozen=True)
class JoinedRow:
    """One bundle row with its terms entry, or with the reason it has none."""
    source: str                      # "positions" | "contracts"
    row: Dict[str, str]
    entry: Optional[TermsEntry]
    unjoined_reason: Optional[str] = None

    @property
    def is_joined(self) -> bool:
        return self.entry is not None


@dataclass(frozen=True)
class TermsJoin:
    """The whole bundle's rows, each joined or explicitly unjoined."""
    rows: Tuple[JoinedRow, ...]
    has_terms_artifact: bool

    @property
    def joined(self) -> Tuple[JoinedRow, ...]:
        return tuple(r for r in self.rows if r.is_joined)

    @property
    def unjoined(self) -> Tuple[JoinedRow, ...]:
        return tuple(r for r in self.rows if not r.is_joined)


def _parse_entry(raw: Dict, index: int, terms_schema: str = TERMS_SCHEMA_V1) -> TermsEntry:
    for key in ("identity", "instrumentType", "terms", "missingTerms"):
        if key not in raw:
            raise TermsJoinError(f"terms entry [{index}] is missing required key {key!r}")

    identity = raw["identity"]
    if not isinstance(identity, dict) or "source" not in identity:
        raise TermsJoinError(f"terms entry [{index}] has no identity.source")

    missing = raw["missingTerms"]
    if not isinstance(missing, list) or not all(isinstance(m, str) for m in missing):
        raise TermsJoinError(
            f"terms entry [{index}] missingTerms must be a list of strings; got {missing!r}"
        )

    # An accrualBasis under a v1 schema label contradicts the document's own version, so it
    # is refused.
    raw_basis = raw.get("accrualBasis")
    if raw_basis is not None and terms_schema == TERMS_SCHEMA_V1:
        raise TermsJoinError(
            f"terms entry [{index}] carries an accrualBasis block, but the artifact "
            f"declares schema {TERMS_SCHEMA_V1!r}, which has no such field. Either "
            f"the artifact should declare {TERMS_SCHEMA_V2!r} or the block should be "
            f"absent -- a document that contradicts its own version marker cannot be "
            f"parsed on that marker's assumptions."
        )

    return TermsEntry(
        instrument_type=raw["instrumentType"],
        terms=raw["terms"],
        # Verbatim and in order (see module docstring).
        missing_terms=tuple(missing),
        provenance=raw.get("provenance", {}),
        identity=identity,
        accrual_basis=_parse_accrual_basis(raw_basis, index),
        terms_schema=terms_schema,
    )


def _entry_key(identity: Dict, index: int) -> Tuple:
    """Join key: `security` for positions; `(contractId, clusterEpoch)` for contracts."""
    source = identity["source"]
    if source == "positions":
        if "security" not in identity:
            raise TermsJoinError(f"terms entry [{index}] identity.source=positions needs 'security'")
        return ("positions", identity["security"])
    if source == "contracts":
        for key in ("contractId", "clusterEpoch"):
            if key not in identity:
                raise TermsJoinError(
                    f"terms entry [{index}] identity.source=contracts needs {key!r}"
                )
        return ("contracts", identity["contractId"], identity["clusterEpoch"])
    raise TermsJoinError(
        f"terms entry [{index}] has unknown identity.source {source!r}; "
        f"expected 'positions' or 'contracts'"
    )


def _index_entries(terms: Dict, bundle_epoch: str) -> Dict[Tuple, TermsEntry]:
    """Identity -> entry index; rejects duplicates and contract entries from another epoch."""
    # The terms schema is checked independently of the bundle version.
    schema = terms.get("schema")
    if schema not in SUPPORTED_TERMS_SCHEMAS:
        raise TermsJoinError(
            f"unsupported terms schema {schema!r}; this consumer accepts "
            f"{list(SUPPORTED_TERMS_SCHEMAS)}"
        )

    index: Dict[Tuple, TermsEntry] = {}
    for i, raw in enumerate(terms.get("entries", [])):
        entry = _parse_entry(raw, i, terms_schema=schema)
        key = _entry_key(entry.identity, i)

        if key[0] == "contracts" and key[2] != bundle_epoch:
            raise TermsJoinError(
                f"terms entry [{i}] for contract {key[1]!r} declares clusterEpoch "
                f"{key[2]!r} but the bundle's epoch is {bundle_epoch!r}. A contract id "
                f"is unique only within its epoch, so this entry may describe a "
                f"different contract."
            )

        if key in index:
            raise TermsJoinError(
                f"terms entry [{i}] duplicates identity {key!r}. Every unique position "
                f"security and every contract needs exactly one entry."
            )
        index[key] = entry
    return index


def join_terms(bundle: Bundle) -> TermsJoin:
    """Join the bundle's position and contract rows onto its terms artifact. Every row is
    joined or carries its reason; `TermsJoinError` only when the artifact is unusable."""
    if not bundle.has_terms:
        # v1: no terms artifact, every row unjoined.
        rows = tuple(
            JoinedRow(source=source, row=row, entry=None, unjoined_reason=NO_TERMS_ARTIFACT)
            for source, artifact in (("positions", bundle.positions), ("contracts", bundle.contracts))
            for row in artifact.rows
        )
        return TermsJoin(rows=rows, has_terms_artifact=False)

    index = _index_entries(bundle.terms, bundle.cluster_epoch)

    joined: List[JoinedRow] = []
    used: set = set()
    for row in bundle.positions.rows:
        key = ("positions", row["security"])
        entry = index.get(key)
        if entry is not None:
            used.add(key)
        joined.append(JoinedRow(
            source="positions", row=row, entry=entry,
            unjoined_reason=None if entry else TERMS_ENTRY_NOT_FOUND,
        ))

    for row in bundle.contracts.rows:
        key = ("contracts", row["contractId"], bundle.cluster_epoch)
        entry = index.get(key)
        if entry is not None:
            used.add(key)
        joined.append(JoinedRow(
            source="contracts", row=row, entry=entry,
            unjoined_reason=None if entry else TERMS_ENTRY_NOT_FOUND,
        ))

    # An entry matching no row describes an instrument not in the bundle, and cannot be
    # attributed to any row, so it is raised.
    unused = set(index) - used
    if unused:
        raise TermsJoinError(
            f"terms artifact carries {len(unused)} entry/entries matching no bundle row: "
            f"{sorted(unused)}"
        )

    return TermsJoin(rows=tuple(joined), has_terms_artifact=True)

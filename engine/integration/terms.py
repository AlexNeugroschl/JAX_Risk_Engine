"""
W0.2 -- the `instrument-terms.json` join.

Attaches each bundle row to its reference terms entry, or establishes that
it has none. **The hash is verified before these bytes are parsed** -- that
happens in `engine.integration.bundle`, W0.2 step 1, so anything reaching
this module already verified.

**Two different join keys, because securities and contracts have different
identity.**

  - *Securities* join by **security identity alone**. One `UST-NOTE-20261215`
    terms entry serves both the long account's row and the short account's
    row: the terms describe the *instrument*, while the amounts and signs
    stay in the position rows. The delivered note fixture is exactly this
    shape, and it is why `join_terms` returns a per-row mapping rather than
    a per-entry one -- the validator "checks each joined row, not only the
    first account" (bundle-v2-and-terms.md).
  - *Contracts* join by **`contractId` + `clusterEpoch`**. A contract id is
    "unique within the cluster epoch by construction"; across epochs it is
    not. Joining on `contractId` alone would silently match a contract from
    a different epoch that happens to reuse the id.

**`missingTerms` is an authoritative refusal input, not a hint** (W0.2 step
3). When an entry lists missing terms, that list is carried *verbatim* into
the refusal reason. This module does not judge whether the listed terms
matter, does not attempt to fill them, and does not reorder them -- the
exporter enumerated exactly what it could not supply, and that enumeration
*is* the agenda (plan §W2). The SOFR fixture's 13 entries are the whole
reason this path exists.

**v1 bundles have no terms artifact** (W0.2 step 5). `join_terms` returns a
join in which every row is unjoined, with reason `NO_TERMS_ARTIFACT`. That
is not an error -- v1 is a valid bundle version -- but it does mean every
instrument needing terms is `unsupported` downstream. This module reports
that state; `engine.integration.conventions` acts on it.

**W1.6.1 -- `traderx.instrument-terms.v2`, and its `accrualBasis`.**

The terms artifact now has two accepted schema versions, and **the terms
version is independent of the bundle version**: a `traderx.eod-bundle.v2`
bundle may carry either `instrument-terms.v1` or `.v2`. They are separately
versioned documents that happen to travel together, so pinning one to the
other would reject a valid combination. `SUPPORTED_TERMS_SCHEMAS` is
therefore its own list, checked on its own.

v2 adds one thing this consumer reads: an optional per-entry `accrualBasis`
block (`traderx.accrual-basis.v1`) making the exporter's accrual convention
machine-readable rather than conventional prose. Its `fractionDecimals` is
what turns W1.3's reconciliation tolerance from a negotiated constant into a
*derived* one -- see `engine.integration.note.accrual_mismatch_tolerance`.

**Unrecognized enum values are refused, not parsed optimistically.** This
was an open question to TraderX (response v4 §1.3: do new `dateBasis` /
`settlementAdjustment` values land in `accrual-basis.v1`, or force a `.v2`?)
and it was never answered. The strict reading is the safe one and is what
the plan's settled row asks for: pin the exact accepted value set and refuse
anything outside it. The failure mode this prevents is the one this whole
boundary exists to prevent -- a real-market settlement basis silently
inheriting the synthetic fixture's same-day semantics, which would shift
accrued interest with no error anywhere. If TraderX later confirms values
are added in place, widening a tuple here is a one-line change; recovering
from months of optimistically-parsed wrong accruals is not.

> **This rule is an ASSUMPTION, not a confirmed contract -- see I-23 in
> `docs/known-issues.md`.** The question above is still unanswered. If
> TraderX intends to add values *in place*, this module will refuse bundles
> they consider valid on the day they first export a real settlement
> calendar. That fails safe (a loud refusal, not a wrong number) but it is
> an operational break that will arrive without warning and will look like
> a defect. Before treating a `dateBasis` / `settlementAdjustment` /
> `rounding` refusal as a bad export, check it against the pinned sets
> below -- the allowlist may simply be narrower than their vocabulary.

**A v2 entry with no `accrualBasis` is legal.** The block is optional, and
its absence means "the exporter did not state one", which is exactly the v1
state. It is not an error, and it does not become a default -- consumers
that need one ask `TermsEntry.accrual_basis` and handle `None`.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from engine.integration.bundle import Bundle

#: Join failure reasons, carried into the refusal rather than raised, so an
#: unjoinable row still produces an *identified* refusal (plan §W0.7 step 3)
#: instead of vanishing.
NO_TERMS_ARTIFACT = "NO_TERMS_ARTIFACT"
TERMS_ENTRY_NOT_FOUND = "TERMS_ENTRY_NOT_FOUND"

#: Terms artifact schemas this consumer accepts (W1.6.1).
#:
#: **Independent of the bundle schema version** -- see the module docstring.
#: v1 has no `accrualBasis`; v2 may carry one per entry.
TERMS_SCHEMA_V1 = "traderx.instrument-terms.v1"
TERMS_SCHEMA_V2 = "traderx.instrument-terms.v2"
SUPPORTED_TERMS_SCHEMAS = (TERMS_SCHEMA_V1, TERMS_SCHEMA_V2)

#: The `accrualBasis` block's own schema. Versioned separately again,
#: because the exporter's accrual convention can change without the terms
#: document's shape changing.
ACCRUAL_BASIS_SCHEMA_V1 = "traderx.accrual-basis.v1"
SUPPORTED_ACCRUAL_BASIS_SCHEMAS = (ACCRUAL_BASIS_SCHEMA_V1,)

#: The exact accepted value set for each `accrualBasis` enum, pinned rather
#: than parsed open-endedly (module docstring, and response v4 §1.3).
#:
#: `SESSION_DATE` means the exporter accrues to the session date itself,
#: with no settlement offset -- which is why `engine.integration.note`
#: supports a settlement lag of zero only. A `dateBasis` this consumer does
#: not recognize is refused, because the alternative is pricing accrued
#: interest to a date that is not the one the number describes.
#:
#: **These sets encode an unconfirmed assumption (I-23).** They are correct
#: if TraderX versions the schema on every new value, and too narrow if they
#: add values in place. Widening is deliberately a one-line change per
#: value -- but it is a change to *which* values are allowlisted, never to
#: *whether* there is an allowlist.
SUPPORTED_DATE_BASES = ("SESSION_DATE",)
SUPPORTED_SETTLEMENT_ADJUSTMENTS = ("NONE",)
SUPPORTED_ACCRUAL_ROUNDING = ("HALF_EVEN",)

#: Bounds on `fractionDecimals`. Not a style check: this value *scales the
#: reconciliation tolerance*, so a nonsensical one silently widens or
#: collapses the check that catches accrual bugs. Zero would make the
#: tolerance a full unit of face; an absurdly large value would make it
#: tighter than float64 can represent.
MIN_FRACTION_DECIMALS = 1
MAX_FRACTION_DECIMALS = 12


class TermsJoinError(Exception):
    """Raised for a terms artifact that is structurally unusable: a
    duplicate identity, a malformed entry, an entry whose epoch contradicts
    the bundle's, or an entry that joins to nothing.

    Distinct from an *unjoined row*, which is not an error: a row with no
    terms entry is refused downstream with full identity (see this module's
    docstring). The difference is that an unjoined row is a knowable state
    of a well-formed artifact, whereas everything here means the artifact
    itself cannot be trusted to describe the bundle.
    """


@dataclass(frozen=True)
class AccrualBasis:
    """The exporter's stated accrual convention, from a v2 terms entry.

    **This describes what the exporter did, not what any model supports.**
    It makes an otherwise-conventional assumption explicit: that accrued
    interest was computed to the session date, unadjusted, and rounded
    HALF_EVEN at `fraction_decimals`. Accepting it is not an assertion that
    those are correct market conventions -- it is a record of the basis the
    exported number was produced on, which is precisely what makes the
    number reconcilable.

    `fraction_decimals` is the load-bearing field: it *derives* W1.3's
    reconciliation tolerance. If the exporter ever publishes more precision,
    the tolerance tightens automatically instead of staying at a stale
    constant.
    """
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
    """Parses and *validates* one entry's `accrualBasis`.

    Returns `None` when the block is absent, which is legal (see the module
    docstring). Every other failure raises `TermsJoinError` naming the
    field and the accepted set -- an unrecognized value is refused rather
    than carried through, because a basis this consumer does not understand
    means the accrued number was produced on terms it cannot reconcile
    against.
    """
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

    # Each enum is pinned to its accepted set. An unrecognized value is the
    # case response v4 §1.3 asked about and never got an answer to; refusing
    # is the reading that cannot silently reinterpret an accrued number.
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
    # bool is an int subclass; True would otherwise pass as 1 decimal.
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
    """One `instrument-terms.json` entry, parsed but not interpreted.

    `terms` and `missing_terms` are carried through as supplied. This class
    deliberately does not normalize units or validate financial content --
    that is `engine.integration.normalize` (W0.3) and
    `engine.integration.conventions` (W0.4) respectively. Keeping the parse
    dumb means a malformed value fails where it is *used*, naming the field,
    rather than here where the context is gone.
    """
    instrument_type: str
    terms: Dict
    missing_terms: Tuple[str, ...]
    provenance: Dict
    identity: Dict
    #: The exporter's stated accrual convention (v2 only), or `None` when
    #: the entry is v1 or a v2 entry that omitted the optional block.
    #: **Optional with a default**, so every existing construction site --
    #: tests included -- stays valid unchanged (W1.6.1).
    accrual_basis: Optional[AccrualBasis] = None
    #: Which terms schema this entry came from. Carried so a consumer can
    #: tell "v1, so no basis was possible" from "v2 that omitted one".
    terms_schema: str = TERMS_SCHEMA_V1

    @property
    def is_complete(self) -> bool:
        """True when the exporter listed no missing terms.

        **This is not a claim that the instrument is priceable.** An entry
        with no missing fields still "does not prove a model can represent
        it" (bundle-v2-and-terms.md) -- that judgement belongs to
        `engine.integration.conventions`, against this engine's own
        allowlist. Complete means "the exporter supplied everything it
        owed", nothing more.
        """
        return not self.missing_terms

    @property
    def origin(self) -> Optional[str]:
        return self.provenance.get("origin")


@dataclass(frozen=True)
class JoinedRow:
    """One bundle row paired with its terms entry, or with the reason it
    has none. Both cases carry full source identity, because a refusal
    nobody can attribute to a position is useless (plan §W0.7 step 3)."""
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

    # `accrualBasis` is a v2 feature. An entry carrying one under a v1
    # schema label is refused rather than accepted leniently: the document
    # is then self-contradictory, and the schema label is the thing every
    # other parsing decision here keys on. Accepting it would mean trusting
    # a version marker the document has already disproved.
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
        # Carried verbatim and in supplied order -- this list is an
        # authoritative refusal input (see module docstring), not a set to
        # be tidied.
        missing_terms=tuple(missing),
        provenance=raw.get("provenance", {}),
        identity=identity,
        accrual_basis=_parse_accrual_basis(raw_basis, index),
        terms_schema=terms_schema,
    )


def _entry_key(identity: Dict, index: int) -> Tuple:
    """The join key for one terms entry.

    Securities key on `security` alone (shared across accounts); contracts
    key on `(contractId, clusterEpoch)` -- see this module's docstring for
    why the epoch is part of the contract key and not the security one.
    """
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
    """Builds the identity -> entry index, rejecting duplicates and
    wrong-epoch contract entries (W0.2 step 4)."""
    # Terms version is checked independently of the bundle version: a v2
    # bundle may legitimately carry either terms schema (W1.6.1).
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
    """Joins `bundle`'s position and contract rows onto its terms artifact.

    Returns a `TermsJoin` in which every row is either joined to its entry
    or carries the reason it is not. Raises `TermsJoinError` only when the
    artifact itself is unusable -- see `TermsJoinError`'s own docstring for
    that distinction.
    """
    if not bundle.has_terms:
        # v1: valid bundle, no terms artifact, every row unjoined (step 5).
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

    # An entry joining to nothing means the artifact describes an instrument
    # this bundle does not contain -- "extra ... entries fail"
    # (bundle-v2-and-terms.md). Unlike an unjoined row, this cannot be
    # attributed to any position, so it is raised rather than carried.
    unused = set(index) - used
    if unused:
        raise TermsJoinError(
            f"terms artifact carries {len(unused)} entry/entries matching no bundle row: "
            f"{sorted(unused)}"
        )

    return TermsJoin(rows=tuple(joined), has_terms_artifact=True)

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
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from engine.integration.bundle import Bundle

#: Join failure reasons, carried into the refusal rather than raised, so an
#: unjoinable row still produces an *identified* refusal (plan §W0.7 step 3)
#: instead of vanishing.
NO_TERMS_ARTIFACT = "NO_TERMS_ARTIFACT"
TERMS_ENTRY_NOT_FOUND = "TERMS_ENTRY_NOT_FOUND"


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


def _parse_entry(raw: Dict, index: int) -> TermsEntry:
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

    return TermsEntry(
        instrument_type=raw["instrumentType"],
        terms=raw["terms"],
        # Carried verbatim and in supplied order -- this list is an
        # authoritative refusal input (see module docstring), not a set to
        # be tidied.
        missing_terms=tuple(missing),
        provenance=raw.get("provenance", {}),
        identity=identity,
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
    schema = terms.get("schema")
    if schema != "traderx.instrument-terms.v1":
        raise TermsJoinError(
            f"unsupported terms schema {schema!r}; expected 'traderx.instrument-terms.v1'"
        )

    index: Dict[Tuple, TermsEntry] = {}
    for i, raw in enumerate(terms.get("entries", [])):
        entry = _parse_entry(raw, i)
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

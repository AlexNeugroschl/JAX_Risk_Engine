"""
Convention allowlist and refusal path (part of I-05).

Checks run on the terms alone, before any pricing object exists; this module imports no
pricer, ORE builder or curve. Once a booking reaches `build_vanilla_swap` it has been given
ACT/365 on both legs, a TARGET calendar and a `SimIndex<N>M` term index, whatever its real
conventions. A USD-SOFR booking (overnight index, ACT/360, compounded in arrears, US
calendar, lookback/lockout) priced that way would be wrong without warning: ACT/360 vs
ACT/365 alone moves every accrual by 1.389%.

So the allowlist is a positive list of what the engine implements, not a blocklist. A
booking with no stated conventions is refused like a wrong one, never defaulted. When the
exporter lists `missingTerms`, that list is the refusal reason, verbatim.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from engine.integration.terms import JoinedRow, TermsEntry

#: Refusal reason codes.
CONVENTION_NOT_SUPPORTED = "CONVENTION_NOT_SUPPORTED"
TERMS_NOT_SUPPLIED = "TERMS_NOT_SUPPLIED"
INSTRUMENT_TYPE_NOT_SUPPORTED = "INSTRUMENT_TYPE_NOT_SUPPORTED"

#: The allowlist: only the generic term-Ibor / ACT/365 profile the engine implements. An
#: entry is a claim that `engine/instruments/` prices such a booking faithfully, so adding
#: one should come with a parity test against an independent reference.

#: Floating indices implemented: `SimIndex`, the generic term index of
#: `build_vanilla_swap`. USD-SOFR (overnight compounded) is not.
SUPPORTED_FLOAT_INDICES = ("SimIndex",)

#: Swap leg day counts implemented. The generic builder stays ACT/365, consistent with the
#: simulation time axis.
SUPPORTED_SWAP_DAY_COUNTS = ("ACT/365",)

#: Overnight compounding is not implemented; a booking naming any method is refused.
SUPPORTED_OVERNIGHT_COMPOUNDING: Tuple[str, ...] = ()

#: Instrument types whose conventions this module can establish. That is not the same as
#: being priced: TREASURY is priced; SWAP is refused downstream (no faithful SOFR build,
#: I-05); EQUITY is refused downstream for want of a spot (I-18), which points the caller at
#: missing data rather than calling it out of scope.
SUPPORTED_INSTRUMENT_TYPES = ("SWAP", "TREASURY", "EQUITY")

#: Swap terms that must be present (and allowlisted); a booking missing any is refused.
REQUIRED_SWAP_CONVENTIONS = (
    "floatIndex",
    "fixedDayCount",
    "floatingDayCount",
    "effectiveDate",
    "maturityDate",
)


@dataclass(frozen=True)
class ConventionRefusal:
    """Why one instrument cannot be priced faithfully. `missing_terms` is the exporter's
    own list of terms it did not supply (fixed by a better export); `offending` lists
    supplied terms the engine does not implement (fixed by new engine work)."""
    reason: str
    detail: str
    #: The exporter's `missingTerms`, verbatim and in supplied order.
    missing_terms: Tuple[str, ...] = ()
    #: `{term: supplied_value}` for terms outside the allowlist.
    offending: Tuple[Tuple[str, str], ...] = ()

    @property
    def offending_fields(self) -> Tuple[str, ...]:
        return tuple(name for name, _ in self.offending)

    def to_dict(self) -> Dict:
        return {
            "reason": self.reason,
            "detail": self.detail,
            "missingTerms": list(self.missing_terms),
            "offendingFields": [
                {"field": name, "value": value} for name, value in self.offending
            ],
        }


def _check_swap(entry: TermsEntry) -> Optional[ConventionRefusal]:
    """Check a swap's conventions. Absent conventions are checked first, so a booking that
    states nothing is refused for that rather than passing vacuously."""
    terms = entry.terms

    absent = tuple(t for t in REQUIRED_SWAP_CONVENTIONS if not terms.get(t))
    if absent:
        # Refuse to infer absent conventions.
        return ConventionRefusal(
            reason=CONVENTION_NOT_SUPPORTED,
            detail=(
                "booking does not state the conventions required to price it: "
                + ", ".join(absent)
                + ". Absent conventions are refused, never defaulted into the "
                "generic term-IBOR builder."
            ),
            missing_terms=entry.missing_terms,
            offending=tuple((t, "<absent>") for t in absent),
        )

    offending = []

    float_index = str(terms["floatIndex"])
    if not any(float_index.startswith(ok) for ok in SUPPORTED_FLOAT_INDICES):
        offending.append(("floatIndex", float_index))

    for field in ("fixedDayCount", "floatingDayCount"):
        value = str(terms[field])
        if value not in SUPPORTED_SWAP_DAY_COUNTS:
            offending.append((field, value))

    compounding = terms.get("overnightCompounding")
    if compounding and str(compounding) not in SUPPORTED_OVERNIGHT_COMPOUNDING:
        offending.append(("overnightCompounding", str(compounding)))

    if offending:
        return ConventionRefusal(
            reason=CONVENTION_NOT_SUPPORTED,
            detail=(
                "booking names conventions this engine does not implement: "
                + ", ".join(f"{name}={value}" for name, value in offending)
                + ". Supported today: floatIndex in "
                + f"{list(SUPPORTED_FLOAT_INDICES)}, leg day counts in "
                + f"{list(SUPPORTED_SWAP_DAY_COUNTS)}."
            ),
            missing_terms=entry.missing_terms,
            offending=tuple(offending),
        )
    return None


def check_conventions(joined: JoinedRow) -> Optional[ConventionRefusal]:
    """A `ConventionRefusal` if the row cannot be priced faithfully, else `None`. `None`
    says only that the conventions are supported, not that a pricer exists."""
    entry = joined.entry

    if entry is None:
        # No terms, whatever the reason (v1 bundle, or an unjoinable row).
        return ConventionRefusal(
            reason=TERMS_NOT_SUPPLIED,
            detail=(
                f"no instrument terms available for this row "
                f"({joined.unjoined_reason}); conventions cannot be established "
                f"and will not be inferred."
            ),
        )

    if entry.instrument_type not in SUPPORTED_INSTRUMENT_TYPES:
        return ConventionRefusal(
            reason=INSTRUMENT_TYPE_NOT_SUPPORTED,
            detail=(
                f"instrumentType={entry.instrument_type!r} is outside this engine's "
                f"scope; supported: {list(SUPPORTED_INSTRUMENT_TYPES)}"
            ),
            missing_terms=entry.missing_terms,
        )

    if entry.missing_terms:
        # The exporter's own list of missing terms is the refusal reason.
        return ConventionRefusal(
            reason=CONVENTION_NOT_SUPPORTED,
            detail=(
                f"instrument terms are incomplete: the export enumerates "
                f"{len(entry.missing_terms)} required term(s) it does not supply. "
                f"An incompletely specified instrument is refused, not "
                f"approximated from the terms that were supplied."
            ),
            missing_terms=entry.missing_terms,
        )

    if entry.instrument_type == "SWAP":
        return _check_swap(entry)

    # TREASURY or EQUITY with complete terms. Whether it prices is decided downstream
    # (an equity is refused by engine.integration.equity for its missing spot).
    return None

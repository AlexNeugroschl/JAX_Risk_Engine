"""
W0.4 -- the convention allowlist and the refusal path. Closes part of
**I-05**.

**Refuse before constructing any pricing object.** Every check here runs on
the terms alone; nothing in this module imports a pricer, an ORE builder, or
a curve. That ordering is the whole point -- once a booking reaches
`build_vanilla_swap`, it has already been given ACT/365 on both legs, a
TARGET calendar, a `SimIndex<N>M` term index, and a tenor-derived schedule,
whether or not any of those are its actual conventions.

---

**The bug this prevents.**

`engine.models.ore_builders.build_vanilla_swap` produces a generic term-IBOR
swap: `SimIndex6M`, ACT/365 both legs, TARGET calendar, schedule derived from
a tenor *string*. A USD-SOFR booking is none of those things -- it is an
overnight index, ACT/360, compounded daily in arrears, on a US calendar, with
explicit effective/maturity dates plus lookback/lockout/payment-lag terms
that signature cannot even express.

Routing the SOFR booking through that builder prices it **confidently and
wrongly**. ACT/360 vs ACT/365 alone shifts every accrual factor by 1.389%:
roughly **$1,906 on a $1mm 5Y fixed leg, about 46x a 1bp DV01**. And **no
existing test catches it**, because every test in this repository builds its
inputs with that same builder -- so the wrong convention is applied
identically on both sides of every comparison and cancels out.

That is why the allowlist is defined as an explicit *positive* list of what
the engine actually implements, rather than a blocklist of what it does not.
A blocklist silently admits everything nobody thought to add to it.

---

**Refuse to infer** (W0.4 step 4). A booking with no stated conventions is
`unsupported` -- *never* defaulted into the generic builder. This is the case
that most wants a default, because the generic builder would happily accept
it and return a plausible number. `check_conventions` treats absent
conventions exactly like wrong ones.

**`missingTerms` short-circuits everything.** When the exporter enumerated
terms it could not supply, that list is the refusal reason, carried verbatim
(plan §W0.2 step 3). The engine does not additionally opine on whether the
*supplied* terms would have been acceptable -- the instrument is incompletely
specified, full stop.
"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from engine.integration.terms import JoinedRow, TermsEntry

#: Refusal reason codes.
CONVENTION_NOT_SUPPORTED = "CONVENTION_NOT_SUPPORTED"
TERMS_NOT_SUPPLIED = "TERMS_NOT_SUPPLIED"
INSTRUMENT_TYPE_NOT_SUPPORTED = "INSTRUMENT_TYPE_NOT_SUPPORTED"

#: ---------------------------------------------------------------------
#: The allowlist.
#:
#: **Today it contains only the generic term-IBOR / ACT-365 profile the
#: engine actually implements** (plan §W0.4 step 2). Every entry here is a
#: claim that `engine/instruments/` can faithfully price a booking with
#: these conventions -- not that ORE could represent it, and not that the
#: engine would return *a* number for it (it would return one for anything).
#: Distinguishing those is plan working rule 4.
#:
#: Adding an entry to this dict is therefore a financial assertion, and
#: should come with a parity test against an independent reference, not
#: against this engine's own suite.
#: ---------------------------------------------------------------------

#: Floating indices the engine implements. `SimIndex` is the generic term
#: IBOR that `build_vanilla_swap` constructs. USD-SOFR is deliberately
#: ABSENT: it is an overnight compounded index and W2 is blocked on the
#: D03/D04 convention agreement.
SUPPORTED_FLOAT_INDICES = ("SimIndex",)

#: Day counts the engine implements for swap legs. ACT/365 only, and it must
#: stay that way for the generic builder -- every swaption pricer depends on
#: `build_vanilla_swap`'s ACT/365 consistency with the simulation time axis
#: (plan §W1.1). ACT/360 is not "not yet added"; it is a different
#: instrument the engine does not build.
SUPPORTED_SWAP_DAY_COUNTS = ("ACT/365",)

#: Overnight-compounding is not implemented at all. A booking naming any
#: compounding method is refused rather than treated as a term-index reset.
SUPPORTED_OVERNIGHT_COMPOUNDING: Tuple[str, ...] = ()

#: Instrument types this boundary knows how to reason about.
#:
#: **"Supported" here means "the engine can establish its conventions",
#: not "the engine prices it".** Those are different claims and this
#: module only makes the first (see `check_conventions`). TREASURY passes
#: and is priced (W1.2/W1.3); SWAP passes and is refused downstream for
#: want of a faithful USD-SOFR build (I-05); EQUITY passes and is refused
#: downstream for want of a spot source (W1.4, I-18).
#:
#: **Why EQUITY is here rather than absent.** Absent, it refused as
#: `INSTRUMENT_TYPE_NOT_SUPPORTED` -- "outside this engine's scope", which
#: is the wrong fact. A cash equity *is* in scope, is fully understood, and
#: needs one market input nobody has supplied. Those two refusals point at
#: different remedies: the first says "this engine will never do that", the
#: second says "supply a spot". Conflating them tells a coordinator to give
#: up when it should be sending data.
SUPPORTED_INSTRUMENT_TYPES = ("SWAP", "TREASURY", "EQUITY")

#: Terms that must be present and allowlisted before a swap could be priced.
#: Used to detect the "no stated conventions" case (step 4) -- a booking
#: missing any of these is refused, never defaulted.
REQUIRED_SWAP_CONVENTIONS = (
    "floatIndex",
    "fixedDayCount",
    "floatingDayCount",
    "effectiveDate",
    "maturityDate",
)


@dataclass(frozen=True)
class ConventionRefusal:
    """Why one instrument cannot be faithfully priced.

    `missing_terms` and `offending` are separate on purpose. The first is
    the exporter's own enumeration of what it did not supply; the second is
    this engine's list of terms that were supplied but name something it
    does not implement. A consumer acting on the refusal needs to know which
    it is looking at: the first is closed by a better export, the second by
    a convention agreement and new engine work.
    """
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
    """Checks a swap booking's conventions against the allowlist.

    Order matters: absent conventions are checked *before* supplied ones, so
    a booking stating nothing is refused for stating nothing rather than
    passing vacuously.
    """
    terms = entry.terms

    absent = tuple(t for t in REQUIRED_SWAP_CONVENTIONS if not terms.get(t))
    if absent:
        # Step 4: refuse to infer. This is the branch a generic default
        # would have swallowed.
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
    """Returns a `ConventionRefusal` if this row cannot be faithfully
    priced, or `None` if its conventions are allowlisted.

    `None` means only that the *conventions* are supported. It is not a
    statement that a pricer exists -- in W0 none does, and the caller
    (`engine.integration.pipeline`) still reports `unsupported` for every
    calculation. Keeping those two refusals distinct is what lets W1 add
    pricers without touching this module.
    """
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
        # The exporter enumerated what it could not supply. That list IS the
        # refusal reason, verbatim -- this is the SOFR fixture's path, and
        # its 13 entries are W2's agenda.
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

    # TREASURY or EQUITY with complete terms: conventions are fine. That
    # is NOT a claim either one prices -- a Treasury does (W1.2/W1.3), an
    # equity does not, and the equity's refusal is raised by
    # `engine.integration.equity` naming the missing market input rather
    # than here naming its type.
    return None

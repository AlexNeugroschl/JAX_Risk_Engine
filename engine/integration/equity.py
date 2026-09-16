"""
W1.4 -- the equity position pricer, which **refuses**.

A cash equity position is arithmetically the simplest thing in this
package:

    NPV = signedQuantity x contractMultiplier x spot x fx

No schedule, no day count, no discounting, no optionality. It is also the
only W1 instrument this engine **cannot honestly price today**, and this
module exists to say so precisely rather than to leave the gap implicit.

---

**Why a formula this simple is refused.**

Two of its four factors have no source at this boundary.

| Factor | Source | Status |
|---|---|---|
| `signedQuantity` | positions CSV `quantity` | ✅ present |
| `contractMultiplier` | terms / CSV `contractMultiplier` | ✅ present |
| **`spot`** | a market-data input | ❌ **none exists** |
| **`fx`** | a market-data input | ❌ **none exists** |

`engine.integration.market_inputs` registers *flat interest-rate profiles*
and nothing else: an `AssumedProfile` is a single `flat_rate`. There is no
equity spot in it, no FX rate, and no `mode` that supplies either.
`SimulationConfig.equities` is **not** a substitute -- the plan says so
outright (§W1.4) and so does **I-07**: it drives correlated risk-factor
*paths* for a Monte Carlo, and nothing in it takes a signed share count and
returns a position value. It is also in `engine.simulation`, which this
package is forbidden to import.

**The tempting wrong answer is `closingMark`.** The positions extract
carries one, and `quantity x closingMark x contractMultiplier` reproduces
the exporter's own `marketValue` column exactly. That is precisely what
makes it dangerous:

  - it would be an **echo, not a valuation**. The engine would be handing
    back TraderX's own number as though it had priced it, and a
    reconciliation against it would always agree -- proving nothing, while
    looking like independent confirmation;
  - `closingMark` is an *observation at the session cut*, not a curve this
    run was priced against. Publishing it under `npv` with a
    `marketProvenance` derived from the requested rate profile would label
    an observed number with a provenance it does not have;
  - it silently answers a **different question** than every other `npv` in
    this result. The bill and note NPVs are present values off an
    explicitly requested curve; an equity "NPV" taken from the mark is a
    mark. Summing them into one portfolio total would mix two
    incompatible quantities under one heading.

So this module refuses, and names the reason. That is the W0 rule applied
to a case where returning *a* number would have been trivially easy --
which is exactly when the rule earns its keep (working rule 1: "an explicit
`unsupported` is recoverable, a plausible wrong number is not").

---

**What it still does, rather than refusing blankly.**

A refusal that says only "no" is hard to act on, so this module validates
everything it *can* and reports it:

  - it **identifies** the position and echoes the inputs it does have
    (`signedQuantity`, `contractMultiplier`), so a coordinator can see the
    engine understood the row;
  - it applies the multiplier **exactly once** in that echo, and
    `multiplied_quantity` exists so the plan's "multiplier applied exactly
    once" test has something to assert against;
  - it refuses a **malformed** multiplier or quantity distinctly
    (`TERMS_INCOMPLETE`) from the missing spot, because those are different
    problems with different fixes;
  - it reports a **non-reporting-currency** position as
    `FX_SOURCE_NOT_SUPPLIED` rather than `SPOT_SOURCE_NOT_SUPPLIED`, since
    that row needs two things this engine lacks and a consumer fixing only
    the spot would still not get a number.

**Closing this needs a market-data decision, not engine work.** Either
`marketInputs` grows a registered spot/FX surface (the W0.6 contract
extends), or TraderX supplies one in the bundle. Both are outside W1.4,
and **I-18** records that.
"""
from dataclasses import dataclass
from typing import Dict, Optional

from engine.integration.terms import TermsEntry

#: Reason codes. Each names a condition this pricer refuses rather than
#: approximates.
NOT_AN_EQUITY = "NOT_AN_EQUITY"
SPOT_SOURCE_NOT_SUPPLIED = "SPOT_SOURCE_NOT_SUPPLIED"
FX_SOURCE_NOT_SUPPLIED = "FX_SOURCE_NOT_SUPPLIED"
TERMS_INCOMPLETE = "TERMS_INCOMPLETE"

#: The currency this engine reports in. A position in any other currency
#: needs an FX rate, which is a second thing this boundary does not have --
#: reported distinctly so a consumer knows fixing the spot alone is not
#: enough.
REPORTING_CURRENCY = "USD"

#: The instrument type this module owns.
EQUITY = "EQUITY"

#: What a priced equity *would* be, echoed in the refusal so the contract
#: is visible before the pricer exists. Deliberately not `METHOD` -- there
#: is no method, because nothing is computed.
INTENDED_METHOD = "spot-revaluation"


class EquityPricingError(Exception):
    """An equity position could not be priced. Carries a reason code and a
    detail, shaped for a `CalculationOutcome` refusal rather than a stack
    trace -- one unpriceable row must not fail the other 200.

    Its own type rather than a shared one, for the reason **I-17**
    documents: a refusal raised by one module and caught as another's
    escapes the handler entirely and fails the whole bundle.
    """

    def __init__(self, reason: str, detail: str, payload: Optional[Dict] = None):
        self.reason = reason
        self.detail = detail
        #: Everything the engine *could* establish about the row, carried
        #: into the refusal so it is diagnosable rather than merely
        #: negative.
        self.payload = payload or {}
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class EquityPosition:
    """One equity position's *inputs*, validated -- deliberately not its
    value.

    There is no `npv` field, and that absence is the design. A dataclass
    with an `npv` that is always `None`, or always zero, is an invitation
    to read it; this type cannot express a price at all, so no caller can
    accidentally publish one.
    """
    signed_quantity: float
    contract_multiplier: float
    currency: str
    #: `signed_quantity x contract_multiplier` -- the exposure in shares,
    #: which is everything the formula can evaluate without a spot. Named
    #: explicitly so "multiplier applied exactly once" is checkable.
    multiplied_quantity: float
    security: Optional[str] = None

    def to_payload(self) -> Dict:
        """The inputs, echoed into the refusal payload."""
        return {
            "intendedMethod": INTENDED_METHOD,
            "signedQuantity": self.signed_quantity,
            "contractMultiplier": self.contract_multiplier,
            "multipliedQuantity": self.multiplied_quantity,
            "currency": self.currency,
            # Named so a consumer can tell *which* inputs are missing
            # rather than re-deriving it from the reason code.
            "missingInputs": self.missing_inputs,
        }

    @property
    def missing_inputs(self):
        """The formula factors this engine has no source for."""
        missing = ["spot"]
        if self.currency != REPORTING_CURRENCY:
            missing.append("fx")
        return missing


def is_equity(entry: Optional[TermsEntry]) -> bool:
    """Whether the *terms* describe a cash equity position.

    Keyed on `instrumentType`, which is the field that states it. Never on
    the absence of bond columns: a blank `coupon`/`maturityDate` is how the
    CSV represents "not applicable to this row", and inferring an
    instrument from which columns are empty is the blank-reading mistake
    `engine.integration.normalize` exists to prevent.
    """
    if entry is None:
        return False
    return str(entry.instrument_type).upper() == EQUITY


def _position_float(row: Dict, entry: TermsEntry, field: str,
                    default: Optional[float] = None) -> float:
    """Reads a numeric position/terms field, preferring the terms.

    The terms are the reference statement and the CSV is the booking, so
    for a *static* property like the multiplier the terms win when both are
    present -- the same precedence the note pricer uses for its schedule.
    A present-but-unparseable value in either is a refusal, never a
    fallback to the other: that would silently pick whichever source
    happened to parse.
    """
    for source, raw in ((f"terms.{field}", entry.terms.get(field)),
                        (f"positions.{field}", row.get(field))):
        if raw is None or (isinstance(raw, str) and not str(raw).strip()):
            continue
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            raise EquityPricingError(
                TERMS_INCOMPLETE, f"{source}={raw!r} is not a number",
            ) from exc
    if default is not None:
        return default
    raise EquityPricingError(
        TERMS_INCOMPLETE,
        f"{field} is absent from both the instrument terms and the position "
        f"row. It is required to establish the position's size and will not "
        f"be defaulted.",
    )


def read_position(entry: TermsEntry, row: Dict) -> EquityPosition:
    """Validates one equity row's inputs, or refuses.

    **Does not price.** It establishes what the engine knows, so the
    refusal that follows can carry it. Raises `EquityPricingError` for a
    row it cannot even read.
    """
    if not is_equity(entry):
        raise EquityPricingError(
            NOT_AN_EQUITY,
            f"instrumentType={entry.instrument_type!r} is not {EQUITY}, so the "
            f"cash-equity position model does not apply.",
        )

    signed_quantity = _position_float(row, entry, "quantity")
    # Absent multiplier defaults to 1: the overwhelmingly common case for a
    # cash equity, and both the delivered fixture and the terms state it
    # explicitly anyway. A present-but-unparseable one still refuses.
    multiplier = _position_float(row, entry, "contractMultiplier", default=1.0)
    currency = str(
        entry.terms.get("currency") or row.get("currency") or ""
    ).upper()

    return EquityPosition(
        signed_quantity=signed_quantity,
        contract_multiplier=multiplier,
        currency=currency,
        # Applied exactly ONCE, here and nowhere else in this module.
        multiplied_quantity=signed_quantity * multiplier,
        security=row.get("security"),
    )


def price_equity(entry: TermsEntry, row: Dict) -> EquityPosition:
    """**Always raises.** There is no equity pricer at this stage.

    Named `price_equity` deliberately, and kept next to `price_bill` /
    `price_note`, so the absence is visible where a reader looks for the
    pricer rather than discovered by its silence. When a spot source
    exists, this function's body changes and its name and call site do not.

    Raises `EquityPricingError` with `SPOT_SOURCE_NOT_SUPPLIED` (or
    `FX_SOURCE_NOT_SUPPLIED` for a non-reporting-currency position), always
    carrying the inputs it was able to validate.
    """
    position = read_position(entry, row)
    payload = position.to_payload()

    if position.currency != REPORTING_CURRENCY:
        raise EquityPricingError(
            FX_SOURCE_NOT_SUPPLIED,
            f"this position is denominated in {position.currency or '<absent>'} and "
            f"this engine reports in {REPORTING_CURRENCY}, but no FX source is "
            f"available at this boundary -- and no equity spot source is either. "
            f"Valuing it would require assuming both a share price and a currency "
            f"conversion; neither is supplied and neither will be assumed. "
            f"Supplying a spot alone would still not make this row priceable.",
            payload=payload,
        )

    raise EquityPricingError(
        SPOT_SOURCE_NOT_SUPPLIED,
        f"a cash equity is worth signedQuantity x contractMultiplier x spot, and "
        f"this engine has no source for spot. marketInputs registers flat "
        f"interest-rate profiles only; SimulationConfig.equities drives simulated "
        f"risk-factor paths, not position valuation. The position's own "
        f"closingMark is an exported observation at the session cut, not a price "
        f"this run computed -- returning it as an npv would echo TraderX's own "
        f"number back as though the engine had valued it, under a provenance it "
        f"does not have. Closed by supplying a spot source (see I-18), not by "
        f"engine work on this module.",
        payload=payload,
    )

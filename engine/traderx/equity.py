"""
Cash equity positions: validated and refused (I-18).

    NPV = signedQuantity x contractMultiplier x spot x fx

Quantity and multiplier come from the positions CSV / terms; spot and fx have no source at
this boundary (`marketInputs` holds flat rate profiles only, and
the simulation's equity components simulate paths, not position values). So the position is
refused, naming what is missing.

`closingMark` is not used as the spot: quantity x mark x multiplier reproduces the
exporter's own `marketValue`, so it would echo TraderX's number back as a valuation
under a provenance it does not have.

The refusal still carries what could be validated (quantity, multiplier applied once,
currency). A malformed input is `TERMS_INCOMPLETE`; a non-USD position is
`FX_SOURCE_NOT_SUPPLIED`, since a spot alone would not make it priceable.
"""
from dataclasses import dataclass
from typing import Dict, Optional

from engine.traderx.terms import TermsEntry

#: Reason codes for refusals.
NOT_AN_EQUITY = "NOT_AN_EQUITY"
SPOT_SOURCE_NOT_SUPPLIED = "SPOT_SOURCE_NOT_SUPPLIED"
FX_SOURCE_NOT_SUPPLIED = "FX_SOURCE_NOT_SUPPLIED"
TERMS_INCOMPLETE = "TERMS_INCOMPLETE"

#: Reporting currency; any other currency also needs an FX rate.
REPORTING_CURRENCY = "USD"

#: The instrument type this module owns.
EQUITY = "EQUITY"

#: The method a priced equity would use, echoed in the refusal (nothing is computed).
INTENDED_METHOD = "spot-revaluation"


class EquityPricingError(Exception):
    """An equity position could not be priced: reason code, detail and the validated
    inputs, reported as that row's refusal. Its own exception type, so it is caught by the
    equity handler and cannot fail the bundle (I-17)."""

    def __init__(self, reason: str, detail: str, payload: Optional[Dict] = None):
        self.reason = reason
        self.detail = detail
        #: What could be established about the row.
        self.payload = payload or {}
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class EquityPosition:
    """One equity position's validated inputs. It has no `npv` field, so no caller can
    publish a price from it."""
    signed_quantity: float
    contract_multiplier: float
    currency: str
    #: signed_quantity x contract_multiplier, applied exactly once.
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
            # Which inputs are missing.
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
    """Whether the terms say `instrumentType` is EQUITY (never inferred from blank bond
    columns)."""
    if entry is None:
        return False
    return str(entry.instrument_type).upper() == EQUITY


def _position_float(row: Dict, entry: TermsEntry, field: str,
                    default: Optional[float] = None) -> float:
    """A numeric field, from the terms if present, else the position row. An unparseable
    value in either raises (no fallback to the other source); absent in both uses `default`
    or raises."""
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
    """Validate one equity row's inputs (does not price); raises `EquityPricingError` if
    unreadable."""
    if not is_equity(entry):
        raise EquityPricingError(
            NOT_AN_EQUITY,
            f"instrumentType={entry.instrument_type!r} is not {EQUITY}, so the "
            f"cash-equity position model does not apply.",
        )

    signed_quantity = _position_float(row, entry, "quantity")
    # An absent multiplier defaults to 1; an unparseable one raises.
    multiplier = _position_float(row, entry, "contractMultiplier", default=1.0)
    currency = str(
        entry.terms.get("currency") or row.get("currency") or ""
    ).upper()

    return EquityPosition(
        signed_quantity=signed_quantity,
        contract_multiplier=multiplier,
        currency=currency,
        # The multiplier is applied here only.
        multiplied_quantity=signed_quantity * multiplier,
        security=row.get("security"),
    )


def price_equity(entry: TermsEntry, row: Dict) -> EquityPosition:
    """Always raises `EquityPricingError`: `SPOT_SOURCE_NOT_SUPPLIED`, or
    `FX_SOURCE_NOT_SUPPLIED` for a non-USD position, with the validated inputs. Kept next
    to `price_bill`/`price_note` so the gap is visible where the pricer would be."""
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
        "a cash equity is worth signedQuantity x contractMultiplier x spot, and "
        "this engine has no source for spot. marketInputs registers flat "
        "interest-rate profiles only; the simulation's equity components drive "
        "simulated risk-factor paths, not position valuation. The position's own "
        "closingMark is an exported observation at the session cut, not a price "
        "this run computed -- returning it as an npv would echo TraderX's own "
        "number back as though the engine had valued it, under a provenance it "
        "does not have. Closed by supplying a spot source (see I-18), not by "
        "engine work on this module.",
        payload=payload,
    )

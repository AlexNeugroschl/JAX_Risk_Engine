"""
Bundle path in, `RiskResult` out. Wiring only; each step's rules live in its own module:

    load_bundle            verify hashes, parse artifacts
      -> join_terms        attach reference terms, or record their absence
      -> identity_for      identify every row, joined or not
      -> check_conventions refuse what cannot be priced faithfully
      -> normalize_position convert units
      -> price_bill        zero-coupon Treasury: npv
      -> price_note        coupon Treasury: npv and rateSensitivity
      -> price_equity      cash equity: a refusal naming the missing spot (I-18)
      -> RiskResult        per-calculation coverage

Dispatch is on the terms, never the CSV. Equities are dispatched first, since their
`quantity` is a share count, not a face amount. Calculations no pricer produces
(`rateGamma`, `theta`, a bill's `rateSensitivity`) are `unsupported`; vega is
`not-applicable` for non-optional instruments; `varEs` is portfolio-level, so
`not-applicable` per item.

Pricing needs explicitly requested market inputs; without them nothing is priced and no
curve is substituted. `accruedInterest` is answered regardless, as a unit conversion of the
exported value (see `engine.integration.normalize`).

A convention refusal takes precedence over "no pricer": it is the more specific, more
actionable fact.
"""
from typing import Dict, Optional, Sequence, Tuple

import ORE

from engine.integration.bill import BillPricingError, is_bill, price_bill
from engine.integration.bundle import Bundle, load_bundle
from engine.integration.capabilities import ENGINE_VERSION
from engine.integration.conventions import ConventionRefusal, check_conventions
from engine.integration.equity import (
    EquityPricingError,
    is_equity,
    price_equity,
)
from engine.integration.identity import identity_for
from engine.integration.market_inputs import (
    ENGINE_RISK_MEASURE,
    MarketInputs,
    MarketInputsNotSupplied,
    resolve_market_inputs,
)
from engine.integration.normalize import (
    CONVERTED as NORMALIZE_CONVERTED,
    MAPPING_VERSION,
    NO_TERMS_ARTIFACT,
    STRUCTURAL_ZERO as NORMALIZE_STRUCTURAL_ZERO,
    NormalizationError,
    normalize_position,
)
from engine.integration.note import (
    ACCRUAL_EXPORTED,
    DEFAULT_FRACTION_DECIMALS,
    RATE_BUMP,
    SENSITIVITY_METHOD,
    NotePricingError,
    is_note,
    price_note,
    rate_sensitivity,
)
from engine.integration.result import (
    CALCULATIONS,
    CalculationOutcome,
    ItemResult,
    RiskResult,
    sensitivity_payload,
)
from engine.integration.terms import JoinedRow, join_terms

#: Reason for a calculation refused because no pricer for it exists yet (closed by build
#: work, unlike a convention refusal).
NO_PRICER_AT_THIS_STAGE = "NO_PRICER_AT_THIS_STAGE"

#: Calculations meaningless without optionality: `not-applicable`, not a coverage gap.
_NON_OPTIONAL_INSTRUMENTS = ("TREASURY", "SWAP", "EQUITY")
_OPTIONALITY_CALCULATIONS = ("vega",)

#: `varEs` is portfolio-level: `not-applicable` per item, so coverage still sums.
_PORTFOLIO_LEVEL_CALCULATIONS = ("varEs",)


def _instrument_type(joined: JoinedRow) -> str:
    """Instrument type from the terms, else the CSV column, else the source artifact (the
    contracts extract carries only OTC swaps)."""
    if joined.entry is not None:
        return joined.entry.instrument_type
    csv_type = joined.row.get("instrumentType") or joined.row.get("productType")
    if csv_type:
        return csv_type
    return "SWAP" if joined.source == "contracts" else "UNKNOWN"


def _refused_calculations(
    refusal: ConventionRefusal, instrument_type: str,
    accrued: Optional[CalculationOutcome] = None,
) -> Dict[str, CalculationOutcome]:
    """Outcomes for a refused item: `unsupported` with the refusal's reason, except where
    `not-applicable` genuinely holds, and `accruedInterest`, which keeps its own verdict
    (a unit conversion, unaffected by conventions)."""
    detail = refusal.detail
    outcomes = {}
    for name in CALCULATIONS:
        if name == "accruedInterest" and accrued is not None:
            outcomes[name] = accrued
        elif name in _PORTFOLIO_LEVEL_CALCULATIONS:
            outcomes[name] = CalculationOutcome.not_applicable(
                "VaR/ES is a portfolio-level statistic, not a per-item one"
            )
        elif name in _OPTIONALITY_CALCULATIONS and instrument_type in _NON_OPTIONAL_INSTRUMENTS:
            outcomes[name] = CalculationOutcome.not_applicable(
                f"{instrument_type} has no optionality, so vega is not defined for it"
            )
        else:
            outcomes[name] = CalculationOutcome.unsupported(
                reason=refusal.reason, detail=detail,
                # The missing terms travel with each refused calculation.
                payload={"missingTerms": list(refusal.missing_terms)} if refusal.missing_terms else None,
            )
    return outcomes


def _unpriced_calculations(
    instrument_type: str, accrued: Optional[CalculationOutcome],
    priced: Optional[Dict[str, CalculationOutcome]] = None,
) -> Dict[str, CalculationOutcome]:
    """Outcomes for an item whose conventions are supported: `accruedInterest` as
    normalized, whatever a pricer produced (`priced`), `not-applicable` where it holds, and
    `unsupported` (`NO_PRICER_AT_THIS_STAGE`) for everything else."""
    priced = priced or {}
    outcomes = {}
    for name in CALCULATIONS:
        if name == "accruedInterest" and accrued is not None:
            outcomes[name] = accrued
        elif name in priced:
            outcomes[name] = priced[name]
        elif name in _PORTFOLIO_LEVEL_CALCULATIONS:
            outcomes[name] = CalculationOutcome.not_applicable(
                "VaR/ES is a portfolio-level statistic, not a per-item one"
            )
        elif name in _OPTIONALITY_CALCULATIONS and instrument_type in _NON_OPTIONAL_INSTRUMENTS:
            outcomes[name] = CalculationOutcome.not_applicable(
                f"{instrument_type} has no optionality, so vega is not defined for it"
            )
        else:
            outcomes[name] = CalculationOutcome.unsupported(
                reason=NO_PRICER_AT_THIS_STAGE,
                detail=(
                    f"conventions for this {instrument_type} are supported, but no "
                    f"pricer for this calculation has been delivered yet. Closed by "
                    f"W1 build work; no external decision is required."
                ),
            )
    return outcomes


def _exported_accrued_fraction(joined: JoinedRow) -> Optional[float]:
    """The row's `accruedInterestFraction`, or `None` if blank or unparseable (normalization
    reports a malformed value as `failed`)."""
    raw = joined.row.get("accruedInterestFraction")
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _accrual_source(provenance: Optional[str]) -> Optional[str]:
    """`accrualSource` label for a normalized accrued value, matching the note NPV
    payload's vocabulary: `exported-fraction` for a converted value, `structural-zero`
    kept distinct for a bill, else `None`."""
    if provenance == NORMALIZE_CONVERTED:
        # Converted from the extract's own accruedInterestFraction.
        return ACCRUAL_EXPORTED
    if provenance == NORMALIZE_STRUCTURAL_ZERO:
        return NORMALIZE_STRUCTURAL_ZERO
    return None


def _fraction_decimals(entry) -> int:
    """The exporter's declared accrual precision: a v2 entry's
    `accrualBasis.fractionDecimals` (validated by `terms`), else
    `DEFAULT_FRACTION_DECIMALS` (6, as the positions preamble states)."""
    basis = getattr(entry, "accrual_basis", None)
    if basis is None:
        return DEFAULT_FRACTION_DECIMALS
    return basis.fraction_decimals


def _priced_outcomes(
    joined: JoinedRow, market: Optional[MarketInputs], valuation_date: Optional[str],
) -> Dict[str, CalculationOutcome]:
    """Every calculation a pricer answers for this row (empty if none). Bills and notes
    need a resolved `market`; without one nothing is priced. Dispatch is on the terms."""
    if joined.entry is None or joined.source != "positions":
        return {}

    # Equities first: their refusal does not depend on a curve, and `quantity` is a share
    # count, not a face amount.
    if is_equity(joined.entry):
        return _equity_outcomes(joined)

    if market is None or market.profile is None:
        return {}

    is_a_bill = is_bill(joined.entry)
    is_a_note = is_note(joined.entry)
    if not (is_a_bill or is_a_note):
        return {}

    signed_face = _parse_signed_face(joined)
    if signed_face is None:
        return {}

    if is_a_bill:
        return _bill_outcomes(joined, signed_face, market, valuation_date)
    return _note_outcomes(joined, signed_face, market, valuation_date)


def _bill_outcomes(
    joined: JoinedRow, signed_face: float, market: MarketInputs,
    valuation_date: Optional[str],
) -> Dict[str, CalculationOutcome]:
    """A bill's outcome: `npv` only."""
    try:
        priced = price_bill(
            joined.entry, signed_face, _ore_date(valuation_date), market.profile,
        )
    except BillPricingError as exc:
        # An explicit refusal (matured, incomplete terms): unsupported, not failed.
        return {"npv": CalculationOutcome.unsupported(reason=exc.reason, detail=exc.detail)}
    except (ValueError, TypeError, ArithmeticError, RuntimeError) as exc:
        # Attempted and broke: `failed`, confined to this row. RuntimeError is needed: SWIG
        # raises QuantLib errors (e.g. an impossible date) as RuntimeError, which would
        # otherwise abort the whole bundle.
        return {"npv": CalculationOutcome.failed(
            reason="PRICING_FAILED", detail=f"{type(exc).__name__}: {exc}",
        )}

    return {"npv": CalculationOutcome.ok(priced.npv, payload=priced.to_payload())}


def _note_outcomes(
    joined: JoinedRow, signed_face: float, market: MarketInputs,
    valuation_date: Optional[str],
) -> Dict[str, CalculationOutcome]:
    """A note's outcomes: `npv` and `rateSensitivity`. A refusal refuses both. The
    reconciliation tolerance uses the terms' declared `fractionDecimals` when present."""
    valuation = _ore_date(valuation_date)
    exported_accrued = _exported_accrued_fraction(joined)
    fraction_decimals = _fraction_decimals(joined.entry)

    try:
        priced = price_note(
            joined.entry, signed_face, valuation, market.profile, exported_accrued,
            fraction_decimals=fraction_decimals,
        )
        sensitivity = rate_sensitivity(
            joined.entry, signed_face, valuation, market.profile, exported_accrued,
        )
    except NotePricingError as exc:
        refusal = CalculationOutcome.unsupported(reason=exc.reason, detail=exc.detail)
        return {"npv": refusal, "rateSensitivity": refusal}
    except (ValueError, TypeError, ArithmeticError, RuntimeError) as exc:
        failure = CalculationOutcome.failed(
            reason="PRICING_FAILED", detail=f"{type(exc).__name__}: {exc}",
        )
        return {"npv": failure, "rateSensitivity": failure}

    return {
        "npv": CalculationOutcome.ok(priced.npv, payload=priced.to_payload()),
        "rateSensitivity": CalculationOutcome.ok(
            sensitivity,
            payload=sensitivity_payload(
                method=SENSITIVITY_METHOD,
                derivative="dNPV/dZeroRate",
                # A flat profile can only shift in parallel.
                shocked_factor="zero-curve-parallel",
                bump=RATE_BUMP,
                value=sensitivity,
                currency=joined.row.get("currency") or "",
            ),
        ),
    }


def _equity_outcomes(joined: JoinedRow) -> Dict[str, CalculationOutcome]:
    """An equity's outcome: an explicit `npv` refusal (`SPOT_SOURCE_NOT_SUPPLIED` or
    `FX_SOURCE_NOT_SUPPLIED`) with the validated inputs, rather than falling through to
    `NO_PRICER_AT_THIS_STAGE`, since sending a spot would fix it. Needs no `market`."""
    try:
        price_equity(joined.entry, joined.row)
    except EquityPricingError as exc:
        refusal = CalculationOutcome.unsupported(
            reason=exc.reason, detail=exc.detail,
            # Validated inputs travel with the refusal.
            payload=exc.payload or None,
        )
        return {"npv": refusal}
    except (ValueError, TypeError, ArithmeticError, RuntimeError) as exc:
        return {"npv": CalculationOutcome.failed(
            reason="PRICING_FAILED", detail=f"{type(exc).__name__}: {exc}",
        )}

    # price_equity always raises; fail loudly if it ever returns.
    raise AssertionError(
        "price_equity returned instead of raising; W1.4 ships no equity "
        "pricer, so this path should be unreachable"
    )


def _parse_signed_face(joined: JoinedRow) -> Optional[float]:
    """The row's signed face amount, or `None` if unreadable (normalization reports that)."""
    try:
        return normalize_position(joined).signed_face_amount
    except NormalizationError:
        return None


def _ore_date(iso: Optional[str]) -> ORE.Date:
    """Bundle session date (ISO) -> `ORE.Date`, the valuation date of the run."""
    if not iso:
        raise ValueError("bundle carries no session date to value against")
    year, month, day = (int(part) for part in str(iso).split("-"))
    return ORE.Date(day, month, year)


def _accrued_outcome(joined: JoinedRow) -> Tuple[Optional[CalculationOutcome], Optional[str], Optional[str]]:
    """Normalize a position row and turn its accrued interest into an outcome:
    `(outcome, currency, mapping_version)`. A malformed row yields `failed`, not an
    exception."""
    if joined.source != "positions":
        return None, joined.row.get("currency"), None

    try:
        normalized = normalize_position(joined)
    except NormalizationError as exc:
        return (
            CalculationOutcome.failed(
                reason="NORMALIZATION_FAILED", detail=str(exc),
            ),
            joined.row.get("currency"),
            MAPPING_VERSION,
        )

    accrued = normalized.accrued_interest
    if accrued.is_ok:
        payload = {
            "provenance": accrued.provenance,
            "currency": normalized.currency,
            # Echoed in the source unit, for reconciliation against the extract.
            "observedCleanPrice": normalized.observed_clean_price,
            "signedFaceAmount": normalized.signed_face_amount,
        }
        # Same label as the note NPV payload's `accrualSource`.
        accrual_source = _accrual_source(accrued.provenance)
        if accrual_source is not None:
            payload["accrualSource"] = accrual_source
        outcome = CalculationOutcome.ok(accrued.value, payload=payload)
    elif accrued.reason == NO_TERMS_ARTIFACT:
        outcome = CalculationOutcome.unavailable(
            reason=accrued.reason,
            detail=(
                "accruedInterestFraction was blank and this bundle carries no "
                "instrument-terms artifact, so the blank is uninterpretable: it "
                "cannot be distinguished from a bill's structural zero without "
                "terms. Closed by re-exporting as a v2 bundle, not by engine work."
            ),
        )
    else:
        outcome = CalculationOutcome.unavailable(
            reason=accrued.reason,
            detail=(
                "accruedInterestFraction was blank and the instrument's terms do "
                "not establish it as structurally zero. A blank is not 0.0 for a "
                "coupon-bearing instrument."
            ),
        )
    return outcome, normalized.currency, normalized.mapping_version


def _build_item(
    joined: JoinedRow, cluster_epoch: str,
    market: Optional[MarketInputs] = None, valuation_date: Optional[str] = None,
) -> ItemResult:
    identity = identity_for(joined.source, joined.row, cluster_epoch)
    instrument_type = _instrument_type(joined)

    accrued, currency, mapping_version = _accrued_outcome(joined)
    refusal = check_conventions(joined)

    if refusal is not None:
        # A refused row is never priced.
        return ItemResult(
            identity=identity,
            calculations=_refused_calculations(refusal, instrument_type, accrued),
            currency=currency or joined.row.get("currency"),
            mapping_version=mapping_version,
            refusal=refusal.to_dict(),
        )

    priced = _priced_outcomes(joined, market, valuation_date)
    return ItemResult(
        identity=identity,
        calculations=_unpriced_calculations(instrument_type, accrued, priced),
        currency=currency or joined.row.get("currency"),
        mapping_version=mapping_version,
    )


def price_bundle(bundle_or_path, market_inputs: Optional[Dict] = None) -> RiskResult:
    """Run the full path over a `Bundle` or bundle path and return its `RiskResult`.

    Raises `BundleIntegrityError` or `TermsJoinError` when the input itself cannot be
    trusted; an unpriceable item comes back as a refusal instead.

    `market_inputs` (e.g. `{"mode": "assumed-profile", "assumedProfileId":
    "flat-3pct-v1"}`) is resolved by `resolve_market_inputs`, which raises
    `MarketInputsNotSupplied` (failing the whole job) rather than substituting a curve.
    Omitted, nothing is priced, `marketProvenance` is null (not "observed"), and a warning
    says so.
    """
    bundle = bundle_or_path if isinstance(bundle_or_path, Bundle) else load_bundle(bundle_or_path)
    joined = join_terms(bundle)

    warnings = []
    if not bundle.has_terms:
        warnings.append(
            "bundle is v1 and carries no instrument-terms artifact: every "
            "instrument requiring reference terms is unsupported."
        )

    # The bundle's own market-data declaration, surfaced as a warning.
    bundle_market_status = (bundle.manifest.get("marketInputs") or {}).get("status")
    if bundle_market_status == "NOT_SUPPLIED":
        warnings.append(
            "bundle declares marketInputs.status=NOT_SUPPLIED: it carries no observed "
            "market data. Any pricing must therefore name an assumed profile "
            "explicitly, and its results will be labelled marketProvenance='assumed'."
        )

    resolved = None
    market_provenance = None
    measure = None
    if market_inputs is not None:
        # Raises for anything unresolvable; no fallback.
        resolved = resolve_market_inputs(market_inputs)
        market_provenance = resolved.market_provenance
        # The measure is stated only when a market basis exists.
        measure = ENGINE_RISK_MEASURE
    else:
        warnings.append(
            "no marketInputs were requested: this result was computed against no "
            "curve at all, and marketProvenance is null rather than 'observed'. "
            "Legal only because this delivery stage (W0) prices nothing."
        )

    # Items are built after market inputs resolve; an unresolvable request has already
    # raised, so no partial result is published.
    items = tuple(
        _build_item(row, bundle.cluster_epoch, resolved, bundle.session_date)
        for row in joined.rows
    )

    return RiskResult(
        bundle_id=bundle.bundle_id,
        cluster_epoch=bundle.cluster_epoch,
        session_date=bundle.session_date,
        valuation_time=bundle.valuation_time,
        items=items,
        mapping_version=MAPPING_VERSION,
        engine_version=ENGINE_VERSION,
        # None when no curve was consulted.
        market_provenance=market_provenance,
        market_inputs=resolved.to_dict() if resolved is not None else None,
        measure=measure,
        warnings=tuple(warnings),
    )

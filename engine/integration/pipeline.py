"""
The W0 composition: bundle path in, `RiskResult` out.

This is the "★ first real result" of the plan's execution order (§6) -- the
SOFR `CONVENTION_NOT_SUPPORTED` output, which needs no pricer, only
W0.1/0.2/0.4/0.5/0.7. Everything here is wiring; each step's judgement lives
in its own module.

    load_bundle      W0.1  verify hashes, parse artifacts
      -> join_terms  W0.2  attach reference terms, or record their absence
      -> identity_for W0.7 identify every row, joined or not
      -> check_conventions W0.4  refuse what cannot be faithfully priced
      -> normalize_position W0.3 convert units for what survives
      -> RiskResult  W0.5  per-calculation coverage

**W0 computes nothing, and says so per calculation.** Every outcome is
`unsupported`, `unavailable` or `not-applicable` -- never `ok`, never
`failed`. `failed` would mean something was attempted and errored; nothing
is attempted here. That distinction is what lets a coordinator tell "this
engine does not do that yet" from "this engine tried and broke", which are
different operational responses.

**`accruedInterest` is the exception, and deliberately so.** It is the one
calculation W0 can answer honestly, because it is a *unit conversion of an
exported value*, not a model output: the extract supplies
`accruedInterestFraction` and the terms supply enough to interpret it. So a
note's accrued interest comes back `ok`, a bill's comes back `ok` at a
structural zero, and a coupon-bearing row with a blank field comes back
`unavailable` -- the W0.3 table, reported rather than assumed. Everything
requiring a curve or a model stays `unsupported`.

**Ordering of the two refusals matters.** A row can be refused for its
conventions (W0.4) *and* lack a pricer (W0 has none). The convention refusal
wins and is reported, because it is the more specific and more actionable
fact: a coordinator learning `CONVENTION_NOT_SUPPORTED` with 13 named terms
can act on it, while `no pricer` tells it only to wait.
"""
from typing import Dict, Optional, Sequence, Tuple

from engine.integration.bundle import Bundle, load_bundle
from engine.integration.capabilities import ENGINE_VERSION
from engine.integration.conventions import ConventionRefusal, check_conventions
from engine.integration.identity import identity_for
from engine.integration.market_inputs import (
    ENGINE_RISK_MEASURE,
    MarketInputsNotSupplied,
    resolve_market_inputs,
)
from engine.integration.normalize import (
    MAPPING_VERSION,
    NO_TERMS_ARTIFACT as NORMALIZE_NO_TERMS_ARTIFACT,
    NormalizationError,
    normalize_position,
)
from engine.integration.result import (
    CALCULATIONS,
    CalculationOutcome,
    ItemResult,
    RiskResult,
)
from engine.integration.terms import JoinedRow, join_terms

#: Reason code for a calculation that is refused because W0 ships no pricer.
#: Distinct from a convention refusal: this one is closed by W1 build work,
#: with no external decision needed.
NO_PRICER_AT_THIS_STAGE = "NO_PRICER_AT_THIS_STAGE"

#: Calculations that are meaningless for an instrument with no optionality.
#: Reported `not-applicable`, which does NOT count against coverage -- vega
#: on a vanilla swap or a Treasury is not a gap (plan §W0.5).
_NON_OPTIONAL_INSTRUMENTS = ("TREASURY", "SWAP")
_OPTIONALITY_CALCULATIONS = ("vega",)

#: `varEs` is a portfolio-level statistic, not a per-item one. Reported
#: `not-applicable` per item rather than silently omitted, so the coverage
#: block still sums to the item count.
_PORTFOLIO_LEVEL_CALCULATIONS = ("varEs",)


def _instrument_type(joined: JoinedRow) -> str:
    """Best available instrument type for a row.

    Prefers the terms entry (authoritative), falls back to the CSV column,
    and finally to the row's source artifact. A contracts row with no terms
    is still a swap booking -- the contracts extract carries only OTC
    derivatives -- so the fallback is sound rather than a guess.
    """
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
    """Every calculation for an item this engine refuses to price.

    Even here, `not-applicable` is applied where it genuinely holds: a
    refused Treasury still has no vega. Marking those `unsupported` would
    inflate the gap counts with calculations that were never owed.

    **`accruedInterest` keeps its own answer even under a refusal.** It is
    a unit conversion of an exported value, not a model output, so a
    convention refusal has no bearing on it -- and the W0.3 table gives a
    precise verdict the refusal would otherwise flatten. A v1 bundle's
    blank accrued field is `unavailable` ("uninterpretable without terms"),
    which tells the coordinator a resend with terms would fix it;
    reporting `unsupported` there would wrongly imply engine work was
    needed. The two statuses point at different remedies, so the more
    specific one wins.
    """
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
                # The offending terms travel on the calculation that was
                # refused, not only in a summary block, so a consumer
                # reading one calculation sees why it has no number.
                payload={"missingTerms": list(refusal.missing_terms)} if refusal.missing_terms else None,
            )
    return outcomes


def _unpriced_calculations(
    instrument_type: str, accrued: Optional[CalculationOutcome],
) -> Dict[str, CalculationOutcome]:
    """Calculations for an item whose conventions are fine but which W0 has
    no pricer for. `accruedInterest` is answered for real (see this module's
    docstring); everything model-driven is `unsupported`."""
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
                reason=NO_PRICER_AT_THIS_STAGE,
                detail=(
                    f"conventions for this {instrument_type} are supported, but this "
                    f"delivery stage (W0) implements contract and refusal machinery "
                    f"only and ships no pricer. Closed by W1 build work; no external "
                    f"decision is required."
                ),
            )
    return outcomes


def _accrued_outcome(joined: JoinedRow) -> Tuple[Optional[CalculationOutcome], Optional[str], Optional[str]]:
    """Normalizes a position row and turns its accrued interest into an
    outcome. Returns `(outcome, currency, mapping_version)`.

    A malformed row produces `failed` rather than an exception: one broken
    row must not cost the other 200 their results, and `failed` is the
    honest status for something that was attempted and errored.
    """
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
        outcome = CalculationOutcome.ok(
            accrued.value,
            payload={
                "provenance": accrued.provenance,
                "currency": normalized.currency,
                # Echoed in its source unit so a consumer can reconcile
                # against the extract without re-deriving the conversion.
                "observedCleanPrice": normalized.observed_clean_price,
                "signedFaceAmount": normalized.signed_face_amount,
            },
        )
    elif accrued.reason == NORMALIZE_NO_TERMS_ARTIFACT:
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


def _build_item(joined: JoinedRow, cluster_epoch: str) -> ItemResult:
    identity = identity_for(joined.source, joined.row, cluster_epoch)
    instrument_type = _instrument_type(joined)

    accrued, currency, mapping_version = _accrued_outcome(joined)
    refusal = check_conventions(joined)

    if refusal is not None:
        return ItemResult(
            identity=identity,
            calculations=_refused_calculations(refusal, instrument_type, accrued),
            currency=currency or joined.row.get("currency"),
            mapping_version=mapping_version,
            refusal=refusal.to_dict(),
        )

    return ItemResult(
        identity=identity,
        calculations=_unpriced_calculations(instrument_type, accrued),
        currency=currency or joined.row.get("currency"),
        mapping_version=mapping_version,
    )


def price_bundle(bundle_or_path, market_inputs: Optional[Dict] = None) -> RiskResult:
    """Runs the full W0 path over a bundle and returns its `RiskResult`.

    Accepts a loaded `Bundle` or a path to one. Raises
    `BundleIntegrityError` if the bundle does not verify, and
    `TermsJoinError` if its terms artifact is structurally unusable --
    both of which mean the *input* cannot be trusted, as distinct from an
    item that cannot be priced, which comes back as a refusal.

    `market_inputs` is the W0.6 request block, e.g.
    `{"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}`. When
    supplied it is resolved through
    `engine.integration.market_inputs.resolve_market_inputs`, which
    **raises `MarketInputsNotSupplied` rather than substituting a curve**
    for anything it cannot resolve -- an unregistered profile id, a
    `package` mode with no package, a missing mode. That failure fails the
    whole job, because market data is the shared basis every price is
    measured against rather than a property of one instrument.

    **Omitting `market_inputs` is legal at W0 and only at W0**, because W0
    prices nothing and therefore uses no curve. The result then reports
    `marketProvenance: null` and carries a warning saying so -- *not*
    `"observed"`, which would claim a market basis that was never
    consulted. W1's pricers require the block, and the day a calculation
    needs a discount factor is the day this argument stops being optional.

    Despite the name, **this prices nothing at W0.** The name is the one
    the W1 pricers will keep, so callers written against this boundary do
    not change when pricing arrives.
    """
    bundle = bundle_or_path if isinstance(bundle_or_path, Bundle) else load_bundle(bundle_or_path)
    joined = join_terms(bundle)

    items = tuple(_build_item(row, bundle.cluster_epoch) for row in joined.rows)

    warnings = []
    if not bundle.has_terms:
        warnings.append(
            "bundle is v1 and carries no instrument-terms artifact: every "
            "instrument requiring reference terms is unsupported."
        )

    # The bundle's own declaration, surfaced rather than silently accepted:
    # all three delivered YU18 fixtures say NOT_SUPPLIED, which is expected
    # (TraderX exports positions and terms, not curves -- plan §1) but is
    # exactly the condition W1 will have to fail on.
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
        # Raises MarketInputsNotSupplied for anything unresolvable. No
        # fallback, no default curve, no nearest-match on the profile id.
        resolved = resolve_market_inputs(market_inputs)
        market_provenance = resolved.market_provenance
        # Stated only when a market basis exists -- an exposure measure for
        # a run that consulted no curve would be a label with nothing under
        # it.
        measure = ENGINE_RISK_MEASURE
    else:
        warnings.append(
            "no marketInputs were requested: this result was computed against no "
            "curve at all, and marketProvenance is null rather than 'observed'. "
            "Legal only because this delivery stage (W0) prices nothing."
        )

    return RiskResult(
        bundle_id=bundle.bundle_id,
        cluster_epoch=bundle.cluster_epoch,
        session_date=bundle.session_date,
        valuation_time=bundle.valuation_time,
        items=items,
        mapping_version=MAPPING_VERSION,
        engine_version=ENGINE_VERSION,
        # None when no curve was consulted. Deliberately not "assumed" --
        # asserting a provenance for a computation that never happened
        # would be its own small lie.
        market_provenance=market_provenance,
        market_inputs=resolved.to_dict() if resolved is not None else None,
        measure=measure,
        warnings=tuple(warnings),
    )

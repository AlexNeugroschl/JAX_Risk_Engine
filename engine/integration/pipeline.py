"""
The composition: bundle path in, `RiskResult` out.

Everything here is wiring; each step's judgement lives in its own module.

    load_bundle      W0.1  verify hashes, parse artifacts
      -> join_terms  W0.2  attach reference terms, or record their absence
      -> identity_for W0.7 identify every row, joined or not
      -> check_conventions W0.4  refuse what cannot be faithfully priced
      -> normalize_position W0.3 convert units for what survives
      -> price_bill  W1.2  NPV for a zero-coupon Treasury
      -> price_note  W1.3  NPV + rateSensitivity for a coupon-bearing one
      -> price_equity W1.4 a REFUSAL naming the missing market input
      -> RiskResult  W0.5  per-calculation coverage

**What computes, and what still refuses.** W0 computed nothing: every
outcome was `unsupported`, `unavailable` or `not-applicable`. W1.2 added
one real number -- a zero-coupon Treasury's `npv`. W1.3 adds a
coupon-bearing Treasury's `npv` **and** its `rateSensitivity`, both priced
against an explicitly requested curve. Everything else is unchanged, and
the statuses stay precise: `failed` still means something was attempted
and errored, which is a different operational fact from "this engine does
not do that yet".

**Answering one calculation does not make the rest answerable.** A priced
bill returns `npv` alone; a priced note adds `rateSensitivity` and stops
there. `rateGamma` and `theta` are `unsupported` for both. Reporting a
zero, or omitting them, would be exactly the silent approximation this
boundary exists to prevent.

**Dispatch is on the terms, never on the CSV.** Three branches now exist,
so "which model applies" is a real decision rather than a single `if`.
`is_bill` and `is_note` both key on `couponFrequency` and the presence of
an explicit schedule -- so a note can never be routed into the bill's
single-cashflow model, which is the dangerous misprice W1.2's tests
already pin and which a second pricer makes newly reachable. `is_equity`
keys on `instrumentType`, and is dispatched **first**: an equity's
`quantity` is a signed share count rather than a currency face, so it must
not reach `_parse_signed_face`.

**W1.4 adds a branch that deliberately produces no number.** A cash equity
is `signedQuantity x multiplier x spot x fx`, and this boundary has a
source for neither spot nor fx -- `marketInputs` registers flat
interest-rate profiles only. The row's own `closingMark` is not a
substitute: returning it would echo TraderX's own number back as an engine
valuation under a provenance it does not have. So the equity branch returns
an explicit `SPOT_SOURCE_NOT_SUPPLIED` (or `FX_SOURCE_NOT_SUPPLIED`)
refusal carrying the inputs it *could* validate, rather than falling
through to `NO_PRICER_AT_THIS_STAGE` -- the two point at different
remedies, and only "send a spot" is actionable by the coordinator. See
`engine.integration.equity` and **I-18**.

**Pricing requires market inputs, with no fallback.** Omitting the
`marketInputs` block was legal at W0 because nothing was priced; it is
still legal, but now it means nothing *gets* priced. No curve is ever
substituted -- see `engine.integration.market_inputs`.

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
    MAPPING_VERSION,
    NO_TERMS_ARTIFACT as NORMALIZE_NO_TERMS_ARTIFACT,
    NormalizationError,
    normalize_position,
)
from engine.integration.note import (
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

#: Reason code for a calculation that is refused because W0 ships no pricer.
#: Distinct from a convention refusal: this one is closed by W1 build work,
#: with no external decision needed.
NO_PRICER_AT_THIS_STAGE = "NO_PRICER_AT_THIS_STAGE"

#: Calculations that are meaningless for an instrument with no optionality.
#: Reported `not-applicable`, which does NOT count against coverage -- vega
#: on a vanilla swap or a Treasury is not a gap (plan §W0.5).
_NON_OPTIONAL_INSTRUMENTS = ("TREASURY", "SWAP", "EQUITY")
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
    priced: Optional[Dict[str, CalculationOutcome]] = None,
) -> Dict[str, CalculationOutcome]:
    """Calculations for an item whose conventions are fine.
    `accruedInterest` is answered for real (see this module's docstring),
    every calculation a pricer produced is taken from `priced`, and
    everything still model-driven is `unsupported`.

    **A calculation being answered does not make the others answerable.**
    A priced bill returns `npv` alone; a priced note adds
    `rateSensitivity` and nothing further. `rateGamma` and `theta` stay
    `unsupported` for both, because no pricer here produces them and
    reporting a zero or an absence would be the silent-approximation
    failure this boundary exists to prevent.
    """
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
    """The row's `accruedInterestFraction`, or `None` when it is blank.

    `None` is a legitimate state, not an error: the blank-accrual
    compatibility fixture is exactly this, and `price_note` handles it by
    reporting the recomputed value under an explicit
    `recomputed-schedule` label. A malformed value is left to the
    normalization path, which already reports it as `failed` naming the
    field.
    """
    raw = joined.row.get("accruedInterestFraction")
    if raw is None or not str(raw).strip():
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _priced_outcomes(
    joined: JoinedRow, market: Optional[MarketInputs], valuation_date: Optional[str],
) -> Dict[str, CalculationOutcome]:
    """Every calculation a pricer can answer for this row.

    Returns a (possibly empty) mapping from calculation name to outcome;
    anything absent from it falls through to `NO_PRICER_AT_THIS_STAGE`.
    An empty mapping is not a failure -- it is an instrument this delivery
    stage does not price yet.

    **Requires an explicitly resolved `market`** for the bond branches.
    Without one they return nothing rather than pricing against a default:
    W0's "no marketInputs requested" path is legal precisely because
    nothing was priced, and the moment something *is* priced, a curve is
    mandatory.

    **Dispatch is on the terms, never on the CSV** -- see this module's
    docstring.
    """
    if joined.entry is None or joined.source != "positions":
        return {}

    # An equity is dispatched BEFORE the market-input check and before the
    # face-amount parse: its refusal does not depend on a curve (a rate
    # profile is not a spot), and its `quantity` is a signed share count
    # rather than a currency face.
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
    """W1.2's outcomes: `npv` alone.

    Unchanged by W1.3 deliberately. A bill's `rateSensitivity` is still
    `unsupported` -- W1.3 delivers a note sensitivity, and extending the
    claim to the bill without a test that earns it would be exactly the
    overclaim working rule 4 names.
    """
    try:
        priced = price_bill(
            joined.entry, signed_face, _ore_date(valuation_date), market.profile,
        )
    except BillPricingError as exc:
        # A refusal this pricer states explicitly (matured, incomplete
        # terms). `unsupported`, not `failed`: nothing errored, the engine
        # declined.
        return {"npv": CalculationOutcome.unsupported(reason=exc.reason, detail=exc.detail)}
    except (ValueError, TypeError, ArithmeticError) as exc:
        # Something was genuinely attempted and broke. `failed` is the
        # honest status, and one bad row must not cost the rest theirs.
        return {"npv": CalculationOutcome.failed(
            reason="PRICING_FAILED", detail=f"{type(exc).__name__}: {exc}",
        )}

    return {"npv": CalculationOutcome.ok(priced.npv, payload=priced.to_payload())}


def _note_outcomes(
    joined: JoinedRow, signed_face: float, market: MarketInputs,
    valuation_date: Optional[str],
) -> Dict[str, CalculationOutcome]:
    """W1.3's outcomes: `npv` **and** `rateSensitivity`.

    The note is the first instrument here to answer more than one
    calculation, and the pair is deliberate: a bond price without a rate
    sensitivity tells a consumer what it is worth but nothing about what
    moves it, and the sensitivity is cheap once the schedule is built.

    **A refusal refuses both.** When `price_note` declines, the same
    reason is reported for the sensitivity: a sensitivity of a price the
    engine would not publish is meaningless, and returning one would imply
    a valuation that was explicitly refused.
    """
    valuation = _ore_date(valuation_date)
    exported_accrued = _exported_accrued_fraction(joined)

    try:
        priced = price_note(
            joined.entry, signed_face, valuation, market.profile, exported_accrued,
        )
        sensitivity = rate_sensitivity(
            joined.entry, signed_face, valuation, market.profile, exported_accrued,
        )
    except NotePricingError as exc:
        refusal = CalculationOutcome.unsupported(reason=exc.reason, detail=exc.detail)
        return {"npv": refusal, "rateSensitivity": refusal}
    except (ValueError, TypeError, ArithmeticError) as exc:
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
                # A flat profile has one rate, so the only shift it can
                # express is a parallel one. Naming the factor "parallel"
                # rather than a pillar keeps the claim to what the curve
                # can actually distinguish.
                shocked_factor="zero-curve-parallel",
                bump=RATE_BUMP,
                value=sensitivity,
                currency=joined.row.get("currency") or "",
            ),
        ),
    }


def _equity_outcomes(joined: JoinedRow) -> Dict[str, CalculationOutcome]:
    """W1.4's outcome: an `npv` refusal that names the missing input.

    **This is the only pricer branch that returns no number, deliberately.**
    A cash equity is `signedQuantity x multiplier x spot x fx`, and this
    boundary has no source for the last two. The row's own `closingMark`
    is not a substitute -- returning it would echo TraderX's own number
    back as an engine valuation, under a provenance it does not have. See
    `engine.integration.equity`.

    It returns an explicit refusal rather than `{}` (which would fall
    through to `NO_PRICER_AT_THIS_STAGE`) because the two say different
    things. `NO_PRICER_AT_THIS_STAGE` means "engine work is scheduled";
    `SPOT_SOURCE_NOT_SUPPLIED` means "send a spot and this prices". Only
    the second is actionable by the coordinator, and it is the true one.

    **It does not require `market`.** Unlike the bond branches, this
    refusal is correct whether or not a curve was requested -- a rate
    profile is not an equity spot, so having one changes nothing about
    this row.
    """
    try:
        price_equity(joined.entry, joined.row)
    except EquityPricingError as exc:
        refusal = CalculationOutcome.unsupported(
            reason=exc.reason, detail=exc.detail,
            # The validated inputs travel with the refusal, so a consumer
            # can confirm the engine read the position correctly even
            # though it would not value it.
            payload=exc.payload or None,
        )
        return {"npv": refusal}
    except (ValueError, TypeError, ArithmeticError) as exc:
        return {"npv": CalculationOutcome.failed(
            reason="PRICING_FAILED", detail=f"{type(exc).__name__}: {exc}",
        )}

    # Unreachable: `price_equity` always raises. Guarded rather than
    # assumed, so the day it gains a real pricer this fails loudly here
    # instead of silently returning no outcome.
    raise AssertionError(
        "price_equity returned instead of raising; W1.4 ships no equity "
        "pricer, so this path should be unreachable"
    )


def _parse_signed_face(joined: JoinedRow) -> Optional[float]:
    """The row's signed face amount, or `None` if it cannot be read.

    Deliberately tolerant: a row whose quantity will not parse is left to
    the normalization path, which already reports it as `failed` with a
    field-level message. Duplicating that judgement here would produce two
    different errors for one cause.
    """
    try:
        return normalize_position(joined).signed_face_amount
    except NormalizationError:
        return None


def _ore_date(iso: Optional[str]) -> ORE.Date:
    """Bundle session date -> `ORE.Date`, the valuation date every price in
    the run is measured at."""
    if not iso:
        raise ValueError("bundle carries no session date to value against")
    year, month, day = (int(part) for part in str(iso).split("-"))
    return ORE.Date(day, month, year)


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


def _build_item(
    joined: JoinedRow, cluster_epoch: str,
    market: Optional[MarketInputs] = None, valuation_date: Optional[str] = None,
) -> ItemResult:
    identity = identity_for(joined.source, joined.row, cluster_epoch)
    instrument_type = _instrument_type(joined)

    accrued, currency, mapping_version = _accrued_outcome(joined)
    refusal = check_conventions(joined)

    if refusal is not None:
        # Pricing is not attempted for a refused row, deliberately. The
        # convention refusal already says the engine cannot represent this
        # instrument faithfully; pricing it anyway would produce exactly the
        # confident wrong number the refusal exists to prevent.
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
    """Runs the full path over a bundle and returns its `RiskResult`.

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

    **W1 prices through this same entry point.** A zero-coupon Treasury
    returns an `npv` (W1.2), a coupon-bearing one an `npv` and a
    `rateSensitivity` (W1.3), and a cash equity an explicit
    `SPOT_SOURCE_NOT_SUPPLIED` refusal (W1.4). Callers written against the
    W0 boundary did not change when pricing arrived.
    """
    bundle = bundle_or_path if isinstance(bundle_or_path, Bundle) else load_bundle(bundle_or_path)
    joined = join_terms(bundle)

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

    # Items are built AFTER market inputs resolve, because pricing needs the
    # curve. An unresolvable request raises above and fails the whole job --
    # no item is built, and no partial result is published (see
    # engine.integration.market_inputs on why market data fails the job
    # rather than refusing an item).
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
        # None when no curve was consulted. Deliberately not "assumed" --
        # asserting a provenance for a computation that never happened
        # would be its own small lie.
        market_provenance=market_provenance,
        market_inputs=resolved.to_dict() if resolved is not None else None,
        measure=measure,
        warnings=tuple(warnings),
    )

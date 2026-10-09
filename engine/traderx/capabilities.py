"""
The capability document: the supported product x convention x calculation matrix, so a
coordinator can tell before submitting whether a bundle is priceable.

Derived from `engine.traderx.conventions` and `result`, never hand-maintained, so it
cannot promise more than the code does; the tests pin the derivation. Known limitations
that change what a consumer should do are advertised with it.
"""
from typing import Dict

from engine.traderx.conventions import (
    SUPPORTED_FLOAT_INDICES,
    SUPPORTED_INSTRUMENT_TYPES,
    SUPPORTED_OVERNIGHT_COMPOUNDING,
    SUPPORTED_SWAP_DAY_COUNTS,
)
from engine.traderx.market_inputs import (
    ASSUMED_PROFILES,
    ENGINE_RISK_MEASURE,
    MEASURES,
    MODES,
)
from engine.traderx.normalize import MAPPING_VERSION
from engine.traderx.result import CALCULATIONS, STATUSES
from engine.traderx.schema_version import (
    CAPABILITY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
)
from engine.traderx.terms import (
    SUPPORTED_ACCRUAL_BASIS_SCHEMAS,
    SUPPORTED_TERMS_SCHEMAS,
)

#: Bundle schemas the ingestion path accepts.
from engine.traderx.bundle import SUPPORTED_BUNDLE_SCHEMAS

ENGINE_VERSION = "0.1.0"

#: Delivery stage of this build (see the integration plan).
DELIVERY_STAGE = "W1.6"

#: What computes a number today, per instrument type and shape: the source of the
#: `products` block. A bill answers `npv`; a note `npv` and `rateSensitivity`. Kept per
#: shape so a bill sensitivity is not advertised. Nothing offers `rateGamma`/`theta`;
#: corporate bonds are refused (I-07); a v1 bundle prices nothing (without terms the shape
#: cannot be established).
PRICED_CALCULATIONS_BY_SHAPE: Dict[str, Dict[str, tuple]] = {
    "TREASURY": {
        "zero-coupon": ("npv",),
        "coupon-bearing": ("npv", "rateSensitivity"),
    },
}

#: Union over shapes per instrument type, derived from the table above.
PRICED_CALCULATIONS: Dict[str, tuple] = {
    instrument_type: tuple(
        sorted({calc for calcs in shapes.values() for calc in calcs})
    )
    for instrument_type, shapes in PRICED_CALCULATIONS_BY_SHAPE.items()
}

#: Calculations blocked on a market input rather than a missing pricer, as
#: {instrument type: {calculation: reason}}, so a coordinator knows it can close the gap by
#: sending data. Today only equity `npv` (needs spot, and fx if non-USD; I-18).
BLOCKED_ON_MARKET_INPUT = {
    "EQUITY": {
        "npv": {
            "reason": "SPOT_SOURCE_NOT_SUPPLIED",
            "requires": ["spot", "fx (non-USD positions only)"],
            "detail": (
                "a cash equity needs a spot price, and marketInputs registers "
                "flat interest-rate profiles only. The position's own closingMark "
                "is an exported observation, not a price this engine computed, so "
                "it is not substituted."
            ),
        },
    },
}

#: Limitations that change what a consumer should do, by id in docs/planning/known-issues.md (a
#: subset of the register).
KNOWN_LIMITATIONS = (
    {
        "id": "I-04",
        "status": "OPEN",
        "summary": (
            "A seasoned swap needs the fixings of its coupons fixed before the as-of date, "
            "which the position export does not carry; such a swap is refused "
            "(MissingFixingError) rather than priced on an assumed fixing."
        ),
        "blockedOn": "historical published fixings, which no current input source supplies",
    },
    {
        "id": "I-05",
        "status": "OPEN",
        "summary": (
            "No faithful USD-SOFR / ACT-360 construction. Such bookings are REFUSED "
            "with CONVENTION_NOT_SUPPORTED rather than routed through the generic "
            "term-IBOR builder."
        ),
        "blockedOn": "D03/D04 convention agreement",
    },
    {
        "id": "I-07",
        "status": "OPEN",
        "summary": (
            "Partial. Both Treasury shapes in a v2 bundle now price: a zero-coupon "
            "bill (W1.2, npv) and a coupon-bearing note (W1.3, npv and "
            "rateSensitivity). An equity position is understood but REFUSED for "
            "want of a spot source (W1.4, SPOT_SOURCE_NOT_SUPPLIED -- see I-18). "
            "A corporate bond and a listed option still have no pricer at all."
        ),
        "blockedOn": (
            "equity needs a spot/FX market-data source (I-18); corporate bonds "
            "need a credit/spread model; listed options need a vol surface"
        ),
    },
    {
        "id": "I-18",
        "status": "OPEN",
        "summary": (
            "No equity spot or FX source. A cash equity position is refused with "
            "SPOT_SOURCE_NOT_SUPPLIED (or FX_SOURCE_NOT_SUPPLIED when its currency "
            "is not USD) rather than valued at its exported closingMark, which "
            "would echo the exporter's own number back as an engine valuation."
        ),
        "blockedOn": (
            "a market-data decision: either marketInputs grows a registered "
            "spot/FX surface, or the bundle supplies one"
        ),
    },
)


def capabilities() -> Dict:
    """The capability document. Priced calculations come from
    `PRICED_CALCULATIONS_BY_SHAPE`; only what has a parity-tested pricer is advertised."""
    return {
        # First, so a consumer can check it understands the document.
        "capabilitySchema": CAPABILITY_SCHEMA_VERSION,
        "engineVersion": ENGINE_VERSION,
        "mappingVersion": MAPPING_VERSION,
        "deliveryStage": DELIVERY_STAGE,
        "stageSummary": (
            "Contract and refusal machinery, plus the Treasury pricers. Bundles "
            "are ingested, hash-verified, joined, normalized and identified. Both "
            "Treasury shapes in a v2 bundle are priced as discounted cashflows "
            "against an explicitly requested curve -- a zero-coupon bill returns "
            "npv, a coupon-bearing note returns npv and a bumped-revaluation "
            "rateSensitivity. A cash equity is understood and identified but "
            "REFUSED: it needs a spot price, and this boundary has no spot or FX "
            "source. Every other calculation is still refused. W1.6 added the "
            "contract interface: instrument-terms v2 with a validated accrualBasis, "
            "versioned result and capability documents with machine-readable JSON "
            "Schema, and HTTP routes under /eod."
        ),
        "bundleSchemas": list(SUPPORTED_BUNDLE_SCHEMAS),
        # Terms schemas are versioned independently of bundle schemas.
        "termsSchemas": list(SUPPORTED_TERMS_SCHEMAS),
        "accrualBasisSchemas": list(SUPPORTED_ACCRUAL_BASIS_SCHEMAS),
        # Published document versions, to pin a validator before submitting.
        "schemas": {
            "resultSchema": RESULT_SCHEMA_VERSION,
            "capabilitySchema": CAPABILITY_SCHEMA_VERSION,
            # Where the JSON Schemas are served.
            "resultSchemaUrl": "/eod/schemas/result",
            "capabilitySchemaUrl": "/eod/schemas/capabilities",
        },
        "calculations": {
            "names": list(CALCULATIONS),
            "statuses": list(STATUSES),
            # Some calculations price; most still refuse.
            "mode": "partial",
        },
        "products": {
            instrument_type: {
                "calculations": list(PRICED_CALCULATIONS.get(instrument_type, ())),
                "priced": bool(PRICED_CALCULATIONS.get(instrument_type)),
                # Per shape (see PRICED_CALCULATIONS_BY_SHAPE).
                "byShape": {
                    shape: list(calcs)
                    for shape, calcs in PRICED_CALCULATIONS_BY_SHAPE.get(
                        instrument_type, {}
                    ).items()
                },
                # Awaiting a market input rather than a release.
                "blockedOnMarketInput": {
                    calc: dict(spec)
                    for calc, spec in BLOCKED_ON_MARKET_INPUT.get(
                        instrument_type, {}
                    ).items()
                },
            }
            for instrument_type in SUPPORTED_INSTRUMENT_TYPES
        },
        "conventions": {
            "swap": {
                "floatIndex": list(SUPPORTED_FLOAT_INDICES),
                "legDayCount": list(SUPPORTED_SWAP_DAY_COUNTS),
                "overnightCompounding": list(SUPPORTED_OVERNIGHT_COMPOUNDING),
            },
        },
        "marketInputs": {
            # An assumed curve must be requested by profile id; there is no fallback.
            "modes": list(MODES),
            "fallbackOnMissingInputs": False,
            # The registered profiles, by id, so a coordinator can pick one
            # Registered profiles, each with the provenance a result against it echoes.
            "assumedProfiles": [
                {
                    "assumedProfileId": profile.profile_id,
                    "description": profile.description,
                    "curveProvenance": profile.provenance().to_dict(),
                }
                for profile in sorted(ASSUMED_PROFILES.values(), key=lambda p: p.profile_id)
            ],
            # `package` mode is advertised but not implemented yet.
            "packageModeImplemented": False,
        },
        "riskMeasure": {
            # Every figure is under the pricing measure, not a loss forecast (I-11).
            "measure": ENGINE_RISK_MEASURE,
            "supported": list(MEASURES),
            # Tail statistics come with convergence diagnostics.
            "tailDiagnostics": ["tailCount", "standardError"],
        },
        "knownLimitations": [dict(limitation) for limitation in KNOWN_LIMITATIONS],
    }

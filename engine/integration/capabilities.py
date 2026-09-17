"""
W0.9 -- the capability matrix.

Reports the supported (product x convention x calculation) matrix, so a
coordinator can determine **before submitting** whether a bundle is
priceable. That is what makes "no silent exclusions" enforceable rather than
aspirational: without it, the only way to discover that a portfolio is
half-refusable is to submit it and read the coverage block afterwards.

**Derived from the allowlist, never hand-maintained.** Every value here
reads out of `engine.integration.conventions` and
`engine.integration.result`. A capability document that is written by hand
drifts from the code it describes, and a *stale* capability document is
worse than none -- it makes a promise the engine no longer keeps. The test
suite pins the derivation rather than the contents, so adding a convention
to the allowlist updates this automatically.

**Known limitations are advertised, not hidden.** The engine's real defects
(the register in `docs/known-issues.md`) are part of its capability surface.
A consumer deciding whether to trust an exposure profile needs to know about
I-04 before submitting, not after reconciling.
"""
from typing import Dict

from engine.integration.conventions import (
    SUPPORTED_FLOAT_INDICES,
    SUPPORTED_INSTRUMENT_TYPES,
    SUPPORTED_OVERNIGHT_COMPOUNDING,
    SUPPORTED_SWAP_DAY_COUNTS,
)
from engine.integration.market_inputs import (
    ASSUMED_PROFILES,
    ENGINE_RISK_MEASURE,
    MEASURES,
    MODES,
)
from engine.integration.normalize import MAPPING_VERSION
from engine.integration.result import CALCULATIONS, STATUSES
from engine.integration.schema_version import (
    CAPABILITY_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
)
from engine.integration.terms import (
    SUPPORTED_ACCRUAL_BASIS_SCHEMAS,
    SUPPORTED_TERMS_SCHEMAS,
)

#: Bundle schemas the ingestion path accepts.
from engine.integration.bundle import SUPPORTED_BUNDLE_SCHEMAS

ENGINE_VERSION = "0.1.0"

#: The delivery stage this build implements. W1 began when the first pricer
#: landed (W1.2, the bill), W1.3 added the note, W1.4 resolved the equity
#: case to a *refusal* rather than a price, and W1.6 made the whole boundary
#: reachable over HTTP with versioned documents. The stage is not "W1"
#: complete -- the portfolio wire-through (W1.5) is still to come -- which
#: is what `stageSummary` spells out.
DELIVERY_STAGE = "W1.6"

#: What actually computes a number today, per instrument type.
#:
#: **The single source of truth for the `products` block**, so the advertised
#: matrix cannot drift from the code: adding a pricer means adding it here,
#: and a consumer reading this document before submitting sees exactly what
#: it will get back.
#:
#: Deliberately narrow, and **stated per instrument shape** rather than per
#: instrument type, because the two Treasury shapes no longer answer the
#: same set. A bill answers `npv` alone (W1.2); a note answers `npv` and
#: `rateSensitivity` (W1.3). Collapsing them into one TREASURY entry would
#: advertise a bill sensitivity that does not exist.
#:
#: It does NOT imply:
#:
#: - that `rateGamma`/`theta` are available for anything -- they are not,
#:   for either shape;
#: - that a corporate bond prices. It is refused (I-07): a
#:   Treasury-discounted corporate is not credit pricing.
#:
#: A v1 bundle also prices nothing, whatever this says: without a terms
#: artifact the engine cannot establish which shape a row IS, and it will
#: not infer that from a coupon column.
PRICED_CALCULATIONS_BY_SHAPE: Dict[str, Dict[str, tuple]] = {
    "TREASURY": {
        "zero-coupon": ("npv",),
        "coupon-bearing": ("npv", "rateSensitivity"),
    },
}

#: The union per instrument type -- what a consumer sees if it does not
#: distinguish the shapes. Derived, never hand-written, so it cannot drift
#: from the per-shape table above.
PRICED_CALCULATIONS: Dict[str, tuple] = {
    instrument_type: tuple(
        sorted({calc for calcs in shapes.values() for calc in calcs})
    )
    for instrument_type, shapes in PRICED_CALCULATIONS_BY_SHAPE.items()
}

#: Calculations this engine understands but cannot compute for want of a
#: **market input**, keyed by instrument type -> {calculation: reason}.
#:
#: **Distinct from "no pricer", and the distinction is the point.** A
#: coordinator reading `NO_PRICER_AT_THIS_STAGE` should wait for a release;
#: one reading `SPOT_SOURCE_NOT_SUPPLIED` should send a spot. Advertising
#: the second as the first would tell them to do nothing when they hold the
#: fix.
#:
#: EQUITY is the only entry today (W1.4). The arithmetic is trivial and
#: fully understood -- `signedQuantity x multiplier x spot x fx` -- and two
#: of those four factors have no source at this boundary. See
#: `engine.integration.equity` and I-18.
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

#: Known limitations a coordinator should weigh before submitting, keyed by
#: their id in `docs/known-issues.md`. Kept deliberately short: these are
#: the ones that change what a consumer should *do*, not the full register.
KNOWN_LIMITATIONS = (
    {
        "id": "I-04",
        "status": "FLAGGED",
        "summary": (
            "Aged swaps are mispriced at every simulated step past first accrual. "
            "t=0 NPV is exact; exposure profiles and any VaR/ES derived from them "
            "carry a known inaccuracy."
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
    """The capability document.

    **What is priced is derived from `PRICED_CALCULATIONS_BY_SHAPE`, not
    asserted here.** At W0 every entry was empty and `priced: False`
    throughout, which was then the honest description. W1.2 priced a
    bill's `npv`; W1.3 adds a note's `npv` and `rateSensitivity`, so
    exactly those appear -- and nothing else does.

    **The per-shape split is load-bearing.** A bill's `rateSensitivity` is
    absent from its own list even though a note now has one: W1.3 earned
    the note sensitivity with a parity test and earned nothing for the
    bill. Advertising a capability the engine has not demonstrated is the
    overclaim this whole integration exists to avoid (plan working rule 4:
    "ORE can represent it" is not "my engine prices it").
    """
    return {
        # W1.6.2: first, so a consumer can decide whether it understands
        # this document before reading anything else in it.
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
        # W1.6.1: advertised separately from `bundleSchemas` because the two
        # are independently versioned -- a v2 bundle may carry either terms
        # version, so collapsing them would misdescribe what is accepted.
        "termsSchemas": list(SUPPORTED_TERMS_SCHEMAS),
        "accrualBasisSchemas": list(SUPPORTED_ACCRUAL_BASIS_SCHEMAS),
        # The versions of the two published documents, so a coordinator can
        # pin its validator before submitting rather than discovering a
        # schema change from a parse failure (W1.6.2).
        "schemas": {
            "resultSchema": RESULT_SCHEMA_VERSION,
            "capabilitySchema": CAPABILITY_SCHEMA_VERSION,
            # Where the machine-readable definitions are served.
            "resultSchemaUrl": "/eod/schemas/result",
            "capabilitySchemaUrl": "/eod/schemas/capabilities",
        },
        "calculations": {
            "names": list(CALCULATIONS),
            "statuses": list(STATUSES),
            # "partial": some calculations price, most still refuse.
            "mode": "partial",
        },
        "products": {
            instrument_type: {
                "calculations": list(PRICED_CALCULATIONS.get(instrument_type, ())),
                "priced": bool(PRICED_CALCULATIONS.get(instrument_type)),
                # Per-shape, because the two Treasury shapes answer
                # different sets and the union above would advertise a
                # bill sensitivity that does not exist.
                "byShape": {
                    shape: list(calcs)
                    for shape, calcs in PRICED_CALCULATIONS_BY_SHAPE.get(
                        instrument_type, {}
                    ).items()
                },
                # Understood, but awaiting a market input rather than a
                # release. Advertised separately from "not priced" so a
                # coordinator can tell which gaps it can close itself.
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
            # Plan §W0.6: an assumed curve must be explicitly requested by
            # profile id; there is no fallback. Advertised so a coordinator
            # knows a submission without market inputs will fail rather than
            # quietly assume something.
            "modes": list(MODES),
            "fallbackOnMissingInputs": False,
            # The registered profiles, by id, so a coordinator can pick one
            # BEFORE submitting rather than discovering the valid set from a
            # rejection. Each carries its own provenance, which is what a
            # result computed against it will echo.
            "assumedProfiles": [
                {
                    "assumedProfileId": profile.profile_id,
                    "description": profile.description,
                    "curveProvenance": profile.provenance().to_dict(),
                }
                for profile in sorted(ASSUMED_PROFILES.values(), key=lambda p: p.profile_id)
            ],
            # `package` is an advertised mode but is not consumable yet --
            # said plainly here rather than discovered via a failed job.
            "packageModeImplemented": False,
        },
        "riskMeasure": {
            # Every risk figure this engine produces is under the pricing
            # measure. A consumer expecting a real-world loss forecast needs
            # to know that before submitting, not after reconciling (I-11).
            "measure": ENGINE_RISK_MEASURE,
            "supported": list(MEASURES),
            # Tail statistics carry their own convergence diagnostics.
            "tailDiagnostics": ["tailCount", "standardError"],
        },
        "knownLimitations": [dict(limitation) for limitation in KNOWN_LIMITATIONS],
    }

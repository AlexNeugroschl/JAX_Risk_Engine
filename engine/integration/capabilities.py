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

#: Bundle schemas the ingestion path accepts.
from engine.integration.bundle import SUPPORTED_BUNDLE_SCHEMAS

ENGINE_VERSION = "0.1.0"

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
        "summary": "No bond, equity or listed-option pricer. W1 adds bill, note and equity.",
        "blockedOn": "nothing; scheduled work",
    },
)


def capabilities() -> Dict:
    """The capability document.

    W0 prices nothing, and this says so plainly: every calculation is
    advertised as `refusal-only`. That is the honest description of an
    engine whose contract machinery works and whose pricers do not exist
    yet -- claiming otherwise here would be the overclaim the whole
    integration is built to avoid (plan working rule 4: "ORE can represent
    it" is not "my engine prices it").
    """
    return {
        "engineVersion": ENGINE_VERSION,
        "mappingVersion": MAPPING_VERSION,
        "deliveryStage": "W0",
        "stageSummary": (
            "Contract and refusal machinery. Bundles are ingested, hash-verified, "
            "joined, normalized and identified; every calculation is refused. No "
            "pricing is performed at this stage."
        ),
        "bundleSchemas": list(SUPPORTED_BUNDLE_SCHEMAS),
        "calculations": {
            "names": list(CALCULATIONS),
            "statuses": list(STATUSES),
            # W0: the machinery is real, the pricers are not.
            "mode": "refusal-only",
        },
        "products": {
            instrument_type: {
                # No product computes anything in W0.
                "calculations": [],
                "priced": False,
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

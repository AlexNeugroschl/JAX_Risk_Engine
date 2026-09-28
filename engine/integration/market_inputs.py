"""
Market-input selection (part of I-11). A job must say where its market data comes from:

```jsonc
"marketInputs": {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}
```

Absent, or `mode: "package"` (not implemented), the whole job fails with
`MARKET_INPUTS_NOT_SUPPLIED`; no curve is ever substituted. Unlike a per-item refusal, a
missing curve leaves nothing to price anything against. TraderX bundles currently carry no
market data, so a delivered bundle prices only when the caller requests an assumed profile.

Assumed profiles are registered and requested by id, so the result records what it was
priced against; the id is part of the workload key. Any result on an assumed curve carries
top-level `marketProvenance: "assumed"`.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

#: Failure code when a job supplies no usable market inputs.
MARKET_INPUTS_NOT_SUPPLIED = "MARKET_INPUTS_NOT_SUPPLIED"

#: `marketInputs.mode` values.
MODE_ASSUMED_PROFILE = "assumed-profile"
MODE_PACKAGE = "package"
MODES = (MODE_ASSUMED_PROFILE, MODE_PACKAGE)

#: `inputOrigin` values. `assumed` (a chosen stand-in for an observation) and `synthetic`
#: (fabricated test data) are kept distinct; both count as not observed.
ORIGIN_OBSERVED = "observed"
ORIGIN_ASSUMED = "assumed"
ORIGIN_MIXED = "mixed"
ORIGIN_SYNTHETIC = "synthetic"
INPUT_ORIGINS = (ORIGIN_OBSERVED, ORIGIN_ASSUMED, ORIGIN_MIXED, ORIGIN_SYNTHETIC)

#: Origins that make a result's `marketProvenance` "assumed".
_NOT_OBSERVED = (ORIGIN_ASSUMED, ORIGIN_MIXED, ORIGIN_SYNTHETIC)

# Risk-measure vocabulary, duplicated from `engine.risk.var_es` because this package must
# not import `engine.risk` (it pulls in JAX). Pinned together by
# `tests/test_integration_market_inputs.py::TestMeasureVocabularyMatchesVarEs`.
MEASURE_RISK_NEUTRAL = "risk-neutral-pricing"
MEASURE_HISTORICAL = "historical-forecast"
MEASURE_STRESS = "deterministic-stress"
MEASURES = (MEASURE_RISK_NEUTRAL, MEASURE_HISTORICAL, MEASURE_STRESS)

#: The measure the engine's simulation runs under (a fact, not a default).
ENGINE_RISK_MEASURE = MEASURE_RISK_NEUTRAL


class MarketInputsNotSupplied(Exception):
    """The job supplied no usable market inputs. Fails the job (see the module docstring)."""

    def __init__(self, detail: str):
        self.reason = MARKET_INPUTS_NOT_SUPPLIED
        self.detail = detail
        super().__init__(f"{MARKET_INPUTS_NOT_SUPPLIED}: {detail}")


@dataclass(frozen=True)
class CurveProvenance:
    """Where one curve's numbers came from, attached to a `ZeroCurveConfig` so assumed and
    observed curves stay distinguishable inside the engine."""
    curve_id: str
    input_origin: str
    construction: str
    #: Hashes of the curve's input data; empty for an assumed profile (a named constant).
    input_hashes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.input_origin not in INPUT_ORIGINS:
            raise ValueError(
                f"unknown inputOrigin {self.input_origin!r}; expected one of "
                f"{list(INPUT_ORIGINS)}"
            )

    @property
    def is_observed(self) -> bool:
        return self.input_origin == ORIGIN_OBSERVED

    def to_dict(self) -> Dict:
        return {
            "curveId": self.curve_id,
            "inputOrigin": self.input_origin,
            "construction": self.construction,
            "inputHashes": list(self.input_hashes),
        }


@dataclass(frozen=True)
class AssumedProfile:
    """A named, versioned assumed curve: flat, a constant rather than a fit."""
    profile_id: str
    description: str
    flat_rate: float
    #: Pillar times the profile is materialized on.
    times: Tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 30.0)

    def provenance(self) -> CurveProvenance:
        return CurveProvenance(
            curve_id=self.profile_id,
            input_origin=ORIGIN_ASSUMED,
            construction="flat-constant",
            # A named constant: the id is its provenance.
            input_hashes=(),
        )

    def rates(self) -> Tuple[float, ...]:
        return tuple(self.flat_rate for _ in self.times)


#: Registered profiles. Never change an existing profile's numbers (results computed
#: against it would become unreproducible under the same id); add a new versioned id.
ASSUMED_PROFILES: Dict[str, AssumedProfile] = {
    profile.profile_id: profile
    for profile in (
        AssumedProfile(
            profile_id="flat-3pct-v1",
            description="Flat 3% continuously-compounded zero curve.",
            flat_rate=0.03,
        ),
    )
}


@dataclass(frozen=True)
class MarketInputs:
    """A resolved market-input selection (failure raises instead)."""
    mode: str
    profile: Optional[AssumedProfile] = None

    @property
    def provenance(self) -> CurveProvenance:
        if self.profile is None:
            raise ValueError("a package-mode MarketInputs has no single curve provenance")
        return self.profile.provenance()

    @property
    def market_provenance(self) -> str:
        """Top-level `marketProvenance` for results computed against this."""
        return market_provenance_of((self.provenance,))

    def to_dict(self) -> Dict:
        payload: Dict = {"mode": self.mode}
        if self.profile is not None:
            payload["assumedProfileId"] = self.profile.profile_id
            payload["curveProvenance"] = self.provenance.to_dict()
        return payload


def market_provenance_of(provenances) -> str:
    """Top-level `marketProvenance`: "observed" only if every curve was observed, else
    "assumed"."""
    provenances = tuple(provenances)
    if not provenances:
        raise ValueError("market_provenance_of needs at least one curve provenance")
    if any(p.input_origin in _NOT_OBSERVED for p in provenances):
        return ORIGIN_ASSUMED
    return ORIGIN_OBSERVED


def resolve_market_inputs(spec: Optional[Dict]) -> MarketInputs:
    """Resolve a `marketInputs` block or raise `MarketInputsNotSupplied`; never
    substitutes a curve."""
    if not spec:
        raise MarketInputsNotSupplied(
            "no marketInputs block was supplied. Market inputs must be requested "
            "explicitly -- there is no default curve, and none will be substituted. "
            f"Supply {{'mode': '{MODE_ASSUMED_PROFILE}', 'assumedProfileId': ...}} "
            f"to price against a named assumed profile, or {{'mode': '{MODE_PACKAGE}', "
            "'package': ...}} to supply observed market data."
        )

    mode = spec.get("mode")
    if mode not in MODES:
        raise MarketInputsNotSupplied(
            f"marketInputs.mode={mode!r} is not one of {list(MODES)}. The mode must "
            f"be stated explicitly; it is not inferred from which other fields are "
            f"present."
        )

    if mode == MODE_PACKAGE:
        package = spec.get("package")
        if not package:
            # Falling back to an assumed curve here would publish an assumption with
            # observed provenance.
            raise MarketInputsNotSupplied(
                "marketInputs.mode='package' was requested but no market data "
                "package was supplied. The job fails rather than falling back to "
                "an assumed curve: a result computed against an assumption must "
                "never be published with observed provenance."
            )
        raise MarketInputsNotSupplied(
            "marketInputs.mode='package' is not supported yet: this engine does not "
            "consume an observed market-data package today (plan §1 -- TraderX "
            "supplies dated observations, curve construction is mine to build). "
            "Request an assumed profile by id instead, which labels the result "
            "honestly as assumed."
        )

    profile_id = spec.get("assumedProfileId")
    if not profile_id:
        raise MarketInputsNotSupplied(
            f"marketInputs.mode='{MODE_ASSUMED_PROFILE}' requires an "
            f"'assumedProfileId'. An assumed curve must be requested BY NAME so the "
            f"result records what it was priced against; an unnamed assumption is "
            f"not reconcilable. Registered: {sorted(ASSUMED_PROFILES)}."
        )

    profile = ASSUMED_PROFILES.get(profile_id)
    if profile is None:
        raise MarketInputsNotSupplied(
            f"unknown assumedProfileId {profile_id!r}. Registered profiles: "
            f"{sorted(ASSUMED_PROFILES)}. An unregistered id is refused rather than "
            f"resolved to a nearest match."
        )

    return MarketInputs(mode=MODE_ASSUMED_PROFILE, profile=profile)

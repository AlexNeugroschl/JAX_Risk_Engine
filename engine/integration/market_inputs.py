"""
W0.6 -- market input selection. Closes part of **I-11**.

**Explicit market-input mode; no silent fallback.** A job must say where its
market data comes from:

```jsonc
"marketInputs": {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}
```

Absent, or `mode: "package"` with no package, the **job fails** with
`MARKET_INPUTS_NOT_SUPPLIED`. It does not fall back to a default curve, a
flat curve, or the last curve it saw.

---

**Why a missing curve fails the job instead of refusing the item.**

Every other gap in this boundary produces a per-item refusal: an unmapped
convention makes *that instrument* `unsupported`, and the rest of the
portfolio still prices. Market inputs are different in kind. They are not a
property of any one instrument -- they are the shared basis every price in
the run is measured against. A run with no curve has nothing to price
*anything* against, so there is no partial result worth publishing, and
publishing one would invite a consumer to reconcile a total that was never
computed.

So this module raises rather than returning a refusal, and
`MarketInputsNotSupplied` is deliberately **not** a `ConventionRefusal`.

**Why every fixture currently hits this path.** All three delivered YU18
bundles carry `marketInputs: {"status": "NOT_SUPPLIED"}` -- TraderX exports
positions and terms, not curves. That is expected and agreed (plan §1,
"Market data: TraderX supplies dated observations; I own curve
construction"). Until they ship a package, the only way to price a delivered
bundle is for the *caller* to request an assumed profile explicitly, by id.

---

**Assumed profiles are named, versioned, and registered -- not passed in.**

`assumedProfileId` resolves against `ASSUMED_PROFILES` below. A caller
cannot hand over arbitrary curve numbers through this path, and that is the
point: an assumed curve has to be a *thing with a name* that appears
verbatim in the result, so "what was this priced against?" has an answer
that survives into the published artifact. Passing a bare rate would make
every assumed run look identical in the output while being different in
fact.

The id is part of the workload key, so two runs against different profiles
never share a cached result.

**Every result computed against any assumed curve carries top-level
`marketProvenance: "assumed"`** -- see `engine.integration.result.RiskResult`.
Top-level, not buried per-curve, because a consumer reading only the summary
must not be able to miss it.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

#: Failure code when a job supplies no usable market inputs.
MARKET_INPUTS_NOT_SUPPLIED = "MARKET_INPUTS_NOT_SUPPLIED"

#: `marketInputs.mode` values.
MODE_ASSUMED_PROFILE = "assumed-profile"
MODE_PACKAGE = "package"
MODES = (MODE_ASSUMED_PROFILE, MODE_PACKAGE)

#: `inputOrigin` values for curve provenance (plan §W0.6).
#:
#: `synthetic` is kept distinct from `assumed` deliberately -- it is open
#: item §7.4 with TraderX. An *assumed* curve is a deliberate modelling
#: choice standing in for an observation; a *synthetic* one is fabricated
#: test data. Both are "not observed", and collapsing them would lose the
#: difference between "we chose this stand-in" and "this number was made up
#: for a fixture".
ORIGIN_OBSERVED = "observed"
ORIGIN_ASSUMED = "assumed"
ORIGIN_MIXED = "mixed"
ORIGIN_SYNTHETIC = "synthetic"
INPUT_ORIGINS = (ORIGIN_OBSERVED, ORIGIN_ASSUMED, ORIGIN_MIXED, ORIGIN_SYNTHETIC)

#: Origins that are not a real observation of the market. A result touched
#: by any of these carries top-level `marketProvenance: "assumed"`.
_NOT_OBSERVED = (ORIGIN_ASSUMED, ORIGIN_MIXED, ORIGIN_SYNTHETIC)

# ---------------------------------------------------------------------
# Risk measure vocabulary (plan §W0.6; part of I-11).
#
# **Duplicated deliberately from `engine.risk.var_es`.** These three
# strings are a CONTRACT vocabulary -- they appear on the wire in every
# published result -- and this package must not import `engine.risk`,
# which pulls in JAX and would break the "imports no pricer" invariant
# that `engine/integration/` is built on (asserted by
# `TestPackageImportsNoPricer`). Copying three constants is the cheaper
# price than inverting that dependency.
#
# `tests/test_integration_market_inputs.py::TestMeasureVocabularyMatchesVarEs`
# pins the two definitions together, so they cannot drift silently.
# ---------------------------------------------------------------------
MEASURE_RISK_NEUTRAL = "risk-neutral-pricing"
MEASURE_HISTORICAL = "historical-forecast"
MEASURE_STRESS = "deterministic-stress"
MEASURES = (MEASURE_RISK_NEUTRAL, MEASURE_HISTORICAL, MEASURE_STRESS)

#: What this engine actually produces: it simulates under the pricing
#: measure. Not a caller-overridable default -- a statement of fact about
#: the simulation. Reporting a risk-neutral exposure where a consumer
#: expects a real-world loss forecast is a category error no amount of
#: numerical accuracy fixes.
ENGINE_RISK_MEASURE = MEASURE_RISK_NEUTRAL


class MarketInputsNotSupplied(Exception):
    """The job supplied no usable market inputs.

    **Fails the job; not a per-item refusal.** See this module's docstring
    for why market data is different in kind from a per-instrument gap.
    """

    def __init__(self, detail: str):
        self.reason = MARKET_INPUTS_NOT_SUPPLIED
        self.detail = detail
        super().__init__(f"{MARKET_INPUTS_NOT_SUPPLIED}: {detail}")


@dataclass(frozen=True)
class CurveProvenance:
    """Where one curve's numbers came from.

    Attached to a `ZeroCurveConfig` so assumed and observed curves are
    distinguishable **inside the engine**, not only at its edges. Before
    this existed, `ZeroCurveConfig` carried times and rates and nothing
    else -- a flat 3% assumption and a bootstrapped market curve were the
    same object, and no downstream code could tell them apart.
    """
    curve_id: str
    input_origin: str
    construction: str
    #: Hashes of the inputs this curve was built from. Empty for an assumed
    #: profile, which is built from a named constant rather than from data.
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
    """A named, versioned assumed-curve profile.

    Flat curves only, today, and the id says so (`flat-3pct-v1`). A profile
    is a *constant*, not a fit -- it stands in for market data rather than
    approximating it, so there is nothing to bootstrap and nothing to hash.
    """
    profile_id: str
    description: str
    flat_rate: float
    #: Pillar times the profile is materialized on. Chosen to span the
    #: maturities the delivered fixtures actually need.
    times: Tuple[float, ...] = (0.0, 1.0, 2.0, 5.0, 10.0, 30.0)

    def provenance(self) -> CurveProvenance:
        return CurveProvenance(
            curve_id=self.profile_id,
            input_origin=ORIGIN_ASSUMED,
            construction="flat-constant",
            # Nothing to hash: a named constant IS its own provenance, and
            # the id is what a consumer reconciles against.
            input_hashes=(),
        )

    def rates(self) -> Tuple[float, ...]:
        return tuple(self.flat_rate for _ in self.times)


#: The registered profiles. `flat-3pct-v1` is the one the plan names.
#:
#: Adding a profile here is a deliberate act: it becomes requestable by id
#: and will appear verbatim in published results. Changing an existing
#: profile's numbers is NOT allowed -- add a new id with a bumped version
#: instead, or every result ever computed against the old one becomes
#: unreproducible while still claiming the same provenance.
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
    """A resolved market-input selection. Reaching this object means the
    job HAS usable market data; the failure path raises instead."""
    mode: str
    profile: Optional[AssumedProfile] = None

    @property
    def provenance(self) -> CurveProvenance:
        if self.profile is None:
            raise ValueError("a package-mode MarketInputs has no single curve provenance")
        return self.profile.provenance()

    @property
    def market_provenance(self) -> str:
        """The top-level label for every result computed against this."""
        return market_provenance_of((self.provenance,))

    def to_dict(self) -> Dict:
        payload: Dict = {"mode": self.mode}
        if self.profile is not None:
            payload["assumedProfileId"] = self.profile.profile_id
            payload["curveProvenance"] = self.provenance.to_dict()
        return payload


def market_provenance_of(provenances) -> str:
    """The top-level `marketProvenance` for a run using these curves.

    `"observed"` only when **every** curve was observed. One assumed curve
    in a portfolio makes the whole result assumed -- a consumer cannot act
    on "mostly observed", and the conservative label is the honest one.
    """
    provenances = tuple(provenances)
    if not provenances:
        raise ValueError("market_provenance_of needs at least one curve provenance")
    if any(p.input_origin in _NOT_OBSERVED for p in provenances):
        return ORIGIN_ASSUMED
    return ORIGIN_OBSERVED


def resolve_market_inputs(spec: Optional[Dict]) -> MarketInputs:
    """Resolves a `marketInputs` request block, or **fails the job**.

    This is the single place the no-silent-fallback rule is enforced. Every
    branch that cannot produce real market data raises
    `MarketInputsNotSupplied`; none of them substitutes a curve.
    """
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
            # The plan's named case: mode says observed data, none arrives.
            # Quietly downgrading to an assumed profile here would be the
            # single most dangerous fallback in the system -- the result
            # would claim observed provenance it does not have.
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

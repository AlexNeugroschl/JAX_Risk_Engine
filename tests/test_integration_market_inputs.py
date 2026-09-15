"""
W0.6 -- market input selection and curve provenance
(`docs/planning/traderx-integration-plan.md` §W0.6). Closes part of I-11.

The plan's named tests: missing market inputs → fail, **no curve
substituted**; assumed profile → provenance on the curve **and** at top
level; `measure` present on every risk figure.

`TestNoSilentFallback` is the one guarding the principle: every branch that
cannot produce real market data must raise, and none may quietly supply a
default curve. A fallback here would be the single most dangerous one in
the system — the result would carry observed provenance it does not have.
"""
import json
from pathlib import Path

import pytest

from engine.integration import price_bundle
from engine.integration.market_inputs import (
    ASSUMED_PROFILES,
    ENGINE_RISK_MEASURE,
    INPUT_ORIGINS,
    MARKET_INPUTS_NOT_SUPPLIED,
    MEASURES,
    MODE_ASSUMED_PROFILE,
    MODE_PACKAGE,
    AssumedProfile,
    CurveProvenance,
    MarketInputs,
    MarketInputsNotSupplied,
    market_provenance_of,
    resolve_market_inputs,
)

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"
ASSUMED = {"mode": MODE_ASSUMED_PROFILE, "assumedProfileId": "flat-3pct-v1"}


class TestNoSilentFallback:
    """Plan §W0.6: 'Absent, or `mode: "package"` with no package → job
    **fails** with `MARKET_INPUTS_NOT_SUPPLIED`.'

    Every test here asserts two things: that it raised, and that no curve
    came back. The second is the substance — a fallback would make these
    pass as "successful" runs.
    """

    def test_absent_block_fails(self):
        with pytest.raises(MarketInputsNotSupplied) as exc:
            resolve_market_inputs(None)
        assert exc.value.reason == MARKET_INPUTS_NOT_SUPPLIED

    def test_empty_block_fails(self):
        with pytest.raises(MarketInputsNotSupplied):
            resolve_market_inputs({})

    def test_package_mode_with_no_package_fails(self):
        """The plan's explicitly named case. Quietly downgrading to an
        assumed profile here would publish a result claiming observed
        provenance it does not have."""
        with pytest.raises(MarketInputsNotSupplied) as exc:
            resolve_market_inputs({"mode": MODE_PACKAGE})
        assert "no market data package was supplied" in exc.value.detail
        assert "never be published with observed provenance" in exc.value.detail

    def test_package_mode_with_a_package_is_still_refused_today(self):
        """`package` is advertised but not implemented. Refused explicitly
        rather than silently ignored -- a consumer must not believe it
        submitted observed data that was then quietly dropped."""
        with pytest.raises(MarketInputsNotSupplied) as exc:
            resolve_market_inputs({"mode": MODE_PACKAGE, "package": {"curves": []}})
        assert "not supported yet" in exc.value.detail

    def test_missing_mode_fails(self):
        with pytest.raises(MarketInputsNotSupplied, match="not one of"):
            resolve_market_inputs({"assumedProfileId": "flat-3pct-v1"})

    def test_unknown_mode_fails(self):
        with pytest.raises(MarketInputsNotSupplied, match="not one of"):
            resolve_market_inputs({"mode": "whatever"})

    def test_mode_is_not_inferred_from_other_fields(self):
        """Supplying an `assumedProfileId` without a `mode` must not be
        read as assumed-profile mode. Inferring the mode from which fields
        happen to be present is exactly the kind of helpfulness this
        contract refuses."""
        with pytest.raises(MarketInputsNotSupplied):
            resolve_market_inputs({"assumedProfileId": "flat-3pct-v1"})

    def test_assumed_mode_without_an_id_fails(self):
        """An unnamed assumption is not reconcilable."""
        with pytest.raises(MarketInputsNotSupplied, match="requires an 'assumedProfileId'"):
            resolve_market_inputs({"mode": MODE_ASSUMED_PROFILE})

    def test_unknown_profile_id_fails(self):
        with pytest.raises(MarketInputsNotSupplied, match="unknown assumedProfileId"):
            resolve_market_inputs({"mode": MODE_ASSUMED_PROFILE, "assumedProfileId": "flat-4pct-v1"})

    def test_unknown_profile_is_not_resolved_to_a_nearest_match(self):
        """A near-miss id must fail, not silently resolve to the one
        registered profile. With exactly one profile registered, 'just use
        the only one' is the tempting shortcut."""
        with pytest.raises(MarketInputsNotSupplied) as exc:
            resolve_market_inputs({"mode": MODE_ASSUMED_PROFILE, "assumedProfileId": "flat-3pct"})
        assert "nearest match" in exc.value.detail

    @pytest.mark.parametrize("spec", [
        None, {}, {"mode": MODE_PACKAGE}, {"mode": MODE_ASSUMED_PROFILE},
        {"mode": MODE_ASSUMED_PROFILE, "assumedProfileId": "nope"},
    ])
    def test_no_branch_ever_returns_a_curve(self, spec):
        """The invariant, stated once over every failing branch: an
        unresolvable request produces an exception, never a MarketInputs."""
        with pytest.raises(MarketInputsNotSupplied):
            result = resolve_market_inputs(spec)
            pytest.fail(f"expected a raise, got {result!r}")

    def test_failure_is_not_a_convention_refusal(self):
        """Market data is the shared basis of the whole run, not a property
        of one instrument. It fails the job rather than refusing an item --
        there is no partial result worth publishing."""
        from engine.integration.conventions import ConventionRefusal
        with pytest.raises(MarketInputsNotSupplied) as exc:
            resolve_market_inputs(None)
        assert not isinstance(exc.value, ConventionRefusal)


class TestAssumedProfileResolves:
    def test_registered_profile_resolves(self):
        resolved = resolve_market_inputs(ASSUMED)
        assert isinstance(resolved, MarketInputs)
        assert resolved.profile.profile_id == "flat-3pct-v1"

    def test_flat_3pct_is_registered_as_the_plan_names_it(self):
        assert "flat-3pct-v1" in ASSUMED_PROFILES
        assert ASSUMED_PROFILES["flat-3pct-v1"].flat_rate == pytest.approx(0.03)

    def test_profile_materializes_a_flat_curve(self):
        profile = ASSUMED_PROFILES["flat-3pct-v1"]
        rates = profile.rates()
        assert len(rates) == len(profile.times)
        assert all(r == pytest.approx(0.03) for r in rates)

    def test_profile_id_is_versioned(self):
        """Changing a profile's numbers under a stable id would make every
        result ever computed against it unreproducible while still claiming
        the same provenance. The version in the id is the guard."""
        for profile_id in ASSUMED_PROFILES:
            assert profile_id.endswith(("-v1", "-v2", "-v3")), profile_id


class TestCurveProvenance:
    """Plan §W0.6: `ZeroCurveConfig` had no metadata field at all, so
    assumed and observed curves were indistinguishable inside the engine."""

    def test_assumed_profile_carries_assumed_origin(self):
        provenance = resolve_market_inputs(ASSUMED).provenance
        assert provenance.input_origin == "assumed"
        assert provenance.is_observed is False

    def test_provenance_names_the_curve_and_its_construction(self):
        provenance = resolve_market_inputs(ASSUMED).provenance
        payload = provenance.to_dict()
        assert payload["curveId"] == "flat-3pct-v1"
        assert payload["construction"] == "flat-constant"
        assert payload["inputHashes"] == []

    def test_unknown_origin_is_rejected(self):
        with pytest.raises(ValueError, match="unknown inputOrigin"):
            CurveProvenance(curve_id="c", input_origin="vibes", construction="x")

    def test_synthetic_is_distinct_from_assumed(self):
        """Open item §7.4 with TraderX. An *assumed* curve is a deliberate
        modelling choice; a *synthetic* one is fabricated test data. Both
        are 'not observed', but collapsing them loses the difference."""
        assert "synthetic" in INPUT_ORIGINS
        assert "assumed" in INPUT_ORIGINS

    def test_zero_curve_config_accepts_provenance(self):
        """The field the plan says was missing entirely."""
        from engine.simulation.market_model import ZeroCurveConfig
        provenance = resolve_market_inputs(ASSUMED).provenance
        curve = ZeroCurveConfig(times=[0.0, 1.0], rates=[0.03, 0.03], provenance=provenance)
        assert curve.provenance.input_origin == "assumed"

    def test_zero_curve_config_provenance_is_optional_and_additive(self):
        """The 69 existing construction sites must be unaffected. Both the
        positional and keyword forms keep working, and provenance defaults
        to None -- which is honestly 'unstated', not 'observed'."""
        from engine.simulation.market_model import ZeroCurveConfig
        assert ZeroCurveConfig([0.0, 1.0], [0.03, 0.03]).provenance is None
        assert ZeroCurveConfig(times=[0.0], rates=[0.03]).provenance is None

    def test_provenance_does_not_reach_the_simulation_math(self):
        """It is metadata. `ZeroCurve.from_config` reads times and rates
        only, so attaching provenance cannot move a number."""
        import jax.numpy as jnp
        from engine.models.hull_white import ZeroCurve
        from engine.simulation.market_model import ZeroCurveConfig

        provenance = resolve_market_inputs(ASSUMED).provenance
        plain = ZeroCurve.from_config(ZeroCurveConfig(times=[0.0, 1.0], rates=[0.03, 0.04]))
        tagged = ZeroCurve.from_config(
            ZeroCurveConfig(times=[0.0, 1.0], rates=[0.03, 0.04], provenance=provenance)
        )
        assert jnp.array_equal(plain.pillar_rates, tagged.pillar_rates)
        assert jnp.array_equal(plain.pillar_times, tagged.pillar_times)


class TestTopLevelMarketProvenance:
    """Plan §W0.6: 'Every result computed against any assumed curve carries
    top-level `marketProvenance: "assumed"`.'"""

    def test_assumed_run_is_labelled_at_top_level(self):
        result = price_bundle(FIXTURES / "note" / "v2", ASSUMED)
        assert result.market_provenance == "assumed"
        assert result.is_assumed is True

    def test_label_is_in_the_serialized_result(self):
        """Top-level, so a consumer reading only the summary cannot miss
        it."""
        payload = price_bundle(FIXTURES / "note" / "v2", ASSUMED).to_dict()
        assert payload["marketProvenance"] == "assumed"

    def test_provenance_appears_on_the_curve_and_at_top_level(self):
        """The plan's named test: provenance on curve **and** top level --
        both, not either."""
        payload = price_bundle(FIXTURES / "note" / "v2", ASSUMED).to_dict()
        assert payload["marketProvenance"] == "assumed"
        assert payload["marketInputs"]["curveProvenance"]["inputOrigin"] == "assumed"

    def test_one_assumed_curve_makes_the_whole_result_assumed(self):
        """A consumer cannot act on 'mostly observed'; the conservative
        label is the honest one."""
        observed = CurveProvenance("c1", "observed", "bootstrap-v2", ("sha256:a",))
        assumed = CurveProvenance("c2", "assumed", "flat-constant")
        assert market_provenance_of([observed]) == "observed"
        assert market_provenance_of([observed, assumed]) == "assumed"

    @pytest.mark.parametrize("origin", ("assumed", "mixed", "synthetic"))
    def test_every_not_observed_origin_labels_the_result_assumed(self, origin):
        provenance = CurveProvenance("c", origin, "x")
        assert market_provenance_of([provenance]) == "assumed"

    def test_empty_provenance_list_is_rejected(self):
        """'No curves' is not 'observed'."""
        with pytest.raises(ValueError, match="at least one"):
            market_provenance_of([])

    def test_no_market_inputs_yields_null_not_observed(self):
        """W0 prices nothing, so omitting market inputs is legal -- but the
        result must say `null`, never `observed`. Claiming a market basis
        that was never consulted is the exact failure this labels against.
        """
        result = price_bundle(FIXTURES / "note" / "v2")
        assert result.market_provenance is None
        assert result.market_provenance != "observed"

    def test_omitting_market_inputs_warns(self):
        result = price_bundle(FIXTURES / "note" / "v2")
        assert any("no curve at all" in w for w in result.warnings)


class TestMeasureIsReported:
    """Plan §W0.6: '`measure` present on every risk figure.' Part of I-11:
    'Risk measure unlabelled'."""

    def test_measure_is_reported_when_a_market_basis_exists(self):
        result = price_bundle(FIXTURES / "note" / "v2", ASSUMED)
        assert result.measure == ENGINE_RISK_MEASURE

    def test_engine_measure_is_risk_neutral(self):
        """This engine simulates under the pricing measure. It does not
        produce a real-world loss forecast, and must not be read as one."""
        assert ENGINE_RISK_MEASURE == "risk-neutral-pricing"

    def test_measure_is_in_the_serialized_result(self):
        payload = price_bundle(FIXTURES / "note" / "v2", ASSUMED).to_dict()
        assert payload["measure"] == "risk-neutral-pricing"

    def test_measure_is_null_when_no_curve_was_consulted(self):
        """An exposure measure for a run that used no curve would be a
        label with nothing under it."""
        assert price_bundle(FIXTURES / "note" / "v2").measure is None

    def test_measure_vocabulary(self):
        assert MEASURES == (
            "risk-neutral-pricing", "historical-forecast", "deterministic-stress",
        )


class TestMeasureVocabularyMatchesVarEs:
    """The measure constants are deliberately duplicated: this package must
    not import `engine.risk.var_es`, which pulls in JAX and would break the
    'imports no pricer' invariant.

    Duplication is only safe if it cannot drift, which is what this pins.
    """

    def test_definitions_agree(self):
        from engine.risk import var_es
        assert MEASURES == var_es.RISK_MEASURES
        assert ENGINE_RISK_MEASURE == var_es.ENGINE_RISK_MEASURE


class TestSerializedShape:
    def test_result_is_json_serializable_with_market_inputs(self):
        payload = price_bundle(FIXTURES / "note" / "v2", ASSUMED).to_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_market_inputs_echoes_what_was_requested(self):
        """'What was this priced against?' must be answerable from the
        published result alone, without re-deriving it from the request."""
        payload = price_bundle(FIXTURES / "note" / "v2", ASSUMED).to_dict()
        assert payload["marketInputs"]["assumedProfileId"] == "flat-3pct-v1"
        assert payload["marketInputs"]["mode"] == MODE_ASSUMED_PROFILE

    def test_market_inputs_is_null_when_none_requested(self):
        payload = price_bundle(FIXTURES / "note" / "v2").to_dict()
        assert payload["marketInputs"] is None


class TestBundleDeclaredMarketStatus:
    """Every delivered YU18 fixture declares
    `marketInputs.status: NOT_SUPPLIED`. That is expected (TraderX exports
    positions and terms, not curves) but is exactly the condition W1 will
    have to fail on, so it is surfaced rather than passed over."""

    @pytest.mark.parametrize("case", ("bill", "note", "sofr"))
    @pytest.mark.parametrize("version", ("v1", "v2"))
    def test_fixture_declares_not_supplied(self, case, version):
        from engine.integration.bundle import load_bundle
        bundle = load_bundle(FIXTURES / case / version)
        assert bundle.manifest["marketInputs"]["status"] == "NOT_SUPPLIED"

    def test_pipeline_warns_about_it(self):
        result = price_bundle(FIXTURES / "note" / "v2", ASSUMED)
        assert any("NOT_SUPPLIED" in w for w in result.warnings)

    def test_warning_says_pricing_must_name_a_profile(self):
        result = price_bundle(FIXTURES / "note" / "v2", ASSUMED)
        warning = next(w for w in result.warnings if "NOT_SUPPLIED" in w)
        assert "assumed" in warning


class TestCapabilitiesAdvertisesMarketInputs:
    """W0.9 must agree with what W0.6 actually enforces -- a coordinator
    has to be able to pick a profile BEFORE submitting."""

    def test_no_fallback_is_advertised(self):
        from engine.integration import capabilities
        assert capabilities()["marketInputs"]["fallbackOnMissingInputs"] is False

    def test_registered_profiles_are_advertised(self):
        from engine.integration import capabilities
        advertised = {
            p["assumedProfileId"] for p in capabilities()["marketInputs"]["assumedProfiles"]
        }
        assert advertised == set(ASSUMED_PROFILES)

    def test_advertised_profiles_are_actually_resolvable(self):
        """The matrix must not promise a profile the resolver rejects."""
        from engine.integration import capabilities
        for entry in capabilities()["marketInputs"]["assumedProfiles"]:
            resolved = resolve_market_inputs({
                "mode": MODE_ASSUMED_PROFILE,
                "assumedProfileId": entry["assumedProfileId"],
            })
            assert resolved.profile.profile_id == entry["assumedProfileId"]

    def test_package_mode_is_advertised_as_unimplemented(self):
        """Advertised as a mode but flagged unimplemented, so the
        discovery is not via a failed job."""
        from engine.integration import capabilities
        assert capabilities()["marketInputs"]["packageModeImplemented"] is False

    def test_risk_measure_is_advertised(self):
        from engine.integration import capabilities
        assert capabilities()["riskMeasure"]["measure"] == "risk-neutral-pricing"
        assert capabilities()["riskMeasure"]["tailDiagnostics"] == ["tailCount", "standardError"]

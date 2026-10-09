"""
Market-input selection and curve provenance (`engine.traderx.market_inputs`, part of
I-11): missing market inputs fail with no curve substituted; an assumed profile's
provenance appears on the curve and at the top level; `measure` is reported.
"""
import json
from pathlib import Path

import pytest

from engine.traderx import price_bundle
from engine.traderx.market_inputs import (
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
    """Every unresolvable request raises `MARKET_INPUTS_NOT_SUPPLIED` and no curve comes
    back (a fallback would make these look like successful runs)."""

    def test_absent_block_fails(self):
        with pytest.raises(MarketInputsNotSupplied) as exc:
            resolve_market_inputs(None)
        assert exc.value.reason == MARKET_INPUTS_NOT_SUPPLIED

    def test_empty_block_fails(self):
        with pytest.raises(MarketInputsNotSupplied):
            resolve_market_inputs({})

    def test_package_mode_with_no_package_fails(self):
        """`package` mode with no package fails (downgrading to an assumed profile would
        claim observed provenance)."""
        with pytest.raises(MarketInputsNotSupplied) as exc:
            resolve_market_inputs({"mode": MODE_PACKAGE})
        assert "no market data package was supplied" in exc.value.detail
        assert "never be published with observed provenance" in exc.value.detail

    def test_package_mode_with_a_package_is_still_refused_today(self):
        """`package` mode is not implemented and is refused explicitly."""
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
        """An `assumedProfileId` without a `mode` is not read as assumed-profile mode."""
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
        """A near-miss id fails rather than resolving to the one registered profile."""
        with pytest.raises(MarketInputsNotSupplied) as exc:
            resolve_market_inputs({"mode": MODE_ASSUMED_PROFILE, "assumedProfileId": "flat-3pct"})
        assert "nearest match" in exc.value.detail

    @pytest.mark.parametrize("spec", [
        None, {}, {"mode": MODE_PACKAGE}, {"mode": MODE_ASSUMED_PROFILE},
        {"mode": MODE_ASSUMED_PROFILE, "assumedProfileId": "nope"},
    ])
    def test_no_branch_ever_returns_a_curve(self, spec):
        """No failing branch returns a `MarketInputs`."""
        with pytest.raises(MarketInputsNotSupplied):
            result = resolve_market_inputs(spec)
            pytest.fail(f"expected a raise, got {result!r}")

    def test_failure_is_not_a_convention_refusal(self):
        """Missing market data fails the job (it is shared by every item), not a
        convention refusal of one item."""
        from engine.traderx.conventions import ConventionRefusal
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
        """Profile ids are versioned, so a profile's numbers never change under one id."""
        for profile_id in ASSUMED_PROFILES:
            assert profile_id.endswith(("-v1", "-v2", "-v3")), profile_id


class TestCurveProvenance:
    """Curve provenance, carried on `ZeroCurveConfig`."""

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
        """Assumed (a chosen stand-in) and synthetic (fabricated test data) are distinct."""
        assert "synthetic" in INPUT_ORIGINS
        assert "assumed" in INPUT_ORIGINS

    def test_zero_curve_config_accepts_provenance(self):
        """`ZeroCurveConfig` accepts provenance."""
        from engine.market_data.market import ZeroCurveConfig
        provenance = resolve_market_inputs(ASSUMED).provenance
        curve = ZeroCurveConfig(times=[0.0, 1.0], rates=[0.03, 0.03], provenance=provenance)
        assert curve.provenance.input_origin == "assumed"

    def test_zero_curve_config_provenance_is_optional_and_additive(self):
        """Provenance is optional and defaults to None ("unstated", not "observed")."""
        from engine.market_data.market import ZeroCurveConfig
        assert ZeroCurveConfig([0.0, 1.0], [0.03, 0.03]).provenance is None
        assert ZeroCurveConfig(times=[0.0], rates=[0.03]).provenance is None

    def test_provenance_does_not_reach_the_simulation_math(self):
        """Provenance is metadata; `ZeroCurve.from_config` reads only times and rates."""
        import jax.numpy as jnp
        from engine.market_data.curves import ZeroCurve
        from engine.market_data.market import ZeroCurveConfig

        provenance = resolve_market_inputs(ASSUMED).provenance
        plain = ZeroCurve.from_config(ZeroCurveConfig(times=[0.0, 1.0], rates=[0.03, 0.04]))
        tagged = ZeroCurve.from_config(
            ZeroCurveConfig(times=[0.0, 1.0], rates=[0.03, 0.04], provenance=provenance)
        )
        assert jnp.array_equal(plain.pillar_rates, tagged.pillar_rates)
        assert jnp.array_equal(plain.pillar_times, tagged.pillar_times)


class TestTopLevelMarketProvenance:
    """Results on an assumed curve carry top-level `marketProvenance: "assumed"`."""

    def test_assumed_run_is_labelled_at_top_level(self):
        result = price_bundle(FIXTURES / "note" / "v2", ASSUMED)
        assert result.market_provenance == "assumed"
        assert result.is_assumed is True

    def test_label_is_in_the_serialized_result(self):
        """The label is in the serialized result."""
        payload = price_bundle(FIXTURES / "note" / "v2", ASSUMED).to_dict()
        assert payload["marketProvenance"] == "assumed"

    def test_provenance_appears_on_the_curve_and_at_top_level(self):
        """Provenance is on the curve and at the top level."""
        payload = price_bundle(FIXTURES / "note" / "v2", ASSUMED).to_dict()
        assert payload["marketProvenance"] == "assumed"
        assert payload["marketInputs"]["curveProvenance"]["inputOrigin"] == "assumed"

    def test_one_assumed_curve_makes_the_whole_result_assumed(self):
        """One assumed curve makes the whole result assumed."""
        observed = CurveProvenance("c1", "observed", "bootstrap-v2", ("sha256:a",))
        assumed = CurveProvenance("c2", "assumed", "flat-constant")
        assert market_provenance_of([observed]) == "observed"
        assert market_provenance_of([observed, assumed]) == "assumed"

    @pytest.mark.parametrize("origin", ("assumed", "mixed", "synthetic"))
    def test_every_not_observed_origin_labels_the_result_assumed(self, origin):
        provenance = CurveProvenance("c", origin, "x")
        assert market_provenance_of([provenance]) == "assumed"

    def test_empty_provenance_list_is_rejected(self):
        """No curves is not "observed"."""
        with pytest.raises(ValueError, match="at least one"):
            market_provenance_of([])

    def test_no_market_inputs_yields_null_not_observed(self):
        """Without market inputs `marketProvenance` is null, never "observed"."""
        result = price_bundle(FIXTURES / "note" / "v2")
        assert result.market_provenance is None
        assert result.market_provenance != "observed"

    def test_omitting_market_inputs_warns(self):
        result = price_bundle(FIXTURES / "note" / "v2")
        assert any("no curve at all" in w for w in result.warnings)


class TestMeasureIsReported:
    """`measure` is reported (I-11)."""

    def test_measure_is_reported_when_a_market_basis_exists(self):
        result = price_bundle(FIXTURES / "note" / "v2", ASSUMED)
        assert result.measure == ENGINE_RISK_MEASURE

    def test_engine_measure_is_risk_neutral(self):
        """The engine's measure is risk-neutral pricing, not a loss forecast."""
        assert ENGINE_RISK_MEASURE == "risk-neutral-pricing"

    def test_measure_is_in_the_serialized_result(self):
        payload = price_bundle(FIXTURES / "note" / "v2", ASSUMED).to_dict()
        assert payload["measure"] == "risk-neutral-pricing"

    def test_measure_is_null_when_no_curve_was_consulted(self):
        """`measure` is null when no curve was consulted."""
        assert price_bundle(FIXTURES / "note" / "v2").measure is None

    def test_measure_vocabulary(self):
        assert MEASURES == (
            "risk-neutral-pricing", "historical-forecast", "deterministic-stress",
        )


class TestMeasureVocabularyMatchesVarEs:
    """The measure constants are duplicated from `engine.risk.market.var_es` (this package must not
    import JAX); this pins them together."""

    def test_definitions_agree(self):
        from engine.risk.market import var_es
        assert MEASURES == var_es.RISK_MEASURES
        assert ENGINE_RISK_MEASURE == var_es.ENGINE_RISK_MEASURE


class TestSerializedShape:
    def test_result_is_json_serializable_with_market_inputs(self):
        payload = price_bundle(FIXTURES / "note" / "v2", ASSUMED).to_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_market_inputs_echoes_what_was_requested(self):
        """The resolved market inputs are echoed in the result."""
        payload = price_bundle(FIXTURES / "note" / "v2", ASSUMED).to_dict()
        assert payload["marketInputs"]["assumedProfileId"] == "flat-3pct-v1"
        assert payload["marketInputs"]["mode"] == MODE_ASSUMED_PROFILE

    def test_market_inputs_is_null_when_none_requested(self):
        payload = price_bundle(FIXTURES / "note" / "v2").to_dict()
        assert payload["marketInputs"] is None


class TestBundleDeclaredMarketStatus:
    """The fixtures declare `marketInputs.status: NOT_SUPPLIED`; the pipeline warns."""

    @pytest.mark.parametrize("case", ("bill", "note", "sofr"))
    @pytest.mark.parametrize("version", ("v1", "v2"))
    def test_fixture_declares_not_supplied(self, case, version):
        from engine.traderx.bundle import load_bundle
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
    """The capability document agrees with what is enforced."""

    def test_no_fallback_is_advertised(self):
        from engine.traderx import capabilities
        assert capabilities()["marketInputs"]["fallbackOnMissingInputs"] is False

    def test_registered_profiles_are_advertised(self):
        from engine.traderx import capabilities
        advertised = {
            p["assumedProfileId"] for p in capabilities()["marketInputs"]["assumedProfiles"]
        }
        assert advertised == set(ASSUMED_PROFILES)

    def test_advertised_profiles_are_actually_resolvable(self):
        """Every advertised profile resolves."""
        from engine.traderx import capabilities
        for entry in capabilities()["marketInputs"]["assumedProfiles"]:
            resolved = resolve_market_inputs({
                "mode": MODE_ASSUMED_PROFILE,
                "assumedProfileId": entry["assumedProfileId"],
            })
            assert resolved.profile.profile_id == entry["assumedProfileId"]

    def test_package_mode_is_advertised_as_unimplemented(self):
        """`package` mode is advertised as unimplemented."""
        from engine.traderx import capabilities
        assert capabilities()["marketInputs"]["packageModeImplemented"] is False

    def test_risk_measure_is_advertised(self):
        from engine.traderx import capabilities
        assert capabilities()["riskMeasure"]["measure"] == "risk-neutral-pricing"
        assert capabilities()["riskMeasure"]["tailDiagnostics"] == ["tailCount", "standardError"]

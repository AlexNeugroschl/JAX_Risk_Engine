"""
Tests for engine.risk.exposure -- ORE's ExposureCalculator statistics.

ORE's `ExposureCalculator` has no constructor in the Python bindings, so it
cannot be called here directly. The reference below is a line-by-line
transcription of its per-trade loop
(`OREAnalytics/orea/aggregation/exposurecalculator.cpp`, lines 150-230):
scalar Python, one path at a time, in the same order as the C++. The
vectorized implementation must agree with it to round-off.
"""
import math

import jax.numpy as jnp
import numpy as np
import pytest

from engine.risk.exposure import ExposureProfile, exposure_profile, netting_set_profile


def _ore_reference(npv, npv0, numeraire, discount, quantile):
    """ExposureCalculator::build for one trade, transcribed. `npv` is the
    undiscounted [S, T] cube; ORE's cube holds NPV / numeraire."""
    samples, dates = npv.shape
    epe = [max(npv0, 0.0)]
    ene = [max(-npv0, 0.0)]
    ee_b = [epe[0]]
    eee_b = [ee_b[0]]
    pfe = [max(npv0, 0.0)]
    for j in range(dates):
        epe_j = ene_j = 0.0
        distribution = []
        for k in range(samples):
            value = npv[k, j] / numeraire[k, j]
            epe_j += max(value, 0.0) / samples
            ene_j += max(-value, 0.0) / samples
            distribution.append(value)
        epe.append(epe_j)
        ene.append(ene_j)
        ee_b.append(epe_j / discount[j])
        eee_b.append(max(eee_b[-1], ee_b[-1]))
        distribution.sort()
        index = int(math.floor(quantile * (samples - 1) + 0.5))
        pfe.append(max(distribution[index], 0.0))
    return {"epe": epe, "ene": ene, "ee_b": ee_b, "eee_b": eee_b, "pfe": pfe}


@pytest.fixture(scope="module")
def cube():
    rng = np.random.default_rng(7)
    samples, dates = 501, 6
    times = np.linspace(0.5, 3.0, dates)
    npv = rng.normal(loc=1_000.0, scale=40_000.0, size=(samples, dates))
    numeraire = np.exp(0.03 * times)[None, :] * np.exp(rng.normal(0.0, 0.01, size=(samples, dates)))
    discount = np.exp(-0.03 * times)
    return npv, numeraire, discount, times


class TestMatchesOreTranscription:
    @pytest.mark.parametrize("quantile", [0.5, 0.9, 0.95, 0.975, 0.99])
    def test_every_statistic_matches(self, cube, quantile):
        npv, numeraire, discount, times = cube
        npv0 = 1_234.5
        got = exposure_profile(
            jnp.asarray(npv), npv0, jnp.asarray(numeraire), jnp.asarray(discount), times, quantiles=(quantile,),
        )
        expected = _ore_reference(npv, npv0, numeraire, discount, quantile)
        for name in ("epe", "ene", "ee_b", "eee_b"):
            np.testing.assert_allclose(np.asarray(getattr(got, name)), expected[name], rtol=1e-12, atol=1e-9)
        (pfe,) = got.pfe.values()
        np.testing.assert_allclose(np.asarray(pfe), expected["pfe"], rtol=1e-12, atol=1e-9)

    def test_negative_t0_npv(self, cube):
        npv, numeraire, discount, times = cube
        got = exposure_profile(jnp.asarray(npv), -500.0, jnp.asarray(numeraire), jnp.asarray(discount), times)
        assert float(got.epe[0]) == 0.0
        assert float(got.ene[0]) == 500.0
        assert all(float(p[0]) == 0.0 for p in got.pfe.values())


class TestDefinitions:
    def test_times_start_at_zero(self, cube):
        npv, numeraire, discount, times = cube
        got = exposure_profile(jnp.asarray(npv), 0.0, jnp.asarray(numeraire), jnp.asarray(discount), times)
        np.testing.assert_array_equal(got.times, np.concatenate([[0.0], times]))
        assert got.epe.shape == (len(times) + 1,)

    def test_epe_minus_ene_is_the_mean_deflated_npv(self, cube):
        npv, numeraire, discount, times = cube
        got = exposure_profile(jnp.asarray(npv), 0.0, jnp.asarray(numeraire), jnp.asarray(discount), times)
        np.testing.assert_allclose(
            np.asarray(got.epe - got.ene)[1:], np.mean(npv / numeraire, axis=0), rtol=1e-12,
        )

    def test_effective_ee_never_decreases(self, cube):
        npv, numeraire, discount, times = cube
        # A shrinking exposure, so EE_B falls and EEE_B must hold its peak.
        shrinking = npv * np.linspace(1.0, 0.1, len(times))[None, :]
        got = exposure_profile(jnp.asarray(shrinking), 0.0, jnp.asarray(numeraire), jnp.asarray(discount), times)
        eee = np.asarray(got.eee_b)
        assert np.all(np.diff(eee) >= 0.0)
        assert eee[-1] == pytest.approx(np.max(np.asarray(got.ee_b)))

    def test_ee_b_undoes_the_discounting(self):
        """With a deterministic numeraire equal to 1/P(0,t), EE_B is the
        plain undiscounted mean positive NPV."""
        times = np.array([1.0, 2.0])
        discount = np.exp(-0.04 * times)
        npv = np.array([[100.0, -50.0], [300.0, 250.0]])
        numeraire = np.tile(1.0 / discount, (2, 1))
        got = exposure_profile(jnp.asarray(npv), 0.0, jnp.asarray(numeraire), jnp.asarray(discount), times)
        np.testing.assert_allclose(np.asarray(got.ee_b)[1:], [200.0, 125.0], rtol=1e-12)

    def test_pfe_uses_ores_rounded_index(self):
        """ORE picks sorted[floor(q*(S-1)+0.5)] -- nearest rank, not an
        interpolated percentile. 11 samples 0..10 at q=0.95 -> index 10."""
        values = np.arange(11, dtype=np.float64)[:, None]
        ones = np.ones_like(values)
        got = exposure_profile(jnp.asarray(values), 0.0, jnp.asarray(ones), jnp.ones(1), [1.0], quantiles=(0.95, 0.5))
        assert float(got.pfe["PFE_95"][1]) == 10.0
        assert float(got.pfe["PFE_50"][1]) == 5.0

    def test_pfe_is_floored_at_zero(self):
        values = -np.arange(1, 11, dtype=np.float64)[:, None]
        ones = np.ones_like(values)
        got = exposure_profile(jnp.asarray(values), 0.0, jnp.asarray(ones), jnp.ones(1), [1.0])
        assert float(got.pfe["PFE_99"][1]) == 0.0

    def test_quantile_keys_keep_fractional_percent(self, cube):
        npv, numeraire, discount, times = cube
        got = exposure_profile(
            jnp.asarray(npv), 0.0, jnp.asarray(numeraire), jnp.asarray(discount), times, quantiles=(0.975, 0.98),
        )
        assert set(got.pfe) == {"PFE_97.5", "PFE_98"}


class TestNettingSet:
    def test_offsetting_trades_net_to_zero(self, cube):
        npv, numeraire, discount, times = cube
        both = np.stack([npv, -npv], axis=-1)
        got = netting_set_profile(
            jnp.asarray(both), [100.0, -100.0], jnp.asarray(numeraire), jnp.asarray(discount), times,
        )
        np.testing.assert_allclose(np.asarray(got.epe), 0.0, atol=1e-9)
        np.testing.assert_allclose(np.asarray(got.ene), 0.0, atol=1e-9)

    def test_netting_never_exceeds_the_sum_of_standalone_epe(self, cube):
        npv, numeraire, discount, times = cube
        other = np.roll(npv, 17, axis=0) * -0.5
        both = np.stack([npv, other], axis=-1)
        netted = netting_set_profile(jnp.asarray(both), [0.0, 0.0], jnp.asarray(numeraire), jnp.asarray(discount), times)
        standalone = [
            exposure_profile(jnp.asarray(x), 0.0, jnp.asarray(numeraire), jnp.asarray(discount), times)
            for x in (npv, other)
        ]
        assert np.all(np.asarray(netted.epe) <= np.asarray(standalone[0].epe + standalone[1].epe) + 1e-9)


class TestPrecisionAndValidation:
    def test_statistics_keep_the_input_dtype(self, cube):
        npv, numeraire, discount, times = cube
        got = exposure_profile(
            jnp.asarray(npv, dtype=jnp.float32), 0.0, jnp.asarray(numeraire), jnp.asarray(discount), times,
        )
        assert isinstance(got, ExposureProfile)
        for arr in [got.epe, got.ene, got.ee_b, got.eee_b, *got.pfe.values()]:
            assert arr.dtype == jnp.float32

    def test_rejects_mismatched_numeraire(self, cube):
        npv, numeraire, discount, times = cube
        with pytest.raises(ValueError, match="numeraire shape"):
            exposure_profile(jnp.asarray(npv), 0.0, jnp.asarray(numeraire[:, :-1]), jnp.asarray(discount), times)

    def test_rejects_mismatched_dates(self, cube):
        npv, numeraire, discount, times = cube
        with pytest.raises(ValueError, match="one entry per"):
            exposure_profile(jnp.asarray(npv), 0.0, jnp.asarray(numeraire), jnp.asarray(discount), times[:-1])

    @pytest.mark.parametrize("quantile", [0.0, 1.0, 1.5])
    def test_rejects_a_quantile_outside_the_open_interval(self, cube, quantile):
        npv, numeraire, discount, times = cube
        with pytest.raises(ValueError, match="quantile"):
            exposure_profile(
                jnp.asarray(npv), 0.0, jnp.asarray(numeraire), jnp.asarray(discount), times, quantiles=(quantile,),
            )

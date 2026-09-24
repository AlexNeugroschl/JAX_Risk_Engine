"""
W0.6's tail-statistic diagnostics: effective sample size, Monte Carlo
standard error, and the risk-measure label
(`docs/planning/traderx-integration-plan.md` §W0.6). Closes part of I-11.

**The gap these close.** `ES_99 = 1,240,000` reads as a precise figure.
Computed from 3 tail observations it is not one, and before this nothing in
the result said so — a sparse estimate and a well-converged one were the
same number on the wire.

`TestAdditiveOnly` is the safety net: every pre-existing key and value must
be untouched, since `compute_risk_metrics` is consumed by
`engine/api/schemas.py`, `engine/portfolio/request.py` and ~8 test files.
"""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from engine.risk.var_es import (
    ENGINE_RISK_MEASURE,
    RISK_MEASURES,
    compute_risk_metrics,
    expected_shortfall,
    expected_shortfall_standard_error,
    portfolio_pnl,
    tail_sample_size,
    value_at_risk,
)


@pytest.fixture(scope="module")
def normal_pnl():
    """[10000 scenarios, 3 steps] of N(0, 1e5) P&L -- enough scenarios that
    the tail counts are exact and predictable."""
    rng = np.random.default_rng(20260915)
    return jnp.asarray(rng.normal(0.0, 1e5, size=(10_000, 3)))


class TestTailSampleSize:
    """The effective sample size for a tail statistic."""

    def test_counts_the_strict_tail(self, normal_pnl):
        """At 99% over 10,000 scenarios, ~100 observations carry the
        estimate -- 1% of the sample, all from the least-sampled part of
        the distribution."""
        counts = np.asarray(tail_sample_size(normal_pnl, 0.99))
        assert counts.shape == (3,)
        assert all(90 <= c <= 110 for c in counts)

    def test_lower_percentile_has_a_larger_tail(self, normal_pnl):
        at_95 = np.asarray(tail_sample_size(normal_pnl, 0.95))
        at_99 = np.asarray(tail_sample_size(normal_pnl, 0.99))
        assert all(at_95 > at_99)

    def test_matches_the_observations_es_actually_averages(self, normal_pnl):
        """Counts the STRICT value-based tail (`pnl < -VaR`), i.e. exactly
        what `expected_shortfall` means over -- not a positional
        `floor(N*(1-p))` slice, which disagrees whenever there are ties at
        the VaR boundary."""
        var = np.asarray(value_at_risk(normal_pnl, 0.99))
        pnl = np.asarray(normal_pnl)
        for step in range(pnl.shape[1]):
            expected = int((pnl[:, step] < -var[step]).sum())
            assert int(np.asarray(tail_sample_size(normal_pnl, 0.99))[step]) == expected

    def test_ties_at_the_boundary_give_an_empty_tail(self):
        """All observations tied exactly at VaR: the strict tail is empty,
        which is the case ORE's own expectedShortfall raises on."""
        tied = jnp.zeros((10, 1))
        assert int(np.asarray(tail_sample_size(tied, 0.95))[0]) == 0
        assert bool(np.isnan(np.asarray(expected_shortfall(tied, 0.95))[0]))

    def test_positional_slice_would_disagree_on_ties(self):
        """Pins the distinction: a positional implementation would report
        floor(N*(1-p)) here regardless of the ties, and be wrong."""
        tied = jnp.zeros((100, 1))
        positional = int(np.floor(100 * (1 - 0.95)))  # 5
        assert positional == 5
        assert int(np.asarray(tail_sample_size(tied, 0.95))[0]) == 0


class TestExpectedShortfallStandardError:
    def test_matches_the_textbook_formula(self, normal_pnl):
        """s / sqrt(n) with the sample standard deviation (ddof=1),
        cross-checked against numpy on the same tail."""
        pnl = np.asarray(normal_pnl)
        var = np.asarray(value_at_risk(normal_pnl, 0.99))
        actual = np.asarray(expected_shortfall_standard_error(normal_pnl, 0.99))

        for step in range(pnl.shape[1]):
            tail = pnl[:, step][pnl[:, step] < -var[step]]
            expected = tail.std(ddof=1) / np.sqrt(tail.size)
            assert actual[step] == pytest.approx(expected, rel=1e-9)

    def test_uses_the_sample_not_population_standard_deviation(self, normal_pnl):
        """The tail IS a sample; the population form (ddof=0) would
        understate the spread. With n~100 the two differ by ~0.5%, which
        is small but systematic and always in the same direction."""
        pnl = np.asarray(normal_pnl)
        var = np.asarray(value_at_risk(normal_pnl, 0.99))
        tail = pnl[:, 0][pnl[:, 0] < -var[0]]

        sample = tail.std(ddof=1) / np.sqrt(tail.size)
        population = tail.std(ddof=0) / np.sqrt(tail.size)
        actual = float(np.asarray(expected_shortfall_standard_error(normal_pnl, 0.99))[0])

        assert actual == pytest.approx(sample, rel=1e-9)
        assert actual != pytest.approx(population, rel=1e-12)

    def test_shrinks_as_the_tail_grows(self, normal_pnl):
        """The whole point of reporting it: a larger tail is a better
        estimate, and the number says so."""
        at_95 = np.asarray(expected_shortfall_standard_error(normal_pnl, 0.95))
        at_99 = np.asarray(expected_shortfall_standard_error(normal_pnl, 0.99))
        assert all(at_95 < at_99)

    def test_empty_tail_is_nan(self):
        tied = jnp.zeros((10, 1))
        assert bool(np.isnan(np.asarray(expected_shortfall_standard_error(tied, 0.95))[0]))

    def test_single_observation_is_nan_not_zero(self):
        """**The important edge case.** With one tail observation the ES is
        defined but its spread is not. Returning 0.0 would read as
        'perfectly converged' for exactly the case where the estimate is
        least trustworthy; NaN is the honest answer."""
        pnl = jnp.asarray(np.concatenate([[-100.0], np.zeros(19)]).reshape(20, 1))
        assert int(np.asarray(tail_sample_size(pnl, 0.95))[0]) == 1

        es = np.asarray(expected_shortfall(pnl, 0.95))[0]
        se = np.asarray(expected_shortfall_standard_error(pnl, 0.95))[0]
        assert es == pytest.approx(100.0)
        assert bool(np.isnan(se))
        assert se != 0.0

    def test_is_positive_where_defined(self, normal_pnl):
        se = np.asarray(expected_shortfall_standard_error(normal_pnl, 0.99))
        assert all(s > 0 for s in se)


class TestComputeRiskMetricsDiagnostics:
    def test_diagnostics_are_included_by_default(self, normal_pnl):
        cube = normal_pnl[:, :, None]
        metrics = compute_risk_metrics(cube, 0.0, percentiles=(0.95, 0.99))

        for label in ("95", "99"):
            assert f"ES_{label}_tailCount" in metrics
            assert f"ES_{label}_standardError" in metrics

    def test_diagnostics_can_be_turned_off(self, normal_pnl):
        """`include_diagnostics=False` returns exactly the pre-W0.6 key
        set."""
        cube = normal_pnl[:, :, None]
        metrics = compute_risk_metrics(cube, 0.0, percentiles=(0.95,), include_diagnostics=False)
        assert set(metrics) == {"VaR_95", "ES_95"}

    def test_diagnostics_agree_with_the_standalone_functions(self, normal_pnl):
        cube = normal_pnl[:, :, None]
        metrics = compute_risk_metrics(cube, 0.0, percentiles=(0.99,))
        pnl = portfolio_pnl(cube, 0.0)

        assert np.array_equal(
            np.asarray(metrics["ES_99_tailCount"]), np.asarray(tail_sample_size(pnl, 0.99))
        )

    def test_every_percentile_gets_its_own_diagnostics(self, normal_pnl):
        cube = normal_pnl[:, :, None]
        metrics = compute_risk_metrics(cube, 0.0, percentiles=(0.90, 0.95, 0.99))
        for label in ("90", "95", "99"):
            assert metrics[f"ES_{label}_tailCount"].shape == (3,)


class TestAdditiveOnly:
    """`compute_risk_metrics` is consumed by `engine/api/schemas.py`,
    `engine/portfolio/request.py` and ~8 test files. The diagnostics sit
    BESIDE the statistics; no existing key or value may move."""

    def test_existing_keys_are_unchanged(self, normal_pnl):
        cube = normal_pnl[:, :, None]
        metrics = compute_risk_metrics(cube, 0.0, percentiles=(0.95, 0.99))
        assert {"VaR_95", "ES_95", "VaR_99", "ES_99"} <= set(metrics)

    def test_existing_values_are_unchanged(self, normal_pnl):
        """Byte-identical to the pre-W0.6 computation."""
        cube = normal_pnl[:, :, None]
        with_diagnostics = compute_risk_metrics(cube, 0.0, percentiles=(0.95, 0.99))
        without = compute_risk_metrics(
            cube, 0.0, percentiles=(0.95, 0.99), include_diagnostics=False
        )
        for key in without:
            assert np.array_equal(np.asarray(with_diagnostics[key]), np.asarray(without[key]))

    def test_diagnostic_keys_cannot_collide_with_a_percentile(self, normal_pnl):
        """`ES_95_tailCount` must not be mistakable for the statistic at
        some percentile -- the suffix keeps the namespaces apart."""
        cube = normal_pnl[:, :, None]
        metrics = compute_risk_metrics(cube, 0.0, percentiles=(0.95,))
        statistics = {k for k in metrics if not k.endswith(("_tailCount", "_standardError"))}
        assert statistics == {"VaR_95", "ES_95"}


class TestDiagnosticsRespectInputPrecision:
    """**Regression.** The first implementation of
    `expected_shortfall_standard_error` promoted float32 P&L to a float64
    result, because `jnp.maximum(count, 2)` is integer-typed and the
    integer arithmetic in the Bessel correction promoted the whole
    expression under `jax_enable_x64`.

    That silently defeated a float32 precision request: precision is applied
    by casting the input, so a statistic's output dtype IS how a caller
    observes it. Every other statistic here inherits the input dtype by
    construction; this one had to be told.
    """

    @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
    def test_standard_error_matches_the_input_dtype(self, dtype):
        rng = np.random.default_rng(7)
        pnl = jnp.asarray(rng.normal(0.0, 1e5, size=(2000, 3)), dtype=dtype)
        assert expected_shortfall_standard_error(pnl, 0.95).dtype == dtype

    @pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
    def test_standard_error_agrees_with_the_other_statistics(self, dtype):
        """The contract it broke: every statistic in this module carries
        the P&L's own dtype."""
        rng = np.random.default_rng(7)
        pnl = jnp.asarray(rng.normal(0.0, 1e5, size=(2000, 3)), dtype=dtype)

        assert value_at_risk(pnl, 0.95).dtype == dtype
        assert expected_shortfall(pnl, 0.95).dtype == dtype
        assert expected_shortfall_standard_error(pnl, 0.95).dtype == dtype

    def test_tail_count_stays_integral_at_every_precision(self):
        """The deliberate exception. A count is not a statistic: float32
        cannot represent integers exactly above 2**24, so casting it to
        follow the precision override would let a large-scenario count
        silently round."""
        rng = np.random.default_rng(7)
        for dtype in (jnp.float32, jnp.float64):
            pnl = jnp.asarray(rng.normal(0.0, 1e5, size=(2000, 3)), dtype=dtype)
            assert jnp.issubdtype(tail_sample_size(pnl, 0.95).dtype, jnp.integer)

    def test_float32_values_still_reconcile(self):
        """Honouring the dtype must not change the number beyond float32's
        own resolution."""
        rng = np.random.default_rng(7)
        sample = rng.normal(0.0, 1e5, size=(4000, 1))
        wide = expected_shortfall_standard_error(jnp.asarray(sample, dtype=jnp.float64), 0.95)
        narrow = expected_shortfall_standard_error(jnp.asarray(sample, dtype=jnp.float32), 0.95)
        assert float(narrow[0]) == pytest.approx(float(wide[0]), rel=1e-4)


class TestDiagnosticsReachTheHttpBoundary:
    """`RiskMetricsSchema` is a generic `Dict[str, List[Optional[float]]]`,
    so the new keys pass through with no schema change. Pinned because
    "it happens to work" and "it is contracted to work" are different, and
    a future typed schema would silently drop them."""

    def test_diagnostics_survive_serialization(self, normal_pnl):
        from engine.api.schemas import RiskMetricsSchema
        cube = normal_pnl[:, :, None]
        payload = RiskMetricsSchema.from_dataclass(
            compute_risk_metrics(cube, 0.0, percentiles=(0.99,))
        ).model_dump()

        assert "ES_99_tailCount" in payload["values"]
        assert "ES_99_standardError" in payload["values"]

    def test_nan_standard_error_serializes_as_null(self):
        """A NaN standard error must reach a consumer as `null`, not as a
        float they might read as a real value. The schema's existing
        NaN -> None conversion covers this; asserted, not assumed."""
        from engine.api.schemas import RiskMetricsSchema
        tied = jnp.zeros((10, 1, 1))
        payload = RiskMetricsSchema.from_dataclass(
            compute_risk_metrics(tied, 0.0, percentiles=(0.95,))
        ).model_dump()

        assert payload["values"]["ES_95_standardError"] == [None]
        assert payload["values"]["ES_95_tailCount"] == [0.0]


class TestRiskMeasureVocabulary:
    """Part of I-11: 'Risk measure unlabelled'."""

    def test_three_measures(self):
        assert RISK_MEASURES == (
            "risk-neutral-pricing", "historical-forecast", "deterministic-stress",
        )

    def test_engine_produces_risk_neutral(self):
        """This engine simulates under the pricing measure. It is NOT a
        calibrated forecast of tomorrow's loss, and reporting it where one
        is expected is a category error no numerical accuracy fixes."""
        assert ENGINE_RISK_MEASURE == "risk-neutral-pricing"

    def test_engine_measure_is_one_of_the_three(self):
        assert ENGINE_RISK_MEASURE in RISK_MEASURES

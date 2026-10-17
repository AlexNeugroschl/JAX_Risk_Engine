"""I-111: finite samples do not certify unobserved tails or nominal intervals."""
import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from engine.precision.estimate import two_level_mean
from tests.support.precision_harness import standardized_errors


def test_zero_width_miss_is_not_dropped_from_coverage():
    # Exact/roundoff equality is distinct from an unseen-event miss. NaN is unavailable.
    z = standardized_errors(np.array([0., 1e-16, 2., -3., 2., 1.]),
                      np.array([0., 0., 0., 0., 2., np.nan]), scale=1.)
    np.testing.assert_array_equal(z, [np.inf, -np.inf, 1.])


@pytest.mark.parametrize("replicates", [1, 8, 32])
def test_empirical_zero_does_not_certify_a_constant_or_an_interval(replicates):
    from engine.precision.report import PrecisionReport
    from engine.precision import Precision
    report = PrecisionReport.of(Precision(), (), {}, (), 128 * replicates)
    assert dataclasses.asdict(report.interval) == {
        "status": "unavailable", "reason": "no_validated_tail_bound",
        "confidence_level": None, "lower": None, "upper": None,
    }
    # P(no observed event) = (1-p)**R > .95, yet E[X] = 1. Even finite variance
    # and many nonzero observations in other examples cannot certify a tail bound.
    p = .001
    rng = np.random.default_rng(817)
    samples = (rng.uniform(size=(256, replicates)) < p) / p
    zero_runs = 0
    for sample in samples:
        estimate = two_level_mean(jnp.asarray(sample), jnp.asarray(sample), replicates=replicates)
        if not sample.any():
            zero_runs += 1
            assert estimate.value == 0.
            assert np.isnan(estimate.standard_error) if replicates == 1 else estimate.standard_error == 0.
            assert report.interval.status == "unavailable"
    assert zero_runs > 230


def test_legacy_es_name_explicitly_deprecates_the_conditional_diagnostic():
    from engine.risk.market.var_es import conditional_tail_standard_error, expected_shortfall_standard_error
    pnl = jnp.arange(-100., 100.)[:, None]
    expected = conditional_tail_standard_error(pnl, .95)
    with pytest.warns(DeprecationWarning, match="conditional_tail_standard_error"):
        actual = expected_shortfall_standard_error(pnl, .95)
    np.testing.assert_array_equal(actual, expected)


def test_diagnostic_criteria_keep_infinite_misses_and_no_checks_failing():
    from tests.support.precision_harness import CoverageRow
    row = CoverageRow("c", 64, "netting_set/ENE", "netting_set", 32, 32, 1., 32, 32, 1.)
    assert row.precision_passes and row.total_passes
    assert not dataclasses.replace(row, precision_z_rms=np.inf).precision_passes
    assert not dataclasses.replace(row, precision_checks=0, precision_covered=0).precision_passes
    assert not dataclasses.replace(row, total_covered=28).total_passes
    assert not dataclasses.replace(row, total_z_rms=.74).total_passes
    assert not dataclasses.replace(row, total_z_rms=1.34).total_passes

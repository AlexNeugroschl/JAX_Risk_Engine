"""
The timing harness (`tests/support/precision_timing.py`; F-07's research result, precision design
§19): its accuracy and equal-accuracy reading on timings built by hand, and one case timed in a
process of its own.

  * a case's error at a path count is its figure's root mean square over the seeds against
    float64's mean at the largest count, relative to the figure's scale, its entries that float64
    holds fixed left out;
  * a case reaches float64's error where its own falls to it, log-linearly between the measured
    counts that bracket it, and costs its warm time there, interpolated alike; a case that never
    reaches it is reported so, never extrapolated;
  * the speedup is float64's time over the case's.
"""
import math

import numpy as np
import pytest

from tests.support.precision_timing import (
    FLOAT64, CaseTiming, equal_accuracy, figure_errors, reached_at, record, run_case, summary,
)


def _timing(case, paths, values, warm=1.0, divisor=None):
    """A `CaseTiming` of `case` at `paths` whose figure "f" has the per-seed `values`
    (`[seeds][entries]`) and whose warm runs took `warm` seconds."""
    return CaseTiming(case=case, paths=paths, budget_divisor=divisor, cold_seconds=2 * warm, compile_seconds=warm,
                      compiles=3, warm_seconds=(warm, warm), warm_compiles=0, device_peak_bytes=None,
                      induction_budget=1, products={}, figures={"f": [list(v) for v in values]})


class TestErrors:
    def test_a_figures_error_is_its_rms_against_float64s_mean_at_the_largest_count(self):
        truth = [[10.0, 4.0], [12.0, 4.0], [11.0, 4.0]]          # entry 1 is held fixed by float64
        timings = [_timing(FLOAT64, 4, truth), _timing(FLOAT64, 1, [[9.0, 4.0], [13.0, 4.0], [11.0, 4.0]]),
                   _timing("fp8", 4, [[10.5, 9.0], [12.5, 9.0], [11.5, 9.0]])]
        errors = figure_errors(timings, fixed=1e-9)
        scale = 11.0                                              # the truth's largest entry
        assert errors[(FLOAT64, 4, "f")] == pytest.approx(math.sqrt(2 / 3) / scale)
        assert errors[(FLOAT64, 1, "f")] == pytest.approx(math.sqrt(8 / 3) / scale)
        # 0.5 off on every seed on top of the spread, the fixed entry's move left out.
        assert errors[("fp8", 4, "f")] == pytest.approx(math.sqrt((0.25 + 2.25 + 0.25) / 3) / scale)

    def test_a_timing_at_another_budget_is_not_an_accuracy_point(self):
        timings = [_timing(FLOAT64, 4, [[1.0], [2.0], [3.0]]), _timing("fp8", 4, [[1.0], [2.0], [3.0]], divisor=2)]
        assert set(figure_errors(timings, fixed=1e-9)) == {(FLOAT64, 4, "f")}


class TestReachedAt:
    def test_log_linear_between_the_bracketing_counts(self):
        assert reached_at([(1024, 4e-3), (4096, 1e-3)], 2e-3) == pytest.approx(2048.0)

    def test_the_first_count_where_it_is_there_already(self):
        assert reached_at([(1024, 1e-3), (4096, 1e-4)], 2e-3) == 1024.0

    def test_a_case_measured_at_one_count_so_far_reads_its_own_time(self):
        """The record is written after every case: a case may have one count yet."""
        timings = [_timing(FLOAT64, 1, [[1.0], [3.0], [2.0]]), _timing(FLOAT64, 4, [[2.0], [2.5], [1.5]]),
                   _timing("fp8", 1, [[1.5], [2.5], [2.0]], warm=0.25)]
        (row,) = equal_accuracy(timings, fixed=1e-9)
        assert (row.paths, row.seconds) == (1.0, 0.25)

    def test_never_reached_is_none_not_an_extrapolation(self):
        assert reached_at([(1024, 4e-3), (4096, 3e-3)], 1e-3) is None


class TestEqualAccuracy:
    def test_uncertain_independent_reference_cannot_produce_a_speedup(self):
        reference = _timing("float64:reference", 1024, [[1.], [2.], [3.]])
        timings = [_timing(FLOAT64, 64, [[1.], [2.], [3.]]),
                   _timing("fp8", 64, [[1.], [2.], [3.]], warm=.1)]
        (row,) = equal_accuracy(timings, fixed=1e-9, references=[reference])
        assert row.reference_resolved is False and row.reference_error > .1 * row.target
        assert row.paths is None and row.speedup is None

    def test_the_speedup_is_float64s_time_over_the_cases_where_its_error_reaches_float64s(self):
        """float64's error falls as 1/N, the case's too but twice as large: it reaches float64's
        error at 1,024 paths at twice the paths, and runs four times faster per path."""
        rng = np.random.default_rng(1)
        noise = rng.standard_normal((64, 1))
        noise = (noise - noise.mean()) / noise.std()

        def values(scale):
            return (100.0 + scale * noise).tolist()

        timings = []
        for paths in (1024, 2048, 4096, 8192):
            timings.append(_timing(FLOAT64, paths, values(1024 / paths), warm=paths / 1000))
            timings.append(_timing("fp8", paths, values(2048 / paths), warm=paths / 4000))
        (row,) = [r for r in equal_accuracy(timings, fixed=1e-9) if r.float64_paths == 1024]
        assert row.paths == pytest.approx(2048, rel=0.02)
        assert row.speedup == pytest.approx((1024 / 1000) / (2048 / 4000), rel=0.03)
        largest = [r for r in equal_accuracy(timings, fixed=1e-9) if r.float64_paths == 8192]
        assert largest == []   # the truth's own count is never a target

    def test_the_summary_names_the_worst_figure_and_those_not_reached(self):
        timings = [_timing(FLOAT64, 1, [[1.0], [3.0], [2.0]], warm=1.0), _timing(FLOAT64, 4, [[2.0], [2.5], [1.5]]),
                   _timing("fp8", 1, [[9.0], [9.0], [9.0]]), _timing("fp8", 4, [[9.0], [9.0], [9.0]])]
        text = summary(record("compute", timings, fixed=1e-9))
        assert "| fp8 | 1 |  |  | 1 of 1 |" in text


@pytest.mark.slow
def test_one_case_is_timed_in_a_process_of_its_own():
    """The float64 reference of the smallest market-risk campaign at 64 scenarios, three seeds: a
    cold run that compiles, warm runs that compile nothing, every figure for every seed."""
    timing = run_case("compute_market_risk", FLOAT64, 64, (1, 2, 3))
    assert timing.compiles > 0 and timing.warm_compiles == 0
    assert len(timing.warm_seconds) == 2 and timing.cold_seconds > max(timing.warm_seconds)
    assert timing.products == {"float64": "native"}
    assert all(len(seeds) == 3 for seeds in timing.figures.values())
    assert "portfolio/VaR_99" in timing.figures

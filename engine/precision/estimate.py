"""
Estimates from a paired float64 sample (docs/planning/details/precision.md §9; decision A-13).

A run at reduced precision can re-run its first `n` paths (or scenarios) at float64 throughout:
the same random numbers, since a scrambled Sobol sequence's first `n` points do not depend on
how many follow, and every kernel is per path, so the float64 re-run equals those paths of a
float64 run bit for bit. Each figure then compares the two on the same paths.

**Means** (EPE, ENE): the two-level estimator. With `f` the per-path quantity at the run's
precision on all `N` paths and `g` the same at float64 on the first `n`,

    F = mean_N(f) + mean_n(g - f)

is an unbiased estimate of the float64 figure: the precision error becomes variance, which
shrinks as `n` grows, instead of bias. Its variance counts both terms and their covariance
(the paired paths are among the `N`): with `d = g - f`,

    Var(F) = Var(f) / N + Var(d) / n + 2 Cov(f, d) / N,

each moment the sample's (`ddof=1`; NaN with fewer than two paths). This is two-level Monte
Carlo as Giles and Sheridan-Methven use it for reduced-precision random variables. The
corrected value is the figure the run reports.

**Quantiles** (PFE, VaR, ES): not means, so not corrected (A-13; multilevel quantile
estimation is roadmap 6.3). The figure is the run's own, from every path at its precision;
the paired sample measures the precision error as the same statistic at the run's precision
and at float64 on the paired paths, and their difference.

Every estimate is computed in float64 (decision A-10: reductions over paths are float64).
Depends on JAX and NumPy only.
"""
import math
from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from engine.precision.storage import BLOCK


def paired_paths(num_paths: int, fraction: float) -> int:
    """How many of `num_paths` are re-run at float64 for `Precision.paired_fraction`: none at
    0, else `fraction` of them rounded up to whole blocks of `BLOCK` paths (at least one block,
    at most every path). Whole blocks keep the paired paths' block scales those of the run, and
    keep the number of distinct paired shapes, each compiled once, small."""
    if fraction == 0:
        return 0
    return min(num_paths, BLOCK * math.ceil(fraction * num_paths / BLOCK))


@dataclass(frozen=True)
class MeanEstimate:
    """A mean figure from the two-level estimator (see the module docstring). Every array has
    the figure's shape (one entry per simulation date for an exposure profile).

    value: the corrected figure, the one the run reports.
    uncorrected: the mean over the `paths` at the run's precision.
    correction: the mean over the `paired_paths` of float64 minus the run's precision.
    standard_error: of `value`, both terms and their covariance.
    uncorrected_standard_error: the Monte Carlo standard error of `uncorrected`.
    correction_standard_error: of `correction`, which tests the precision bias: a correction
        several of these from zero is a bias the uncorrected figure carries.
    max_difference: the largest paired difference, |float64 - run| on one path.
    """
    value: jax.Array
    uncorrected: jax.Array
    correction: jax.Array
    standard_error: jax.Array
    uncorrected_standard_error: jax.Array
    correction_standard_error: jax.Array
    max_difference: jax.Array
    paths: int
    paired_paths: int


@dataclass(frozen=True)
class QuantileEstimate:
    """A quantile figure measured on the paired sample (see the module docstring).

    value: the figure from every path at the run's precision, the one the run reports.
    paired: the same statistic on the `paired_paths` at the run's precision.
    paired_float64: the same statistic on the `paired_paths` at float64.
    difference: `paired - paired_float64`, the measured precision error.
    """
    value: jax.Array
    paired: jax.Array
    paired_float64: jax.Array
    difference: jax.Array
    paths: int
    paired_paths: int


def two_level_mean(low, high) -> MeanEstimate:
    """The two-level estimate of the mean over axis 0 of `low` (`[N, ...]`, every path at the
    run's precision), corrected by `high` (`[n, ...]`, the first `n` paths at float64)."""
    low, high = _paired(low, high)
    return MeanEstimate(*_two_level(low, high), paths=low.shape[0], paired_paths=high.shape[0])


def paired_quantile(statistic: Callable[[jax.Array], jax.Array], low, high) -> QuantileEstimate:
    """`statistic` (a sample `[k, ...]` -> the figure) on every path of `low` (`[N, ...]`, the
    run's precision), and on the first `n` paths of `low` and of `high` (`[n, ...]`, float64)."""
    low, high = _paired(low, high)
    paired, paired_float64 = statistic(low[:high.shape[0]]), statistic(high)
    return QuantileEstimate(value=statistic(low), paired=paired, paired_float64=paired_float64,
                            difference=paired - paired_float64, paths=low.shape[0], paired_paths=high.shape[0])


def _paired(low, high) -> Tuple[jax.Array, jax.Array]:
    """Both samples at float64; the paired one no longer than the run, with the same figure shape."""
    low, high = jnp.asarray(low, dtype=jnp.float64), jnp.asarray(high, dtype=jnp.float64)
    if low.ndim == 0 or high.shape[1:] != low.shape[1:] or not 0 < high.shape[0] <= low.shape[0]:
        raise ValueError(f"a paired sample [n, ...] pairs the first n of the run's [N, ...] paths, 0 < n <= N; "
                         f"got {high.shape} against {low.shape}")
    return low, high


@jax.jit
def _two_level(low, high):
    """`MeanEstimate`'s arrays, in field order; one program per pair of shapes."""
    num, paired = low.shape[0], high.shape[0]
    low_paired = low[:paired]
    difference = high - low_paired
    uncorrected, correction = jnp.mean(low, axis=0), jnp.mean(difference, axis=0)
    var_low = jnp.var(low, axis=0, ddof=1)
    var_difference = jnp.var(difference, axis=0, ddof=1)
    covariance = (jnp.sum((low_paired - jnp.mean(low_paired, axis=0)) * (difference - correction), axis=0)
                  / (paired - 1))
    variance = var_low / num + var_difference / paired + 2.0 * covariance / num
    return (uncorrected + correction, uncorrected, correction, jnp.sqrt(jnp.maximum(variance, 0.0)),
            jnp.sqrt(var_low / num), jnp.sqrt(var_difference / paired), jnp.max(jnp.abs(difference), axis=0))

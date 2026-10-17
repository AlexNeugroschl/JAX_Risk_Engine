"""
Estimates from a paired float64 sample (docs/planning/details/precision.md §9; decision A-13).

A run at reduced precision can re-run a prefix within each independent scramble at float64:
the same random numbers, since a scrambled Sobol sequence's first `n` points do not depend on
how many follow, and every kernel is per path, so the float64 re-run equals those paths of a
float64 run bit for bit. Each figure then compares the two on the same paths.

**Means** (EPE, ENE): the two-level estimator. With `f` the per-path quantity at the run's
precision on all `N` paths and `g` the same at float64 on the `n` paired paths,

    F = mean_N(f) + mean_n(g - f)

is an unbiased estimate of the float64 figure: the precision error becomes variance, which
shrinks as `n` grows, instead of bias. For standalone IID samples only, its variance
counts both terms and their covariance (the paired paths are among the `N`): with `d = g - f`,

    Var(F) = Var(f) / N + Var(d) / n + 2 Cov(f, d) / N,

each moment the sample's (`ddof=1`; NaN with fewer than two paths). This is two-level Monte
Carlo as Giles and Sheridan-Methven use it for reduced-precision random variables. The
corrected value is the figure the run reports.

**Quantiles** (PFE, VaR, ES): not means, so not corrected (A-13; multilevel quantile
estimation is a separate conditional F-07 experiment). The figure is the run's own, from every path at its precision;
the paired sample measures the precision error as the same statistic at the run's precision
and at float64 on the paired paths, and their difference.

With `replicates=R`, samples are concatenated equal-sized independent scrambles.
Pair a prefix within EACH scramble. Compute the low mean and paired correction within
each, then their average and sample standard deviation / sqrt(R). This measures both
terms and their covariance without assuming independence of paths or storage blocks.
One scramble returns NaN standard errors. These are variance diagnostics; Student t
intervals are not certified for unobserved tails (I-111). `replicates=None` retains the explicitly IID estimator for
standalone callers with independent observations.

Every estimate is computed in float64 (decision A-10: reductions over paths are float64).
Depends on JAX and NumPy only.
"""
import math
from dataclasses import dataclass
from typing import Callable, Tuple
from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp

from engine.precision.storage import BLOCK


def validate_replicates(paths: int, replicates: int, name: str = "replicates") -> None:
    """Equal, nonempty independent replicates, checked on the host before any work."""
    if (not isinstance(replicates, Integral) or isinstance(replicates, bool)
            or replicates < 1 or paths % replicates or paths < replicates):
        raise ValueError(f"{name} must be a positive integer dividing the sample count {paths}; got {replicates!r}")


def paired_count(paths: int, fraction: float, replicates: int = 1) -> int:
    """A whole-block prefix of each equal-sized replicate, concatenated in replicate order."""
    validate_replicates(paths, replicates)
    return replicates * paired_paths(paths // replicates, fraction)


def paired_indices(paths: int, paired: int, replicates: int = 1):
    """Indices of the prefix paired within each replicate; no random draw or repricing."""
    validate_replicates(paths, replicates)
    validate_replicates(paired, replicates)
    if paired > paths:
        raise ValueError("paired sample cannot exceed the run's sample count")
    return (jnp.arange(replicates)[:, None] * (paths // replicates)
            + jnp.arange(paired // replicates)[None, :]).reshape(-1)


@jax.jit
def replicate_standard_error(estimates):
    """SE of a mean of independent equal-sized replicate estimates, in float64.

    One replicate cannot estimate its sampling uncertainty. This function returns one
    empirical SE, not a nominal confidence bound; see `IntervalAvailability` (I-111).
    """
    estimates = jnp.asarray(estimates, dtype=jnp.float64)
    if estimates.shape[0] < 2:
        return jnp.full(estimates.shape[1:], jnp.nan, dtype=jnp.float64)
    return jnp.std(estimates, axis=0, ddof=1) / jnp.sqrt(estimates.shape[0])


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

    Pipeline standard errors use independent scrambles (I-111), with NaN when there is
    only one. Standalone `two_level_mean(..., replicates=None)` is the IID estimator.
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


def two_level_mean(low, high, replicates=None) -> MeanEstimate:
    """The two-level estimate of the mean over axis 0 of `low` (`[N, ...]`, every path at the
    run's precision), corrected by `high` (`[n, ...]`, the paired prefixes at float64).
    With `replicates=None`, the paired sample is a single prefix of IID observations."""
    low, high = _paired(low, high)
    arrays = _two_level(low, high) if replicates is None else _two_level_replicates(low, high, replicates)
    return MeanEstimate(*arrays, paths=low.shape[0], paired_paths=high.shape[0])


def paired_quantile(statistic: Callable[[jax.Array], jax.Array], low, high, replicates=None) -> QuantileEstimate:
    """`statistic` (a sample `[k, ...]` -> the figure) on every path of `low` (`[N, ...]`, the
    run's precision), and on the paired prefixes of `low` and `high` (`[n, ...]`, float64).
    With `replicates=None`, the paired sample is a single prefix."""
    low, high = _paired(low, high)
    sample = low[:high.shape[0]] if replicates is None else low[paired_indices(len(low), len(high), replicates)]
    paired, paired_float64 = statistic(sample), statistic(high)
    return QuantileEstimate(value=statistic(low), paired=paired, paired_float64=paired_float64,
                            difference=paired - paired_float64, paths=low.shape[0], paired_paths=high.shape[0])


def _paired(low, high) -> Tuple[jax.Array, jax.Array]:
    """Both samples at float64; the paired one no longer than the run, with the same figure shape."""
    low, high = jnp.asarray(low, dtype=jnp.float64), jnp.asarray(high, dtype=jnp.float64)
    if low.ndim == 0 or high.ndim == 0 or high.shape[1:] != low.shape[1:] or not 0 < high.shape[0] <= low.shape[0]:
        raise ValueError(f"a paired sample [n, ...] must match the run's [N, ...] figure shape, 0 < n <= N; "
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


def _two_level_replicates(low, high, replicates):
    validate_replicates(len(low), replicates)
    validate_replicates(len(high), replicates)
    return _replicated_mean(low, high, replicates)


@partial(jax.jit, static_argnames="replicates")
def _replicated_mean(low, high, replicates):
    low = low.reshape((replicates, -1) + low.shape[1:])
    high = high.reshape((replicates, -1) + high.shape[1:])
    difference = high - low[:, :high.shape[1]]
    means, corrections = jnp.mean(low, axis=1), jnp.mean(difference, axis=1)
    uncorrected, correction = jnp.mean(means, axis=0), jnp.mean(corrections, axis=0)
    return (uncorrected + correction, uncorrected, correction, replicate_standard_error(means + corrections),
            replicate_standard_error(means), replicate_standard_error(corrections),
            jnp.max(jnp.abs(difference), axis=(0, 1)))

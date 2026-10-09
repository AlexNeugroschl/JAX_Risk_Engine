"""
Exposure profiles over a simulated NPV cube: EPE, ENE, EE_B, EEE_B and PFE.

Counterparty exposure through time, from the multi-step risk-neutral cube of
`price_portfolio`. Not market-risk VaR (see `engine.risk.market`).

The statistics are ORE's `ExposureCalculator` definitions
(OREAnalytics/orea/aggregation/exposurecalculator.cpp), for one trade or one netting set,
without collateral:

    V_k(t)    NPV on path k at t divided by the numeraire N(t) (ORE's cube stores NPV / N)
    EPE(t)    mean_k max(V_k(t), 0)          discounted expected positive exposure
    ENE(t)    mean_k max(-V_k(t), 0)         discounted expected negative exposure
    EE_B(t)   EPE(t) / P(0,t)                undiscounted expected exposure
    EEE_B(t)  max(EEE_B(t-), EE_B(t))        effective (non-decreasing) EE
    EPE_B(t)  sum_{s <= t} EE_B(s) dt_s / t  time-weighted EE up to t (up to maturity)
    EEPE_B(t) sum_{s <= t} EEE_B(s) dt_s / t time-weighted effective EE
    PFE_q(t)  max(sorted_k V_k(t)[i], 0),    i = floor(q * (S - 1) + 0.5)

EPE_B/EEPE_B weigh with ActualActual(ISDA) year fractions when the dates are given, as ORE
does, and are 0 after the trade's maturity; the Basel figures are their values at the last
date on or before `WeekendsOnly().adjust(asof + 1Y + 4D)` (ORE's `baselMaxEEPDate`).

Each profile includes t=0, where ORE sets EPE = EE_B = EEE_B = PFE = max(NPV0, 0) and
ENE = max(-NPV0, 0). A netting set sums paths across trades before the statistics.

The numeraire is whatever the simulation supplies: in the pipeline ORE's LGM numeraire of
the base currency (`engine.market_simulation.scenario_market`), so E[1/N(t)] = P(0,t) and EE_B is
exact; on the Hull-White path a discretely accrued bank account (I-45).

With a paired float64 sample (`paired`: the first n paths again at float64, of a cube priced
at reduced precision; decision A-13), EPE and ENE are the two-level estimates, means over the
run's paths corrected by the paired paths' float64 differences (`engine.precision.estimate`),
and EE_B, EEE_B, EPE_B, EEPE_B and the Basel figures follow from the corrected EPE. PFE is the
run's own, measured on the pair. `ExposureProfile.estimates` holds each figure's estimate.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.precision.estimate import MeanEstimate, QuantileEstimate, paired_quantile, two_level_mean
from engine.risk.market.var_es import quantile_label

#: A paired float64 sample: the NPVs `[n, T]` (or the cube `[n, T, N]`) and the numeraire
#: `[n, T]` of the first n paths, run at float64 throughout.
PairedPaths = Tuple[jnp.ndarray, jnp.ndarray]


@dataclass(frozen=True)
class ExposureProfile:
    """Exposure statistics on a date grid. Every array is `[T+1]`, indexed
    like `times`, whose first entry is t=0."""
    times: np.ndarray
    epe: jnp.ndarray
    ene: jnp.ndarray
    ee_b: jnp.ndarray
    eee_b: jnp.ndarray
    #: `"PFE_95"`-style key -> profile, one per requested quantile.
    pfe: Dict[str, jnp.ndarray]
    #: ORE's time-weighted EPE_B / EEPE_B profiles (`exposure_profile` always sets them).
    epe_b: Optional[jnp.ndarray] = None
    eepe_b: Optional[jnp.ndarray] = None
    #: ORE's Basel EPE_B / EEPE_B (the profiles at the one-year horizon); `None` without dates.
    basel_epe: Optional[float] = None
    basel_eepe: Optional[float] = None
    #: With a paired sample, each figure's estimate over the simulated dates (t=0 excluded):
    #: `"EPE"`, `"ENE"` (two-level, `MeanEstimate`) and each `"PFE_95"` (`QuantileEstimate`).
    #: Empty without one.
    estimates: Dict[str, Union[MeanEstimate, QuantileEstimate]] = field(default_factory=dict)


def exposure_profile(
    npv: jnp.ndarray,
    npv0: float,
    numeraire: jnp.ndarray,
    discount: jnp.ndarray,
    times: Sequence[float],
    quantiles: Sequence[float] = (0.95, 0.99),
    dates: Optional[Sequence[ORE.Date]] = None,
    asof: Optional[ORE.Date] = None,
    maturity: Optional[ORE.Date] = None,
    paired: Optional[PairedPaths] = None,
) -> ExposureProfile:
    """ORE exposure statistics for one trade or netting set.

    npv: `[S, T]` undiscounted NPV on each path at each simulated date.
    npv0: the t=0 NPV.
    numeraire: `[S, T]` numeraire on each path at each date, `N(0) = 1`.
    discount: `[T]` today's discount factor `P(0,t)` to each date, from the
        curve the numeraire accrues on.
    times: `[T]` the simulated dates as year fractions (t=0 excluded).
    quantiles: PFE quantiles, each in (0, 1).
    dates / asof / maturity: the simulated dates, the as-of date and the trade's maturity, for
        ORE's time weights and Basel horizon (without them EPE_B/EEPE_B weigh by `times`, and
        there is no Basel figure).
    paired: the paired float64 sample `(npv [n, T], numeraire [n, T])` of the first n paths
        (see the module docstring), or None.

    Statistics are computed in `npv`'s dtype, the paired estimates in float64.
    """
    npv = jnp.asarray(npv)
    dtype = npv.dtype
    num_paths, num_dates = npv.shape
    if numeraire.shape != npv.shape:
        raise ValueError(f"numeraire shape {numeraire.shape} does not match npv shape {npv.shape}")
    if len(times) != num_dates or jnp.shape(discount) != (num_dates,):
        raise ValueError(
            f"times ({len(times)}) and discount {jnp.shape(discount)} must have one entry per "
            f"simulated date ({num_dates})"
        )
    for q in quantiles:
        if not 0.0 < q < 1.0:
            raise ValueError(f"PFE quantile must lie in (0, 1); got {q}")

    deflated = npv / jnp.asarray(numeraire, dtype=dtype)
    zero = jnp.zeros((), dtype=dtype)
    npv0 = jnp.asarray(npv0, dtype=dtype)

    estimates = {}
    if paired is None:
        positive = jnp.mean(jnp.maximum(deflated, zero), axis=0)
        negative = jnp.mean(jnp.maximum(-deflated, zero), axis=0)
    else:
        paired_npv, paired_numeraire = (jnp.asarray(a) for a in paired)
        if paired_npv.shape != paired_numeraire.shape or paired_npv.shape[1:] != npv.shape[1:]:
            raise ValueError(f"paired npv {paired_npv.shape} and numeraire {paired_numeraire.shape} must be [n, "
                             f"{num_dates}]")
        paired_deflated = paired_npv / paired_numeraire
        positive_of = lambda x: jnp.maximum(x, 0.0)  # noqa: E731
        estimates["EPE"] = two_level_mean(positive_of(deflated), positive_of(paired_deflated))
        estimates["ENE"] = two_level_mean(positive_of(-deflated), positive_of(-paired_deflated))
        positive, negative = (estimates[k].value.astype(dtype) for k in ("EPE", "ENE"))

    epe = jnp.concatenate([jnp.maximum(npv0, zero)[None], positive])
    ene = jnp.concatenate([jnp.maximum(-npv0, zero)[None], negative])
    discount_with_t0 = jnp.concatenate([jnp.ones((1,), dtype=dtype), jnp.asarray(discount, dtype=dtype)])
    ee_b = epe / discount_with_t0
    eee_b = jax.lax.cummax(ee_b)

    ordered = jnp.sort(deflated, axis=0)
    pfe = {}
    for q in quantiles:
        key = f"PFE_{quantile_label(q)}"
        pfe[key] = jnp.concatenate([jnp.maximum(npv0, zero)[None], _pfe(ordered, q)])
        if paired is not None:
            estimates[key] = paired_quantile(lambda x, q=q: _pfe(jnp.sort(x, axis=0), q), deflated, paired_deflated)

    epe_b, eepe_b, basel_epe, basel_eepe = _time_weighted(ee_b, eee_b, times, dates, asof, maturity)
    return ExposureProfile(
        times=np.concatenate([[0.0], np.asarray(times, dtype=np.float64)]),
        epe=epe, ene=ene, ee_b=ee_b, eee_b=eee_b, pfe=pfe, epe_b=epe_b, eepe_b=eepe_b,
        basel_epe=basel_epe, basel_eepe=basel_eepe, estimates=estimates,
    )


def _pfe(ordered, quantile: float):
    """ORE's PFE at `quantile` of paths sorted along axis 0: the order statistic
    `floor(q (S - 1) + 0.5)`, floored at 0."""
    index = int(np.floor(quantile * (ordered.shape[0] - 1) + 0.5))
    return jnp.maximum(ordered[index], jnp.zeros((), dtype=ordered.dtype))


def _time_weighted(ee_b, eee_b, times, dates, asof, maturity):
    """ORE's `epe_bTimeWeighted_`/`eepe_bTimeWeighted_` and the Basel scalars (see the
    module docstring)."""
    if dates is not None:
        day_count = ORE.ActualActual(ORE.ActualActual.ISDA)
        weights = np.asarray([day_count.yearFraction(asof, d) for d in dates], dtype=np.float64)
        alive = np.asarray([maturity is None or d <= maturity for d in dates])
    else:
        weights, alive = np.asarray(times, dtype=np.float64), np.ones(len(times), dtype=bool)
    deltas = np.diff(np.concatenate([[0.0], weights]))
    dtype = ee_b.dtype
    scale = jnp.asarray(np.where(alive, deltas, 0.0), dtype=dtype)
    per_date = lambda profile: jnp.where(  # noqa: E731
        jnp.asarray(alive), jnp.cumsum(profile[1:] * scale) / jnp.asarray(weights, dtype=dtype), 0.0)
    epe_b = jnp.concatenate([ee_b[:1], per_date(ee_b)])
    eepe_b = jnp.concatenate([eee_b[:1], per_date(eee_b)])
    if dates is None:
        return epe_b, eepe_b, None, None
    horizon = ORE.WeekendsOnly().adjust(asof + ORE.Period(1, ORE.Years) + 4)
    inside = [j for j, d in enumerate(dates) if d <= horizon and alive[j]]
    if not inside:
        return epe_b, eepe_b, 0.0, 0.0
    return epe_b, eepe_b, float(epe_b[inside[-1] + 1]), float(eepe_b[inside[-1] + 1])


def netting_set_profile(
    npv_cube: jnp.ndarray,
    npv0_per_trade: Sequence[float],
    numeraire: jnp.ndarray,
    discount: jnp.ndarray,
    times: Sequence[float],
    quantiles: Sequence[float] = (0.95, 0.99),
    dates: Optional[Sequence[ORE.Date]] = None,
    asof: Optional[ORE.Date] = None,
    paired: Optional[PairedPaths] = None,
) -> ExposureProfile:
    """Exposure of a single netting set holding every trade in `npv_cube`
    (`[S, T, N]`), without collateral: paths are summed across trades before
    any statistic is taken, so offsetting trades net. `paired`, if given, is the paired
    sample's cube `[n, T, N]` and numeraire `[n, T]`, netted the same way."""
    if paired is not None:
        paired = (jnp.sum(paired[0], axis=-1), paired[1])
    return exposure_profile(
        jnp.sum(npv_cube, axis=-1), float(np.sum(npv0_per_trade)),
        numeraire, discount, times, quantiles, dates=dates, asof=asof, paired=paired,
    )

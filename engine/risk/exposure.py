"""
Exposure profiles over a simulated NPV cube: EPE, ENE, EE_B, EEE_B and PFE.

Counterparty exposure through time, from the multi-step risk-neutral cube of
`price_portfolio`. Not market-risk VaR (see `engine.market_risk`).

The statistics are ORE's `ExposureCalculator` definitions
(OREAnalytics/orea/aggregation/exposurecalculator.cpp), for one trade or one netting set,
without collateral:

    V_k(t)    NPV on path k at t divided by the numeraire N(t) (ORE's cube stores NPV / N)
    EPE(t)    mean_k max(V_k(t), 0)          discounted expected positive exposure
    ENE(t)    mean_k max(-V_k(t), 0)         discounted expected negative exposure
    EE_B(t)   EPE(t) / P(0,t)                undiscounted expected exposure
    EEE_B(t)  max(EEE_B(t-), EE_B(t))        effective (non-decreasing) EE
    PFE_q(t)  max(sorted_k V_k(t)[i], 0),    i = floor(q * (S - 1) + 0.5)

Each profile includes t=0, where ORE sets EPE = EE_B = EEE_B = PFE = max(NPV0, 0) and
ENE = max(-NPV0, 0). A netting set sums paths across trades before the statistics.

Differs from ORE: ORE's time-weighted EPE_B/EEPE_B are not computed. The numeraire is the
simulation's discretely accrued money-market account on rate factor 0, not ORE's LGM
numeraire (I-45), and the cube inherits the simulation's limitations (I-42, I-43, I-44;
I-04).
"""
from dataclasses import dataclass
from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from engine.risk.var_es import quantile_label


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


def exposure_profile(
    npv: jnp.ndarray,
    npv0: float,
    numeraire: jnp.ndarray,
    discount: jnp.ndarray,
    times: Sequence[float],
    quantiles: Sequence[float] = (0.95, 0.99),
) -> ExposureProfile:
    """ORE exposure statistics for one trade or netting set.

    npv: `[S, T]` undiscounted NPV on each path at each simulated date.
    npv0: the t=0 NPV.
    numeraire: `[S, T]` numeraire on each path at each date, `N(0) = 1`.
    discount: `[T]` today's discount factor `P(0,t)` to each date, from the
        curve the numeraire accrues on.
    times: `[T]` the simulated dates as year fractions (t=0 excluded).
    quantiles: PFE quantiles, each in (0, 1).

    Statistics are computed in `npv`'s dtype.
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

    epe = jnp.concatenate([jnp.maximum(npv0, zero)[None], jnp.mean(jnp.maximum(deflated, zero), axis=0)])
    ene = jnp.concatenate([jnp.maximum(-npv0, zero)[None], jnp.mean(jnp.maximum(-deflated, zero), axis=0)])
    discount_with_t0 = jnp.concatenate([jnp.ones((1,), dtype=dtype), jnp.asarray(discount, dtype=dtype)])
    ee_b = epe / discount_with_t0
    eee_b = jax.lax.cummax(ee_b)

    ordered = jnp.sort(deflated, axis=0)
    pfe = {}
    for q in quantiles:
        index = int(np.floor(q * (num_paths - 1) + 0.5))
        pfe[f"PFE_{quantile_label(q)}"] = jnp.concatenate(
            [jnp.maximum(npv0, zero)[None], jnp.maximum(ordered[index], zero)]
        )

    return ExposureProfile(
        times=np.concatenate([[0.0], np.asarray(times, dtype=np.float64)]),
        epe=epe, ene=ene, ee_b=ee_b, eee_b=eee_b, pfe=pfe,
    )


def netting_set_profile(
    npv_cube: jnp.ndarray,
    npv0_per_trade: Sequence[float],
    numeraire: jnp.ndarray,
    discount: jnp.ndarray,
    times: Sequence[float],
    quantiles: Sequence[float] = (0.95, 0.99),
) -> ExposureProfile:
    """Exposure of a single netting set holding every trade in `npv_cube`
    (`[S, T, N]`), without collateral: paths are summed across trades before
    any statistic is taken, so offsetting trades net."""
    return exposure_profile(
        jnp.sum(npv_cube, axis=-1), float(np.sum(npv0_per_trade)),
        numeraire, discount, times, quantiles,
    )

"""
Bootstrap calibration of a piecewise-constant LGM `Sigma` to a co-terminal swaption basket,
the equivalent of ORE's `LgmBuilder::calibrate()` bootstrap path
(`calibrateVolatilitiesIterative`, OREData/ored/model/lgmbuilder.cpp).

As in `LgmBuilder`, the sigma breakpoints are the basket's expiries minus the last, so there
is one bucket per instrument. Instrument `i` depends only on buckets `0..i`, so the
instruments are fitted one at a time in expiry order, each solving for one new bucket
value with the earlier ones held fixed.

Mean reversion `a` is an input, not calibrated (ORE's default, `calibrateH == false`).

Differs from ORE: ORE fits each instrument with its configured optimizer
(Levenberg-Marquardt by default). Here each is a 60-step bisection on a fixed bracket of
[1e-6, 0.20]. The price is increasing in the new bucket's sigma, so bisection converges to
the same root when one exists in the bracket. Market prices are Bachelier (normal-vol)
prices.
"""
from dataclasses import dataclass
from typing import List

import jax
import jax.numpy as jnp

from engine.models.hull_white import ZeroCurve
from engine.models.lgm import Sigma
from engine.calibration.basket import CalibrationTarget, bachelier_swaption_price, price_lgm_swaption


@dataclass
class CalibrationResult:
    """Calibrated `Sigma` plus per-instrument diagnostics, like ORE's
    `LgmCalibrationInfo`: model vs. market price per basket instrument, and their RMSE."""
    sigma: Sigma
    market_prices: jax.Array      # [N], Bachelier price implied by each target's market_vol
    model_prices: jax.Array       # [N], price_lgm_swaption at the final calibrated Sigma
    rmse: float                   # sqrt(mean((model-market)^2))


#: Bisection bracket for one bucket's sigma: [0.01bp, 2000bp] of normal vol.
_SIGMA_BRACKET = (1e-6, 0.20)


def _bisect_bucket_sigma(
    price_fn, market_price: float, lo: float = _SIGMA_BRACKET[0], hi: float = _SIGMA_BRACKET[1],
    iterations: int = 60,
) -> jax.Array:
    """Bisection for the sigma with `price_fn(sigma) == market_price`, on a fixed bracket.

    `price_fn` is increasing in sigma. The bracket is not expanded: a result on the
    ceiling means the market vol is unattainable and `calibrate_lgm_sigma` raises; a
    result on the floor is left to show up in `rmse`."""
    lo_arr, hi_arr = jnp.array(lo), jnp.array(hi)

    def body(carry, _):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        val = price_fn(mid) - market_price
        lo = jnp.where(val < 0.0, mid, lo)
        hi = jnp.where(val < 0.0, hi, mid)
        return (lo, hi), None

    (lo_arr, hi_arr), _ = jax.lax.scan(body, (lo_arr, hi_arr), None, length=iterations)
    return 0.5 * (lo_arr + hi_arr)


def calibrate_lgm_sigma(
    targets: List[CalibrationTarget], curve: ZeroCurve, a: float,
) -> CalibrationResult:
    """
    Bootstrap `targets` (a co-terminal basket in increasing expiry order) into a piecewise
    `Sigma` with one bucket per target.

    For target `i`, the breakpoints are the expiries of targets `0..i-1`; bucket `i`'s value
    is solved so the LGM price matches the target's Bachelier price, keeping earlier
    buckets fixed. A successful bootstrap reprices every instrument up to bisection
    tolerance; `rmse` is materially non-zero only when a bucket hit the bracket floor.

    Works in `curve.pillar_rates.dtype`.
    """
    if len(targets) < 1:
        raise ValueError("calibrate_lgm_sigma requires at least one basket instrument")
    expiries = sorted(t.expiry_time for t in targets)
    if expiries != [t.expiry_time for t in targets]:
        raise ValueError("targets must be supplied in increasing expiry order")

    dtype = curve.pillar_rates.dtype
    bucket_times: List[float] = []       # interior breakpoints calibrated so far
    bucket_values: List[float] = []      # calibrated sigma per bucket so far

    # Jit the closed forms as one program each rather than dispatching every elementwise op
    # (see docs/concepts/profiling.md). The target is closed over, not passed as a static
    # argument, because `engine.risk.greeks.bermudan_vega` puts a tracer in its
    # `market_vol`, so it cannot be hashed.
    def _jit_over_target(fn, target):
        return jax.jit(lambda: fn(target, curve))

    for i, target in enumerate(targets):
        times_arr = jnp.asarray(bucket_times, dtype=dtype)

        def price_fn(new_sigma, _times=times_arr, _values=bucket_values, _target=target):
            values_arr = jnp.asarray(_values + [new_sigma], dtype=dtype)
            sigma = Sigma(times=_times, values=values_arr)
            return price_lgm_swaption(curve, a, sigma, _target)

        market_price = float(_jit_over_target(bachelier_swaption_price, target)())
        new_value = float(_bisect_bucket_sigma(price_fn, market_price))
        # Saturating at the ceiling means the market vol is out of range: refuse.
        # Saturating at the floor happens on a spike-then-dip vol curve, where earlier
        # buckets already carry too much variance; it is reported through `rmse`
        # (test_calibration_edge_cases.py::test_large_dip_after_a_spike_cannot_reprice_exactly).
        if new_value >= _SIGMA_BRACKET[1] * (1.0 - 1e-9):
            raise ValueError(
                f"calibration target {i} (expiry t={target.expiry_time}, market_vol="
                f"{target.market_vol}) is not attainable with a bucket sigma in "
                f"{list(_SIGMA_BRACKET)}; the bisection converged onto the bracket bound "
                f"{new_value}. Check the market vol and the basket."
            )

        bucket_values.append(new_value)
        if i < len(targets) - 1:
            bucket_times.append(target.expiry_time)

    final_sigma = Sigma(
        times=jnp.asarray(bucket_times, dtype=dtype),
        values=jnp.asarray(bucket_values, dtype=dtype),
    )

    # Diagnostics: reprice every instrument at the final Sigma.
    market_prices = jnp.asarray([
        float(_jit_over_target(bachelier_swaption_price, t)()) for t in targets
    ])
    model_prices = jnp.asarray([
        float(jax.jit(lambda _t=t: price_lgm_swaption(curve, a, final_sigma, _t))())
        for t in targets
    ])
    rmse = float(jnp.sqrt(jnp.mean((model_prices - market_prices) ** 2)))

    return CalibrationResult(
        sigma=final_sigma, market_prices=market_prices, model_prices=model_prices, rmse=rmse,
    )

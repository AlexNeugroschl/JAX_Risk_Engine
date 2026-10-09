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
(Levenberg-Marquardt by default). Here each bucket's sigma is the root on a fixed bracket of
[1e-6, 0.20], by the configured solver (`engine.numerics.roots`, decision A-21). The price is
increasing in the new bucket's sigma, so the root is the same when one exists in the bracket.
Market prices are Bachelier (normal-vol) prices.
"""
from dataclasses import dataclass
from functools import partial
from typing import List

import jax
import jax.numpy as jnp

from engine.models.hull_white import ZeroCurve
from engine.models.lgm import Sigma
from engine.calibration.basket import CalibrationTarget, bachelier_swaption_price, price_lgm_swaption
from engine.numerics.roots import DEFAULT_SOLVER, Steps, check_solver, solve


@dataclass
class CalibrationResult:
    """Calibrated `Sigma` plus per-instrument diagnostics, like ORE's
    `LgmCalibrationInfo`: model vs. market price per basket instrument, and their RMSE."""
    sigma: Sigma
    market_prices: jax.Array      # [N], Bachelier price implied by each target's market_vol
    model_prices: jax.Array       # [N], price_lgm_swaption at the final calibrated Sigma
    rmse: float                   # sqrt(mean((model-market)^2))


#: The bracket of one bucket's sigma: [0.01bp, 2000bp] of normal vol.
_SIGMA_BRACKET = (1e-6, 0.20)

#: Steps of one bucket's sigma: Bisection's the count before Newton (A-21), Newton's measured
#: (2026-10-07: a bucket reaches its rounding in 5 steps; two of margin, tests/test_root_solvers.py).
_SIGMA_STEPS = Steps(bisection=60, newton=7)


#: The closed forms, jitted with the target as an argument: one program per target shape.
_market_price = jax.jit(bachelier_swaption_price)
_model_price = jax.jit(price_lgm_swaption, static_argnames="solver")


@partial(jax.jit, static_argnames="solver")
def _solve_bucket_sigma(curve: ZeroCurve, a, times, earlier, target: CalibrationTarget, market_price, start,
                        solver: str) -> jax.Array:
    """The next bucket's sigma on `_SIGMA_BRACKET`, the buckets before it (`earlier`, ending at
    `times`) held fixed, with the target's LGM price equal to `market_price`; Newton starts at
    `start`.

    The price is increasing in sigma. The bracket is not expanded: a result on the ceiling
    means the market vol is unattainable and `calibrate_lgm_sigma` raises; a result on the
    floor is left to show up in `rmse`."""
    def residual(new_sigma, market_price):
        values = jnp.concatenate([earlier, jnp.reshape(new_sigma, (1,))]).astype(earlier.dtype)
        return price_lgm_swaption(curve, a, Sigma(times=times, values=values), target, solver) - market_price

    return solve(residual, market_price, jnp.asarray(start, dtype=earlier.dtype), solver=solver, steps=_SIGMA_STEPS,
                 increasing=True, bracket=_SIGMA_BRACKET)


def calibrate_lgm_sigma(
    targets: List[CalibrationTarget], curve: ZeroCurve, a: float, solver: str = DEFAULT_SOLVER,
) -> CalibrationResult:
    """
    Bootstrap `targets` (a co-terminal basket in increasing expiry order) into a piecewise
    `Sigma` with one bucket per target.

    For target `i`, the breakpoints are the expiries of targets `0..i-1`; bucket `i`'s value
    is solved by `solver` so the LGM price matches the target's Bachelier price, keeping
    earlier buckets fixed (Newton starts each at its target's market vol, never at an edge of
    the bracket, where its slope vanishes). A successful bootstrap reprices every instrument to
    rounding;
    `rmse` is materially non-zero only when a bucket hit the bracket floor.

    Works in `curve.pillar_rates.dtype`.
    """
    check_solver(solver)
    if len(targets) < 1:
        raise ValueError("calibrate_lgm_sigma requires at least one basket instrument")
    expiries = sorted(t.expiry_time for t in targets)
    if expiries != [t.expiry_time for t in targets]:
        raise ValueError("targets must be supplied in increasing expiry order")

    dtype = curve.pillar_rates.dtype
    bucket_times: List[float] = []       # interior breakpoints calibrated so far
    bucket_values: List[float] = []      # calibrated sigma per bucket so far

    for i, target in enumerate(targets):
        market_price = float(_market_price(target, curve))
        new_value = float(_solve_bucket_sigma(curve, a, jnp.asarray(bucket_times, dtype=dtype),
                                              jnp.asarray(bucket_values, dtype=dtype), target, market_price,
                                              target.market_vol, solver))
        # Saturating at the ceiling means the market vol is out of range: refuse.
        # Saturating at the floor happens on a spike-then-dip vol curve, where earlier
        # buckets already carry too much variance; it is reported through `rmse`
        # (test_calibration_edge_cases.py::test_large_dip_after_a_spike_cannot_reprice_exactly).
        if new_value >= _SIGMA_BRACKET[1] * (1.0 - 1e-9):
            raise ValueError(
                f"calibration target {i} (expiry t={target.expiry_time}, market_vol="
                f"{target.market_vol}) is not attainable with a bucket sigma in "
                f"{list(_SIGMA_BRACKET)}; the solve reached the bracket bound "
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
    market_prices = jnp.asarray([float(_market_price(t, curve)) for t in targets])
    model_prices = jnp.asarray([float(_model_price(curve, a, final_sigma, t, solver=solver)) for t in targets])
    rmse = float(jnp.sqrt(jnp.mean((model_prices - market_prices) ** 2)))

    return CalibrationResult(
        sigma=final_sigma, market_prices=market_prices, model_prices=model_prices, rmse=rmse,
    )

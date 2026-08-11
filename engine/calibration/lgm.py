"""
Bootstrap calibration of a piecewise-constant LGM `Sigma` term structure to
a co-terminal basket of market swaption volatilities -- the JAX-native
analogue of `ore::data::LgmBuilder::calibrate()`'s bootstrap path
(`lgmModel->calibrateVolatilitiesIterative(...)`, `OREData/ored/model/
lgmbuilder.cpp` lines 209-212).

**Bootstrap, not joint least-squares -- matching ORE's own default, not a
simplification of it.** ORE's `LgmBuilder::initParametrization` sets the
piecewise sigma's own bucket breakpoints (`aTimes`) to the calibration
basket's OWN swaption expiry times, dropping the basket's last expiry
(`aTimes = swaptionExpiries_[:-1]`, `N` expiries producing `N-1` interior
breakpoints for `N` buckets -- exactly `engine.models.lgm.Sigma`'s own
`len(values) == len(times)+1` invariant). This construction makes the
calibration triangular by design: the `i`-th co-terminal swaption's price
depends on `zeta(t)` only up to its own expiry `T0_i`, which in turn
depends ONLY on sigma buckets `0..i` (buckets `i+1..N-1` cover times AFTER
`T0_i`, contributing nothing to `zeta(T0_i)`). So each new swaption pins
down EXACTLY one new free sigma value via a single 1D root-find, holding
every earlier bucket fixed at its own already-calibrated value -- this is
what `calibrateVolatilitiesIterative` actually does (see
`QuantLib::CalibratedModel::calibrateIterative`, called per-instrument in
basket order), not a rebranding of joint calibration.

**Why mean reversion (kappa/`a`) is not calibrated.** ORE's own default
(`LgmData::calibrateH() == false` unless a trade config explicitly opts
in) fixes mean reversion and calibrates ONLY the piecewise sigma term
structure -- the one-parameter-per-instrument bootstrap this module
implements requires exactly this (jointly calibrating both `a` and
`sigma` from the SAME basket is a fundamentally different, non-bootstrap
problem -- ORE's own `calibrate()` for that case requires `BestFit`, not
`Bootstrap`, and a full multi-parameter optimizer, out of this module's
scope). `a` is therefore always an input to `calibrate_lgm_sigma` below,
never an output.

**Each bucket's root-find**, not a full Levenberg-Marquardt run per
bucket: since `price_lgm_swaption` is monotonically increasing in the
NEWEST sigma bucket's own value (confirmed directly:
`TestPriceLgmSwaptionSanity::test_higher_sigma_gives_higher_price`,
`tests/test_calibration_basket.py`) -- a swaption is long volatility, and
raising the newest bucket's sigma strictly increases the model's total
`zeta(T0_i)` for every `t` in that bucket -- a single well-posed scalar
bisection recovers the unique sigma matching the market price exactly,
with no need for the generality (or cost) of a full LM optimizer for a
1-parameter-at-a-time problem. `engine/calibration/optimizer.py`'s
JAX-native LM implementation is reserved for cases needing a genuinely
joint multi-parameter fit (e.g. a future `BestFit`-style calibration
mode) -- bootstrap does not need it, matching ORE's own architectural
split between the two calibration types.
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
    """Output of `calibrate_lgm_sigma`: the fitted piecewise `Sigma`, plus
    per-instrument diagnostics -- mirrors the fields ORE's own
    `LgmCalibrationInfo`/`getCalibrationDetails` report (model vs. market
    price per basket instrument, and the overall RMSE), so a caller can
    verify calibration quality the same way `LgmBuilder::calibrate`'s own
    DLOG output does."""
    sigma: Sigma
    market_prices: jax.Array      # [N], Bachelier price implied by each target's market_vol
    model_prices: jax.Array       # [N], price_lgm_swaption at the final calibrated Sigma
    rmse: float                   # sqrt(mean((model-market)^2)), ORE's own `error_` metric


def _bisect_bucket_sigma(
    price_fn, market_price: float, lo: float = 1e-6, hi: float = 0.20, iterations: int = 60,
) -> jax.Array:
    """Scalar bisection for one bucket's sigma value: `price_fn(sigma) ==
    market_price`. `price_fn` is strictly increasing in `sigma` (a
    swaption is long volatility -- see module docstring), so the bracket
    `[lo, hi]` = [1bp, 2000bp] safely contains any realistic calibrated
    volatility; the bracket is NOT expanded dynamically (unlike
    `european_swaption._bisect_rstar`) since a piecewise LGM sigma bucket
    calibrating outside a 0.01%-20% annual vol range indicates a
    misconfigured basket (e.g. a market vol far outside typical rates
    levels), not a case this bootstrap should silently paper over."""
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
    Bootstrap-calibrates a piecewise `Sigma` to `targets` (a co-terminal
    basket from `engine.calibration.basket.build_coterminal_basket`, in
    increasing expiry order), matching ORE's own
    `calibrateVolatilitiesIterative` bootstrap (see module docstring):

    For each target `i` (in order):
      1. The bucket breakpoints so far are `targets[0].expiry_time,
         ..., targets[i-1].expiry_time` (every EARLIER target's own
         expiry becomes an interior breakpoint -- exactly
         `LgmBuilder::initParametrization`'s `aTimes = swaptionExpiries_
         [:-1]` convention, since target `i`'s own expiry is never itself
         a breakpoint until a LATER target makes it one).
      2. Solve for bucket `i`'s sigma value (the newest, currently-open
         bucket covering `[targets[i-1].expiry_time, inf)`) such that
         `price_lgm_swaption` at target `i` matches its own Bachelier
         market price, via a 1D bisection (see `_bisect_bucket_sigma`) --
         every earlier bucket's value stays fixed at whatever it was
         already calibrated to.

    Returns a `CalibrationResult` with the final N-bucket `Sigma` (one
    bucket per target) and per-instrument model/market price diagnostics
    at that final Sigma (an exact bootstrap reprices every instrument
    exactly, up to root-find tolerance -- unlike a joint BestFit, whose
    RMSE is generally nonzero even at convergence).
    """
    assert len(targets) >= 1, "calibrate_lgm_sigma requires at least one basket instrument"
    expiries = sorted(t.expiry_time for t in targets)
    assert expiries == [t.expiry_time for t in targets], "targets must be supplied in increasing expiry order"

    bucket_times: List[float] = []       # interior breakpoints calibrated so far
    bucket_values: List[float] = []      # calibrated sigma per bucket so far

    for i, target in enumerate(targets):
        times_arr = jnp.asarray(bucket_times, dtype=jnp.float64)

        def price_fn(new_sigma, _times=times_arr, _values=bucket_values, _target=target):
            values_arr = jnp.asarray(_values + [new_sigma], dtype=jnp.float64)
            sigma = Sigma(times=_times, values=values_arr)
            return price_lgm_swaption(curve, a, sigma, _target)

        market_price = float(bachelier_swaption_price(target, curve))
        new_value = float(_bisect_bucket_sigma(price_fn, market_price))

        bucket_values.append(new_value)
        if i < len(targets) - 1:
            bucket_times.append(target.expiry_time)

    final_sigma = Sigma(
        times=jnp.asarray(bucket_times, dtype=jnp.float64),
        values=jnp.asarray(bucket_values, dtype=jnp.float64),
    )

    market_prices = jnp.asarray([float(bachelier_swaption_price(t, curve)) for t in targets])
    model_prices = jnp.asarray([float(price_lgm_swaption(curve, a, final_sigma, t)) for t in targets])
    rmse = float(jnp.sqrt(jnp.mean((model_prices - market_prices) ** 2)))

    return CalibrationResult(
        sigma=final_sigma, market_prices=market_prices, model_prices=model_prices, rmse=rmse,
    )

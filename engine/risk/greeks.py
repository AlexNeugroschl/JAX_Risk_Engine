"""
Greeks by automatic differentiation (`GreeksConfig.method = "AD"`, decision A-5): the other
method beside ORE's bump-and-revalue sensitivities (`engine.risk.sensitivities`, the default),
with the same keys, for any trade, engine and simulation model.

Per trade, in the base currency (`portfolio_greeks`):

  * `delta:<kind>:<name>` / `gamma:<kind>:<name>`: per pillar of each market curve the trade
    reads (`engine.risk.price_functions.curve_keys`), dNPV/dz_i * shift and
    d^2NPV/dz_i^2 * shift^2 for an absolute zero-rate shift `SensitivityConfig.curve_shift`: the
    shift -> 0 limit of ORE's forward-difference Delta and its Gamma, on the market curve's
    pillars (ORE's sensitivity market samples the curves at `SensitivityConfig.curve_tenors`
    instead). Only the diagonal Gamma is computed, as ORE computes no cross-gammas by default.
  * `vega:<ccy>` `[option tenors, swap tenors]`: per swaption volatility quote,
    dNPV/dquote * `vol_shift`, for a trade whose engine reads the quotes. The volatility read
    is linear in the quotes (`SwaptionVolSurface.weights`); a European's price is
    differentiated in its volatility, a Bermudan's/American's through its calibration by the
    implicit function theorem (`_bootstrap_jacobian`: a quote moves the helpers' volatilities,
    which move every later bucket of the bootstrap).
  * `theta`: ORE's Theta, the same function as the bump method's (a roll in time, not a
    derivative).

Differs from the bump method beyond the shift size: a Bermudan's/American's Delta and Gamma
hold its calibrated LGM fixed while the curve moves, where ORE recalibrates under each bump.
A test of the two agreeing as the shift halves is feature F-01.
"""
from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from engine.calibration.ore_lgm import price_pair
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.market import Market
from engine.models.curves import ZeroCurve
from engine.risk.price_functions import (
    bermudan_price_function, curve_keys, curves_of, european_price_function, market_curve, trade_price_function,
    vol_point,
)
from engine.risk.sensitivities import SensitivityConfig, sensitivity_context, theta_context, trade_theta
from engine.valuation.bermudan import calibration_basket
from engine.valuation.config import PricingConfig
from engine.valuation.portfolio import Trade, reads_swaption_vols, validate_trades, value_on


def _grad_and_hessian_diagonal(price_fn, x, *rest):
    """`(df/dx_i, d^2f/dx_i^2)` for every `i`, without building the Hessian.

    Each diagonal entry is one Hessian-vector product against a basis vector, batched
    with `vmap`; checked against `jnp.diagonal(jax.hessian(...))` in
    `tests/test_profiling_and_jit.py::TestHessianDiagonalEquivalence`.
    Differentiates with respect to the first argument; `rest` is held fixed.

    Both outputs come from one jitted program. `price_fn` is a fresh closure on every
    call, so it recompiles once per call (I-21; read that entry before memoizing it, since
    a wrongly keyed memo returns a program compiled for a different trade).
    """
    def combined(xi, *fixed):
        def f(inner):
            return price_fn(inner, *fixed)

        grad = jax.grad(f)(xi)
        basis = jnp.eye(xi.shape[0], dtype=xi.dtype)

        def hvp(v):
            return jax.jvp(jax.grad(f), (xi,), (v,))[1]

        rows = jax.vmap(hvp)(basis)   # [n, n]; only its diagonal escapes
        return grad, jnp.diagonal(rows)

    return jax.jit(combined)(x, *rest)


def portfolio_greeks(trades: Sequence[Trade], market: Market, base_currency: str,
                     pricing: PricingConfig = PricingConfig(),
                     config: SensitivityConfig = SensitivityConfig()) -> Dict[int, Dict[str, np.ndarray]]:
    """Per trade (by request index), the AD Greeks in the base currency (see the module
    docstring)."""
    validate_trades(trades, market, pricing)
    base_context, theta_ctx = sensitivity_context(market, config), theta_context(market, config)
    result: Dict[int, Dict[str, np.ndarray]] = {}
    for i, cfg in enumerate(trades):
        fx = market.fx_spot(cfg.currency, base_currency)
        greeks = curve_greeks(cfg, market, pricing, config.curve_shift)
        greeks = {key: value * fx for key, value in greeks.items()}
        vega = vega_greek(cfg, market, pricing, config.vol_shift)
        if vega is not None:
            greeks[f"vega:{cfg.currency}"] = vega * fx
        value = lambda context, _cfg=cfg: value_on(_cfg, context, pricing) * fx  # noqa: E731
        greeks["theta"] = trade_theta(value, value(base_context), cfg, theta_ctx, fx)
        result[i] = greeks
    return result


def curve_greeks(cfg, market: Market, pricing: PricingConfig, shift: float) -> Dict[str, np.ndarray]:
    """Delta and Gamma per pillar of each curve the trade reads, in its currency."""
    fn = trade_price_function(cfg, market, pricing)
    rates = curves_of(fn, market, jnp.float64)
    out: Dict[str, np.ndarray] = {}
    for k, (kind, name) in enumerate(fn.curves):
        def along(x, *others, _k=k):
            args = list(others)
            args.insert(_k, x)
            return fn.price(*args)

        delta, gamma = _grad_and_hessian_diagonal(along, rates[k], *(rates[:k] + rates[k + 1:]))
        out[f"delta:{kind}:{name}"] = np.asarray(delta) * shift
        out[f"gamma:{kind}:{name}"] = np.asarray(gamma) * shift ** 2
    return out


def vega_greek(cfg, market: Market, pricing: PricingConfig, shift: float):
    """`[option tenors, swap tenors]` Vega per quote in the trade's currency, or None when its
    engine reads no volatility."""
    if not reads_swaption_vols(cfg, pricing):
        return None
    surface = market.swaption_vols(cfg.currency)
    asof = market.asof
    disc, index = (ZeroCurve.from_config(market_curve(market, key)) for key in curve_keys(cfg))
    if isinstance(cfg, SwaptionConfig):
        if not cfg.exercise_date > asof:
            return np.zeros((len(surface.option_tenors), len(surface.swap_tenors)))
        price = european_price_function(cfg, market, jnp.float64)
        t, swap_len = vol_point(cfg, market)
        volatility = jnp.asarray(float(surface.volatility(asof, t, swap_len)))
        d_npv = float(jax.grad(lambda v: price(disc, index, v))(volatility))
        return d_npv * surface.weights(asof, t, swap_len) * shift
    option = bermudan_price_function(cfg, market, pricing, jnp.float64)
    if option.calibration is None:
        return np.zeros((len(surface.option_tenors), len(surface.swap_tenors)))
    d_npv_d_sigma = np.asarray(jax.grad(lambda values: option.price(disc, index, values))(option.sigma.values))
    engine = pricing.american if isinstance(cfg, AmericanSwaptionConfig) else pricing.bermudan
    basket = calibration_basket(cfg, engine, asof, asof)
    jacobian = _bootstrap_jacobian(basket, disc, index, surface, asof, engine.reversion, option.sigma)
    weights = np.stack([surface.weights(asof, b.vol_option_time, b.vol_swap_length) for b in basket])
    return np.tensordot(d_npv_d_sigma @ jacobian, weights, axes=1) * shift


def _bootstrap_jacobian(basket, disc, index, surface, asof, reversion: float, sigma) -> np.ndarray:
    """`J[j, h] = d sigma_j / d v_h`: how each bucket of the bootstrap (`engine.calibration.
    ore_lgm.bootstrap_sigma`) moves with each helper's volatility v_h. Bucket j solves
    g_j = model_j(zeta_j) - market_j(v_j) = 0 with zeta_j = sum_{k<=j} sigma_k^2 dt_k, so by
    the implicit function theorem, row by row (forward substitution over a lower-triangular
    system),

        J[j] = -(sum_{k<j} dg_j/dsigma_k J[k] + e_j dg_j/dv_j) / (dg_j/dsigma_j),
        dg_j/dsigma_k = dmodel_j/dzeta * 2 sigma_k dt_k.

    dg_j/dv_j includes the model's side: a deal strike beyond 3 ATM standard deviations is
    clipped there, so the helper's strike moves with its volatility."""
    values = np.asarray(sigma.values, dtype=np.float64)
    expiries = np.array([b.expiry_time for b in basket])
    dt = np.diff(np.concatenate([[0.0], expiries]))
    vols = [float(surface.volatility(asof, b.vol_option_time, b.vol_swap_length)) for b in basket]
    n = len(basket)
    J = np.zeros((n, n))
    for j, helper in enumerate(basket):
        zeta = float(np.sum(values[: j + 1] ** 2 * dt[: j + 1]))

        def residual(v, z, _helper=helper):
            market, model = price_pair(_helper, disc, index, v, reversion, z)
            return model - market

        dg_dv, dg_dzeta = (float(d) for d in jax.grad(residual, argnums=(0, 1))(
            jnp.asarray(vols[j]), jnp.asarray(zeta)))
        dg_dsigma = dg_dzeta * 2.0 * values[: j + 1] * dt[: j + 1]
        row = -(dg_dsigma[:j] @ J[:j]) if j > 0 else np.zeros(n)
        row[j] -= dg_dv
        J[j] = row / dg_dsigma[j]
    return J

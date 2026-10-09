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

**Compiled programs**. Each derivative is a module-level jitted function of the
trade's price function as data (`TradePriceFunction.pricer`, static, and its pytree `terms`):
every curve's Delta and Gamma is one program per product and shape (`_curve_derivatives`,
from one linearization of the gradient over all its curves), each Vega gradient another. So a
trade's Greeks are at most two programs, compiled once and shared by every trade of that shape,
whatever its dates, rates or size. Differentiating the jitted pricers eagerly instead compiled
a forward and a backward program per curve and per derivative (about ten per option), and kept
them only while JAX's internal caches of 2,048 traced programs did: a job that traced more
than that evicted the first trades' derivative programs, so the same job compiled them again
on its next run.
"""
from functools import partial
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
    bachelier_price, bachelier_terms, bermudan_price_function, curve_keys, curves_of, market_curve, on_pillars,
    trade_price_function, vol_point,
)
from engine.risk.sensitivities import SensitivityConfig, sensitivity_context, theta_context, trade_theta
from engine.valuation.bermudan import calibration_basket
from engine.valuation.config import PricingConfig
from engine.valuation.portfolio import Trade, reads_swaption_vols, validate_trades, value_on


def _gradients_and_hessian_diagonals(f, xs):
    """Per array `x` of the tuple `xs`, `(df/dx_i, d^2f/dx_i^2)` for every `i`, without
    building the Hessian.

    One linearization of the gradient (`jax.linearize`), so the forward and backward pass run
    once for every array; each diagonal entry is then the Hessian-vector product along a basis
    vector, all of them batched with `vmap`. Checked against `jnp.diagonal(jax.hessian(...))`
    in `tests/test_profiling_and_jit.py::TestHessianDiagonalEquivalence`.
    """
    gradients, hessian_times = jax.linearize(jax.grad(f), xs)
    sizes = [x.shape[0] for x in xs]
    offsets = np.cumsum([0] + sizes)
    # Row j of the identity, split into one block per array: the j-th basis vector of `xs`.
    basis = tuple(jnp.split(jnp.eye(offsets[-1], dtype=xs[0].dtype), offsets[1:-1], axis=1))
    rows = jax.vmap(hessian_times)(basis)   # per array, [sum(sizes), size]; only its diagonal block escapes
    return tuple((gradients[k], jnp.diagonal(rows[k][offsets[k]:offsets[k + 1]])) for k in range(len(xs)))


def portfolio_greeks(trades: Sequence[Trade], market: Market, base_currency: str,
                     pricing: PricingConfig = PricingConfig(),
                     config: SensitivityConfig = SensitivityConfig()) -> Dict[int, Dict[str, np.ndarray]]:
    """Per trade (by request index), the AD Greeks in the base currency (see the module
    docstring)."""
    # Not at module scope: importing engine.portfolio runs its __init__, which imports engine.risk.
    from engine.portfolio.profiling import trade_greeks_phase

    validate_trades(trades, market, pricing)
    base_context, theta_ctx = sensitivity_context(market, config), theta_context(market, config)
    result: Dict[int, Dict[str, np.ndarray]] = {}
    for i, cfg in enumerate(trades):
        with trade_greeks_phase(i, cfg):
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
    derivatives = _curve_derivatives(fn.pricer, fn.terms, fn.times, tuple(curves_of(fn, market, jnp.float64)))
    out: Dict[str, np.ndarray] = {}
    for (kind, name), (delta, gamma) in zip(fn.curves, derivatives):
        out[f"delta:{kind}:{name}"] = np.asarray(delta) * shift
        out[f"gamma:{kind}:{name}"] = np.asarray(gamma) * shift ** 2
    return out


@partial(jax.jit, static_argnums=0)
def _curve_derivatives(pricer, terms, times, rates):
    """Per curve, `(dNPV/dz, diag d^2NPV/dz^2)` in its pillar rates `z`: one program per
    pricer and trade shape for every curve (see the module docstring)."""
    return _gradients_and_hessian_diagonals(lambda moved: pricer(terms, *on_pillars(times, moved)), rates)


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
        t, swap_len = vol_point(cfg, market)
        volatility = jnp.asarray(float(surface.volatility(asof, t, swap_len)))
        d_npv = float(_bachelier_vega(bachelier_terms(cfg, market, jnp.float64), disc, index, volatility))
        return d_npv * surface.weights(asof, t, swap_len) * shift
    option = bermudan_price_function(cfg, market, pricing, jnp.float64)
    if option.calibration is None:
        return np.zeros((len(surface.option_tenors), len(surface.swap_tenors)))
    d_npv_d_sigma = np.asarray(_option_vega(option.pricer, option.terms, disc, index, option.sigma.values))
    engine = pricing.american if isinstance(cfg, AmericanSwaptionConfig) else pricing.bermudan
    basket = calibration_basket(cfg, engine, asof, asof)
    jacobian = _bootstrap_jacobian(basket, disc, index, surface, asof, engine.reversion, option.sigma, engine.solver)
    weights = np.stack([surface.weights(asof, b.vol_option_time, b.vol_swap_length) for b in basket])
    return np.tensordot(d_npv_d_sigma @ jacobian, weights, axes=1) * shift


@jax.jit
def _bachelier_vega(terms, disc, index, volatility):
    """dNPV/d(normal volatility) of a European on Bachelier's engine."""
    return jax.grad(lambda v: bachelier_price(terms, disc, index, v))(volatility)


@partial(jax.jit, static_argnums=0)
def _option_vega(pricer, terms, disc, index, values):
    """dNPV/d(LGM volatility bucket) of a Bermudan/American on its grid engine."""
    return jax.grad(lambda v: pricer(terms, disc, index, v))(values)


def _bootstrap_jacobian(basket, disc, index, surface, asof, reversion: float, sigma, solver: str) -> np.ndarray:
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
        dg_dv, dg_dzeta = (float(d) for d in _residual_gradient(jnp.asarray(vols[j]), jnp.asarray(zeta), helper,
                                                                disc, index, reversion, solver))
        dg_dsigma = dg_dzeta * 2.0 * values[: j + 1] * dt[: j + 1]
        row = -(dg_dsigma[:j] @ J[:j]) if j > 0 else np.zeros(n)
        row[j] -= dg_dv
        J[j] = row / dg_dsigma[j]
    return J


def _residual(v, z, helper, disc, index, reversion, solver):
    """A bootstrap bucket's g = model - market as a function of the helper's volatility `v`
    and zeta at its expiry `z` (y* by the engine's `solver`)."""
    market, model = price_pair(helper, disc, index, v, reversion, z, solver)
    return model - market


#: `(dg/dv, dg/dzeta)`, one program per helper shape and solver.
_residual_gradient = jax.jit(jax.grad(_residual, argnums=(0, 1)), static_argnums=6)

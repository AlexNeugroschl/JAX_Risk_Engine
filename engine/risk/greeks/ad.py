"""
Greeks by automatic differentiation (`GreeksConfig.method = "AD"`, decision A-5): ORE's
sensitivity analysis (`engine.risk.greeks.bump`, the default) in the limit of a vanishing shift,
with the same keys and factors, for any trade, engine and simulation model.

ORE computes AD sensitivities the same way it bumps (`UseAD` with a `SensitivityDelta` run: the
engine's gradient in its market's parameters, taken once on the sensitivity market, then each
shift scenario's NPV is the base plus the gradient times the scenario's move,
OREData/ored/scripting/engines/amccgbaseengine.cpp), so an AD Greek here is the derivative of the
price on ORE's sensitivity market (`trade_price_function(..., tenors=...)`: every curve sampled at
`SensitivityConfig.curve_tenors`, log-linear in the discount factor) in what ORE shifts. Per
trade, in the base currency (`portfolio_greeks`):

  * `delta:<kind>:<name>` / `gamma:<kind>:<name>` `[K]`: per tenor of each curve the trade reads
    (`engine.risk.greeks.price_functions.curve_keys`), dNPV/dz_k * shift and
    d^2NPV/dz_k^2 * shift^2 in the tenor's zero rate z_k, for an absolute shift
    `SensitivityConfig.curve_shift`. ORE's forward-difference Delta is
    `Delta + Gamma / 2 + O(shift^3)` of these and its Gamma `Gamma + O(shift^4)`
    (tests/test_greeks.py's `TestAgainstTheBumpMethod`, tests/test_sensitivity_ore_parity.py for
    the bump Greeks against ORE). Only the diagonal Gamma, as ORE computes no cross-gammas by
    default.
  * `vega:<ccy>` `[option tenors, swap tenors]`: per swaption volatility quote, dNPV/dquote *
    `vol_shift`, for a trade whose engine reads the quotes: a European's volatility is linear in
    them; a calibrated Bermudan's/American's LGM moves with them through its calibration.
  * `theta`: ORE's Theta, the same function as the bump method's (a roll in time, not a
    derivative).

A calibrated Bermudan/American calibrates inside its price function, so its Delta and Gamma move
the calibration with the curve, as ORE's sensitivity analysis recalibrates under every shift, and
its Vega moves it with the quotes; each by the implicit function theorem on the whole calibration
(`engine.calibration.ore_lgm.calibrated_values`).

**Where AD is not ORE's Greek: the grid engine** (I-78). A Bermudan's/American's price on the
LGM grid (ORE's own, to 1e-11) is piecewise smooth in the rates: its exercise switches on the
state grid's nodes. Its AD Greeks are the derivatives of that price at the market, exact
(central differences at steps small enough to stay on one smooth piece agree to 1e-9), while
ORE's 1bp bump is a secant across the grid's kinks: on the shared portfolio the AD Delta was
within 0.4% of each curve's largest bump Delta less half its Gamma, and the bump Gamma, a
second difference across kinks, is not a curvature at all (tests/test_greeks_bermudan.py). An AD
run with such a trade says so (`ad_warnings`). Swaps, bonds and Europeans are smooth, and their
AD Greeks are ORE's in the limit.

**Compiled programs**. A trade's Delta, Gamma and Vega are one module-level jitted program
(`_derivatives`) of its price function as data (`TradePriceFunction.pricer`, static, and its
pytree `terms`), by forward differentiation along every node value and quote
(`_directional_derivatives`). So a product and shape compiles once, whatever the trade's dates,
rates or size, and is shared by every trade of that shape. Differentiating the jitted pricers
eagerly instead compiled a forward and a backward program per curve and per derivative (about
ten per option), and kept them only while JAX's internal caches of 2,048 traced programs did: a
job that traced more than that evicted the first trades' derivative programs, so the same job
compiled them again on its next run.
"""
from functools import partial
from typing import Dict, List, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.market_data.market import Market
from engine.pricing.config import PricingConfig
from engine.pricing.cube import Trade, validate_trades, value_on
from engine.risk.greeks.bump import SensitivityConfig, sensitivity_context, theta_context, trade_theta
from engine.risk.greeks.price_functions import curves_on, trade_price_function


def _directional_derivatives(f, xs, q):
    """Per array `x` of the tuple `xs`, `(df/dx_i, d^2f/dx_i^2)` for every `i`, and `df/dq` of
    every entry of `q` (None when `q` is), all by forward differentiation.

    Along each basis direction of `xs`, one forward pass of a forward pass (`jax.jvp` of
    `jax.jvp`) gives the first and the second directional derivative together; along each entry
    of `q`, one forward pass. The directions are batched with `vmap`, the price itself not: its
    inputs are the same in every direction, only the tangents carry the batch. Forward
    differentiation holds no tape: an option recalibrated under the shift moves its LGM's grid
    and rollback operator with the curve, and the reverse pass held their tangents on every row
    of its induction for every direction (13 GB for an American on ORE's 301-node grid, against
    2 GB forward; 2026-10-17). Checked against `jax.grad` and `jnp.diagonal(jax.hessian(...))`
    in `tests/test_profiling_and_jit.py::TestHessianDiagonalEquivalence`.
    """
    sizes = [x.shape[0] for x in xs]
    cuts = list(np.cumsum(sizes)[:-1])
    flat = jnp.concatenate(xs)
    price = lambda x: f(tuple(jnp.split(x, cuts)), q)  # noqa: E731

    def along(direction):
        return jax.jvp(lambda x: jax.jvp(price, (x,), (direction,))[1], (flat,), (direction,))

    first, second = jax.vmap(along)(jnp.eye(flat.shape[0], dtype=flat.dtype))
    curves = tuple(zip(jnp.split(first, cuts), jnp.split(second, cuts)))
    if q is None:
        return curves, None
    entries = q.reshape(-1)
    vega = jax.vmap(lambda d: jax.jvp(lambda v: f(xs, v.reshape(q.shape)), (entries,), (d,))[1])(
        jnp.eye(entries.shape[0], dtype=entries.dtype))
    return curves, vega.reshape(q.shape)


def portfolio_greeks(trades: Sequence[Trade], market: Market, base_currency: str,
                     pricing: PricingConfig = PricingConfig(),
                     config: SensitivityConfig = SensitivityConfig()) -> Dict[int, Dict[str, np.ndarray]]:
    """Per trade (by request index), the AD Greeks in the base currency (see the module
    docstring)."""
    # Not at module scope: importing engine.run runs its __init__, which imports engine.risk.
    from engine.run.trace import trade_greeks_phase

    validate_trades(trades, market, pricing)
    base_context, theta_ctx = sensitivity_context(market, config), theta_context(market, config)
    result: Dict[int, Dict[str, np.ndarray]] = {}
    for i, cfg in enumerate(trades):
        with trade_greeks_phase(i, cfg):
            fx = market.fx_spot(cfg.currency, base_currency)
            greeks = {key: value * fx for key, value in trade_greeks(cfg, market, pricing, config).items()}
            value = lambda context, _cfg=cfg: value_on(_cfg, context, pricing) * fx  # noqa: E731
            greeks["theta"] = trade_theta(value, value(base_context), cfg, theta_ctx, fx)
            result[i] = greeks
    return result


def ad_warnings(trades: Sequence[Trade]) -> List[str]:
    """The label of an AD run's Greeks that are not ORE's (see the module docstring): one
    naming every live Bermudan/American, priced on the LGM grid."""
    gridded = [cfg.trade_id for cfg in trades
               if isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)) and not cfg.is_expired()]
    if not gridded:
        return []
    return [f"AD Greeks of trades {gridded} (the LGM grid engine) are the derivatives of the grid price at the "
            f"market, which is piecewise smooth in the rates: they are not ORE's 1bp bump-and-revalue Greeks, a "
            f"secant across the grid's kinks (the Delta measured within 0.4% of the curve's largest, the Gamma not "
            f"comparable); method='Bump' gives ORE's (I-78)."]


def trade_greeks(cfg, market: Market, pricing: PricingConfig = PricingConfig(),
                 config: SensitivityConfig = SensitivityConfig()) -> Dict[str, np.ndarray]:
    """Delta and Gamma per tenor of each curve the trade reads and, where its engine reads the
    quotes, Vega per quote, in the trade's currency (see the module docstring)."""
    fn = trade_price_function(cfg, market, pricing, tenors=config.curve_tenors)
    rates = tuple(jnp.asarray(r) for r in fn.rates)
    quotes = None if fn.quotes is None else jnp.asarray(fn.quotes)
    curves, vega = _derivatives(fn.pricer, fn.terms, fn.nodes, rates, quotes)
    out: Dict[str, np.ndarray] = {}
    for (kind, name), (delta, gamma) in zip(fn.curves, curves):
        out[f"delta:{kind}:{name}"] = np.asarray(delta) * config.curve_shift
        out[f"gamma:{kind}:{name}"] = np.asarray(gamma) * config.curve_shift ** 2
    if vega is not None:
        out[f"vega:{cfg.currency}"] = np.asarray(vega) * config.vol_shift
    return out


@partial(jax.jit, static_argnums=0)
def _derivatives(pricer, terms, nodes, rates, quotes):
    """Per curve `(dNPV/dz, diag d^2NPV/dz^2)` in its node values `z`, and `dNPV/dquotes`: one
    program per pricer and trade shape (see the module docstring)."""
    return _directional_derivatives(lambda moved, q: pricer(terms, curves_on(nodes, moved), q), rates, quotes)

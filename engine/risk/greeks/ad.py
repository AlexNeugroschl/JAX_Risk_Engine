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
(`engine.calibration.lgm.calibrated_values`).

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

**A calibrated option's derivatives are two programs** (`TradePriceFunction.stages`, I-115): its
calibration's first and second derivatives along every direction (`_model_derivatives`, the
implicit function theorem on the basket's residuals), then its grid engine's along each
direction combined with the calibration's response (`_staged_derivatives`). Along a direction
`d` of the node values, with the calibration's buckets `v(x)` moving by `v'd` and `v''[d, d]`,
the price `g(x, v(x))` has the derivatives of `t -> g(x + t d, v + t v'd + t^2 v''[d, d] / 2)`
at 0, its first and second, by the chain rule; along a quote only the buckets move. Each program
traces its price once for every direction (`_directional_derivatives`), and the calibration
prices its basket's helpers as one (`engine.calibration.lgm.stack_helpers`): an American of 16
helpers on ORE's grid compiles in 22 s where the one program took 72 s (CPU, 2026-10-19,
docs/concepts/profiling.md §3.10).
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
from engine.pricing.cube import Trade
from engine.risk.greeks.bump import SensitivityConfig, greeks_by_trade
from engine.risk.greeks.price_functions import curves_on, trade_price_function


def _directional_derivatives(f, xs, q):
    """`f(xs, q)` and, per array `x` of the tuple `xs`, `(df/dx_i, d^2f/dx_i^2)` for every `i`,
    and `df/dq` of every entry of `q` (None when `q` is), all by forward differentiation; `f`'s
    value may be an array, whose axes then follow the directions'.

    Along each basis direction of the node values and the quotes together, one forward pass of
    a forward pass (`jax.jvp` of `jax.jvp`) gives the value and the first and second directional
    derivatives at once (a quote's second derivative is not kept: one trace of `f` serves every
    direction, where a quote's own forward pass traced it again, the calibration of a calibrated
    option alone 16 s to compile; 2026-10-19). The directions are batched with `vmap`, the price
    itself not: its inputs are the same in every direction, only the tangents carry the batch.
    Forward differentiation holds no tape: an option recalibrated under the shift moves its LGM's
    grid and rollback operator with the curve, and the reverse pass held their tangents on every
    row of its induction for every direction (13 GB for an American on ORE's 301-node grid,
    against 2 GB forward; 2026-10-17). Checked against `jax.grad` and
    `jnp.diagonal(jax.hessian(...))` in `tests/test_profiling_and_jit.py::TestHessianDiagonalEquivalence`.
    """
    parts = xs if q is None else (*xs, q.reshape(-1))
    cuts = _cuts(parts)
    flat = jnp.concatenate(parts)

    def at(x):
        split = jnp.split(x, cuts)
        return f(tuple(split[:len(xs)]), None if q is None else split[-1].reshape(q.shape))

    def along(direction):
        (value, first), (_, second) = jax.jvp(lambda x: jax.jvp(at, (x,), (direction,)), (flat,), (direction,))
        return value, first, second

    value, first, second = jax.vmap(along, out_axes=(None, 0, 0))(jnp.eye(flat.shape[0], dtype=flat.dtype))
    firsts, seconds = jnp.split(first, cuts), jnp.split(second, cuts)
    curves = tuple(zip(firsts[:len(xs)], seconds[:len(xs)]))
    return value, curves, None if q is None else firsts[-1].reshape(q.shape + first.shape[1:])


def _cuts(xs):
    """Where the concatenation of the arrays `xs` splits back into them."""
    return list(np.cumsum([x.shape[0] for x in xs])[:-1])


def portfolio_greeks(trades: Sequence[Trade], market: Market, base_currency: str,
                     pricing: PricingConfig = PricingConfig(),
                     config: SensitivityConfig = SensitivityConfig()) -> Dict[int, Dict[str, np.ndarray]]:
    """Per trade (by request index), the AD Greeks in the base currency (see the module
    docstring)."""
    return greeks_by_trade(trade_greeks, trades, market, base_currency, pricing, config)


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
    if fn.stages is None:
        curves, vega = _derivatives(fn.pricer, fn.terms, fn.nodes, rates, quotes)
    else:
        model = _model_derivatives(fn.stages.model, fn.terms, fn.nodes, rates, quotes)
        curves, vega = _staged_derivatives(fn.stages.priced, fn.terms, fn.nodes, rates, model)
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
    _, curves, vega = _directional_derivatives(lambda moved, q: pricer(terms, curves_on(nodes, moved), q), rates,
                                               quotes)
    return curves, vega


@partial(jax.jit, static_argnums=0)
def _model_derivatives(model, terms, nodes, rates, quotes):
    """`(values, curves, quotes)` of a two-stage price's first stage (`Stages.model`): its values
    `v` `[n]` at the market, per curve `(dv/dz, diag d^2v/dz^2)` `[K, n]` each, and `dv/dquotes`
    `[*quotes, n]` (None when the price reads no quotes): one program per model and trade shape
    (see the module docstring)."""
    return _directional_derivatives(lambda moved, q: model(terms, curves_on(nodes, moved), q), rates, quotes)


@partial(jax.jit, static_argnums=0)
def _staged_derivatives(priced, terms, nodes, rates, model):
    """Per curve `(dNPV/dz, diag d^2NPV/dz^2)` and `dNPV/dquotes` of `g = priced(terms, curves,
    v(z, quotes))` (`Stages.priced`), from the first stage's `_model_derivatives` (see the module
    docstring): along each node value `e`, with `u = (e, dv)`, the first derivative `Dg[u]` and
    the second `D^2g[u, u] + d2v . grad_v g`, a forward pass of a forward pass along `u` and the
    gradient in the buckets; along each quote `dv . grad_v g`. The primal is the same in every
    direction, only the tangents are batched, so the induction runs once for the price. One
    program per stage and trade shape."""
    values, by_curve, by_quote = model
    cuts = _cuts(rates)
    flat = jnp.concatenate(rates)
    dv = jnp.concatenate([first for first, _ in by_curve])                # [D, n]
    d2v = jnp.concatenate([second for _, second in by_curve])             # [D, n]

    def price(x, v):
        return priced(terms, curves_on(nodes, tuple(jnp.split(x, cuts))), v)

    def along(direction, dv_i):
        tangent = (direction, dv_i)
        return jax.jvp(lambda x, v: jax.jvp(price, (x, v), tangent)[1], (flat, values), tangent)

    first, curvature = jax.vmap(along)(jnp.eye(flat.shape[0], dtype=flat.dtype), dv)
    by_value = jax.vmap(lambda e: jax.jvp(lambda v: price(flat, v), (values,), (e,))[1])(
        jnp.eye(values.shape[-1], dtype=values.dtype))                       # [n]: grad_v g
    second = curvature + jnp.sum(d2v * by_value, axis=-1)
    curves = tuple(zip(jnp.split(first, cuts), jnp.split(second, cuts)))
    if by_quote is None:
        return curves, None
    return curves, jnp.sum(by_quote * by_value, axis=-1)

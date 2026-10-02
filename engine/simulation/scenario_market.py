"""
The simulated market on every path and date: ORE's `CrossAssetModelScenarioGenerator`
(OREAnalytics/orea/scenario/crossassetmodelscenariogenerator.cpp, `nextPath`) feeding a
`ScenarioSimMarket`.

On date t with the currency's LGM state z (`nextPath`):

  * discount curve: the model-implied curve (`ModelImpliedYieldTermStructure`), sampled at
    the simulation-market tenors,
        P(t, t + tau | z) = P(0, t+tau) / P(0, t)
                            * exp(-(H(t+tau) - H(t)) z - 1/2 (H(t+tau)^2 - H(t)^2) zeta(t));
  * index (forwarding) curve: the same with the index's own t=0 curve in place of P(0, .)
    (`ModelImpliedYtsFwdFwdCorrected`), so the index-discount basis is deterministic;
  * every discount factor floored at 1e-5;
  * numeraire: the domestic LGM numeraire N(t, z_0) = exp(H z_0 + 1/2 H^2 zeta) / P(0, t);
  * zeta is the component's own (`IrComponent.zeta`): the Hagan LGM's or the Hull-White
    adaptor's, so a Hull-White currency's curves are its own bond prices;
  * FX and equity spots: exp of their states.

Tenor times are measured from the scenario date, `dc.yearFraction(date, date + tenor)` on the
model's ACT/365, and the scenario curve between tenors is `engine.models.curves.DiscountCurve`
(log-linear, flat forward). Model time on the grid is ACT/365 from the as-of date, so ORE's
discount-curve time `t` and index-curve time `t_dc` coincide (plan V-10).
"""
import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.models.curves import DiscountCurve, ZeroCurve, log_discount
from engine.models.lgm import H as lgm_H
from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER
from engine.simulation.cam import CrossAssetModel, IrComponent

#: ORE's floor on every simulated discount factor (`nextPath`: `std::max(..., 0.00001)`).
DISCOUNT_FLOOR = 1e-5


@jax.tree_util.register_pytree_node_class
@dataclass
class ScenarioCurves:
    """One curve on every path and date: log discount factors `[S, D, K+1]` at tenor times
    `[D, K+1]` measured from each date (column 0 is t = 0, log discount 0)."""
    tenor_times: jax.Array    # [D, K+1]
    log_discounts: jax.Array  # [S, D, K+1]

    def tree_flatten(self):
        return (self.tenor_times, self.log_discounts), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def on_date(self, j: int) -> DiscountCurve:
        """The curve on date j, batched over paths."""
        return DiscountCurve(times=self.tenor_times[j], log_discounts=self.log_discounts[:, j, :])


@dataclass
class ScenarioMarket:
    """The simulated market on `dates` (after the as-of date), every array path-major.

    numeraire: [S, D], domestic LGM numeraire.
    discount: currency -> `ScenarioCurves`.
    index: index name -> `ScenarioCurves`.
    fx: foreign currency -> [S, D], domestic units per unit of it.
    equity: name -> [S, D] spot.
    states: [S, D, d], the CAM states (for pricers that condition on them).
    simulation_formats: the formats `simulate` stored the shocks and the states in, read from
        those arrays (`engine.precision.format_name`), which it does not keep: `{"shocks": ...,
        "states": ...}`, for the precision report. Empty for a market not from `simulate`.

    As `simulate` returns it, each path array is stored at the market stage's storage format:
    a plain array, or an `engine.precision.Stored` (values and block scales) in a scaled
    format; the pricers read the market loaded at their compute dtype (`map_arrays` with
    `engine.precision.load`).
    """
    asof: ORE.Date
    dates: Tuple[ORE.Date, ...]
    times: np.ndarray
    numeraire: jax.Array
    discount: Dict[str, ScenarioCurves]
    index: Dict[str, ScenarioCurves]
    fx: Dict[str, jax.Array]
    equity: Dict[str, jax.Array]
    states: jax.Array
    simulation_formats: Mapping[str, str] = dataclasses.field(default_factory=dict)

    @property
    def num_paths(self) -> int:
        return int(self.numeraire.shape[0])

    def map_arrays(self, fn: Callable, grid: Optional[Callable] = None) -> "ScenarioMarket":
        """The same market with `fn` applied to every path array (numeraire, log discounts,
        FX, equity, states, in that order) and `grid` (`fn` if omitted) to the curves' tenor
        times, which have no scenario axis; the dates and model times stay as they are.
        `store`/`load` of the whole market go through it."""
        grid = grid or fn
        curves = lambda c: ScenarioCurves(tenor_times=grid(c.tenor_times), log_discounts=fn(c.log_discounts))  # noqa: E731
        return dataclasses.replace(
            self, numeraire=fn(self.numeraire),
            discount={k: curves(c) for k, c in self.discount.items()},
            index={k: curves(c) for k, c in self.index.items()},
            fx={k: fn(v) for k, v in self.fx.items()}, equity={k: fn(v) for k, v in self.equity.items()},
            states=fn(self.states))

    def path_arrays(self) -> List:
        """Every path array, in `map_arrays`' order (the tenor grids are not path arrays)."""
        arrays = []
        self.map_arrays(lambda a: arrays.append(a) or a, grid=lambda grid: grid)
        return arrays


def tenor_times(dates: Sequence[ORE.Date], tenors: Sequence[str]) -> np.ndarray:
    """`[D, K+1]` times of each tenor from each date, with a leading 0 column:
    `dc.yearFraction(date, date + tenor)` as `nextPath` caches them."""
    periods = [ORE.Period(t) for t in tenors]
    rows = [[0.0] + [TIME_AXIS_DAY_COUNTER.yearFraction(d, d + p) for p in periods] for d in dates]
    times = np.asarray(rows, dtype=np.float64)
    if np.any(np.diff(times, axis=1) <= 0.0):
        raise ValueError(f"simulation-market tenors must be positive and increasing; got {list(tenors)}")
    return times


def implied_log_discounts(target: ZeroCurve, model: IrComponent, t: np.ndarray,
                          tenors: np.ndarray, z: jax.Array) -> jax.Array:
    """ln P(t, t + tau | z) of the LGM `model` with `target` as its t=0 curve (ORE's
    `LinearGaussMarkovModel::discountBond(t, T, x, targetCurve)`), floored at
    `DISCOUNT_FLOOR`. Shapes: t [D], tenors [D, K+1], z [S, D] -> [S, D, K+1], in z's dtype.

    The path-independent parts are computed in float64 and cast, so a float32 run carries
    only the state in float32."""
    t64 = jnp.asarray(t, dtype=jnp.float64)
    T = t64[:, None] + jnp.asarray(tenors, dtype=jnp.float64)
    Ht, HT = lgm_H(model.reversion, t64)[:, None], lgm_H(model.reversion, T)
    zeta_t = model.zeta(t64)[:, None]
    deterministic = (log_discount(target, T) - log_discount(target, t64)[:, None]
                     - 0.5 * (HT ** 2 - Ht ** 2) * zeta_t).astype(z.dtype)
    dH = (HT - Ht).astype(z.dtype)
    values = deterministic[None, :, :] - dH[None, :, :] * z[:, :, None]
    return jnp.maximum(values, jnp.asarray(np.log(DISCOUNT_FLOOR), dtype=z.dtype))


def lgm_numeraire(model: IrComponent, t: np.ndarray, z: jax.Array) -> jax.Array:
    """N(t, z) = exp(H(t) z + 1/2 H(t)^2 zeta(t)) / P(0, t) (`LinearGaussMarkovModel::
    numeraire`) of the LGM `model` on its own curve, for t [D] and z [S, D], with the
    path-independent parts in float64."""
    t64 = jnp.asarray(t, dtype=jnp.float64)
    Ht = lgm_H(model.reversion, t64)
    log_deterministic = (0.5 * Ht ** 2 * model.zeta(t64) - log_discount(model.curve, t64)).astype(z.dtype)
    return jnp.exp(Ht.astype(z.dtype)[None, :] * z + log_deterministic[None, :])


def build_scenario_market(
    model: CrossAssetModel,
    asof: ORE.Date,
    dates: Sequence[ORE.Date],
    states: jax.Array,
    tenors: Sequence[str],
    index_curves: Mapping[str, Tuple[str, ZeroCurve]],
) -> ScenarioMarket:
    """The scenario market from the CAM states `[S, D, d]` on `dates`.

    `index_curves`: index name -> (currency, the index's t=0 forwarding curve)."""
    times = np.asarray([TIME_AXIS_DAY_COUNTER.yearFraction(asof, d) for d in dates], dtype=np.float64)
    taus = tenor_times(dates, tenors)

    def curves_for(currency: str, target: ZeroCurve) -> ScenarioCurves:
        i = model.ir_index(currency)
        log_dfs = implied_log_discounts(target, model.ir[i], times, taus, states[:, :, i])
        return ScenarioCurves(tenor_times=jnp.asarray(taus, dtype=states.dtype), log_discounts=log_dfs)

    numeraire = lgm_numeraire(model.ir[0], times, states[:, :, 0])
    return ScenarioMarket(
        asof=asof,
        dates=tuple(dates),
        times=times,
        numeraire=numeraire,
        discount={c.currency: curves_for(c.currency, c.curve) for c in model.ir},
        index={name: curves_for(ccy, curve) for name, (ccy, curve) in index_curves.items()},
        fx={c.currency: jnp.exp(states[:, :, model.fx_state(j)]) for j, c in enumerate(model.fx)},
        equity={c.name: jnp.exp(states[:, :, model.eq_state(k)]) for k, c in enumerate(model.eq)},
        states=states,
    )

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

Each curve's discount factors are the model's at the tenors measured from the scenario date,
`dc.yearFraction(date, date + tenor)` (`nextPath`'s `time_dsc_`), but the simulation market
holds them at the tenors measured from the as-of date, `dc.yearFraction(asof, asof + tenor)`:
`ScenarioSimMarket::addYieldCurve` builds each curve once, on those times and a reference date
that moves with the evaluation date, and every scenario only replaces its discount factors
(`as_of_tenor_times`). Between tenors the curve is `engine.market_data.curves.DiscountCurve`
(log-linear, flat forward), ORE's `LogLinear` interpolation and `FlatFwd` extrapolation. Until
2026-10-07 (I-84) the engine held each value at the time it was computed for; on a 6M tenor that is
two days apart, and the swaps of the shared portfolio moved by up to 0.3% of their largest
path value against ORE's own simulation (`tests/test_ore_xva_parity.py`). Model time on the
grid is ACT/365 from the as-of date, so ORE's discount-curve time `t` and index-curve time
`t_dc` coincide (plan V-10).
"""
import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import ORE

from engine.market_data.curves import DiscountCurve, ZeroCurve, log_discount
from engine.market_data.day_counts import TIME_AXIS_DAY_COUNTER
from engine.models.cam import CrossAssetModel, IrComponent, PathMoments, path_moments, step_moments
from engine.models.lgm import H as lgm_H
from engine.precision import Arithmetic, add_level, deviation_from

#: ORE's floor on every simulated discount factor (`nextPath`: `std::max(..., 0.00001)`).
DISCOUNT_FLOOR = 1e-5


@jax.tree_util.register_pytree_node_class
@dataclass
class ScenarioCurves:
    """One curve on every path and date: log discount factors `[S, D, K+1]` at the simulation
    market's tenor times `[D, K+1]`, measured from each date (column 0 is t = 0, log discount
    0); since 2026-10-07 every date's are the as-of date's (`as_of_tenor_times`). `level`
    `[D, K+1]` is a curve per date that no path depends on (the model's at its mean state):
    a pricer in difference form multiplies each path's deviation from it, never the curve
    itself (F-07; any level gives the same values, a near one keeps the deviations small). The
    tenor times and the level have no scenario axis: they are float64, as every level is, and
    the precision policy neither stores nor loads them."""
    tenor_times: jax.Array    # [D, K+1]
    log_discounts: jax.Array  # [S, D, K+1]
    level: jax.Array          # [D, K+1]

    def tree_flatten(self):
        return (self.tenor_times, self.log_discounts, self.level), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def on_date(self, j: int) -> DiscountCurve:
        """The curve on date j, batched over paths, its times in its values' format."""
        return DiscountCurve(times=self.tenor_times[j].astype(self.log_discounts.dtype),
                             log_discounts=self.log_discounts[:, j, :])


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
        times and levels, which have no scenario axis; the dates and model times stay as they are.
        `store`/`load` of the whole market go through it."""
        grid = grid or fn
        curves = lambda c: ScenarioCurves(tenor_times=grid(c.tenor_times), log_discounts=fn(c.log_discounts),  # noqa: E731
                                          level=grid(c.level))
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


def as_of_tenor_times(asof: ORE.Date, tenors: Sequence[str]) -> np.ndarray:
    """`[K+1]`: the times the simulation market holds its tenor points at on every date,
    `dc.yearFraction(asof, asof + tenor)` with a leading 0 (`ScenarioSimMarket::addYieldCurve`)."""
    return tenor_times([asof], tenors)[0]


def log_discount_split(target: ZeroCurve, model: IrComponent, t: np.ndarray,
                       tenors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ln P(t, t + tau | z) of the LGM `model` with `target` as its t=0 curve (ORE's
    `LinearGaussMarkovModel::discountBond(t, T, x, targetCurve)`, unfloored) as its value at
    z = 0 and its slope in z, `ln P = level - dH z`: `(level, -dH)`, each `[D, K+1]` in float64,
    for t [D] and tenors [D, K+1]."""
    t64 = jnp.asarray(t, dtype=jnp.float64)
    T = t64[:, None] + jnp.asarray(tenors, dtype=jnp.float64)
    Ht, HT = lgm_H(model.reversion, t64)[:, None], lgm_H(model.reversion, T)
    zeta_t = model.zeta(t64)[:, None]
    level = log_discount(target, T) - log_discount(target, t64)[:, None] - 0.5 * (HT ** 2 - Ht ** 2) * zeta_t
    return np.asarray(level), np.asarray(-(HT - Ht))


def log_numeraire_split(model: IrComponent, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ln N(t, z) = H(t) z + 1/2 H(t)^2 zeta(t) - ln P(0, t) (`LinearGaussMarkovModel::
    numeraire`) of the LGM `model` on its own curve, as its value at z = 0 and its slope in z:
    `(1/2 H^2 zeta - ln P(0, t), H)`, each `[D]` in float64."""
    t64 = jnp.asarray(t, dtype=jnp.float64)
    Ht = lgm_H(model.reversion, t64)
    return np.asarray(0.5 * Ht ** 2 * model.zeta(t64) - log_discount(model.curve, t64)), np.asarray(Ht)


def implied_log_discounts(target: ZeroCurve, model: IrComponent, t: np.ndarray,
                          tenors: np.ndarray, z: jax.Array) -> jax.Array:
    """ln P(t, t + tau | z) (`log_discount_split`), floored at `DISCOUNT_FLOOR`, for z [S, D]
    -> [S, D, K+1] in z's dtype, the path-independent parts computed in float64 and cast."""
    level, slope = (jnp.asarray(a, dtype=z.dtype) for a in log_discount_split(target, model, t, tenors))
    return jnp.maximum(level[None] + slope[None] * z[:, :, None], jnp.asarray(np.log(DISCOUNT_FLOOR), dtype=z.dtype))


def lgm_numeraire(model: IrComponent, t: np.ndarray, z: jax.Array) -> jax.Array:
    """N(t, z) (`log_numeraire_split`) for t [D] and z [S, D], in z's dtype, the
    path-independent parts computed in float64 and cast."""
    level, slope = (jnp.asarray(a, dtype=z.dtype) for a in log_numeraire_split(model, t))
    return jnp.exp(level[None] + slope[None] * z)


@dataclass(frozen=True)
class _Layout:
    """The scenario market as one linear map of the CAM state on each date, in float64:
    `values[s, j, c] = level[j, c] + sum_i x[s, j, i] loading[j, i, c]` over its columns `c`,
    the curves' log discount factors (`curves` of them, `tenors` columns each, floored), then
    the log numeraire and the log FX and equity spots (one column each, exponentiated)."""
    level: np.ndarray    # [D, C]
    loading: np.ndarray  # [D, d, C]
    curves: int
    tenors: int
    fx: int
    equity: int


def _layout(model: CrossAssetModel, times: np.ndarray, taus: np.ndarray,
            curves: Sequence[Tuple[int, ZeroCurve]]) -> _Layout:
    """The `_Layout` of `curves` (each the index of its currency's IR state and its t=0 curve),
    the domestic numeraire and every FX and equity spot."""
    num_dates, num_tenors = taus.shape
    levels, loadings = [], []

    def column(level, state, slope):
        level = np.reshape(level, (num_dates, -1))
        loading = np.zeros((num_dates, model.dimension, level.shape[1]))
        loading[:, state, :] = np.reshape(slope, (num_dates, -1)) if np.ndim(slope) else slope
        levels.append(level)
        loadings.append(loading)

    for state, target in curves:
        level, slope = log_discount_split(target, model.ir[state], times, taus)
        column(level, state, slope)
    level, slope = log_numeraire_split(model.ir[0], times)
    column(level, 0, slope)
    for state in [model.fx_state(j) for j in range(len(model.fx))] + [model.eq_state(k) for k in range(len(model.eq))]:
        column(np.zeros(num_dates), state, 1.0)
    return _Layout(np.concatenate(levels, axis=1), np.concatenate(loadings, axis=2), len(curves), num_tenors,
                   len(model.fx), len(model.eq))


@partial(jax.jit, static_argnames=("arithmetic", "counts"))
def _market_values(states, mean, scale, level, loading, arithmetic: Arithmetic, counts: Tuple[int, int, int, int]):
    """The scenario market's path arrays from the states `[S, D, d]` (see `build_scenario_market`):
    `(curves' log discount factors [S, D, K+1] each, numeraire [S, D], FX [S, D] each, equity
    [S, D] each)`, in `arithmetic`'s accumulate format."""
    curves, tenors, fx, equity = counts
    deviation = deviation_from(mean, states) / arithmetic.cast(scale)                        # [S, D, d]
    product = arithmetic.matmul(jnp.swapaxes(deviation, 0, 1), loading)                      # [D, S, C]
    values = add_level(level, jnp.swapaxes(product, 0, 1))                                    # [S, D, C]
    floor = jnp.asarray(np.log(DISCOUNT_FLOOR), dtype=arithmetic.dtype)
    logs = tuple(jnp.maximum(values[:, :, k * tenors:(k + 1) * tenors], floor) for k in range(curves))
    spots = jnp.exp(values[:, :, curves * tenors:])                                           # [S, D, 1 + fx + equity]
    fx_spots = tuple(spots[:, :, 1 + j] for j in range(fx))
    return logs, spots[:, :, 0], fx_spots, tuple(spots[:, :, 1 + fx + k] for k in range(equity))


def build_scenario_market(
    model: CrossAssetModel,
    asof: ORE.Date,
    dates: Sequence[ORE.Date],
    states: jax.Array,
    tenors: Sequence[str],
    index_curves: Mapping[str, Tuple[str, ZeroCurve]],
    arithmetic: Optional[Arithmetic] = None,
    path: Optional[PathMoments] = None,
) -> ScenarioMarket:
    """The scenario market from the CAM states `[S, D, d]` on `dates`: each curve's discount
    factors computed at the tenors from each date, held at the tenors from the as-of date (see
    the module docstring).

    In difference form (F-07): every path array is a level that does not depend on the path,
    computed in float64 at the mean state (`engine.models.cam.path_moments`), plus one matrix
    product of the path's state deviation, in units of its scale, with the market's loadings on
    the state (`_Layout`), in `arithmetic` (the states' own format, float64 or float32, by
    default); the level is added in two terms of the accumulate format
    (`engine.precision.add_level`). The curves keep their level and tenor times in float64
    (`ScenarioCurves`), from which the pricers measure each path's deviation. `path` is computed
    from the model when not given.

    `index_curves`: index name -> (currency, the index's t=0 forwarding curve)."""
    arithmetic = arithmetic or Arithmetic.of(states.dtype)
    states = arithmetic.cast(states)
    times = np.asarray([TIME_AXIS_DAY_COUNTER.yearFraction(asof, d) for d in dates], dtype=np.float64)
    taus = tenor_times(dates, tenors)
    path = path or path_moments(model, step_moments(model, np.concatenate([[0.0], times])))
    curves = ([(model.ir_index(c.currency), c.curve) for c in model.ir]
              + [(model.ir_index(ccy), curve) for ccy, curve in index_curves.values()])
    layout = _layout(model, times, taus, curves)
    mean, scale = path.mean[1:], path.scale[1:]                                               # [D, d]
    level = layout.level + np.einsum("ji,jic->jc", mean, layout.loading)
    loading = layout.loading * scale[:, :, None]
    logs, numeraire, fx, equity = _market_values(states, mean, scale, level, loading, arithmetic,
                                                 (layout.curves, layout.tenors, layout.fx, layout.equity))
    pillars = jnp.asarray(np.broadcast_to(as_of_tenor_times(asof, tenors), taus.shape), dtype=jnp.float64)
    on_curve = [ScenarioCurves(tenor_times=pillars, log_discounts=logs[k],
                               level=jnp.asarray(level[:, k * layout.tenors:(k + 1) * layout.tenors]))
                for k in range(layout.curves)]
    return ScenarioMarket(
        asof=asof,
        dates=tuple(dates),
        times=times,
        numeraire=numeraire,
        discount={c.currency: on_curve[i] for i, c in enumerate(model.ir)},
        index={name: on_curve[len(model.ir) + i] for i, name in enumerate(index_curves)},
        fx={c.currency: fx[j] for j, c in enumerate(model.fx)},
        equity={c.name: equity[k] for k, c in enumerate(model.eq)},
        states=states,
    )

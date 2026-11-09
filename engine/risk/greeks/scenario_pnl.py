"""
A trade's P&L under scenarios of its price function's market, in batches: the one revaluation
both the bump Greeks (`engine.risk.greeks.bump`, each shift a scenario of ORE's sensitivity
market) and market risk (`engine.risk.market.revaluation`, each shock a scenario of today's
market) run.

    price_function_pnl(cfg, fn, rates, moves, pricing, batch_size, arithmetic) -> (value, pnl [S])

`fn` is the trade's `TradePriceFunction` (`engine.risk.greeks.price_functions`), `rates` its
curves' node values at the base, `moves` each scenario's move of them (`Moves.shifts`, one
`[S, K]` per curve, in node values) and of the quotes (`Moves.quotes`, or None to hold them). A
scenario's P&L is `fn.pnl`, in difference form in `arithmetic` (F-07), so a scenario that moves
nothing has exactly 0. An option whose engine calibrates its LGM is recalibrated under every
scenario first, in float64 (A-10), as ORE's historical P&L and sensitivity analysis rebuild the
trade's model (`fn.calibration`, I-112); a scenario whose calibration misses ORE's acceptance is
refused, naming the trade and the scenarios (`CalibrationError`).

**Programs.** The P&L is one jitted program per product, shape, arithmetic and group size
(`_pnls`), with the price function's pricer static and the trades' terms traced, so a repeated
run, or another trade of the same shape, reuses it; trades of one shape are revalued together,
their terms stacked and the programs mapped over them (`price_function_pnls`, TraderX fork); the calibration is the bootstrap's programs, one per helper shape
and batch (`engine.calibration.lgm.bootstrap`), shared by every basket with such a helper. The scenarios
are revalued in batches of at most `batch_size`, fewer where the grid engine's induction would
exceed its per-device memory budget (`scenario_batch_size`); split across the devices of the host
where the scenario count allows (`engine.market_simulation.sharding`), each batch taking its
scenarios from every device's share.
"""
import dataclasses
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.lax.lax import _convert_element_type

from engine.calibration.lgm import CalibrationError
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.market_simulation.sharding import host_array, scenario_device_count, shard_scenarios
from engine.precision import Arithmetic
from engine.pricing.bermudan import prepared_option
from engine.pricing.config import PricingConfig
from engine.pricing.cube import reads_swaption_vols
from engine.pricing.lgm_grid import induction_memory_budget, working_bytes_per_model, working_bytes_per_path
from engine.pricing.options import bound_option_pnl, position_sign
from engine.risk.greeks.price_functions import Moves, TradePriceFunction, curves_on

OPTIONS = (SwaptionConfig, BermudanSwaptionConfig, AmericanSwaptionConfig)


def scenario_batch_size(cfg, pricing: PricingConfig, requested: int, itemsize: int) -> int:
    """Scenarios to revalue at once for `cfg`: `requested`, reduced so one batch stays within
    the induction's per-device budget (`engine.pricing.lgm_grid.induction_memory_budget`). Only the grid
    engine needs it: its induction holds its columns on the state grid per scenario
    (`working_bytes_per_path`), and an option recalibrated under each scenario its own model's
    too (`working_bytes_per_model`)."""
    if not isinstance(cfg, (BermudanSwaptionConfig, AmericanSwaptionConfig)) or cfg.is_expired():
        return requested
    engine = pricing.american if isinstance(cfg, AmericanSwaptionConfig) else pricing.bermudan
    prepared = prepared_option(cfg, engine, engine.volatility)
    per_scenario = working_bytes_per_path(prepared, itemsize)
    if reads_swaption_vols(cfg, pricing):
        per_scenario += working_bytes_per_model(prepared, itemsize)
    return max(1, min(requested, induction_memory_budget() // per_scenario))


#: Trade-scenario pairs one call of `price_function_pnls` revalues at most, where the grid
#: engine's memory budget does not set fewer (`trades_per_call`).
# ponytail: one fixed cap for every device; size it per device if a GPU/TPU measurement asks.
MODELS_PER_CALL = 16384


#: Trades a recalibration is padded to on an accelerator (`calibration_group_size`).
CALIBRATION_GROUP = 16


def calibration_group_size(count: int) -> int:
    """How many trades a group of `count` (a power of two) is recalibrated as: on an accelerator
    at least `CALIBRATION_GROUP`, padded with copies of the last, so a helper shape's bucket is
    one program whatever the group's size; on a CPU the group's own size. A recalibration is
    cheap on the device and dear to compile: with a program per group size the book's first run
    on an A100 compiled for 2,800 s more (2026-11-09)."""
    return count if jax.default_backend() == "cpu" else max(count, CALIBRATION_GROUP)


def price_function_pnl(cfg, fn: TradePriceFunction, rates, moves: Moves, pricing: PricingConfig,
                       batch_size: int, arithmetic: Arithmetic) -> Tuple[float, jax.Array]:
    """The trade's value at the base `rates` and its P&L `[S]` under `moves` (see the module
    docstring), in `arithmetic`: the moves' node values are carried to each curve's node
    deviation (`nodes[i].deviation`); an option's P&L is bounded so that no scenario values it
    below nothing in its position (`engine.pricing.options.bound_option_pnl`)."""
    values, pnl = price_function_pnls([cfg], [fn], rates, moves, pricing, batch_size, arithmetic)
    return float(values[0]), pnl[:, 0]


def group_key(cfg, fn: TradePriceFunction, arithmetic: Arithmetic):
    """Trades of one key share every program of their revaluation and are revalued together
    (`price_function_pnls`): one product, pricer, curves and arithmetic, and terms of one
    structure and shape. None for a trade revalued alone: stochastic rounding's key is the
    trade's own (`Precision.arithmetic_for`)."""
    if arithmetic.key is not None:
        return None
    leaves, structure = jax.tree_util.tree_flatten(fn.terms)
    return (type(cfg), fn.pricer, fn.pnl, fn.calibration, fn.curves, fn.quotes is None, structure,
            tuple((np.shape(leaf), str(getattr(leaf, "dtype", type(leaf).__name__))) for leaf in leaves),
            dataclasses.replace(arithmetic, sample_ids=None))


def _stacked(fns):
    """The terms of `fns` stacked on the host along a leading axis of the trades. A Python
    scalar term (an option's `float_spread`) is stacked weakly typed, as JAX types the scalar
    itself, so a float32 option's arithmetic stays float32."""
    def stack(*leaves):
        if isinstance(leaves[0], (int, float)) and not isinstance(leaves[0], bool):
            # ponytail: JAX's private weak-type cast; jnp has no public one.
            return _convert_element_type(jnp.asarray(np.asarray(leaves)), weak_type=True)
        return np.stack([np.asarray(x) for x in leaves])
    return jax.tree_util.tree_map(stack, *(f.terms for f in fns))


def trades_per_call(cfg, pricing: PricingConfig, batch_size: int, itemsize: int) -> int:
    """How many trades of `cfg`'s group one call of `price_function_pnls` takes: on an
    accelerator, a batch of scenarios for each within `MODELS_PER_CALL` trade-scenario pairs,
    or the grid engine's per-device budget (`scenario_batch_size`), a power of two, as a call's
    trades are padded to one, so calls of every group size share their programs; on a CPU, one.

    A CPU gains nothing from a group, its per-call cost on the host being small, and loses from
    the call's larger working set: four Bermudans recalibrated under 256 scenarios in one call
    took 2.4 times the CPU time of four calls (an 11-core Mac, 2026-11-08), and the same as
    four calls at 64 scenarios a batch."""
    if jax.default_backend() == "cpu":
        return 1
    capacity = scenario_batch_size(cfg, pricing, MODELS_PER_CALL, itemsize)
    return 1 << ((capacity // min(batch_size, capacity)).bit_length() - 1)


def price_function_pnls(cfgs, fns, rates, moves: Moves, pricing: PricingConfig, batch_size: int,
                        arithmetic: Arithmetic) -> Tuple[np.ndarray, jax.Array]:
    """`price_function_pnl` of trades of one `group_key` at once: values `[G]` and P&L `[S, G]`.
    Their terms are stacked on the host and every program is mapped over the trades
    (`jax.vmap`), so the group is one call per scenario batch and one host sync, where each trade
    was its own (TraderX fork). The trades are padded to a power of two with copies of the last,
    so a group's programs are one per shape and power of two. A group of one runs the trade's own
    programs, unmapped: mapped over one trade, a recalibrated Bermudan's took twice the CPU time
    (an 11-core Mac, 2026-11-08).

    A recalibrated option's price stays per trade, as each trade's own. Its recalibration is
    the group's, a group of one included: each bucket one program for its trades and scenarios
    (`ScenarioCalibration.group`), where the scenarios hold the quotes (market risk's); per trade
    where they move them (a Vega bump). Mapped outside a jit (`jax.vmap` called on the
    recalibration, a program per bucket), a group's programs compiled again on the run after the
    first (117 s on an A100, 2026-11-08)."""
    cfg, fn = cfgs[0], fns[0]
    count = len(fns)
    curves = curves_on(fn.nodes, rates)
    quotes = None if fn.quotes is None else jnp.asarray(fn.quotes)
    padded = list(fns) + [fns[-1]] * ((1 << (count - 1).bit_length()) - count)
    if count == 1:
        terms = fn.terms
        values = np.asarray([fn.pricer(terms, curves, quotes)], dtype=np.float64)
    else:
        terms = _stacked(padded)
        if fn.calibration is None:
            values = np.asarray(_prices(fn.pricer, terms, curves, quotes), dtype=np.float64)[:count]
        else:
            values = np.asarray([f.pricer(f.terms, curves, quotes) for f in fns], dtype=np.float64)
    batch = scenario_batch_size(cfg, pricing, batch_size, jnp.dtype(arithmetic.dtype).itemsize)
    moves = moves._replace(shifts=tuple(n.deviation(s) for n, s in zip(fn.nodes, moves.shifts)))
    if fn.calibration is not None:
        if moves.quotes is None:
            size = calibration_group_size(len(padded))
            stacked = terms if count > 1 and size == len(padded) else _stacked(padded + [fns[-1]] * (size - len(padded)))
            calibrated = lambda per: tuple(a[:, :len(padded)] for a in  # noqa: E731
                                           fn.calibration.group(stacked, curves, quotes, per))
        else:
            calibrated = lambda per: jax.tree_util.tree_map(  # noqa: E731
                lambda *a: jnp.stack(a, axis=1), *(f.calibration(f.terms, curves, quotes, per) for f in padded))
        models, misses = _map_scenarios(lambda per, _ids: calibrated(per), moves, batch)   # [S, G, n], [S, G]
        misses = host_array(misses)
        for g, trade in enumerate(cfgs):
            _require_calibrated(trade, pricing, misses[:, g])
        moves = moves._replace(models=models[:, 0] if count == 1 else models)
    pnl = _map_scenarios(
        lambda per, ids: _pnls(fn.pnl, terms, curves, quotes, per, dataclasses.replace(arithmetic, sample_ids=ids),
                               mapped=count > 1),
        moves, batch)
    pnl = pnl[:, :count] if count > 1 else pnl[:, None]
    if isinstance(cfg, OPTIONS):
        signs = np.asarray([position_sign(trade) for trade in cfgs])
        pnl = bound_option_pnl(pnl, values, signs)
    return values, pnl


def _require_calibrated(cfg, pricing: PricingConfig, misses: np.ndarray) -> None:
    """Refuse a revaluation whose option misses ORE's calibration acceptance in a scenario, as
    every calibration that misses is refused (`engine.calibration.lgm.require_accepted`)."""
    engine = pricing.american if isinstance(cfg, AmericanSwaptionConfig) else pricing.bermudan
    failed = np.flatnonzero(misses)
    if failed.size:
        shown = ", ".join(str(i) for i in failed[:10]) + (", ..." if failed.size > 10 else "")
        raise CalibrationError(
            f"{cfg.product} {cfg.trade_id!r}: the LGM recalibrated under {failed.size} of {misses.size} scenarios "
            f"misses its basket (ORE's LgmBuilder acceptance, tolerance {engine.bootstrap_tolerance:g}): scenarios {shown}")


@partial(jax.jit, static_argnames="pricer")
def _prices(pricer, terms, curves, quotes):
    """`pricer(terms, curves, quotes)` of each of a group's trades `[G]`, for a pricer that does not
    calibrate: one program per product, shape and padded group size."""
    return jax.vmap(pricer, in_axes=(0, None, None))(terms, curves, quotes)


@partial(jax.jit, static_argnames=("pnl", "mapped"))
def _pnls(pnl, terms, curves, quotes, moves: Moves, arithmetic: Arithmetic, mapped: bool):
    """`pnl(terms, curves, quotes, moves, arithmetic)` of each of a group's trades, `[S, G]`
    (`terms` and `moves.models` mapped over the trades), or of one trade, `[S]`, unmapped: one
    program per product, shape, arithmetic and padded group size."""
    if not mapped:
        return pnl(terms, curves, quotes, moves, arithmetic)
    axes = Moves(shifts=None, models=None if moves.models is None else 1, quotes=None)
    return jax.vmap(pnl, in_axes=(0, None, None, axes, None), out_axes=1)(terms, curves, quotes, moves, arithmetic)


def _map_scenarios(batched, per_scenario, batch_size: int):
    """`batched(per_scenario[i:j], ids[i:j])` for every batch of `batch_size` scenarios,
    concatenated along the scenarios (`per_scenario` and the result are pytrees of arrays whose
    leading axis is the scenarios). With the scenarios split across devices
    (`engine.market_simulation.sharding`), each batch takes `batch_size` scenarios from every
    device's share, so every device works on every batch and holds one batch's memory."""
    count = jax.tree_util.tree_leaves(per_scenario)[0].shape[0]
    devices = scenario_device_count(count)
    ids = jnp.arange(count, dtype=jnp.uint32)
    tree_map = jax.tree_util.tree_map
    if devices == 1:
        parts = [batched(tree_map(lambda a: a[i:i + batch_size], per_scenario), ids[i:i + batch_size])
                 for i in range(0, count, batch_size)]
        return tree_map(lambda *a: jnp.concatenate(a), *parts)
    shares = tree_map(lambda a: shard_scenarios(a.reshape((devices, -1) + a.shape[1:]), axis=0), per_scenario)
    per_device = jax.vmap(batched)
    ids = ids.reshape(devices, -1)
    share = count // devices
    parts = [per_device(tree_map(lambda a: a[:, i:i + batch_size], shares), ids[:, i:i + batch_size])
             for i in range(0, share, batch_size)]
    return tree_map(lambda *a: jnp.concatenate(a, axis=1).reshape((-1,) + a[0].shape[2:]), *parts)

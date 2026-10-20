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

**Programs.** The P&L is one jitted program per product, shape and arithmetic (`_pnl`), with the
price function's pricer static and the trade's terms traced, so a repeated run, or another trade
of the same shape, reuses it; the calibration is the bootstrap's programs, one per helper shape
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


def price_function_pnl(cfg, fn: TradePriceFunction, rates, moves: Moves, pricing: PricingConfig,
                       batch_size: int, arithmetic: Arithmetic) -> Tuple[float, jax.Array]:
    """The trade's value at the base `rates` and its P&L `[S]` under `moves` (see the module
    docstring), in `arithmetic`: the moves' node values are carried to each curve's node
    deviation (`nodes[i].deviation`); an option's P&L is bounded so that no scenario values it
    below nothing in its position (`engine.pricing.options.bound_option_pnl`)."""
    value = float(fn.price(*rates))
    batch = scenario_batch_size(cfg, pricing, batch_size, jnp.dtype(arithmetic.dtype).itemsize)
    curves = curves_on(fn.nodes, rates)
    quotes = None if fn.quotes is None else jnp.asarray(fn.quotes)
    moves = moves._replace(shifts=tuple(n.deviation(s) for n, s in zip(fn.nodes, moves.shifts)))
    if fn.calibration is not None:
        models, misses = _map_scenarios(lambda per, _ids: fn.calibration(fn.terms, curves, quotes, per), moves, batch)
        _require_calibrated(cfg, pricing, host_array(misses))
        moves = moves._replace(models=models)
    pnl = _map_scenarios(
        lambda per, ids: _pnl(fn.pnl, fn.terms, curves, quotes, per, dataclasses.replace(arithmetic, sample_ids=ids)),
        moves, batch)
    if isinstance(cfg, OPTIONS):
        pnl = bound_option_pnl(pnl, value, position_sign(cfg))
    return value, pnl


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


@partial(jax.jit, static_argnames="pnl")
def _pnl(pnl, terms, curves, quotes, moves: Moves, arithmetic: Arithmetic):
    """`pnl(terms, curves, quotes, moves, arithmetic)`: one program per product, shape and
    arithmetic."""
    return pnl(terms, curves, quotes, moves, arithmetic)


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

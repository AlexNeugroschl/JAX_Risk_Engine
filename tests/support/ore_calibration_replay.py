"""
Where ORE's LGM calibration stops short of its basket (I-126): the market-risk P&L the engine
would have if its calibrations ended where ORE's did, so the parity sweep can still compare the
two at its tolerance.

**Why.** ORE fits each LGM bucket with a Levenberg-Marquardt optimiser
(`LevenbergMarquardt(1e-8, 1e-8, 1e-8)`, OREData/ored/model/irmodelbuilder.cpp) on the parameter
`x` with `alpha = x^2`, its Jacobian a forward difference of step `1e-4 x`. When an iterate (often
the start, the configured volatility) lies within that step of the root, the difference straddles
the kink of the objective `|market - model| / market`, and the optimiser stops on its step
tolerance with the helper off by up to about half the helper's change over the step: up to ~1e-4
of the helper's value for the first bucket, a few 1e-6 for later ones. The engine solves each
bucket to ~1e-14. On the sweep's seed 18, one of ORE's 1,024 fits (scenario 9, helper 31 of
`american_swaption-3`) stopped at 1.8e-6, which moved the option's P&L by 3.3e-10 of its notional,
against the sweep's 1e-11. With the engine's calibration ending where ORE's did, the gap is 4e-13.
Across the sweep's options, a stop like that can move the P&L by up to ~2e-6 of the notional, so
no tolerance both admits it and still catches an engine error (measured 2026-11-09).

**What this does.** ORE's debug log writes every calibration's helpers (`getCalibrationDetails`,
OREData/ored/model/utilities.cpp): each helper's model and market value and their difference, to
six significant digits. `calibration_tables` reads them; `ores_fit_correction` matches a logged
calibration to each scenario by its helpers' market values (each scenario's are its own), solves
the engine's helpers to ORE's logged relative misses instead of to zero, and returns how much
that moves the trade's P&L: the option on ORE's fit less the option on the engine's, in the
scenario, less the same on the base market. Only the calibration's targets change, by ORE's own
misses; the engine prices and calibrates with its own code, so an engine error is not absorbed.

Test tooling: the engine never imports it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from engine.calibration.lgm import bootstrap, helper_residuals
from engine.market_data.curves import with_node_deviation
from engine.risk.greeks.price_functions import curves_on, option_on_model, trade_price_function
from engine.risk.market import RateRiskFactors
from engine.risk.market.revaluation import factor_indices

#: A header of `getCalibrationDetails`' LGM table, and the line that ends the table.
_HEADER, _END = "irlgm1fHwSigma", "t >= "
#: A logged table's columns: the helper's index, then ten numbers.
_COLUMNS = 11
_MARKET, _DIFF = 6, 7
#: How far a logged market value (six significant digits) may be from the engine's, relative,
#: for the table to be that market's calibration.
MATCH = 2e-5


def calibration_tables(lines: Sequence[str]) -> List[np.ndarray]:
    """Every LGM calibration table of ORE's debug log, in the log's order: `[n, 2]`, each
    helper's market value and model value less market value (`valueDiff`)."""
    tables, rows = [], None
    for line in lines:
        if _END in line:   # names the parameters too, so before the header's test
            if rows is not None:
                tables.append(np.asarray(rows, dtype=np.float64).reshape(-1, 2))
            rows = None
        elif _HEADER in line:
            rows = []
        elif rows is not None:
            tokens = line.split()[-_COLUMNS:]
            if len(tokens) == _COLUMNS and tokens[0] == str(len(rows)):
                values = [float(t) for t in tokens]
                rows.append((values[_MARKET], values[_DIFF]))
    return tables


@dataclass(frozen=True)
class Replay:
    """`ores_fit_correction`'s result: per scenario asked for, the P&L's change `correction`
    `[k]`, and ORE's largest relative helper miss `ore_miss` `[k]` in that scenario's calibration
    (the base's in `base_miss`)."""
    scenarios: Tuple[int, ...]
    correction: np.ndarray
    ore_miss: np.ndarray
    base_miss: float

    def describe(self, trade_id: str) -> str:
        return "; ".join(f"{trade_id} scenario {s}: ORE's calibration stopped {m:.2e} from a helper "
                         f"(relative), moving the P&L by {c:+.3e}"
                         for s, m, c in zip(self.scenarios, self.ore_miss, self.correction))


def ores_fit_correction(cfg, market, pricing, factors: RateRiskFactors, shifts, scenarios: Sequence[int],
                        tables: Sequence[np.ndarray]) -> Replay:
    """For each of `scenarios`, how the calibrated option `cfg`'s market-risk P&L moves when its
    LGM is calibrated, in the scenario and on the base market, to where ORE's logged calibrations
    of those markets ended (`tables`, `calibration_tables`), instead of exactly (see the module
    docstring). `shifts` `[S, F]` are the scenarios' factor shifts, `factors`' order."""
    fn = trade_price_function(cfg, market, pricing, jnp.float64)
    if fn.calibration is None:
        raise ValueError(f"{cfg.trade_id}: not a calibrated option; nothing to replay")
    model, positioned = fn.calibration.model, fn.terms
    option, sign = positioned.terms, float(positioned.sign)
    calibration = option.calibration
    base = curves_on(fn.nodes, fn.rates)
    quotes = jnp.asarray(fn.quotes)
    own = [factors.slice_of(i) for i in factor_indices(cfg, factors)]
    shifts = np.asarray(shifts, dtype=np.float64)

    def curves(scenario: Optional[int]):
        if scenario is None:
            return base
        return tuple(with_node_deviation(c, jnp.asarray(shifts[scenario][s], jnp.float64)) for c, s in zip(base, own))

    def change(scenario: Optional[int]):
        """The option on ORE's fit less the option on the engine's, and ORE's largest miss."""
        on = curves(scenario)
        disc, index, vols = model.market(calibration, on, quotes)
        values, helpers, _, _ = bootstrap(calibration.basket, disc, index, vols, calibration.dt, model.reversion,
                                          model.solver)
        table = _table_of(np.asarray(helpers), tables, "the base market" if scenario is None else f"scenario {scenario}")
        relative = table[:, 1] / table[:, 0]
        target = jnp.asarray(relative, jnp.float64) * helpers   # model - market as ORE's ended

        def residual(zeta):
            return helper_residuals(calibration.basket, disc, index, vols, zeta, model.reversion, model.solver) - target

        # Each helper's equation is a root in its own zeta (`helper_residuals`), so a tangent along
        # ones is the diagonal: Newton from the engine's exact fit, a step of ~ORE's miss.
        zeta = jnp.cumsum(jnp.square(values) * calibration.dt)
        for _ in range(8):
            r, slope = jax.jvp(residual, (zeta,), (jnp.ones_like(zeta),))
            zeta = zeta - r / slope
        steps = jnp.diff(zeta, prepend=0.0)
        if not bool(jnp.all(steps > 0.0)) or float(jnp.max(jnp.abs(residual(zeta)) / jnp.abs(helpers))) > 1e-13:
            raise AssertionError(f"{cfg.trade_id}: the engine's helpers cannot reach ORE's fit on "
                                 f"{'the base market' if scenario is None else f'scenario {scenario}'}")
        ores = jnp.sqrt(steps / calibration.dt)
        moved = sign * (float(option_on_model(option, on, ores)) - float(option_on_model(option, on, values)))
        return moved, float(np.max(np.abs(relative)))

    base_change, base_miss = change(None)
    changes = [change(int(s)) for s in scenarios]
    return Replay(tuple(int(s) for s in scenarios), np.asarray([c - base_change for c, _ in changes]),
                  np.asarray([m for _, m in changes]), base_miss)


def _table_of(helpers: np.ndarray, tables: Sequence[np.ndarray], what: str) -> np.ndarray:
    """The last logged calibration whose helpers' market values are `helpers`' (to `MATCH`)."""
    matches = [t for t in tables if t.shape[0] == helpers.shape[0]
               and np.max(np.abs(t[:, 0] - helpers) / np.abs(helpers)) <= MATCH]
    if not matches:
        raise AssertionError(f"ORE's log has no calibration of {what}'s helpers ({len(tables)} tables read)")
    return matches[-1]

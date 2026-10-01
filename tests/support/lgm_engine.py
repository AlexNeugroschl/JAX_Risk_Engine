"""
The Bermudan/American grid engine (`engine.instruments.bermudan_swaption`) with an explicit
model: the LGM's reversion, volatility and term structure given directly, as ORE's engine takes
them with `Calibration=None`. Tests that pin the engine itself (against ORE's own engine,
QuantLib's Hull-White engines or closed forms) price through `grid_npv`; a portfolio prices
through `engine.valuation.bermudan`, which calibrates the model first.
"""
from typing import Optional, Union

import jax.numpy as jnp

from engine.instruments.bermudan_swaption import grid_value, prepare_bermudan
from engine.market import ZeroCurveConfig
from engine.models.curves import ZeroCurve
from engine.models.lgm import Sigma


def prepared(cfg, *, a: float, sigma: Union[float, Sigma], curve: ZeroCurveConfig,
             index_curve: Optional[ZeroCurveConfig] = None, n_per_std: int = 48, std_devs: float = 6.0,
             steps_per_year: int = 24, dtype=jnp.float64):
    """`prepare_bermudan` on the trade's own date with the model `(a, sigma)` on `curve`
    (and `index_curve` for the coupons, default `curve`)."""
    return prepare_bermudan(
        cfg, reversion=a, sigma=sigma, n_per_std=n_per_std, std_devs=std_devs,
        exercise_time_steps_per_year=steps_per_year, curve=ZeroCurve.from_config(curve, dtype=dtype),
        index_curve=None if index_curve is None else ZeroCurve.from_config(index_curve, dtype=dtype))


def grid_npv(cfg, **model) -> float:
    """The engine's t=0 NPV (0 once expired, as ORE's `isExpired`); `model` as `prepared`."""
    if cfg.is_expired():
        return 0.0
    return float(grid_value(prepared(cfg, **model)))

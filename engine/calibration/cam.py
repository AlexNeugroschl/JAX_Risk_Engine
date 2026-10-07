"""
Calibration of the cross-asset model's interest-rate components, as ORE's
`CrossAssetModelBuilder` calibrates them (plan T-14, V-6): each currency's LGM volatility is
bootstrapped (`LgmBuilder`, `Calibration = Bootstrap`) to its `CalibrationSwaptions` basket --
tenor-based expiries and terms, ATM -- on today's discount curve and the swap index's
forwarding curve, with the configured constant reversion. The same helpers and bootstrap as
a trade's calibration (`engine.calibration.ore_lgm`), which match ORE's to ~1e-12.

A Hull-White currency (`HullWhiteConfig`, ORE's `VolatilityType HullWhite`) is calibrated to
the same helpers: a helper's model price depends on the volatility only through zeta at its
expiry, so the Hagan bootstrap is converted bucket by bucket into the short-rate volatility
with the same zeta at every expiry (`engine.models.lgm.hull_white_matching_zeta`); every
helper keeps its price, which is what ORE's bootstrap of the adaptor solves for.

FX and equity volatilities are inputs (ORE's `Calibrate = false`); calibrating them to
options is not implemented.
"""
from dataclasses import dataclass
from typing import Dict, Mapping

import jax.numpy as jnp
import numpy as np

from engine.calibration.ore_lgm import basket_vols, bootstrap_sigma, build_basket
from engine.market import Market, index_name
from engine.models.curves import ZeroCurve
from engine.models.lgm import Sigma, hull_white_matching_zeta


@dataclass
class CamCalibration:
    """One currency's calibrated volatility, in its model's parametrization, and each
    helper's market and model value."""
    sigma: Sigma
    market: np.ndarray
    model: np.ndarray


def calibrate_currency(market: Market, currency: str, lgm) -> CamCalibration:
    """Bootstrap `lgm` (an `engine.simulation.config.LgmConfig` or `HullWhiteConfig` with a
    basket) on today's market."""
    conventions = lgm.swap_index
    basket = build_basket(market.asof, list(lgm.calibration_expiries), list(lgm.calibration_terms), conventions)
    disc = ZeroCurve.from_config(market.currency(currency).discount_curve)
    index = ZeroCurve.from_config(market.index_curve(currency, index_name(currency, conventions.index_tenor_months)))
    result = bootstrap_sigma(basket, disc, index, basket_vols(basket, market.swaption_vols(currency), market.asof),
                             lgm.reversion, lgm.solver)
    if np.any(np.asarray(result.hit_ceiling)):
        raise ValueError(f"{currency} CAM calibration failed: a helper's volatility is not attainable")
    sigma = Sigma(jnp.asarray(result.times), result.values)
    if lgm.volatility_type == "HullWhite":
        sigma = hull_white_matching_zeta(lgm.reversion, sigma, basket[-1].expiry_time)
    return CamCalibration(sigma, np.asarray(result.market), np.asarray(result.model))


def calibrate_cam(market: Market, lgm_configs: Mapping) -> Dict[str, CamCalibration]:
    """Every currency in `lgm_configs` (currency -> `LgmConfig` or `HullWhiteConfig`) with a
    calibration basket."""
    return {currency: calibrate_currency(market, currency, lgm)
            for currency, lgm in lgm_configs.items() if lgm.calibrated}

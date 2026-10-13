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

A calibration that misses its basket by ORE's acceptance (the helpers' relative price errors
against the cross-asset model's `bootstrap_tolerance`, ORE's `BootstrapTolerance`) is refused,
naming the currency, as ORE's builder fails without `ContinueOnCalibrationError` (I-73).

FX and equity volatilities are inputs (ORE's `Calibrate = false`); calibrating them to
options is not implemented.
"""
from dataclasses import dataclass
from typing import Dict, Mapping

import jax.numpy as jnp
import numpy as np

from engine.calibration.ore_lgm import (
    BOOTSTRAP_TOLERANCE, basket_vols, bootstrap_sigma, build_basket, check_tolerance, require_accepted,
)
from engine.market_data.curves import ZeroCurve
from engine.market_data.market import Market, index_name
from engine.models.lgm import Sigma, hull_white_matching_zeta


@dataclass
class CamCalibration:
    """One currency's calibrated volatility, in its model's parametrization, and each
    helper's market and model value."""
    sigma: Sigma
    market: np.ndarray
    model: np.ndarray


def calibrate_currency(market: Market, currency: str, lgm, tolerance: float = BOOTSTRAP_TOLERANCE) -> CamCalibration:
    """Bootstrap `lgm` (an `engine.market_simulation.config.LgmConfig` or `HullWhiteConfig` with a
    basket) on today's market, refused if its error is not below `tolerance` (ORE's
    `BootstrapTolerance`)."""
    conventions = lgm.swap_index
    basket = build_basket(market.asof, list(lgm.calibration_expiries), list(lgm.calibration_terms), conventions)
    disc = ZeroCurve.from_config(market.currency(currency).discount_curve)
    index = ZeroCurve.from_config(market.index_curve(currency, index_name(currency, conventions.index_tenor_months)))
    result = bootstrap_sigma(basket, disc, index, basket_vols(basket, market.swaption_vols(currency), market.asof),
                             lgm.reversion, lgm.solver)
    require_accepted(result, tolerance, f"the {currency} model of the cross-asset model on {market.asof}")
    sigma = Sigma(jnp.asarray(result.times), result.values)
    if lgm.volatility_type == "HullWhite":
        sigma = hull_white_matching_zeta(lgm.reversion, sigma, basket[-1].expiry_time)
    return CamCalibration(sigma, np.asarray(result.market), np.asarray(result.model))


def calibrate_cam(market: Market, lgm_configs: Mapping,
                  tolerance: float = BOOTSTRAP_TOLERANCE) -> Dict[str, CamCalibration]:
    """Every currency in `lgm_configs` (currency -> `LgmConfig` or `HullWhiteConfig`) with a
    calibration basket, each refused if its error is not below `tolerance` (ORE's
    `BootstrapTolerance`, one for the model, `CamConfig.bootstrap_tolerance`)."""
    check_tolerance(tolerance, "bootstrap_tolerance")
    return {currency: calibrate_currency(market, currency, lgm, tolerance)
            for currency, lgm in lgm_configs.items() if lgm.calibrated}

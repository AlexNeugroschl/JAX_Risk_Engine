"""
Today's market: the t=0 inputs every valuation starts from, grouped by currency as ORE's
`TodaysMarket` groups them.

  * `CurrencyMarket`: the discount curve, one forwarding curve per Ibor index, and an ATM
    normal swaption volatility matrix.
  * `Market`: the as-of date, the currency markets, FX spots and equity spots/dividends.

Model parameters (reversions, simulation volatilities, correlations) are not market data;
they belong to the simulation configuration (`engine.simulation.config.CamConfig`), as in
ORE's `simulation.xml`.

Curves are `ZeroCurveConfig`s: continuously compounded ACT/365 zero rates at pillar times,
interpolated and extrapolated as `engine.models.curves.ZeroCurve` (ORE's `ZeroCurve`).
"""
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import ORE

from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER
from engine.simulation.market_model import ZeroCurveConfig

#: Conventions of every swaption volatility matrix: ORE's `SwaptionVolatilityCurveConfig`
#: defaults for a TARGET market (calendar, option date roll, day counter).
VOL_CALENDAR = ORE.TARGET()
VOL_BUSINESS_DAY_CONVENTION = ORE.Following
VOL_DAY_COUNTER = ORE.Actual365Fixed()


def index_name(currency: str, tenor_months: int) -> str:
    """The name of the generic Ibor index of `currency` with a `tenor_months` fixing period,
    e.g. `USD-SIMINDEX-6M` (the index `engine.models.ore_builders.build_vanilla_swap` builds;
    also the name the ORE oracle registers)."""
    return f"{currency}-SIMINDEX-{tenor_months}M"


def swap_length(tenor: str) -> float:
    """QuantLib's `SwaptionVolatilityStructure::swapLength(Period)`: months / 12 or years."""
    period = ORE.Period(tenor)
    if period.units() == ORE.Months:
        return period.length() / 12.0
    if period.units() == ORE.Years:
        return float(period.length())
    raise ValueError(f"swap tenor {tenor!r} must be in months or years")


def swap_length_between(start: ORE.Date, end: ORE.Date) -> float:
    """QuantLib's `swapLength(start, end)`: the day count rounded to whole months (365.25
    days a year), in years."""
    if not end > start:
        raise ValueError(f"swap end {end} must be after its start {start}")
    return float(np.round((end - start) / 365.25 * 12.0)) / 12.0


@dataclass(frozen=True)
class SwaptionVolSurface:
    """ATM normal swaption volatilities, `vols[i][j]` for option tenor i and swap tenor j:
    QuantLib's `SwaptionVolatilityMatrix` with `flatExtrapolation = true` (ORE's ATM
    `SwaptionVolatilityCurveConfig` with `Extrapolation Flat`, `VolatilityType Normal`).

    Option tenors roll to dates with `VOL_CALENDAR`/`VOL_BUSINESS_DAY_CONVENTION` from the
    reference date and become times on `VOL_DAY_COUNTER`; swap tenors become lengths in
    years. Between points the volatility is bilinear in (swap length, option time), and flat
    beyond them. A smile is not supported (plan X-5): the surface is ATM only.
    """
    option_tenors: Tuple[str, ...]
    swap_tenors: Tuple[str, ...]
    vols: Tuple[Tuple[float, ...], ...]

    def __post_init__(self):
        vols = np.asarray(self.vols, dtype=np.float64)
        if vols.shape != (len(self.option_tenors), len(self.swap_tenors)):
            raise ValueError(
                f"vols must be {len(self.option_tenors)} option tenors x {len(self.swap_tenors)} swap "
                f"tenors; got shape {vols.shape}")
        if not np.all(np.isfinite(vols)) or np.any(vols < 0.0):
            raise ValueError("swaption volatilities must be finite and non-negative")
        lengths = [swap_length(t) for t in self.swap_tenors]
        if np.any(np.diff(lengths) <= 0.0):
            raise ValueError(f"swap tenors must increase; got {list(self.swap_tenors)}")
        object.__setattr__(self, "option_tenors", tuple(self.option_tenors))
        object.__setattr__(self, "swap_tenors", tuple(self.swap_tenors))
        object.__setattr__(self, "vols", tuple(tuple(float(v) for v in row) for row in vols))

    def option_times(self, reference_date: ORE.Date) -> np.ndarray:
        """Option times from `reference_date` (`SwaptionVolatilityDiscrete`)."""
        dates = [VOL_CALENDAR.advance(reference_date, ORE.Period(t), VOL_BUSINESS_DAY_CONVENTION)
                 for t in self.option_tenors]
        times = np.asarray([VOL_DAY_COUNTER.yearFraction(reference_date, d) for d in dates])
        if np.any(np.diff(times) <= 0.0):
            raise ValueError(f"option tenors must increase; got {list(self.option_tenors)}")
        return times

    def volatility(self, reference_date: ORE.Date, option_time, swap_len) -> np.ndarray:
        """Bilinear in (swap length, option time), flat outside the grid (host, float64;
        broadcasts over its arguments)."""
        y = _bind_and_locate(self.option_times(reference_date), option_time)
        x = _bind_and_locate(np.asarray([swap_length(t) for t in self.swap_tenors]), swap_len)
        z = np.asarray(self.vols)
        # A one-point axis gets weight 0 on its (repeated) neighbour.
        z = np.pad(z, ((0, int(z.shape[0] == 1)), (0, int(z.shape[1] == 1))), mode="edge")
        (j, u), (i, s) = y, x
        return ((1 - s) * (1 - u) * z[j, i] + s * (1 - u) * z[j, i + 1]
                + (1 - s) * u * z[j + 1, i] + s * u * z[j + 1, i + 1])

    def black_variance(self, reference_date: ORE.Date, option_time, swap_len) -> np.ndarray:
        """vol^2 * option_time (a normal variance; QuantLib's `blackVariance`)."""
        return self.volatility(reference_date, option_time, swap_len) ** 2 * np.asarray(option_time)


def _bind_and_locate(grid: np.ndarray, value):
    """`(index, weight)` of `value` on `grid` for `BilinearInterpolation` after
    `FlatExtrapolator2D`'s binding to the grid range. A one-point axis is constant (index 0,
    weight 0; the caller pads that axis so `index + 1` exists)."""
    value = np.clip(np.asarray(value, dtype=np.float64), grid[0], grid[-1])
    if grid.size == 1:
        return np.zeros(value.shape, dtype=np.int64), np.zeros(value.shape)
    i = np.clip(np.searchsorted(grid, value, side="right") - 1, 0, grid.size - 2)
    return i, (value - grid[i]) / (grid[i + 1] - grid[i])


@dataclass(frozen=True)
class CurrencyMarket:
    """One currency's curves and volatilities.

    discount_curve: the currency's discount (and LGM model) curve.
    index_curves: forwarding curve per Ibor index name (`index_name`), e.g.
        `{"USD-SIMINDEX-6M": curve}`. A trade whose index is missing is refused; there is no
        fallback to the discount curve.
    swaption_vols: ATM normal swaption volatilities (Europeans, calibration); optional when
        no trade or calibration needs them.
    """
    discount_curve: ZeroCurveConfig
    index_curves: Mapping[str, ZeroCurveConfig] = field(default_factory=dict)
    swaption_vols: Optional[SwaptionVolSurface] = None


@dataclass(frozen=True)
class EquityMarket:
    """An equity's spot and currency. `dividend_curve` is its dividend yield curve as a
    zero curve (`None` = no dividends); its forecast curve is its currency's discount
    curve."""
    currency: str
    spot: float
    dividend_curve: Optional[ZeroCurveConfig] = None


@dataclass(frozen=True)
class Market:
    """Today's market on `asof`.

    fx_spots: domestic units per unit of foreign currency, keyed `<foreign><domestic>`
        (e.g. `"EURUSD": 1.10`), ORE's FX quote orientation.
    """
    asof: ORE.Date
    currencies: Mapping[str, CurrencyMarket]
    fx_spots: Mapping[str, float] = field(default_factory=dict)
    equities: Mapping[str, EquityMarket] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.asof, ORE.Date):
            raise TypeError(f"asof must be an ORE.Date; got {self.asof!r}")
        if not self.currencies:
            raise ValueError("a market needs at least one currency")
        for pair, spot in self.fx_spots.items():
            if len(pair) != 6 or pair[:3] not in self.currencies or pair[3:] not in self.currencies:
                raise ValueError(f"FX spot {pair!r} must name two market currencies, e.g. 'EURUSD'")
            if not np.isfinite(spot) or spot <= 0.0:
                raise ValueError(f"FX spot {pair} must be positive; got {spot}")
        for name, eq in self.equities.items():
            if eq.currency not in self.currencies:
                raise ValueError(f"equity {name!r} is in {eq.currency}, which has no currency market")
            if not np.isfinite(eq.spot) or eq.spot <= 0.0:
                raise ValueError(f"equity {name!r} spot must be positive; got {eq.spot}")

    def currency(self, code: str) -> CurrencyMarket:
        if code not in self.currencies:
            raise KeyError(f"no market for currency {code!r}; have {sorted(self.currencies)}")
        return self.currencies[code]

    def index_curve(self, currency: str, index: str) -> ZeroCurveConfig:
        curves = self.currency(currency).index_curves
        if index not in curves:
            raise KeyError(f"no forwarding curve for index {index!r} in {currency}; have {sorted(curves)}")
        return curves[index]

    def swaption_vols(self, currency: str) -> SwaptionVolSurface:
        vols = self.currency(currency).swaption_vols
        if vols is None:
            raise KeyError(f"no swaption volatilities for {currency}")
        return vols

    def fx_spot(self, foreign: str, domestic: str) -> float:
        """Domestic units per unit of `foreign`; 1 for the same currency. Only the quoted
        direction or its inverse is used, never a cross."""
        if foreign == domestic:
            return 1.0
        if foreign + domestic in self.fx_spots:
            return float(self.fx_spots[foreign + domestic])
        if domestic + foreign in self.fx_spots:
            return 1.0 / float(self.fx_spots[domestic + foreign])
        raise KeyError(f"no FX spot for {foreign}{domestic}")


def times_of(asof: ORE.Date, dates: Sequence[ORE.Date]) -> np.ndarray:
    """ACT/365 times of `dates` from `asof`: the model time axis (ORE's CAM term structure
    day counter), shared by the simulation grid and every curve."""
    return np.asarray([TIME_AXIS_DAY_COUNTER.yearFraction(asof, d) for d in dates], dtype=np.float64)


__all__ = [
    "CurrencyMarket", "EquityMarket", "Market", "SwaptionVolSurface", "ZeroCurveConfig",
    "index_name", "swap_length", "swap_length_between", "times_of",
]

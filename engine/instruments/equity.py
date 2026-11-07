"""
Cash equities and listed equity options (TraderX fork, R1 and R2): the trades.

A cash equity (or ETF) position is `quantity` shares, each `multiplier` units of the
equity's spot (`market.equities[symbol]`), in the equity's currency:

    NPV = quantity x multiplier x spot

A listed equity option is `quantity` contracts of `multiplier` shares each, a call or put on
`symbol` struck at `strike`, exercisable on `expiry_date` (European) or on any day up to it
(American, the listed-equity convention), priced by `engine.pricing.equity`.

Both carry no curve or volatility of their own; the market supplies the spot, the dividend
curve and the volatility, and their prices read the equity's spot as a risk factor
(`engine.risk.market.factors.SpotFactor`). They price today and under market-risk shocks
(`npv`, `market_risk`), not on simulated paths or as Greeks (`engine.pricing.cube.t0_only`).
"""
import math
from dataclasses import dataclass, field
from typing import ClassVar

import ORE

from engine.instruments._validation import _validate_identity

OPTION_TYPES = ("Call", "Put")
EXERCISE_STYLES = ("American", "European")


def _finite(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number; got {value!r}")


@dataclass(frozen=True)
class EquityConfig:
    """`quantity` (signed: a short is negative) shares of `symbol`, each `multiplier` units of
    its spot."""
    symbol: str
    quantity: float
    multiplier: float = 1.0
    currency: str = "USD"
    trade_id: str = field(kw_only=True)
    evaluation_date: ORE.Date = field(kw_only=True)
    product: ClassVar[str] = "equity"

    def __post_init__(self):
        _validate_identity(self.trade_id, self.evaluation_date)
        if not self.symbol:
            raise ValueError("symbol must name the equity")
        _finite("quantity", self.quantity)
        _finite("multiplier", self.multiplier)
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be positive; got {self.multiplier}")

    @property
    def units(self) -> float:
        """Shares held, the multiplier applied once."""
        return self.quantity * self.multiplier


@dataclass(frozen=True)
class EquityOptionConfig:
    """`quantity` (signed) contracts of `multiplier` shares: a `option_type` on `symbol` at
    `strike`, `exercise` American (any day to `expiry_date`) or European (on it)."""
    symbol: str
    option_type: str
    strike: float
    expiry_date: ORE.Date
    quantity: float
    multiplier: float = 100.0
    exercise: str = "American"
    currency: str = "USD"
    trade_id: str = field(kw_only=True)
    evaluation_date: ORE.Date = field(kw_only=True)
    product: ClassVar[str] = "equity_option"

    def __post_init__(self):
        _validate_identity(self.trade_id, self.evaluation_date)
        if not self.symbol:
            raise ValueError("symbol must name the underlying equity")
        if self.option_type not in OPTION_TYPES:
            raise ValueError(f"option_type must be one of {OPTION_TYPES}; got {self.option_type!r}")
        if self.exercise not in EXERCISE_STYLES:
            raise ValueError(f"exercise must be one of {EXERCISE_STYLES}; got {self.exercise!r}")
        if not isinstance(self.expiry_date, ORE.Date):
            raise TypeError(f"expiry_date must be an ORE.Date; got {self.expiry_date!r}")
        for name in ("strike", "quantity", "multiplier"):
            _finite(name, getattr(self, name))
        if self.strike <= 0 or self.multiplier <= 0:
            raise ValueError(f"strike and multiplier must be positive; got {self.strike}, {self.multiplier}")

    @property
    def units(self) -> float:
        """Shares controlled, the multiplier applied once."""
        return self.quantity * self.multiplier

    def is_expired(self) -> bool:
        """Expired before the evaluation date: nothing left to exercise (on the expiry date
        itself the option is worth its intrinsic value)."""
        return self.expiry_date < self.evaluation_date


EQUITY_PRODUCTS = (EquityConfig, EquityOptionConfig)

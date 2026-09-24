"""
The risk factors a market-risk run shocks: the pillar zero rates of a set of
named curves.

A factor vector is every curve's pillar rates laid end to end, curve 0 first,
each curve's pillars in time order. Shocks, covariances and histories are all
expressed on that vector, so one ordering is shared by every piece of the
pipeline (`scenarios`, `revaluation`).

Shocks are **absolute** zero-rate moves: a rate can be zero or negative, and a
relative move is undefined there (Basel plan decision D-6).
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from engine.simulation.market_model import ZeroCurveConfig


@dataclass(frozen=True, eq=False)
class RateRiskFactors:
    """Named curves whose pillar zero rates are the risk factors.

    Trades refer to curves by index into `curves`: a swap by its
    `discount_curve_index`/`forward_curve_index`, a swaption by its
    `rate_factor_index`, a bond by its `curve_index`.
    """
    curves: Tuple[ZeroCurveConfig, ...]
    names: Tuple[str, ...]

    def __post_init__(self):
        if not self.curves:
            raise ValueError("at least one curve is required")
        if len(self.names) != len(self.curves):
            raise ValueError(f"{len(self.names)} names for {len(self.curves)} curves")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"curve names must be unique; got {list(self.names)}")
        for name, curve in zip(self.names, self.curves):
            times = np.asarray(curve.times, dtype=np.float64)
            rates = np.asarray(curve.rates, dtype=np.float64)
            if times.ndim != 1 or times.size == 0 or times.shape != rates.shape:
                raise ValueError(f"curve {name!r}: times and rates must be non-empty and the same length")
            if np.any(np.diff(times) <= 0.0):
                raise ValueError(f"curve {name!r}: pillar times must be strictly increasing")
            if not (np.all(np.isfinite(times)) and np.all(np.isfinite(rates))):
                raise ValueError(f"curve {name!r}: pillar times and rates must be finite")

    @classmethod
    def from_curves(cls, curves: Sequence[ZeroCurveConfig], names: Optional[Sequence[str]] = None) -> "RateRiskFactors":
        names = tuple(names) if names is not None else tuple(f"curve{i}" for i in range(len(curves)))
        return cls(curves=tuple(curves), names=names)

    @property
    def sizes(self) -> Tuple[int, ...]:
        """Pillar count of each curve."""
        return tuple(len(c.times) for c in self.curves)

    @property
    def size(self) -> int:
        """Total number of factors."""
        return sum(self.sizes)

    def slice_of(self, curve_index: int) -> slice:
        """Where curve `curve_index`'s pillars sit in the factor vector."""
        start = sum(self.sizes[:curve_index])
        return slice(start, start + self.sizes[curve_index])

    def base_rates(self) -> np.ndarray:
        """Today's pillar rates, as one factor vector `[F]`."""
        return np.concatenate([np.asarray(c.rates, dtype=np.float64) for c in self.curves])

    def labels(self) -> List[str]:
        """One label per factor, `"<curve>/<pillar time>y"`, in vector order."""
        return [f"{name}/{t:g}y" for name, curve in zip(self.names, self.curves) for t in curve.times]

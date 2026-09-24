"""
Shock scenarios for a market-risk run: absolute moves of every risk factor
over the risk horizon, one row per scenario.

Two sources, one shape:

    monte_carlo_scenarios   many draws from a zero-mean Gaussian with a given
                            horizon covariance of factor moves -- scrambled
                            Sobol normals, so different seeds are independent
                            randomized-QMC replicates.
    historical_scenarios    the overlapping `horizon_days` moves observed in a
                            history of factor levels (Basel IMA's historical
                            simulation; plan phase P2).

`covariance_from_history` estimates the Monte Carlo covariance from the same
kind of history, so the two sources can be compared on one data set.

Both are **real-world forecasts** of the horizon move, labelled
`historical-forecast` in the engine's measure vocabulary -- not the
risk-neutral measure the exposure simulation runs under.
"""
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from engine.market_risk.factors import RateRiskFactors
from engine.risk.var_es import RISK_MEASURE_HISTORICAL
from engine.simulation.market_model import generate_sobol_normals

SOURCE_MONTE_CARLO = "monte-carlo"
SOURCE_HISTORICAL = "historical"
SOURCES = (SOURCE_MONTE_CARLO, SOURCE_HISTORICAL)


@dataclass(frozen=True, eq=False)
class ShockScenarios:
    """Absolute risk-factor shifts over one risk horizon.

    shifts: `[S, F]`, row `s` is scenario `s`'s move of every factor, in
        `factors`' vector order.
    windows: for historical scenarios, the `(start, end)` date labels of the
        window each row was observed over; `None` for Monte Carlo.
    """
    factors: RateRiskFactors
    shifts: np.ndarray
    horizon_days: int
    source: str
    measure: str = RISK_MEASURE_HISTORICAL
    windows: Optional[Tuple[Tuple[str, str], ...]] = None

    def __post_init__(self):
        shifts = np.asarray(self.shifts, dtype=np.float64)
        if shifts.ndim != 2 or shifts.shape[1] != self.factors.size:
            raise ValueError(
                f"shifts must be [scenarios, {self.factors.size}] for these factors; got {shifts.shape}"
            )
        if shifts.shape[0] < 2:
            raise ValueError("at least two scenarios are required for a tail statistic")
        if not np.all(np.isfinite(shifts)):
            raise ValueError("shifts must be finite")
        if self.horizon_days < 1:
            raise ValueError(f"horizon_days must be at least 1; got {self.horizon_days}")
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}; got {self.source!r}")
        if self.windows is not None and len(self.windows) != shifts.shape[0]:
            raise ValueError(f"{len(self.windows)} windows for {shifts.shape[0]} scenarios")
        object.__setattr__(self, "shifts", shifts)

    @property
    def num_scenarios(self) -> int:
        return self.shifts.shape[0]


def monte_carlo_scenarios(
    factors: RateRiskFactors,
    covariance,
    horizon_days: int,
    num_scenarios: int,
    seed: int = 42,
) -> ShockScenarios:
    """Zero-mean Gaussian factor moves with the given horizon covariance.

    covariance: `[F, F]` covariance of the factors' `horizon_days` moves, in
        `factors`' vector order. Must be symmetric positive semi-definite; it
        may be singular (perfectly correlated pillars are common), because
        it is factorized by eigendecomposition rather than Cholesky.
    num_scenarios: a power of two keeps the Sobol sequence balanced.
    seed: Sobol scrambling seed.
    """
    cov = _validated_covariance(covariance, factors.size)
    if num_scenarios < 2:
        raise ValueError(f"num_scenarios must be at least 2; got {num_scenarios}")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Eigenvalues below round-off of the largest are zero: a rank-deficient
    # covariance otherwise gains ~1e-10 of spurious noise per move from the
    # square roots of its ~1e-21 numerical eigenvalues.
    eigenvalues = np.where(eigenvalues > 1e-12 * eigenvalues[-1], eigenvalues, 0.0)
    root = eigenvectors * np.sqrt(eigenvalues)[None, :]   # cov = root @ root.T
    normals = np.asarray(generate_sobol_normals(num_scenarios, 1, factors.size, jnp.float64, seed=seed)[0])
    return ShockScenarios(
        factors=factors, shifts=normals @ root.T, horizon_days=horizon_days, source=SOURCE_MONTE_CARLO,
    )


def horizon_moves(history, horizon_days: int) -> np.ndarray:
    """Overlapping `horizon_days` moves of a `[D, F]` history of factor
    levels (oldest first): row `i` is `history[i + h] - history[i]`."""
    levels = np.asarray(history, dtype=np.float64)
    if levels.ndim != 2:
        raise ValueError(f"history must be [dates, factors]; got shape {levels.shape}")
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be at least 1; got {horizon_days}")
    if levels.shape[0] <= horizon_days + 1:
        raise ValueError(
            f"a history of {levels.shape[0]} dates gives fewer than two {horizon_days}-day moves"
        )
    if not np.all(np.isfinite(levels)):
        raise ValueError("history must be finite")
    return levels[horizon_days:] - levels[:-horizon_days]


def historical_scenarios(
    factors: RateRiskFactors,
    history,
    horizon_days: int,
    dates: Optional[Sequence[str]] = None,
) -> ShockScenarios:
    """One scenario per overlapping `horizon_days` window of `history`.

    history: `[D, F]` factor levels on consecutive business dates, oldest
        first, in `factors`' vector order.
    dates: optional `D` date labels; each scenario then records the window
        it came from.
    """
    moves = horizon_moves(history, horizon_days)
    if moves.shape[1] != factors.size:
        raise ValueError(f"history has {moves.shape[1]} factors; expected {factors.size}")
    windows = None
    if dates is not None:
        if len(dates) != len(history):
            raise ValueError(f"{len(dates)} dates for {len(history)} history rows")
        windows = tuple((str(dates[i]), str(dates[i + horizon_days])) for i in range(moves.shape[0]))
    return ShockScenarios(
        factors=factors, shifts=moves, horizon_days=horizon_days, source=SOURCE_HISTORICAL, windows=windows,
    )


def covariance_from_history(history, horizon_days: int) -> np.ndarray:
    """Sample covariance (`ddof=1`) of the overlapping `horizon_days` moves
    of `history` -- the input `monte_carlo_scenarios` needs to draw from the
    same distribution a historical run samples."""
    moves = horizon_moves(history, horizon_days)
    return np.atleast_2d(np.cov(moves, rowvar=False, ddof=1))


def _validated_covariance(covariance, size: int) -> np.ndarray:
    """Square, symmetric and PSD, to a tolerance relative to its own scale:
    rate-move covariances are ~1e-6, so an absolute tolerance would accept
    almost anything."""
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (size, size):
        raise ValueError(f"covariance must be [{size}, {size}] for these factors; got {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance must be finite")
    scale = max(float(np.max(np.abs(np.diag(cov)))), np.finfo(np.float64).tiny)
    tol = 1e-10 * scale
    if np.max(np.abs(cov - cov.T)) > tol:
        raise ValueError("covariance must be symmetric")
    smallest = float(np.min(np.linalg.eigvalsh(cov)))
    if smallest < -1e-8 * scale:
        raise ValueError(f"covariance is not positive semi-definite (smallest eigenvalue {smallest:.3e})")
    return 0.5 * (cov + cov.T)

"""
What a run request checks in its results before returning them (`RunRequestConfig.checks`,
I-127).

**Non-finite values.** A NaN or an infinity in a computed value is never a usable figure, and it
can hide: a P&L partly NaN gives a finite VaR and ES that are too small (the sort puts a NaN after
every gain, and the tail's mean skips it), and two NaN results compare equal in NumPy's
`assert_allclose`. One way to make them is a compute format's range: FP8 E4M3 holds magnitudes
up to 448, and an operand beyond it is NaN when products are not scaled
(`ProductOptions.scaling="none"`, the default, which the engine keeps as the format behaves).

Every run counts them, per array, in `RunResult.nonfinite` (only arrays with any; empty when all
are finite). `ResultChecks.nonfinite` says what the run then does:

    "raise"   refuse the result: `NonFiniteResultError` names every array and its count (the
              default); over HTTP the job fails as a numerical failure
    "warn"    return the result, with the same message in `RunResult.warnings`
    "allow"   return the result (research on what a format does out of range)

Counted are the values the analytics compute: today's values, the NPV cube per trade, the
exposure profiles (netting set and per trade), the Greeks, and the market-risk base values and
P&L per trade and of the portfolio. Not counted are statistics that are undefined by design and
say so with NaN (an ES standard error with one scramble, an ES of an empty tail), whose inputs
are counted, and the cross-asset calibration, whose own acceptance refuses a failed fit. The
direct entry points (`price_portfolio`, `run_market_risk`) return what they compute: the check
is the run request's.
"""
from dataclasses import dataclass, fields
from typing import Dict, Iterable, Tuple

import jax.numpy as jnp
import numpy as np

#: What a run does with non-finite values in its results (see the module docstring).
NONFINITE_MODES = ("raise", "warn", "allow")
#: The profile fields of `ExposureProfile` that hold computed values.
_PROFILE_FIELDS = ("epe", "ene", "ee_b", "eee_b", "epe_b", "eepe_b", "basel_epe", "basel_eepe")
#: How many arrays a message names before it summarizes the rest.
_NAMED = 20


class NonFiniteResultError(FloatingPointError):
    """A run's results hold NaN or infinite values and `ResultChecks.nonfinite` is "raise"."""


@dataclass(frozen=True)
class ResultChecks:
    """The run's checks of its results (see the module docstring)."""
    nonfinite: str = "raise"

    def __post_init__(self):
        if self.nonfinite not in NONFINITE_MODES:
            raise ValueError(f"config.checks.nonfinite must be one of {list(NONFINITE_MODES)}; got {self.nonfinite!r}")


def check_results(result, checks: ResultChecks) -> None:
    """Count `result`'s non-finite values into `result.nonfinite`, then raise, warn or allow as
    `checks` says (see the module docstring)."""
    result.nonfinite = nonfinite_counts(result)
    if not result.nonfinite or checks.nonfinite == "allow":
        return
    message = _message(result.nonfinite)
    if checks.nonfinite == "raise":
        raise NonFiniteResultError(f"{message}. Set config.checks.nonfinite to 'warn' or 'allow' to receive the "
                                   f"results with them")
    result.warnings.append(message)


def nonfinite_counts(result) -> Dict[str, int]:
    """`{array: count}` of the NaN and infinite values of each computed array of a `RunResult`
    with any (see the module docstring)."""
    counts: Dict[str, int] = {}
    ids = result.trade_ids
    for name, values in _arrays(result, ids):
        count = _count(values)
        if count:
            counts[name] = count
    return counts


def _arrays(result, ids) -> Iterable[Tuple[str, object]]:
    portfolio = result.portfolio
    if portfolio is not None:
        for trade, value in zip(ids, portfolio.base_npv_per_trade):
            yield f"npv[{trade}]", value
        cube = np.asarray(portfolio.npv_cube)
        if cube.size:
            for trade, column in zip(ids, np.moveaxis(cube, -1, 0)):
                yield f"exposure.npv_cube[{trade}]", column
        if portfolio.exposure is not None:
            yield from _profile("exposure.netting_set", portfolio.exposure)
        for trade, profile in zip(ids, portfolio.trade_exposures or ()):
            yield from _profile(f"exposure[{trade}]", profile)
        for index, greeks in (portfolio.greeks or {}).items():
            for greek, value in greeks.items():
                yield f"sensitivities[{ids[index]}].{greek}", value
    risk = result.market_risk
    if risk is not None:
        for trade, value in zip(ids, risk.base_npv_per_trade):
            yield f"market_risk.base_npv[{trade}]", value
        pnl = np.asarray(risk.pnl)
        for trade, column in zip(ids, np.moveaxis(pnl, -1, 0)):
            yield f"market_risk.pnl[{trade}]", column
        yield "market_risk.portfolio_pnl", risk.portfolio_pnl


def _profile(prefix: str, profile) -> Iterable[Tuple[str, object]]:
    names = {f.name for f in fields(profile)}
    for name in _PROFILE_FIELDS:
        if name in names:
            yield f"{prefix}.{name}", getattr(profile, name)
    for key, value in (profile.pfe or {}).items():
        yield f"{prefix}.pfe.{key}", value


def _count(values) -> int:
    if values is None:
        return 0
    array = np.asarray(values)
    if not jnp.issubdtype(array.dtype, jnp.inexact):   # JAX's test knows the formats below float32 too
        return 0
    return int(np.count_nonzero(~np.isfinite(array)))


def _message(counts: Dict[str, int]) -> str:
    named = [f"{name}: {count}" for name, count in list(counts.items())[:_NAMED]]
    rest = len(counts) - len(named)
    more = f", and {rest} more arrays" if rest else ""
    return (f"{sum(counts.values())} NaN or infinite values in the run's results ({', '.join(named)}{more}); "
            f"one cause is a compute format's range: FP8 E4M3 holds magnitudes up to 448 when products are not "
            f"scaled (ProductOptions.scaling)")

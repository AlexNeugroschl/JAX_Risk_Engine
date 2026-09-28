"""
Accrual day-count vocabulary: the table of day counts a trade may accrue on.

A leaf module that imports only `ORE`. Both `engine.models.ore_builders` and
`engine.integration.note` need the table, and `engine.integration` is not allowed to import
`engine.models` (tested by
`tests/test_integration_pipeline.py::TestPackageImportsNoSimulationPricer`), which keeps
the generic swap builder out of reach of bookings the EOD boundary refuses (I-05).

Only the instrument-accrual role lives here. The simulation time axis
(`TIME_AXIS_DAY_COUNTER`, always ACT/365) stays in `engine.models.ore_builders`.
"""
import ORE

#: Accrual day counts the engine supports, by name. Any other name is refused, never
#: defaulted.
SUPPORTED_ACCRUAL_DAY_COUNTS = {
    "ACT/365": ORE.Actual365Fixed(),
    # ORE's ISMA variant is ACT/ACT (ICMA), accruing against the coupon period.
    "ACT/ACT (ICMA)": ORE.ActualActual(ORE.ActualActual.ISMA),
}

#: Used when a trade names no accrual day count.
DEFAULT_ACCRUAL_DAY_COUNT = "ACT/365"


class UnsupportedDayCountError(ValueError):
    """A trade named an accrual day count the engine does not implement. Raised rather
    than substituting ACT/365, which would give a wrong accrual without warning."""


def resolve_accrual_day_count(name):
    """Name -> `ORE.DayCounter`, or raise `UnsupportedDayCountError`. `None` means the
    default. An `ORE.DayCounter` is returned unchanged (the allowlist governs names, which
    is what arrives over the wire)."""
    if name is None:
        name = DEFAULT_ACCRUAL_DAY_COUNT
    if not isinstance(name, str):
        return name  # already an ORE.DayCounter
    try:
        return SUPPORTED_ACCRUAL_DAY_COUNTS[name]
    except KeyError:
        raise UnsupportedDayCountError(
            f"accrual day count {name!r} is not supported; this engine implements "
            f"{sorted(SUPPORTED_ACCRUAL_DAY_COUNTS)}. It is refused rather than "
            f"defaulted -- a substituted day count prices confidently and wrongly."
        ) from None

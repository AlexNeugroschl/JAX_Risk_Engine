"""
Day counts: the simulation's time axis, and the table of day counts a trade may accrue on.

A leaf module that imports only `ORE`, so the TraderX path (`engine.traderx`), which may not
import JAX or the pricing layers (I-05; tested by
`tests/test_traderx_pipeline.py::TestPackageImportsNoSimulationPricer`), can use it.

Actual/365Fixed plays two separate roles; they are named apart.

  1. Simulation time axis (`TIME_AXIS_DAY_COUNTER`): converts an ORE.Date to the year
     fraction that indexes the simulated curves, the model and every cashflow. Fixed at
     ACT/365: the model's time, the scenario curves' tenor times and the pricers' cashflow
     times are measured alike (ORE's model day counter), so changing one would silently
     misalign every pricer.
  2. Instrument accrual (`SUPPORTED_ACCRUAL_DAY_COUNTS`): the day count a contract's coupons
     accrue on. Per booking; the `accrual_day_count` argument of
     `engine.instruments.schedules.build_vanilla_swap`.
"""
import ORE

#: Role 1. Other modules import this object rather than constructing their own
#: (identity checked by `tests/test_day_count_roles.py::TestTimeAxisIsOneObject`).
TIME_AXIS_DAY_COUNTER = ORE.Actual365Fixed()


def time_from_reference(evaluation_date: ORE.Date, date: ORE.Date) -> float:
    """A date's position on the simulation time axis, as ORE's
    `timeFromReference(d)` on a curve with day counter `TIME_AXIS_DAY_COUNTER`. Equal
    dates map to identical floats, so exercise dates match accrual dates exactly."""
    return TIME_AXIS_DAY_COUNTER.yearFraction(evaluation_date, date)


#: Role 2. Accrual day counts the engine supports, by name. Any other name is refused, never
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

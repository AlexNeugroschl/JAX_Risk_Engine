"""
Accrual day-count vocabulary -- the convention table, and nothing else.

**Extracted from `engine.models.ore_builders` in W1.3, and the reason is
structural rather than cosmetic.** The table is needed by two callers that
must not be able to reach each other:

  - `engine.models.ore_builders.build_vanilla_swap` (the simulation-layer
    trade builder), and
  - `engine.integration.note` (the EOD boundary's note pricer),

and `engine/integration/` is forbidden from importing `engine.models` --
asserted by
`tests/test_integration_pipeline.py::TestPackageImportsNoSimulationPricer`.
That ban is not bureaucratic: `engine.models.ore_builders` is where
`build_vanilla_swap` lives, and `build_vanilla_swap` is the exact object
W0.4's convention refusal exists to keep unreachable (see **I-05**). A
booking with unmapped conventions that reaches it gets priced confidently
and wrongly. Importing the module to borrow a dictionary would put that
builder one attribute access from the refusal boundary, so the dictionary
moves here instead of the ban being relaxed.

**This module imports `ORE` and nothing else from this repository.** That
is what makes it safe for both sides to depend on: it cannot pull a pricer
in behind it, because there is nothing here to pull.

---

**What belongs here: the *instrument accrual* role only.**

W1.1 split `Actual365Fixed`'s two unrelated jobs apart. The *simulation
time axis* (`TIME_AXIS_DAY_COUNTER`) stays in `engine.models.ore_builders`
and is permanently ACT/365 -- it is a property of the engine's own
simulated curve cube, not of any contract, and it is deliberately **not**
re-homed here. Only the per-instrument accrual convention, which is a
property of a booking, lives in this module. Keeping the two in separate
modules makes the distinction structural rather than a naming convention.
"""
import ORE

#: Accrual day counts this engine can faithfully build a trade with.
#: **Keyed by name, refused if absent** -- a trade naming an unsupported
#: day count is rejected, never silently defaulted (plan §W1.1 test 3, and
#: the same refuse-don't-infer rule as `engine.integration.conventions`).
SUPPORTED_ACCRUAL_DAY_COUNTS = {
    # The engine's default, and the only one every existing test pins.
    "ACT/365": ORE.Actual365Fixed(),
    # ORE's ISMA convention IS ACT/ACT (ICMA) -- the bond-market variant
    # that accrues against the actual coupon period rather than the
    # calendar year. Required by the TraderX note fixture, and first
    # actually exercised by the W1.3 note pricer.
    "ACT/ACT (ICMA)": ORE.ActualActual(ORE.ActualActual.ISMA),
}

#: The default, preserving every existing caller's behavior exactly.
DEFAULT_ACCRUAL_DAY_COUNT = "ACT/365"


class UnsupportedDayCountError(ValueError):
    """A trade named an accrual day count this engine does not implement.

    Raised rather than defaulted. A day count silently replaced by ACT/365
    produces a confident, wrong accrual -- the same failure mode as the
    convention substitution I-05 describes, one layer down.
    """


def resolve_accrual_day_count(name):
    """Name -> `ORE.DayCounter`, or raise.

    Accepts an `ORE.DayCounter` unchanged, so a caller with an exotic
    convention can pass one directly and take responsibility for it; the
    allowlist governs the string form, which is what arrives over the wire.
    """
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

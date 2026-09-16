"""
W1.1 — day count as a per-instrument input
(`docs/planning/traderx-integration-plan.md` §W1.1).

**The subtlety this pins.** `Actual365Fixed` filled two unrelated roles under
one name:

| Role | Must it stay ACT/365? |
|---|---|
| Simulation time axis — dates → year-fractions indexing the curve cube | **YES.** Changing it silently desynchronizes every pricer. |
| Instrument accrual — the coupon day count of the contract | **NO.** Per-instrument; ACT/ACT (ICMA) for the TraderX note. |

`TestOnlyTheAccrualRoleIsConfigurable` is the important one: it pins that the
*time axis* is untouched while the *accrual* became configurable. A change
that made both configurable would pass every other test here and break every
pricer's alignment with the simulated cube — silently, with no error.

The plan's named tests: full suite byte-identical with defaults; ACT/ACT ICMA
reproduces the fixture schedule; an unsupported day count is **refused**, not
silently defaulted.
"""
import numpy as np
import pytest

import ORE

from engine.models.ore_builders import (
    DAY_COUNTER,
    DEFAULT_ACCRUAL_DAY_COUNT,
    SUPPORTED_ACCRUAL_DAY_COUNTS,
    TIME_AXIS_DAY_COUNTER,
    UnsupportedDayCountError,
    build_vanilla_swap,
    fixed_leg_cashflows,
    floating_leg_cashflows,
    resolve_accrual_day_count,
)
from engine.instruments.swap import SwapConfig

EVAL_DATE = ORE.DateParser.parseISO("2025-06-02")


def _swap(accrual=None, tenor="5Y"):
    kwargs = {} if accrual is None else {"accrual_day_count": accrual}
    return build_vanilla_swap(
        notional=1_000_000.0, fixed_rate=0.04, payer=True, swap_tenor=tenor,
        index_tenor_months=6, floating_spread=0.0, evaluation_date=EVAL_DATE,
        **kwargs,
    )


class TestDefaultsAreByteIdentical:
    """Plan §W1.1 step 3: 'Default to ACT/365 so **every existing test is
    byte-identical**'. 38 tests across 10 files pin `ORE.Actual365Fixed()`
    directly, so this is load-bearing, not a nicety."""

    def test_default_equals_explicit_act365(self):
        implicit = fixed_leg_cashflows(_swap(), EVAL_DATE)
        explicit = fixed_leg_cashflows(_swap("ACT/365"), EVAL_DATE)

        assert np.array_equal(implicit.accrual_fractions, explicit.accrual_fractions)
        assert np.array_equal(implicit.payment_times, explicit.payment_times)
        assert np.array_equal(implicit.accrual_start_times, explicit.accrual_start_times)

    def test_default_constant_is_act365(self):
        assert DEFAULT_ACCRUAL_DAY_COUNT == "ACT/365"

    def test_floating_leg_default_is_unchanged_too(self):
        implicit = floating_leg_cashflows(_swap(), EVAL_DATE)
        explicit = floating_leg_cashflows(_swap("ACT/365"), EVAL_DATE)
        assert np.array_equal(implicit.accrual_fractions, explicit.accrual_fractions)

    def test_swap_config_defaults_to_act365(self):
        assert SwapConfig(
            notional=1e6, fixed_rate=0.04, payer=True,
            discount_curve_index=0, forward_curve_index=0,
            evaluation_date=EVAL_DATE,
        ).accrual_day_count == "ACT/365"


class TestOnlyTheAccrualRoleIsConfigurable:
    """**The point of the whole task.** The time axis must stay ACT/365
    while the accrual becomes per-instrument.

    A change that made the time axis configurable too would pass every
    other test in this file and silently desynchronize every pricer from
    the simulated curve cube.
    """

    def test_time_axis_is_act365(self):
        assert TIME_AXIS_DAY_COUNTER.name() == ORE.Actual365Fixed().name()

    def test_time_axis_is_not_a_parameter(self):
        """`build_vanilla_swap` exposes `accrual_day_count` and nothing
        that would let a caller move the time axis."""
        import inspect
        params = set(inspect.signature(build_vanilla_swap).parameters)
        assert "accrual_day_count" in params
        assert not {p for p in params if "time_axis" in p or "time_grid" in p}

    def test_changing_accrual_does_not_move_the_time_axis(self):
        """Cashflow TIMES are the time axis; accrual FRACTIONS are the
        contract. Switching the accrual basis must move the second and
        leave the first exactly where it was -- otherwise every
        `_maturity_indices` lookup against the simulated cube shifts."""
        act365 = fixed_leg_cashflows(_swap("ACT/365"), EVAL_DATE)
        actact = fixed_leg_cashflows(_swap("ACT/ACT (ICMA)"), EVAL_DATE)

        assert np.array_equal(act365.payment_times, actact.payment_times)
        assert np.array_equal(act365.accrual_start_times, actact.accrual_start_times)
        assert np.array_equal(act365.accrual_end_times, actact.accrual_end_times)
        assert not np.allclose(act365.accrual_fractions, actact.accrual_fractions)

    def test_deprecated_alias_still_points_at_the_time_axis(self):
        """`DAY_COUNTER` always meant the time axis. Kept as an alias so no
        import breaks, and it must not have silently become the accrual."""
        assert DAY_COUNTER is TIME_AXIS_DAY_COUNTER


class TestActActIcmaIsSupported:
    """Plan §W1.1 step 4: 'Only then add ACT/ACT (ICMA) as a supported
    value.' Required by the TraderX note, whose terms name
    `dayCount: "ACT/ACT (ICMA)"`."""

    def test_act_act_icma_resolves(self):
        assert "ACT/ACT (ICMA)" in SUPPORTED_ACCRUAL_DAY_COUNTS

    def test_resolves_to_ores_isma_convention(self):
        """ORE's ISMA *is* ACT/ACT (ICMA) — the bond-market variant that
        accrues against the actual coupon period, not the calendar year."""
        resolved = resolve_accrual_day_count("ACT/ACT (ICMA)")
        assert resolved.name() == ORE.ActualActual(ORE.ActualActual.ISMA).name()

    def test_it_genuinely_changes_the_accrual(self):
        """Not just accepted — it must produce different numbers, or the
        wiring is decorative."""
        act365 = fixed_leg_cashflows(_swap("ACT/365"), EVAL_DATE).accrual_fractions
        actact = fixed_leg_cashflows(_swap("ACT/ACT (ICMA)"), EVAL_DATE).accrual_fractions
        assert not np.allclose(act365, actact)

    def test_act_act_icma_gives_exact_unit_periods(self):
        """ACT/ACT (ICMA) measures a coupon period against itself, so a
        regular annual period is exactly 1.0. ACT/365 is not — it reads
        366/365 across a leap day, which is the ~0.55% the two conventions
        disagree by."""
        actact = fixed_leg_cashflows(_swap("ACT/ACT (ICMA)"), EVAL_DATE).accrual_fractions
        assert np.allclose(actact, 1.0)

        act365 = fixed_leg_cashflows(_swap("ACT/365"), EVAL_DATE).accrual_fractions
        assert not np.allclose(act365, 1.0)


class TestUnsupportedDayCountIsRefused:
    """Plan §W1.1: 'a trade requesting an unsupported day count is
    **refused**, not silently defaulted.'

    Same refuse-don't-infer rule as `engine.integration.conventions` (I-05),
    one layer down: a day count quietly replaced by ACT/365 produces a
    confident, wrong accrual.
    """

    @pytest.mark.parametrize("name", ("ACT/360", "30/360", "ACT/ACT", "", "Actual365Fixed"))
    def test_unsupported_names_raise(self, name):
        with pytest.raises(UnsupportedDayCountError, match="not supported"):
            resolve_accrual_day_count(name)

    def test_act360_is_refused_specifically(self):
        """ACT/360 is the USD-SOFR convention. It is refused here for the
        same reason W0.4 refuses the booking: this engine does not
        implement it, and substituting ACT/365 shifts every accrual by
        1.389%."""
        with pytest.raises(UnsupportedDayCountError):
            resolve_accrual_day_count("ACT/360")

    def test_the_error_names_what_is_supported(self):
        with pytest.raises(UnsupportedDayCountError) as exc:
            resolve_accrual_day_count("ACT/360")
        assert "ACT/365" in str(exc.value)
        assert "refused rather than" in str(exc.value)

    def test_builder_refuses_at_build_time(self):
        with pytest.raises(UnsupportedDayCountError):
            _swap("ACT/360")

    def test_swap_config_refuses_at_construction(self):
        """Fails where the offending trade is identifiable, not deep inside
        ORE at pricing time."""
        with pytest.raises(UnsupportedDayCountError):
            SwapConfig(
                notional=1e6, fixed_rate=0.04, payer=True,
                discount_curve_index=0, forward_curve_index=0,
                evaluation_date=EVAL_DATE, accrual_day_count="ACT/360",
            )

    def test_an_ore_daycounter_passes_through(self):
        """An explicit `ORE.DayCounter` is accepted: the allowlist governs
        the string form, which is what arrives over the wire. A caller
        constructing one in Python has taken responsibility for it."""
        resolved = resolve_accrual_day_count(ORE.Thirty360(ORE.Thirty360.BondBasis))
        assert resolved.name() == ORE.Thirty360(ORE.Thirty360.BondBasis).name()

    def test_none_resolves_to_the_default(self):
        assert resolve_accrual_day_count(None).name() == ORE.Actual365Fixed().name()


class TestSwapConfigThreadsItThrough:
    def test_config_reaches_the_builder(self):
        from engine.instruments.swap import _build_ore_swap

        cfg = SwapConfig(
            notional=1e6, fixed_rate=0.04, payer=True,
            discount_curve_index=0, forward_curve_index=0,
            swap_tenor="5Y", evaluation_date=EVAL_DATE,
            accrual_day_count="ACT/ACT (ICMA)",
        )
        built = fixed_leg_cashflows(_build_ore_swap(cfg), EVAL_DATE)
        expected = fixed_leg_cashflows(_swap("ACT/ACT (ICMA)"), EVAL_DATE)

        assert np.array_equal(built.accrual_fractions, expected.accrual_fractions)

    def test_default_config_still_builds_act365(self):
        """The regression that matters: an unmodified `SwapConfig` must
        build exactly what it always did."""
        from engine.instruments.swap import _build_ore_swap

        cfg = SwapConfig(
            notional=1e6, fixed_rate=0.04, payer=True,
            discount_curve_index=0, forward_curve_index=0,
            swap_tenor="5Y", evaluation_date=EVAL_DATE,
        )
        built = fixed_leg_cashflows(_build_ore_swap(cfg), EVAL_DATE)
        expected = fixed_leg_cashflows(_swap(), EVAL_DATE)

        assert np.array_equal(built.accrual_fractions, expected.accrual_fractions)


class TestTimeAxisConstantsAgree:
    """All three `TIME_AXIS_DAY_COUNTER` definitions must stay identical --
    they index the same simulated cube. Three separate constants exist for
    import-cycle reasons, not because they may differ."""

    def test_all_three_are_act365(self):
        from engine.instruments import bermudan_swaption
        from engine.risk import greeks
        from engine.models import ore_builders

        expected = ORE.Actual365Fixed().name()
        assert ore_builders.TIME_AXIS_DAY_COUNTER.name() == expected
        assert bermudan_swaption.TIME_AXIS_DAY_COUNTER.name() == expected
        assert greeks.TIME_AXIS_DAY_COUNTER.name() == expected

    def test_deprecated_aliases_all_still_resolve(self):
        from engine.instruments import bermudan_swaption
        from engine.risk import greeks
        from engine.models import ore_builders

        assert bermudan_swaption.DAY_COUNTER is bermudan_swaption.TIME_AXIS_DAY_COUNTER
        assert greeks.DAY_COUNTER is greeks.TIME_AXIS_DAY_COUNTER
        assert ore_builders.DAY_COUNTER is ore_builders.TIME_AXIS_DAY_COUNTER

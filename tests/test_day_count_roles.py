"""
Day count as a per-instrument input. `Actual365Fixed` plays two roles:

| Role | Must it stay ACT/365? |
|---|---|
| Simulation time axis: dates -> year fractions indexing the curve cube | Yes; changing it would silently misalign every pricer. |
| Instrument accrual: the contract's coupon day count | No; per instrument (ACT/ACT (ICMA) for the TraderX note). |

`TestOnlyTheAccrualRoleIsConfigurable` pins that the time axis is untouched while accrual is
configurable. Also: defaults unchanged, ACT/ACT (ICMA) supported, unsupported day counts
refused rather than defaulted.
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
    resolve_accrual_day_count,
    resolve_swap_dates,
)
from engine.instruments.swap import SwapConfig

EVAL_DATE = ORE.DateParser.parseISO("2025-06-02")


def _swap(accrual=None, tenor="5Y"):
    kwargs = {} if accrual is None else {"accrual_day_count": accrual}
    effective_date, maturity_date = resolve_swap_dates(EVAL_DATE, tenor)
    return build_vanilla_swap(
        notional=1_000_000.0, fixed_rate=0.04, payer=True, effective_date=effective_date,
        maturity_date=maturity_date, index_tenor_months=6, floating_spread=0.0,
        **kwargs,
    )


class TestDefaultsAreByteIdentical:
    """Defaulting to ACT/365 leaves existing results unchanged (many tests pin
    `ORE.Actual365Fixed()` directly)."""

    def test_default_equals_explicit_act365(self):
        implicit = fixed_leg_cashflows(_swap(), EVAL_DATE)
        explicit = fixed_leg_cashflows(_swap("ACT/365"), EVAL_DATE)

        assert np.array_equal(implicit.accrual_fractions, explicit.accrual_fractions)
        assert np.array_equal(implicit.payment_times, explicit.payment_times)
        assert np.array_equal(implicit.accrual_start_times, explicit.accrual_start_times)

    def test_default_constant_is_act365(self):
        assert DEFAULT_ACCRUAL_DAY_COUNT == "ACT/365"

    def test_floating_leg_default_is_unchanged_too(self):
        accruals = lambda swap: [ORE.as_floating_rate_coupon(c).accrualPeriod() for c in swap.floatingLeg()]  # noqa: E731
        assert accruals(_swap()) == accruals(_swap("ACT/365"))

    def test_swap_config_defaults_to_act365(self):
        assert SwapConfig(trade_id="swap-L66", 
            notional=1e6, fixed_rate=0.04, payer=True,
            swap_tenor="5Y", evaluation_date=EVAL_DATE,
        ).accrual_day_count == "ACT/365"


class TestOnlyTheAccrualRoleIsConfigurable:
    """The time axis stays ACT/365 while accrual is per instrument. Making the time axis
    configurable too would pass every other test here and misalign every pricer."""

    def test_time_axis_is_act365(self):
        assert TIME_AXIS_DAY_COUNTER.name() == ORE.Actual365Fixed().name()

    def test_time_axis_is_not_a_parameter(self):
        """`build_vanilla_swap` exposes `accrual_day_count` and nothing that moves the time
        axis."""
        import inspect
        params = set(inspect.signature(build_vanilla_swap).parameters)
        assert "accrual_day_count" in params
        assert not {p for p in params if "time_axis" in p or "time_grid" in p}

    def test_changing_accrual_does_not_move_the_time_axis(self):
        """Changing the accrual basis changes accrual fractions but not cashflow times (the
        time axis), so `_maturity_indices` lookups do not shift."""
        act365 = fixed_leg_cashflows(_swap("ACT/365"), EVAL_DATE)
        actact = fixed_leg_cashflows(_swap("ACT/ACT (ICMA)"), EVAL_DATE)

        assert np.array_equal(act365.payment_times, actact.payment_times)
        assert np.array_equal(act365.accrual_start_times, actact.accrual_start_times)
        assert np.array_equal(act365.accrual_end_times, actact.accrual_end_times)
        assert not np.allclose(act365.accrual_fractions, actact.accrual_fractions)

    def test_deprecated_alias_still_points_at_the_time_axis(self):
        """The `DAY_COUNTER` alias still means the time axis."""
        assert DAY_COUNTER is TIME_AXIS_DAY_COUNTER


class TestActActIcmaIsSupported:
    """ACT/ACT (ICMA), required by the TraderX note's terms."""

    def test_act_act_icma_resolves(self):
        assert "ACT/ACT (ICMA)" in SUPPORTED_ACCRUAL_DAY_COUNTS

    def test_resolves_to_ores_isma_convention(self):
        """ORE's ISMA variant is ACT/ACT (ICMA), accruing against the coupon period."""
        resolved = resolve_accrual_day_count("ACT/ACT (ICMA)")
        assert resolved.name() == ORE.ActualActual(ORE.ActualActual.ISMA).name()

    def test_it_genuinely_changes_the_accrual(self):
        """It changes the accrual fractions."""
        act365 = fixed_leg_cashflows(_swap("ACT/365"), EVAL_DATE).accrual_fractions
        actact = fixed_leg_cashflows(_swap("ACT/ACT (ICMA)"), EVAL_DATE).accrual_fractions
        assert not np.allclose(act365, actact)

    def test_act_act_icma_gives_exact_unit_periods(self):
        """ICMA measures a regular annual period as exactly 1.0; ACT/365 does not (actual
        days over 365)."""
        actact = fixed_leg_cashflows(_swap("ACT/ACT (ICMA)"), EVAL_DATE).accrual_fractions
        assert np.allclose(actact, 1.0)

        act365 = fixed_leg_cashflows(_swap("ACT/365"), EVAL_DATE).accrual_fractions
        assert not np.allclose(act365, 1.0)


class TestUnsupportedDayCountIsRefused:
    """An unsupported day count is refused, not replaced by ACT/365 (which would give a
    wrong accrual)."""

    @pytest.mark.parametrize("name", ("ACT/360", "30/360", "ACT/ACT", "", "Actual365Fixed"))
    def test_unsupported_names_raise(self, name):
        with pytest.raises(UnsupportedDayCountError, match="not supported"):
            resolve_accrual_day_count(name)

    def test_act360_is_refused_specifically(self):
        """ACT/360 (USD-SOFR's convention) is refused: it is not implemented, and ACT/365
        instead would move every accrual by 1.389%."""
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
        """Refused at construction, where the trade is identifiable."""
        with pytest.raises(UnsupportedDayCountError):
            SwapConfig(trade_id="swap-L159", 
                notional=1e6, fixed_rate=0.04, payer=True,
                    swap_tenor="5Y", evaluation_date=EVAL_DATE, accrual_day_count="ACT/360",
            )

    def test_an_ore_daycounter_passes_through(self):
        """An explicit `ORE.DayCounter` passes through (the allowlist governs names, which
        arrive over the wire)."""
        resolved = resolve_accrual_day_count(ORE.Thirty360(ORE.Thirty360.BondBasis))
        assert resolved.name() == ORE.Thirty360(ORE.Thirty360.BondBasis).name()

    def test_none_resolves_to_the_default(self):
        assert resolve_accrual_day_count(None).name() == ORE.Actual365Fixed().name()


class TestSwapConfigThreadsItThrough:
    def test_config_reaches_the_builder(self):
        from engine.instruments.swap import _build_ore_swap

        cfg = SwapConfig(trade_id="swap-L179", 
            notional=1e6, fixed_rate=0.04, payer=True,
            swap_tenor="5Y", evaluation_date=EVAL_DATE,
            accrual_day_count="ACT/ACT (ICMA)",
        )
        built = fixed_leg_cashflows(_build_ore_swap(cfg), EVAL_DATE)
        expected = fixed_leg_cashflows(_swap("ACT/ACT (ICMA)"), EVAL_DATE)

        assert np.array_equal(built.accrual_fractions, expected.accrual_fractions)

    def test_default_config_still_builds_act365(self):
        """An unmodified `SwapConfig` builds exactly as before."""
        from engine.instruments.swap import _build_ore_swap

        cfg = SwapConfig(trade_id="swap-L194", 
            notional=1e6, fixed_rate=0.04, payer=True,
            swap_tenor="5Y", evaluation_date=EVAL_DATE,
        )
        built = fixed_leg_cashflows(_build_ore_swap(cfg), EVAL_DATE)
        expected = fixed_leg_cashflows(_swap(), EVAL_DATE)

        assert np.array_equal(built.accrual_fractions, expected.accrual_fractions)


class TestTimeAxisConstantsAgree:
    """The time-axis references in `bermudan_swaption`, `valuation.legs` and `ore_builders` all
    resolve to ACT/365. (Equality alone cannot catch divergence between modules;
    `TestTimeAxisIsOneObject` asserts identity.)"""

    def test_all_three_are_act365(self):
        from engine.instruments import bermudan_swaption
        from engine.valuation import legs
        from engine.models import ore_builders

        expected = ORE.Actual365Fixed().name()
        assert ore_builders.TIME_AXIS_DAY_COUNTER.name() == expected
        assert bermudan_swaption.TIME_AXIS_DAY_COUNTER.name() == expected
        assert legs.TIME_AXIS_DAY_COUNTER.name() == expected

    def test_deprecated_aliases_all_still_resolve(self):
        from engine.instruments import bermudan_swaption
        from engine.models import ore_builders

        assert bermudan_swaption.DAY_COUNTER is bermudan_swaption.TIME_AXIS_DAY_COUNTER
        assert ore_builders.DAY_COUNTER is ore_builders.TIME_AXIS_DAY_COUNTER


class TestTimeAxisIsOneObject:
    """The time axis is one object engine-wide. Identity, not equality: separately
    constructed ACT/365 objects compare equal by name, so only identity shows there is a
    single definition. A local re-construction in any module fails here."""

    def test_every_module_exposes_the_canonical_object(self):
        from engine.instruments import bermudan_swaption
        from engine.valuation import legs
        from engine.models import ore_builders

        canonical = ore_builders.TIME_AXIS_DAY_COUNTER
        assert bermudan_swaption.TIME_AXIS_DAY_COUNTER is canonical
        assert legs.TIME_AXIS_DAY_COUNTER is canonical
        assert bermudan_swaption.DAY_COUNTER is canonical

    def test_no_module_constructs_its_own_time_axis_day_counter(self):
        """No module other than `ore_builders` constructs its own time-axis day counter
        (read from the source, catching the line that would reintroduce one)."""
        import ast
        from pathlib import Path

        engine_root = Path(__file__).parents[1] / "engine"
        offenders = []
        for source in sorted(engine_root.rglob("*.py")):
            # ore_builders is the one place allowed to construct it.
            if source.name == "ore_builders.py":
                continue
            for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Assign):
                    continue
                targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if "TIME_AXIS_DAY_COUNTER" not in targets and "DAY_COUNTER" not in targets:
                    continue
                # Only a construction is an offence; `DAY_COUNTER = TIME_AXIS_DAY_COUNTER` is
                # an alias of the imported object.
                if isinstance(node.value, ast.Call):
                    offenders.append(f"{source.relative_to(engine_root)}:{node.lineno}")

        assert offenders == [], (
            "the simulation time axis must be imported from "
            "engine.models.ore_builders, never re-constructed -- three "
            "equal-but-distinct copies can drift apart silently: "
            + "; ".join(offenders)
        )

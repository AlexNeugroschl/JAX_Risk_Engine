"""
Tests for engine.portfolio's Phase 1 validation/assembly layer:
validate_portfolio_against_simulation (cross-field consistency between
RatesConfig and each trade's duplicated hw_a/hw_sigma/initial_zero_curve)
and derive_maturity_pillars (automatic maturity-pillar assembly for an
arbitrary multi-trade portfolio) -- see
docs/planning/traderx-integration.md gap items 2, 3, and 5.
"""
import warnings

import numpy as np
import ORE
import pytest

from engine.simulation.market_model import EquityConfig, RatesConfig, SimulationConfig, ZeroCurveConfig
from engine.instruments.swap import SwapConfig, _maturity_indices
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.portfolio import derive_maturity_pillars, validate_portfolio_against_simulation

TODAY = ORE.Date(30, 7, 2026)
FLAT_RATE = 0.03
HW_A = 0.03
HW_SIGMA = 0.01
ZERO_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)


def _sim_config(**overrides) -> SimulationConfig:
    defaults = dict(
        time_grid=[0.0, 0.5, 1.0, 1.5, 2.0],
        scenarios=16,
        equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0]]),
        rates=RatesConfig(
            initial_rates=[FLAT_RATE], theta=[FLAT_RATE], mean_reversion=[HW_A],
            maturities=[0.0, 1.0, 2.0, 5.0], initial_zero_curves=[ZERO_CURVE],
        ),
        joint_covariance=[[0.04, 0.0], [0.0, HW_SIGMA ** 2]],
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)


def _swaption_cfg(**overrides) -> SwaptionConfig:
    defaults = dict(
        notional=1_000_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
        hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
        swap_tenor="3Y", evaluation_date=TODAY,
    )
    defaults.update(overrides)
    return SwaptionConfig(**defaults)


class TestCrossFieldValidation:
    """validate_portfolio_against_simulation (docs/planning/
    traderx-integration.md gap item 2)."""

    def test_matching_config_passes(self):
        sim = _sim_config()
        validate_portfolio_against_simulation(sim, [_swaption_cfg()])  # must not raise

    def test_mismatched_hw_a_raises(self):
        sim = _sim_config()
        cfg = _swaption_cfg(hw_a=0.05)
        with pytest.raises(ValueError, match="hw_a"):
            validate_portfolio_against_simulation(sim, [cfg])

    def test_mismatched_hw_sigma_raises(self):
        sim = _sim_config()
        cfg = _swaption_cfg(hw_sigma=0.02)
        with pytest.raises(ValueError, match="hw_sigma"):
            validate_portfolio_against_simulation(sim, [cfg])

    def test_mismatched_zero_curve_times_raises(self):
        sim = _sim_config()
        bad_curve = ZeroCurveConfig(times=[0.0, 1.0, 3.0], rates=[FLAT_RATE] * 3)
        cfg = _swaption_cfg(initial_zero_curve=bad_curve)
        with pytest.raises(ValueError, match="initial_zero_curve"):
            validate_portfolio_against_simulation(sim, [cfg])

    def test_mismatched_zero_curve_rates_raises(self):
        sim = _sim_config()
        bad_curve = ZeroCurveConfig(times=ZERO_CURVE.times, rates=[0.05] * 6)
        cfg = _swaption_cfg(initial_zero_curve=bad_curve)
        with pytest.raises(ValueError, match="initial_zero_curve"):
            validate_portfolio_against_simulation(sim, [cfg])

    def test_out_of_range_rate_factor_index_raises(self):
        sim = _sim_config()
        cfg = _swaption_cfg(rate_factor_index=5)
        with pytest.raises(ValueError, match="rate_factor_index"):
            validate_portfolio_against_simulation(sim, [cfg])

    def test_swap_config_has_no_cross_field_requirement(self):
        """SwapConfig carries no rate_factor_index/hw_a/hw_sigma -- it must
        pass through untouched (no AttributeError, no spurious raise)."""
        sim = _sim_config()
        swap_cfg = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            discount_curve_index=0, forward_curve_index=0,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        validate_portfolio_against_simulation(sim, [swap_cfg])  # must not raise

    def _two_factor_sim(self):
        curve_a = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6)
        curve_b = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.025] * 6)
        return _sim_config(
            equities=EquityConfig(initial_prices=[100.0], dividend_yields=[0.0], rate_mapping=[[0.0, 0.0]]),
            rates=RatesConfig(
                initial_rates=[0.03, 0.025], theta=[0.03, 0.025], mean_reversion=[0.03, 0.028],
                maturities=[0.0, 1.0, 2.0, 5.0], initial_zero_curves=[curve_a, curve_b],
            ),
            joint_covariance=[[0.04, 0.0, 0.0], [0.0, 0.03 ** 2, 0.0], [0.0, 0.0, 0.027 ** 2]],
        ), curve_a, curve_b

    def test_second_factor_curve_correctly_cross_checked_not_just_factor_zero(self):
        """A trade on rate_factor_index=1 whose own curve is actually
        factor 0's curve must still be rejected -- confirms the validator
        indexes into the RIGHT factor's own curve/mean_reversion/vol at
        factor counts > 1, not always factor 0 by coincidence (every other
        test in this class uses exactly one factor, where this distinction
        is unobservable)."""
        sim, curve_a, curve_b = self._two_factor_sim()
        wrong_curve_for_factor_1 = _swaption_cfg(
            rate_factor_index=1, hw_a=0.028, hw_sigma=0.027, initial_zero_curve=curve_a,
        )
        with pytest.raises(ValueError, match="initial_zero_curve"):
            validate_portfolio_against_simulation(sim, [wrong_curve_for_factor_1])

    def test_second_factor_correctly_matched_config_passes(self):
        sim, curve_a, curve_b = self._two_factor_sim()
        correct = _swaption_cfg(rate_factor_index=1, hw_a=0.028, hw_sigma=0.027, initial_zero_curve=curve_b)
        validate_portfolio_against_simulation(sim, [correct])  # must not raise

    def test_second_factor_mismatched_hw_a_pinpoints_the_right_factor_in_the_message(self):
        sim, curve_a, curve_b = self._two_factor_sim()
        bad = _swaption_cfg(rate_factor_index=1, hw_a=0.099, hw_sigma=0.027, initial_zero_curve=curve_b)
        with pytest.raises(ValueError, match=r"mean_reversion\[1\]"):
            validate_portfolio_against_simulation(sim, [bad])

    def test_mixed_portfolio_pinpoints_the_bad_trade(self):
        sim = _sim_config()
        good = _swaption_cfg()
        bad = _swaption_cfg(hw_a=0.099)
        with pytest.raises(ValueError, match=r"trade\[1\]"):
            validate_portfolio_against_simulation(sim, [good, bad])

    @pytest.mark.parametrize("exercise_date", [ORE.Date(3, 8, 2027), ORE.Date(30, 10, 2027)],
                             ids=["on-an-accrual-start", "mid-period"])
    def test_bermudan_exercise_never_warns(self, exercise_date):
        """A Bermudan exercise date inside an accrual period is priced
        exactly as ORE prices it (into the next whole period), not
        approximated, so there is nothing to warn about -- unlike the
        mid-coupon warning this used to raise before I-06 was closed."""
        sim = _sim_config()
        cfg = BermudanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
            hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
            exercise_dates=[exercise_date], swap_tenor="3Y", evaluation_date=TODAY,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_portfolio_against_simulation(sim, [cfg])  # must not warn/raise

    def test_american_exercise_window_never_warns(self):
        """Nor does an American window, whose broken-period exercise is ORE's
        own `couponRatio` proration."""
        sim = _sim_config()
        cfg = AmericanSwaptionConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True, rate_factor_index=0,
            hw_a=HW_A, hw_sigma=HW_SIGMA, initial_zero_curve=ZERO_CURVE,
            first_exercise_date=TODAY + 365, last_exercise_date=TODAY + 730, exercise_time_steps_per_year=3,
            swap_tenor="3Y", evaluation_date=TODAY,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            validate_portfolio_against_simulation(sim, [cfg])  # must not warn/raise


class TestPillarAssembly:
    """derive_maturity_pillars (docs/planning/traderx-integration.md gap
    item 3)."""

    def test_single_trade_pillars_are_subset_and_accepted_by_maturity_indices(self):
        cfg = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            discount_curve_index=0, forward_curve_index=0,
            swap_tenor="2Y", evaluation_date=TODAY,
        )
        pillars = derive_maturity_pillars([cfg], TODAY)
        assert 0.0 in pillars
        assert pillars == sorted(pillars)

        from engine.instruments.swap import prepare_swap
        prepared = prepare_swap(cfg, np.asarray(pillars))
        # If prepare_swap succeeded without raising, every one of this
        # trade's own cashflow times was accepted as a pillar match --
        # _maturity_indices raises ValueError otherwise.
        assert prepared.fixed_pay_idx.shape[0] > 0

    def test_two_trade_portfolio_pillar_set_strictly_exceeds_either_alone(self):
        """swap_tenor/index_tenor_months chosen so neither trade's own
        pillar set is a superset of the other's (confirmed directly against
        each trade's real ORE-generated schedule) -- a 2Y/6M-index swap and
        an 18M/4M-index swap land on genuinely different reset dates, not
        just a shorter tenor nested inside a longer one at the same
        frequency."""
        cfg_a = SwapConfig(
            notional=1_000_000.0, fixed_rate=0.03, payer=True,
            discount_curve_index=0, forward_curve_index=0,
            swap_tenor="2Y", index_tenor_months=6, evaluation_date=TODAY,
        )
        cfg_b = SwapConfig(
            notional=500_000.0, fixed_rate=0.028, payer=False,
            discount_curve_index=0, forward_curve_index=0,
            swap_tenor="18M", index_tenor_months=4, evaluation_date=TODAY,
        )
        pillars_a = set(derive_maturity_pillars([cfg_a], TODAY))
        pillars_b = set(derive_maturity_pillars([cfg_b], TODAY))
        pillars_both = set(derive_maturity_pillars([cfg_a, cfg_b], TODAY))

        assert not pillars_a > pillars_b and not pillars_b > pillars_a  # genuinely distinct schedules
        assert pillars_both > pillars_a
        assert pillars_both > pillars_b
        assert pillars_both >= pillars_a | pillars_b

    def test_multi_trade_multi_tenor_portfolio_every_trades_dates_accepted(self):
        swap_cfgs = [
            SwapConfig(notional=1_000_000.0, fixed_rate=0.030, payer=True,
                       discount_curve_index=0, forward_curve_index=0,
                       swap_tenor="2Y", evaluation_date=TODAY),
            SwapConfig(notional=2_000_000.0, fixed_rate=0.028, payer=False,
                       discount_curve_index=0, forward_curve_index=0,
                       swap_tenor="3Y", evaluation_date=TODAY),
            SwapConfig(notional=1_500_000.0, fixed_rate=0.032, payer=True,
                       discount_curve_index=0, forward_curve_index=0,
                       swap_tenor="7Y", index_tenor_months=3, evaluation_date=TODAY),
        ]
        pillars = derive_maturity_pillars(swap_cfgs, TODAY)
        pillars_np = np.asarray(pillars)

        from engine.instruments.swap import prepare_swap
        for cfg in swap_cfgs:
            prepare_swap(cfg, pillars_np)  # must not raise ValueError

    def test_swaption_only_portfolio_produces_only_t0_pillar(self):
        """Swaption-family pricers price off simulated hw_paths directly,
        not the yield_curves/maturity-pillar cube -- a portfolio with no
        SwapConfig trades contributes no pillars beyond the anchor t=0."""
        cfg = _swaption_cfg()
        pillars = derive_maturity_pillars([cfg], TODAY)
        assert pillars == [0.0]

    def test_empty_portfolio_returns_just_t0(self):
        assert derive_maturity_pillars([], TODAY) == [0.0]

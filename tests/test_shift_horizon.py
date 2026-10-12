"""
ORE's `ShiftHorizon` on the Bermudan/American grid engine (I-32), in its parts: the shifted
LGM (`engine.models.lgm`), the builder's horizon (`engine.pricing.bermudan.shift_horizon_time`),
the trade as built on the as-of date and valued later (the shift and an American's window), and
the setting. Prices against ORE: `tests/test_ore_lgm_parity.py` (any horizon, today),
`tests/test_ore_lgm_calibration.py` (calibrated), `tests/test_ore_xva_parity.py` (on ORE's
simulated paths).
"""
import dataclasses
import math

import jax
import numpy as np
import ORE
import pytest

from engine.api.requests import LgmEngineSchema
from engine.instruments.american_swaption import AmericanSwaptionConfig
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig, underlying_swap
from engine.market_data.curves import ZeroCurve
from engine.market_data.day_counts import time_from_reference
from engine.market_data.market import ZeroCurveConfig
from engine.models.lgm import H, bond_price, horizon_shift, numeraire, r_from_x, x_from_r, zeta
from engine.pricing.bermudan import prepared_option, shift_horizon_time
from engine.pricing.config import LgmSwaptionEngineConfig
from engine.pricing.lgm_grid import prepare_bermudan
from tests.support.compiles import count_compiles
from tests.support.lgm_engine import grid_npv

jax.config.update("jax_enable_x64", True)

ASOF = ORE.Date(30, 7, 2026)
A = 0.03
SIGMA = 0.012
CURVE = ZeroCurve.from_config(ZeroCurveConfig([0.0, 1.0, 5.0, 10.0], [0.02, 0.025, 0.035, 0.04]))
ACT_ACT_ISDA = ORE.ActualActual(ORE.ActualActual.ISDA)


def _bermudan(**overrides) -> BermudanSwaptionConfig:
    fields = dict(notional=1e6, fixed_rate=0.03, payer=True, exercise_dates=[ASOF + 365, ASOF + 730],
                  swap_tenor="4Y", evaluation_date=ASOF, trade_id="bermudan")
    return BermudanSwaptionConfig(**{**fields, **overrides})


def _american(**overrides) -> AmericanSwaptionConfig:
    fields = dict(notional=1e6, fixed_rate=0.03, payer=True, first_exercise_date=ORE.Date(3, 2, 2027),
                  last_exercise_date=ORE.Date(3, 2, 2029), effective_date=ORE.Date(3, 2, 2027),
                  maturity_date=ORE.Date(3, 2, 2031), evaluation_date=ASOF, trade_id="american")
    return AmericanSwaptionConfig(**{**fields, **overrides})


class TestShiftedModel:
    """`Lgm1fParametrization::shift()`: H + shift wherever H is read, an invariance of the model."""

    def test_the_shift_adds_to_h_and_vanishes_at_its_horizon(self):
        for horizon in (0.7, 2.5, 12.0):
            shift = horizon_shift(A, horizon)
            assert float(H(A, horizon, shift)) == pytest.approx(0.0, abs=1e-16)
            assert float(H(A, 0.0, shift)) == float(shift)
            assert float(H(A, 3.0, shift)) == pytest.approx(float(H(A, 3.0)) + float(shift), abs=1e-16)
        assert float(horizon_shift(A, 0.0)) == 0.0
        assert float(horizon_shift(0.0, 2.0)) == -2.0  # H(t) = t at zero reversion

    @pytest.mark.parametrize("t, T, x", [(0.0, 3.0, 0.0), (1.5, 4.0, 0.013), (4.0, 9.0, -0.021)])
    def test_bond_prices_are_invariant_when_the_state_moves_with_the_shift(self, t, T, x):
        """Shifting H by C and moving the state to x - C zeta(t) leaves every bond price as it
        was: the model is the same, only its state variable is centred elsewhere."""
        shift = horizon_shift(A, 2.0)
        moved = x - shift * zeta(SIGMA, t)
        assert float(bond_price(CURVE, A, SIGMA, t, T, moved, shift)) == pytest.approx(
            float(bond_price(CURVE, A, SIGMA, t, T, x)), rel=1e-14)
        rate = r_from_x(CURVE, A, SIGMA, t, moved, shift)
        assert float(rate) == pytest.approx(float(r_from_x(CURVE, A, SIGMA, t, x)), rel=1e-12)
        if t > 0.0:
            assert float(x_from_r(CURVE, A, SIGMA, t, rate, shift)) == pytest.approx(float(moved), abs=1e-15)

    def test_the_numeraire_changes_measure_by_the_shift(self):
        """N_C(t, x - C zeta) / N(t, x) = exp(C (x - C zeta) + C^2 zeta / 2): the density of the
        measure change, 1 at t = 0 on the state 0, and exp(C x) at t = 0 elsewhere."""
        shift = float(horizon_shift(A, 2.0))
        for t, x in ((0.0, 0.0), (1.5, 0.013), (4.0, -0.021)):
            z = float(zeta(SIGMA, t))
            moved = x - shift * z
            ratio = float(numeraire(CURVE, A, SIGMA, t, moved, shift) / numeraire(CURVE, A, SIGMA, t, x))
            assert ratio == pytest.approx(math.exp(shift * moved + 0.5 * shift ** 2 * z), rel=1e-13)
        assert float(numeraire(CURVE, A, SIGMA, 0.0, 0.01, shift)) == pytest.approx(math.exp(shift * 0.01), rel=1e-15)


class TestBuilderHorizon:
    """`LGMBermudanAmericanSwaptionEngineBuilder::model`: the fraction of the ActualActual(ISDA)
    time from the build date to the underlying's maturity."""

    def test_the_horizon_is_the_fraction_of_the_time_to_the_last_payment(self):
        cfg = _bermudan()
        maturity = underlying_swap(cfg).maturityDate()
        assert shift_horizon_time(cfg, 0.5, ASOF) == 0.5 * ACT_ACT_ISDA.yearFraction(ASOF, maturity)
        assert shift_horizon_time(cfg, 0.0, ASOF) == 0.0

    def test_a_trade_built_on_the_as_of_date_keeps_its_horizon_on_a_later_date(self):
        """ORE shifts the parametrization once, when the trade is built (`initParametrization`);
        a simulation values that trade on later dates."""
        engine = LgmSwaptionEngineConfig(shift_horizon=0.5, calibration="None")
        later = ASOF + 200
        dated = dataclasses.replace(_bermudan(), evaluation_date=later)
        assert prepared_option(dated, engine, SIGMA, built_on=ASOF).shift_horizon == shift_horizon_time(
            dated, 0.5, ASOF)
        assert prepared_option(dated, engine, SIGMA).shift_horizon == shift_horizon_time(dated, 0.5, later)
        assert shift_horizon_time(dated, 0.5, ASOF) > shift_horizon_time(dated, 0.5, later)

    def test_an_american_built_earlier_can_be_exercised_on_the_date_it_is_valued(self):
        """ORE's trade builder opens the window no earlier than the day after its build date; the
        engine starts the grid no earlier than the valuation date. Built on the as-of date and
        valued inside the window, the first exercise is the valuation date itself."""
        inside = dataclasses.replace(_american(), evaluation_date=ORE.Date(10, 6, 2027))
        assert inside.option_times(24, built_on=ASOF)[0] == 0.0
        assert inside.option_times(24)[0] == time_from_reference(inside.evaluation_date, inside.evaluation_date + 1)
        before = _american()
        assert before.option_times(24, built_on=ASOF) == before.option_times(24)
        bermudan = dataclasses.replace(_bermudan(), evaluation_date=ASOF + 100)
        assert bermudan.option_times(24, built_on=ASOF) == bermudan.option_times(24)


class TestSetting:
    def test_the_default_is_ores_builder_default(self):
        """`ShiftHorizon` defaults to 0.5 in ORE's builder (OREData/ored/portfolio/builders/
        swaption.cpp), in the engine and in the API alike."""
        assert LgmSwaptionEngineConfig().shift_horizon == 0.5
        assert LgmEngineSchema().shift_horizon == 0.5
        assert LgmEngineSchema(shift_horizon=1.25).to_dataclass().shift_horizon == 1.25

    @pytest.mark.parametrize("value", [-0.1, math.nan, math.inf])
    def test_a_negative_or_non_finite_horizon_is_refused(self, value):
        with pytest.raises(ValueError, match="shift_horizon"):
            LgmSwaptionEngineConfig(shift_horizon=value)
        with pytest.raises(ValueError, match="shift_horizon"):
            prepare_bermudan(_bermudan(), reversion=A, sigma=SIGMA, shift_horizon=value, n_per_std=8, std_devs=3.0,
                             exercise_time_steps_per_year=24)

    def test_every_horizon_shares_one_program(self):
        """The horizon is traced data of the prepared trade, so a trade compiles its rollback
        once whatever its horizon, and the horizon moves the price (the grid's state)."""
        model = dict(a=A, sigma=SIGMA, curve=ZeroCurveConfig([0.0, 5.0], [0.03, 0.03]), n_per_std=10, std_devs=4.0)
        cfg = _bermudan()
        unshifted = grid_npv(cfg, shift_horizon=0.0, **model)
        with count_compiles() as counter:
            values = [grid_npv(cfg, shift_horizon=h, **model) for h in (0.0, 0.5, 1.5)]
        assert sum(counter.values()) == 0, dict(counter)
        assert values[0] == unshifted
        assert len(set(values)) == 3
        np.testing.assert_allclose(values, unshifted, rtol=1e-3)

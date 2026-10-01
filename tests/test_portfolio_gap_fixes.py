"""
Regression tests for gaps where `price_portfolio` returned a quietly incomplete result
(found while writing the EOD contract proposal; see docs/planning/details/traderx-integration.md).
Each class failed against the code it was written for; since roadmap 1.3 they run on the one
pipeline (a `Market`, any model).

1. Swap Greeks were skipped (I-01): the Greeks dispatcher could not resolve a swap's curves
   and skipped every swap without error.
2. Bermudan Vega was never called from the portfolio path (I-02).
3. There was no per-trade base NPV (I-03).
4. I-13: a trade's curves were validated only on the Greeks path (then curve indices, which
   wrapped when negative); now the curves a trade names are checked before any pricing, and
   the refusal names the trade.

The aged-swap inaccuracy (I-04) is fixed rather than warned about: tests/test_hull_white_model.py.
"""
import dataclasses

import numpy as np
import ORE
import pytest

from demos.demo_scenarios import EVAL_DATE, demo_market
from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.market import index_name
from engine.portfolio import (
    LgmSwaptionEngineConfig, PortfolioRequest, PricingConfig, RunConfig, price_portfolio,
)
from engine.risk.sensitivities import portfolio_sensitivities

FAST = LgmSwaptionEngineConfig(n_per_std=12, std_devs=4.0)
EUR_6M = index_name("EUR", 6)


def _swap(notional=2_000_000.0, fixed_rate=0.032, payer=True, tenor="2Y", currency="USD", trade_id="swap"):
    return SwapConfig(notional=notional, fixed_rate=fixed_rate, payer=payer, swap_tenor=tenor, currency=currency,
                      evaluation_date=EVAL_DATE, trade_id=trade_id)


def _request(trades, compute_greeks=False, market=None, pricing=PricingConfig()):
    return PortfolioRequest(market=market or demo_market(), trades=list(trades), compute_greeks=compute_greeks,
                            scenario_risk=False, config=RunConfig(pricing=pricing))


# =============================================================================
# Gap 1: swap Greeks were silently skipped (I-01)
# =============================================================================
class TestSwapGreeksReachThePortfolioPath:
    """`compute_greeks=True` returns Delta/Gamma/Theta for swaps (the pre-fix code skipped
    them and returned no swap entries, without error)."""

    KEYS = ("delta:discount:USD", "gamma:discount:USD", f"delta:index:{index_name('USD', 6)}",
            f"gamma:index:{index_name('USD', 6)}", "theta")

    def test_swap_greeks_are_present_not_skipped(self):
        result = price_portfolio(_request([_swap()], compute_greeks=True))
        assert result.greeks is not None
        assert 0 in result.greeks, "the swap has no Greeks entry -- the dispatcher skipped it (I-01)"
        for key in self.KEYS:
            assert key in result.greeks[0], f"missing {key!r} for the swap"

    def test_swap_greeks_are_finite_and_nonzero(self):
        g = price_portfolio(_request([_swap()], compute_greeks=True)).greeks[0]
        for key in self.KEYS[:4]:
            assert np.all(np.isfinite(g[key])), f"{key} contains non-finite values"
            assert np.any(np.abs(g[key]) > 0), f"{key} is identically zero"
        assert np.isfinite(float(g["theta"]))

    def test_portfolio_swap_greeks_equal_direct_sensitivities_call(self):
        """Greeks through `price_portfolio` equal a direct `portfolio_sensitivities` call."""
        trades = [_swap()]
        greeks = price_portfolio(_request(trades, compute_greeks=True)).greeks
        direct = portfolio_sensitivities(trades, demo_market(), "USD")
        assert greeks[0].keys() == direct[0].keys()
        for key in direct[0]:
            np.testing.assert_array_equal(greeks[0][key], direct[0][key])

    def test_uses_the_curves_its_currency_names(self):
        """A EUR swap is bumped on the EUR curves (in USD, the reporting currency), and its USD
        Deltas are absent; a dispatcher defaulting to the first currency would fail here."""
        g = price_portfolio(_request([_swap(currency="EUR")], compute_greeks=True)).greeks[0]
        assert "delta:discount:EUR" in g and f"delta:index:{EUR_6M}" in g
        assert not any(key.endswith(":USD") for key in g)

    def test_mixed_portfolio_keys_every_trade_by_its_own_index(self):
        """In a mixed portfolio every trade index gets its own entry; the swap is second, so
        appending rather than keying by index would misalign."""
        european = SwaptionConfig(notional=1_500_000.0, fixed_rate=0.031, payer=True, swap_tenor="1Y",
                                  forward_start=ORE.Period(1, ORE.Years), evaluation_date=EVAL_DATE,
                                  trade_id="european")
        result = price_portfolio(_request([european, _swap()], compute_greeks=True))
        assert set(result.greeks) == {0, 1}
        assert "vega:USD" in result.greeks[0], "index 0 is the swaption"
        assert "vega:USD" not in result.greeks[1], "index 1 is the swap"

    def test_greeks_absent_when_not_requested(self):
        assert price_portfolio(_request([_swap()])).greeks is None


# =============================================================================
# Gap 1b (I-13): a trade's curves were validated only on the Greeks path
# =============================================================================
class TestCurveIndexValidatedBeforeAllPricing:
    """The curves a trade reads are checked before any pricing, whether or not Greeks are
    asked for, and the refusal names the trade. (I-13: curve indices were validated only by
    the Greeks, so with `compute_greeks=False` a negative index wrapped to the last curve and
    priced against it. Trades now name a currency and index, not an index into a list.)"""

    @pytest.mark.parametrize("compute_greeks", [False, True])
    @pytest.mark.parametrize("market, missing", [
        (demo_market(("USD",)), "GBP"),
        (dataclasses.replace(demo_market(("USD",)), currencies={
            "USD": dataclasses.replace(demo_market(("USD",)).currency("USD"), index_curves={})}), "USD-SIMINDEX-6M"),
    ], ids=["currency", "index"])
    def test_a_missing_curve_is_refused_on_both_paths(self, market, missing, compute_greeks):
        trade = _swap(currency="GBP" if missing == "GBP" else "USD", trade_id="the-trade")
        with pytest.raises(KeyError, match=rf"the-trade.*SwapConfig.*{missing}"):
            price_portfolio(_request([trade], compute_greeks=compute_greeks, market=market))

    def test_valid_curves_are_unaffected(self):
        result = price_portfolio(_request([_swap(currency="EUR")]))
        assert np.isfinite(result.base_npv_per_trade[0]) and result.base_npv_per_trade[0] != 0.0


# =============================================================================
# Gap 2: Bermudan Vega was never called (I-02)
# =============================================================================
class TestBermudanVegaReachesThePortfolioPath:
    """The portfolio path reports a calibrated Bermudan's Vega on the volatility quotes it
    calibrates to."""

    @staticmethod
    def _bermudan():
        return BermudanSwaptionConfig(notional=1_000_000.0, fixed_rate=0.030, payer=True,
                                      exercise_dates=[ORE.Date(3, 8, 2027), ORE.Date(3, 8, 2028)], swap_tenor="3Y",
                                      evaluation_date=EVAL_DATE, trade_id="bermudan")

    @pytest.mark.slow
    def test_vega_present_for_a_calibrated_bermudan(self):
        market = demo_market(("USD",))
        g = price_portfolio(_request([self._bermudan()], compute_greeks=True, market=market,
                                     pricing=PricingConfig(bermudan=FAST))).greeks[0]
        assert "vega:USD" in g, "a calibrated Bermudan reported no Vega (I-02)"
        surface = market.swaption_vols("USD")
        assert g["vega:USD"].shape == (len(surface.option_tenors), len(surface.swap_tenors))
        assert np.all(np.isfinite(g["vega:USD"])) and np.any(np.abs(g["vega:USD"]) > 0)

    @pytest.mark.slow
    def test_no_vega_for_an_uncalibrated_bermudan(self):
        """With `calibration="None"` the model's fixed volatility has no market quote to be
        sensitive to: Vega is omitted, not fabricated; Delta/Gamma/Theta remain."""
        engine = dataclasses.replace(FAST, calibration="None")
        g = price_portfolio(_request([self._bermudan()], compute_greeks=True, market=demo_market(("USD",)),
                                     pricing=PricingConfig(bermudan=engine))).greeks[0]
        assert not any(key.startswith("vega") for key in g)
        assert "delta:discount:USD" in g and "theta" in g


# =============================================================================
# Gap 3: per-trade base NPV (I-03)
# =============================================================================
class TestPerTradeBaseNpv:
    """`base_npv_per_trade` lets the total be reconciled against its positions."""

    def _two(self):
        return [_swap(notional=2_000_000.0), _swap(notional=500_000.0, tenor="3Y", trade_id="swap-3y")]

    def test_per_trade_values_are_returned_one_per_trade(self):
        result = price_portfolio(_request(self._two()))
        assert len(result.base_npv_per_trade) == 2
        assert all(np.isfinite(v) for v in result.base_npv_per_trade)
        assert result.trade_ids == ["swap", "swap-3y"]

    def test_total_equals_sum_of_parts_exactly(self):
        result = price_portfolio(_request(self._two()))
        assert result.base_npv == pytest.approx(float(sum(result.base_npv_per_trade)), rel=0.0, abs=1e-9)

    def test_values_are_attributable_to_the_right_trade(self):
        """Order follows `request.trades`: a payer and an identical receiver have opposite
        signs, so misordering fails."""
        payer = _swap(fixed_rate=0.05, payer=True, trade_id="payer")
        receiver = _swap(fixed_rate=0.05, payer=False, trade_id="receiver")
        first, second = price_portfolio(_request([payer, receiver])).base_npv_per_trade
        assert first == pytest.approx(-second, rel=1e-12)
        assert first < 0 < second, "a 5% payer on a ~3.5% curve is out of the money"

    def test_single_trade_total_matches_its_only_row(self):
        result = price_portfolio(_request([_swap()]))
        assert result.base_npv == pytest.approx(result.base_npv_per_trade[0], rel=0.0, abs=1e-12)

    def test_empty_portfolio_has_empty_breakdown(self):
        result = price_portfolio(_request([]))
        assert result.base_npv_per_trade == [] and result.base_npv == 0.0

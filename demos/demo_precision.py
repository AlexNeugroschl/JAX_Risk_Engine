"""
How much precision does Monte Carlo market risk need? FP64 vs FP32 VaR/ES,
through the engine's own market-risk path.

**The question.** Lower precision is cheaper on accelerators. The decision is
not "is FP32 exact?" (it is not) but "is FP32's error small next to the Monte
Carlo sampling error the VaR/ES number already carries?" This demo measures
both on one portfolio.

**What runs.** `engine.market_risk.run_market_risk`, unmodified, at
`precision=64` and `precision=32`:

    10-day Monte Carlo shocks of every pillar of two sloped curves
        -> full revaluation of every trade at t=0, at the requested precision
        -> VaR 99% and ES 97.5% of the portfolio P&L

The same seed gives the same scenarios at both precisions, so the FP32-FP64
difference is pure arithmetic. Two yardsticks measure it:

  * the Monte Carlo standard error of the ES estimate (`ES_*_standardError`);
  * the spread of the FP64 estimate across independent Sobol seeds -- the
    empirical version of the same thing.

**What this replaces.** An earlier version of this demo rebuilt the
risk-neutral exposure simulation from private functions, on a flat curve,
and called its per-step loss quantiles VaR. It never exercised the engine's
own FP32 path, which then carried a finite-difference bug that put ~2.6e-3
relative error into every FP32 discount factor (docs/planning/
engine-audit.md, P-3). It also ran FP16; no pricer here has a float16 path,
so that comparison is not made.

ORE parity of this path is established in
tests/test_market_risk_ore_parity.py; this demo is only about precision.

Run with: .venv/Scripts/python.exe demos/demo_precision.py
"""
import time
import warnings

import numpy as np
import ORE

from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market_risk import MarketRiskRequest, RateRiskFactors, monte_carlo_scenarios, run_market_risk
from engine.simulation.market_model import ZeroCurveConfig

warnings.simplefilter("ignore")  # Sobol balance notices; the demo prints its own caveats

TODAY = ORE.Date(30, 7, 2026)
SCENARIOS = 8192
SEEDS = (1, 2, 3, 4, 5)
QUANTILES = (0.99, 0.975)


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# =============================================================================
# Market: a sloped OIS curve and an IBOR curve 40bp above it.
# =============================================================================
PILLAR_TIMES = [d / 365 for d in (0, 365, 730, 1095, 1825, 2555, 3650, 5475, 7300, 10950)]
OIS = ZeroCurveConfig(PILLAR_TIMES, [0.030, 0.031, 0.032, 0.033, 0.035, 0.037, 0.039, 0.041, 0.042, 0.043])
IBOR = ZeroCurveConfig(PILLAR_TIMES, [r + 0.004 for r in OIS.rates])
FACTORS = RateRiskFactors.from_curves([OIS, IBOR], names=["OIS", "IBOR"])

# 10-day covariance of absolute pillar moves: 8bp daily vol, correlation
# decaying with pillar distance, 0.95 between the two curves' pillars.
n = len(PILLAR_TIMES)
index = np.arange(2 * n)
pillar, curve = index % n, index // n
correlation = np.exp(-np.abs(pillar[:, None] - pillar[None, :]) / 4.0)
correlation *= np.where(curve[:, None] == curve[None, :], 1.0, 0.95)
np.fill_diagonal(correlation, 1.0)
COVARIANCE = correlation * 0.0008 ** 2 * 10

# =============================================================================
# Portfolio: offsetting swaps, options, and a bond -- netting matters, so the
# tail is a difference of large numbers, which is where precision bites.
# =============================================================================
TRADES = [
    SwapConfig(notional=50e6, fixed_rate=0.036, payer=True, discount_curve_index=0,
               forward_curve_index=1, swap_tenor="10Y", evaluation_date=TODAY),
    SwapConfig(notional=45e6, fixed_rate=0.035, payer=False, discount_curve_index=0,
               forward_curve_index=1, swap_tenor="9Y", evaluation_date=TODAY),
    SwaptionConfig(notional=20e6, fixed_rate=0.037, payer=False, rate_factor_index=0, hw_a=0.03,
                   hw_sigma=0.01, initial_zero_curve=OIS, swap_tenor="5Y",
                   forward_start=ORE.Period(2, ORE.Years), evaluation_date=TODAY),
    BermudanSwaptionConfig(notional=15e6, fixed_rate=0.035, payer=True, rate_factor_index=0, hw_a=0.03,
                           hw_sigma=0.01, initial_zero_curve=OIS,
                           exercise_dates=[TODAY + ORE.Period(y, ORE.Years) for y in (1, 2, 3, 4)],
                           swap_tenor="5Y", evaluation_date=TODAY, n_per_std=16, std_devs=5.0),
    BondConfig(face_amount=10e6, maturity_date=ORE.Date(30, 7, 2029), evaluation_date=TODAY,
               initial_zero_curve=OIS, coupon_rate=0.04,
               coupon_schedule=tuple(CouponPeriod(ORE.Date(30, 7, 2026 + k), ORE.Date(30, 7, 2027 + k))
                                     for k in range(3)),
               curve_index=0),
]


def run(precision: int, seed: int):
    scenarios = monte_carlo_scenarios(FACTORS, COVARIANCE, horizon_days=10, num_scenarios=SCENARIOS, seed=seed)
    started = time.time()
    result = run_market_risk(MarketRiskRequest(TRADES, scenarios, quantiles=QUANTILES, precision=precision))
    return result, time.time() - started


# =============================================================================
section("Running the market-risk path at each precision and seed")
# =============================================================================
results = {}
for seed in SEEDS:
    for precision in (64, 32):
        result, seconds = run(precision, seed)
        results[precision, seed] = result
        print(f"  seed {seed}  FP{precision}:  VaR 99% {result.risk['VaR_99']:>14,.2f}   "
              f"ES 97.5% {result.risk['ES_97.5']:>14,.2f}   ({seconds:.1f}s)")

base = results[64, SEEDS[0]].base_npv
print(f"\n  portfolio base NPV {base:,.2f}; {SCENARIOS:,} scenarios; 10-day horizon")

# =============================================================================
section("Precision error vs Monte Carlo noise")
# =============================================================================
rows = []
for metric in ("VaR_99", "ES_97.5"):
    fp64 = np.array([results[64, s].risk[metric] for s in SEEDS])
    fp32 = np.array([results[32, s].risk[metric] for s in SEEDS])
    precision_error = np.max(np.abs(fp32 - fp64))
    seed_spread = np.std(fp64, ddof=1)
    rows.append((metric, precision_error, seed_spread))
standard_error = np.mean([results[64, s].risk["ES_97.5_standardError"] for s in SEEDS])

print(f"  {'metric':<9}{'max |FP32-FP64|':>18}{'FP64 seed std':>16}{'ratio':>10}")
for metric, error, spread in rows:
    print(f"  {metric:<9}{error:>18,.4f}{spread:>16,.2f}{error / spread:>10.1e}")
print(f"\n  ES 97.5% Monte Carlo standard error (mean over seeds): {standard_error:,.2f}")

pnl64 = np.asarray(results[64, SEEDS[0]].portfolio_pnl, dtype=np.float64)
pnl32 = np.asarray(results[32, SEEDS[0]].portfolio_pnl, dtype=np.float64)
scale = np.max(np.abs(pnl64))
print(f"  per-scenario portfolio P&L, max |FP32-FP64| / max |P&L|: {np.max(np.abs(pnl32 - pnl64)) / scale:.1e}")

# =============================================================================
section("Reading the result")
# =============================================================================
worst_ratio = max(error / spread for _, error, spread in rows)
if worst_ratio < 0.01:
    verdict = "far inside the sampling noise the number already carries"
elif worst_ratio < 1.0:
    verdict = "inside the sampling noise, but not negligible next to it"
else:
    verdict = "as large as the sampling noise or larger -- FP32 is not safe here"
print(f"""  The largest FP32 error in a VaR/ES figure is {worst_ratio:.1e} of the spread
  that re-running FP64 with a different Sobol seed produces: at this scenario
  count the arithmetic error of FP32 is {verdict}.

  Scope: one sloped two-curve market, rates-only shocks, 10-day horizon,
  {SCENARIOS:,} scenarios. Heavier netting, more scenarios (which shrink the
  sampling noise) or longer horizons move the balance, so re-measure there
  rather than generalising.""")

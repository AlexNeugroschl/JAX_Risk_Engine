"""
How much precision does Monte Carlo market risk need? FP64 vs FP32, and storage in
FP16, BF16 and FP8, of VaR/ES, through the engine's own market-risk path.

**The question.** Lower precision is cheaper on accelerators. The decision is
not "is FP32 exact?" (it is not) but "is FP32's error small next to the Monte
Carlo sampling error the VaR/ES number already carries?" This demo measures
both on one portfolio.

**What runs.** `engine.risk.market.run_market_risk`, unmodified, at eight
`engine.precision.Precision` settings:

    FP64           Precision(): float64 everywhere, the default
    FP32           Precision.throughout("float32"): shifts, revaluation and P&L in float32
    FP32 stored    Precision(pricing=StagePrecision("float32")): revalued in float64,
                   the P&L stored in float32
    FP32, Berm 64  FP32, but the Bermudan revalued and stored in float64
                   (`by_product`; a single trade would be `by_trade`)
    FP16 stored    the shifts and the P&L stored in float16, revalued in float64
    BF16 stored    the same in bfloat16
    FP8 stored     the same in FP8 (e4m3), rounded to nearest
    FP8 stochastic the same, rounded stochastically

Storage below 32 bits keeps a power-of-two scale per block of 32 scenarios, so each
block uses the format's whole range; see docs/planning/details/precision.md §6.2.

    10-day Monte Carlo shocks of every pillar of two sloped curves
        -> full revaluation of every trade at t=0, at its pricing stage's compute precision
        -> P&L per scenario, stored at its pricing stage's storage precision
        -> VaR 99% and ES 97.5% of the portfolio P&L, reduced in float64

The same seed gives the same scenarios at every precision, so the differences
from FP64 are pure arithmetic. Two yardsticks measure them:

  * the Monte Carlo standard error of the ES estimate (`ES_*_standardError`);
  * the spread of the FP64 estimate across independent Sobol seeds -- the
    empirical version of the same thing.

Every result carries a precision report: the policy as run, the format each
array was actually stored in, the device, and with a paired float64 sample
(`Precision.paired_fraction`) each VaR/ES measured at the run's precision and at float64 on
the same scenarios. The last section prints one.

Compute below float32 is not enabled yet (F-07).

ORE parity of this path is established in
tests/test_market_risk_ore_parity.py; this demo is only about precision.

Run with: .venv/Scripts/python.exe demos/demo_precision.py
"""
import dataclasses
import time
import warnings

import numpy as np
import ORE

from engine.instruments.bermudan_swaption import BermudanSwaptionConfig
from engine.instruments.european_swaption import SwaptionConfig
from engine.instruments.swap import SwapConfig
from engine.instruments.treasury import BondConfig, CouponPeriod
from engine.market_data.market import CurrencyMarket, Market, ZeroCurveConfig, index_name
from engine.precision import Precision, StagePrecision
from engine.pricing.config import JamshidianEngineConfig, LgmSwaptionEngineConfig, PricingConfig
from engine.risk.market import MarketRiskRequest, RateRiskFactors, monte_carlo_scenarios, run_market_risk

warnings.simplefilter("ignore")  # Sobol balance notices; the demo prints its own caveats

TODAY = ORE.Date(30, 7, 2026)
SCENARIOS = 8192
SEEDS = (1, 2, 3, 4, 5)
QUANTILES = (0.99, 0.975)
PRECISIONS = {
    "FP64": Precision(),
    "FP32": Precision.throughout("float32"),
    "FP32 stored": Precision(pricing=StagePrecision("float32", "float64", "float64")),
    "FP32, Berm 64": Precision(simulation=StagePrecision("float32", "float32", "float32"),
                               pricing=StagePrecision("float32", "float32", "float32"),
                               by_product={"bermudan_swaption": StagePrecision()}),
    **{label: Precision(simulation=StagePrecision(name), pricing=StagePrecision(name), rounding=rounding)
       for label, name, rounding in (("FP16 stored", "float16", "nearest"), ("BF16 stored", "bfloat16", "nearest"),
                                     ("FP8 stored", "float8_e4m3fn", "nearest"),
                                     ("FP8 stochastic", "float8_e4m3fn", "stochastic"))},
}


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# =============================================================================
# Market: a sloped OIS discount curve and the 6M index's curve 40bp above it.
# =============================================================================
PILLAR_TIMES = [d / 365 for d in (0, 365, 730, 1095, 1825, 2555, 3650, 5475, 7300, 10950)]
OIS = ZeroCurveConfig(PILLAR_TIMES, [0.030, 0.031, 0.032, 0.033, 0.035, 0.037, 0.039, 0.041, 0.042, 0.043])
IBOR = ZeroCurveConfig(PILLAR_TIMES, [r + 0.004 for r in OIS.rates])
MARKET = Market(TODAY, {"USD": CurrencyMarket(OIS, {index_name("USD", 6): IBOR})})
FACTORS = RateRiskFactors.from_market(MARKET)
# The engines: the European on Jamshidian (Hull-White a = 3%, sigma = 1%: the market has no
# swaption volatilities here), the Bermudan on ORE's LGM grid at a fixed model.
PRICING = PricingConfig(european="Jamshidian", jamshidian=JamshidianEngineConfig(0.03, 0.01),
                        bermudan=LgmSwaptionEngineConfig(reversion=0.03, volatility=0.01, calibration="None",
                                                         n_per_std=16, std_devs=5.0))

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
    SwapConfig(notional=50e6, fixed_rate=0.036, payer=True, swap_tenor="10Y", evaluation_date=TODAY,
               trade_id="payer-10y"),
    SwapConfig(notional=45e6, fixed_rate=0.035, payer=False, swap_tenor="9Y", evaluation_date=TODAY,
               trade_id="receiver-9y"),
    SwaptionConfig(notional=20e6, fixed_rate=0.037, payer=False, swap_tenor="5Y",
                   forward_start=ORE.Period(2, ORE.Years), evaluation_date=TODAY, trade_id="european"),
    BermudanSwaptionConfig(notional=15e6, fixed_rate=0.035, payer=True,
                           exercise_dates=[TODAY + ORE.Period(y, ORE.Years) for y in (1, 2, 3, 4)],
                           swap_tenor="5Y", evaluation_date=TODAY, trade_id="bermudan"),
    BondConfig(face_amount=10e6, maturity_date=ORE.Date(30, 7, 2029), evaluation_date=TODAY, coupon_rate=0.04,
               coupon_schedule=tuple(CouponPeriod(ORE.Date(30, 7, 2026 + k), ORE.Date(30, 7, 2027 + k))
                                     for k in range(3)), trade_id="note"),
]


def run(precision: Precision, seed: int):
    scenarios = monte_carlo_scenarios(FACTORS, COVARIANCE, horizon_days=10, num_scenarios=SCENARIOS, seed=seed)
    started = time.time()
    result = run_market_risk(MarketRiskRequest(TRADES, MARKET, scenarios, PRICING, quantiles=QUANTILES,
                                               precision=precision))
    return result, time.time() - started


# =============================================================================
section("Running the market-risk path at each precision and seed")
# =============================================================================
results = {}
for seed in SEEDS:
    for name, precision in PRECISIONS.items():
        result, seconds = run(precision, seed)
        results[name, seed] = result
        print(f"  seed {seed}  {name:<15} VaR 99% {result.risk['VaR_99']:>14,.2f}   "
              f"ES 97.5% {result.risk['ES_97.5']:>14,.2f}   ({seconds:.1f}s)")

base = results["FP64", SEEDS[0]].base_npv
print(f"\n  portfolio base NPV {base:,.2f}; {SCENARIOS:,} scenarios; 10-day horizon")

# =============================================================================
section("Precision error vs Monte Carlo noise")
# =============================================================================
rows = []
for name in PRECISIONS:
    if name == "FP64":
        continue
    for metric in ("VaR_99", "ES_97.5"):
        fp64 = np.array([results["FP64", s].risk[metric] for s in SEEDS])
        other = np.array([results[name, s].risk[metric] for s in SEEDS])
        rows.append((name, metric, np.max(np.abs(other - fp64)), np.std(fp64, ddof=1)))
standard_error = np.mean([results["FP64", s].risk["ES_97.5_standardError"] for s in SEEDS])

print(f"  {'precision':<16}{'metric':<9}{'max |x - FP64|':>17}{'FP64 seed std':>16}{'ratio':>10}")
for name, metric, error, spread in rows:
    print(f"  {name:<16}{metric:<9}{error:>17,.4f}{spread:>16,.2f}{error / spread:>10.1e}")
print(f"\n  ES 97.5% Monte Carlo standard error (mean over seeds): {standard_error:,.2f}")

pnl64 = np.asarray(results["FP64", SEEDS[0]].portfolio_pnl)
scale = np.max(np.abs(pnl64))
for name in PRECISIONS:
    if name != "FP64":
        pnl = np.asarray(results[name, SEEDS[0]].portfolio_pnl)
        print(f"  per-scenario portfolio P&L, {name}: max |x - FP64| / max |P&L| = "
              f"{np.max(np.abs(pnl - pnl64)) / scale:.1e}")

# =============================================================================
section("The precision report of one run, with a paired float64 sample")
# =============================================================================
paired_policy = dataclasses.replace(PRECISIONS["FP8 stored"], paired_fraction=0.05)
report = run(paired_policy, SEEDS[0])[0].precision
print(f"  policy: FP8 stored, paired_fraction {paired_policy.paired_fraction}; ran on {', '.join(report.devices)} "
      f"({report.backend}, jax {report.jax_version})")
print(f"  stored as: {report.realized}")
print(f"  {report.paired_paths} of {report.paths} scenarios re-run at float64:")
print(f"  {'figure':<20}{'run':>16}{'paired, run':>16}{'paired, FP64':>16}{'difference':>14}")
for key, figure in report.figures.items():
    print(f"  {key:<20}{float(figure.value):>16,.2f}{float(figure.paired):>16,.2f}"
          f"{float(figure.paired_float64):>16,.2f}{float(figure.difference):>14,.2f}")
print("  (VaR and ES are quantiles: measured on the pair, not corrected; decision A-13. ES at a\n"
      "  quantile whose tail is empty on the paired scenarios is nan, as ORE refuses it.)")

# =============================================================================
section("Reading the result")
# =============================================================================
worst_ratio = max(error / spread for _, _, error, spread in rows)
if worst_ratio < 0.01:
    verdict = "far inside the sampling noise the number already carries"
elif worst_ratio < 1.0:
    verdict = "inside the sampling noise, but not negligible next to it"
else:
    verdict = "as large as the sampling noise or larger -- not safe here"
print(f"""  The largest reduced-precision error in a VaR/ES figure is {worst_ratio:.1e} of
  the spread that re-running FP64 with a different Sobol seed produces: at this
  scenario count the arithmetic error is {verdict}.

  Scope: one sloped two-curve market, rates-only shocks, 10-day horizon,
  {SCENARIOS:,} scenarios. Heavier netting, more scenarios (which shrink the
  sampling noise) or longer horizons move the balance, so re-measure there
  rather than generalising.""")

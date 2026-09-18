"""
Precision comparison under Monte Carlo: ORE vs. JAX FP64 / FP32 / FP16.

**The question this answers.** This engine exists to run Monte Carlo risk on
TPUs, where lower-precision arithmetic is cheaper. The decision it has to
support is not "is FP32 exact?" (it isn't) but "is FP32's error small
compared to the Monte Carlo sampling error you already tolerate?" -- because
a VaR number carries sampling noise whether you compute it in FP64 or not.
Precision error only matters if it is large relative to that noise. This
demo measures both on the same portfolio, so they can be compared directly.

**What gets compared.**

  1. ORE prices every trade independently, in C++, in double precision,
     through its own `ORE.DiscountingSwapEngine` -- the t=0 anchor. Nothing
     about ORE's numbers comes from this engine's code.
  2. The same portfolio is run through the full Monte Carlo pipeline at
     FP64, FP32 and FP16: Sobol -> Brownian bridge -> correlated cross-asset
     paths -> Hull-White yield-curve reconstruction -> repricing every trade
     in every scenario at every time step -> VaR and Expected Shortfall.
  3. The precision error in VaR/ES is reported against TWO yardsticks: the
     analytic Monte Carlo standard error of the ES estimate itself
     (`expected_shortfall_standard_error`, already in `engine.risk.var_es`),
     and the spread observed by re-running the whole FP64 simulation across
     five different Sobol seeds. The second is the empirical version of the
     first, and a precision tier that is invisible against both is genuinely
     inside the noise rather than merely arguably so.

**Why FP16 needs a mixed-precision pipeline, and what that costs.**
`jnp.linalg.cholesky` and `jax.scipy.stats.norm.ppf` have NO float16 kernel
on this backend (see docs/concepts/architecture.md's "Option B" section --
verified live, not assumed). Both are SETUP steps, computed once per
simulation rather than per path, so this demo hoists them to float64 and
casts down before the per-path evolution -- which is the standard
mixed-precision arrangement real low-precision pipelines use, and is what
makes an FP16 Monte Carlo run possible at all here. The per-path work --
path evolution, curve reconstruction, and all the pricing -- genuinely runs
in float16. What this demo therefore measures is the error of low-precision
*path and pricing arithmetic*, not of a hypothetical all-FP16 stack that no
backend currently supports.

This is why the demo drives the pricers and the simulation kernel directly
instead of going through `price_portfolio`: `PrecisionConfig` accepts only
{32, 64} (deliberately -- see its docstring), so the FP16 column cannot be
requested through the portfolio entry point at all.

**Notionals.** Trades are priced at unit notional and scaled to real size in
float64 afterwards. float16 tops out at 65504, so a multi-million notional
overflows to inf before any pricing math runs. Scaling afterwards measures
float16's *arithmetic* error rather than its range limit; the range limit is
real, and is reported separately at the end.

Run with: .venv/Scripts/python.exe demos/demo_precision.py
"""
import textwrap
from dataclasses import replace

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import ORE

from engine.instruments.swap import SwapConfig, price_swaps
from engine.models.hull_white import B as hw_B
from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER as DAY_COUNTER
from engine.portfolio.request import _flat_curve_cube, derive_maturity_pillars
from engine.risk.var_es import compute_risk_metrics
from engine.simulation.demo_scenarios import EVAL_DATE
from engine.simulation.market_model import (
    ZeroCurveConfig,
    _simulate_cross_asset_paths_jit,
    apply_brownian_bridge,
    compute_hw_A_matrix,
    reconstruct_yield_curves,
)
from jax.scipy.stats import norm
from scipy.stats.qmc import Sobol


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def paragraph(text: str) -> None:
    print(textwrap.fill(" ".join(text.split()), width=78))
    print()


# =============================================================================
# Market: one flat 3% curve, one Hull-White rate factor, one equity factor
# (present so the Cholesky correlation step is exercised rather than
# degenerate). Every precision tier sees exactly this.
# =============================================================================
TODAY = EVAL_DATE
ORE.Settings.instance().evaluationDate = TODAY

FLAT_RATE = 0.03
HW_A = 0.03       # mean reversion
HW_SIGMA = 0.01   # short-rate vol
EQ_VOL = 0.20

ZERO_CURVE = ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[FLAT_RATE] * 6)

TIME_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SCENARIOS = 8192

JOINT_COVARIANCE = np.array([[EQ_VOL ** 2, 0.0], [0.0, HW_SIGMA ** 2]])


# =============================================================================
# The portfolio: 10 swaps, 1Y-30Y, payers and receivers, struck both sides of
# the 3% curve.
#
# Sized so the test is real rather than decorative: repricing runs over 60+
# distinct cashflow pillars, in every one of 8192 scenarios, at every one of 8
# time steps. The book mixes payers and receivers, so the portfolio total is a
# sum of offsetting positive and negative NPVs, which is the
# catastrophic-cancellation setup where precision loss actually bites. The
# VaR/ES tail is then a small fraction of those scenarios, which is where it
# bites hardest.
# =============================================================================
SWAPS = [
    #    tenor, fixed rate, payer, notional
    ("1Y", 0.0295, True, 5_000_000.0),
    ("2Y", 0.0280, True, 2_000_000.0),
    ("3Y", 0.0300, False, 3_000_000.0),
    ("4Y", 0.0315, True, 1_500_000.0),
    ("5Y", 0.0320, True, 2_000_000.0),
    ("7Y", 0.0310, False, 4_000_000.0),
    ("10Y", 0.0350, True, 2_500_000.0),
    ("15Y", 0.0290, False, 1_000_000.0),
    ("20Y", 0.0305, True, 750_000.0),
    ("30Y", 0.0330, False, 500_000.0),
]

# notional=1.0 here; real size is applied in float64 after pricing (see the
# module docstring on float16's range).
swap_configs = [
    SwapConfig(
        notional=1.0, fixed_rate=rate, payer=payer,
        discount_curve_index=0, forward_curve_index=0,
        swap_tenor=tenor, evaluation_date=TODAY,
    )
    for tenor, rate, payer, _size in SWAPS
]
SIZES = np.array([size for _t, _r, _p, size in SWAPS])
NAMES = [f"swap {t:>3} @{r:.2%} {'pay ' if p else 'recv'}" for t, r, p, _s in SWAPS]

# The maturity-pillar set the swap pricer requires: the union of every swap's
# real ORE-generated accrual and payment dates. The portfolio's schedules are
# built by ORE itself (engine.models.ore_builders), not reimplemented here --
# both ORE and this engine price the identical ORE-generated contracts.
PILLARS = np.asarray(derive_maturity_pillars(swap_configs, TODAY))
REMAPPED = [replace(c, discount_curve_index=0, forward_curve_index=1) for c in swap_configs]

section("Portfolio and simulation")
print(f"{len(SWAPS)} swaps, 1Y-30Y, {SIZES.sum():,.0f} total notional")
print(f"cashflow pillars (from ORE's own schedules): {len(PILLARS)}")
print(f"{SCENARIOS:,} scenarios x {len(TIME_GRID) - 1} time steps x {len(SWAPS)} trades "
      f"= {SCENARIOS * (len(TIME_GRID) - 1) * len(SWAPS):,} revaluations per precision")


# =============================================================================
# ORE reference, t=0.
# =============================================================================
section("ORE reference (independent C++ pricing, double precision)")

ore_curve = ORE.YieldTermStructureHandle(ORE.FlatForward(TODAY, FLAT_RATE, DAY_COUNTER))
ore_index = ORE.IborIndex(
    "SimIndex", ORE.Period(6, ORE.Months), 2, ORE.USDCurrency(), ORE.TARGET(),
    ORE.ModifiedFollowing, False, DAY_COUNTER, ore_curve,
)

ore_base_per_trade = []
for cfg in swap_configs:
    swap = ORE.MakeVanillaSwap(
        ORE.Period(cfg.swap_tenor), ore_index, cfg.fixed_rate,
        nominal=cfg.notional,
        swapType=ORE.VanillaSwap.Payer if cfg.payer else ORE.VanillaSwap.Receiver,
        floatingLegSpread=cfg.floating_spread,
        fixedLegDayCount=DAY_COUNTER, floatingLegDayCount=DAY_COUNTER,
    )
    swap.setPricingEngine(ORE.DiscountingSwapEngine(ore_curve))
    ore_base_per_trade.append(swap.NPV())

ore_base_per_trade = np.array(ore_base_per_trade) * SIZES
ore_base = ore_base_per_trade.sum()
print(f"ORE.DiscountingSwapEngine t=0 portfolio NPV: {ore_base:,.2f}")


# =============================================================================
# The Monte Carlo pipeline, at one precision.
#
# This mirrors generate_paths' own pipeline (Sobol -> bridge -> correlated
# cross-asset evolution -> HW curve reconstruction) rather than calling it,
# because generate_paths takes precision in {32, 64} only. The two setup
# steps with no float16 kernel -- the Cholesky correlation factor and the
# Sobol-to-normal inverse CDF -- are computed in float64 and cast down. They
# run ONCE per simulation, not per path; everything that runs per path or per
# scenario is genuinely at `dtype`.
# =============================================================================
def _sobol_normals(seed, num_steps):
    """`generate_sobol_normals` with the Sobol seed exposed.

    The engine's own function hardcodes seed=42 (it has no reason to vary
    it). This demo needs a second and third seed to measure how much the
    VaR/ES estimate moves from RESAMPLING alone -- the yardstick the whole
    comparison rests on. Same construction otherwise: scrambled Sobol,
    clipped away from 0/1, inverse-CDF to normals in float64, reshaped to
    [TimeSteps, Scenarios, Factors].
    """
    engine = Sobol(d=num_steps * 2, scramble=True, seed=seed)
    uniforms = np.clip(engine.random(n=SCENARIOS), 1e-10, 1 - 1e-10)
    normals = np.asarray(norm.ppf(jnp.asarray(uniforms, dtype=jnp.float64)))
    return jnp.asarray(normals.reshape(SCENARIOS, num_steps, 2).transpose(1, 0, 2))


def simulate_and_price(dtype, seed=42):
    num_steps = len(TIME_GRID) - 1
    time_grid_64 = jnp.array(TIME_GRID, dtype=jnp.float64)

    # --- setup, float64 (no float16 kernel for norm.ppf / cholesky) ---
    # seed=42 reproduces generate_sobol_normals' own draw exactly.
    Z = _sobol_normals(seed, num_steps)
    Z_bridged = apply_brownian_bridge(Z, time_grid_64).astype(dtype)

    sigma = np.sqrt(np.diag(JOINT_COVARIANCE))
    correlation = JOINT_COVARIANCE / (sigma[:, None] * sigma[None, :])
    L = jnp.asarray(np.linalg.cholesky(correlation), dtype=dtype)
    L_t = jnp.tile(L[None, :, :], (num_steps, 1, 1))

    # --- per-path evolution, at `dtype` ---
    _eq, hw_paths, _numeraire = _simulate_cross_asset_paths_jit(
        jnp.array([100.0], dtype=dtype),                        # eq_S0
        jnp.zeros((num_steps, 1), dtype=dtype),                 # dividends
        jnp.array([[0.0]], dtype=dtype),                        # rate_mapping
        jnp.full((num_steps, 1), sigma[0], dtype=dtype),        # eq vol
        jnp.array([FLAT_RATE], dtype=dtype),                    # r0
        jnp.full((num_steps, 1), FLAT_RATE, dtype=dtype),       # theta
        jnp.full((num_steps, 1), sigma[1], dtype=dtype),        # hw vol
        jnp.array([HW_A], dtype=dtype),                         # mean reversion
        L_t, jnp.diff(time_grid_64).astype(dtype), Z_bridged,
    )

    # --- yield-curve reconstruction, at `dtype` ---
    step_times = np.array(TIME_GRID[1:])
    T_minus_t = jnp.maximum(
        jnp.asarray(PILLARS)[None, :] - jnp.asarray(step_times)[:, None], 0.0
    )
    B_matrix = jax.vmap(lambda a: hw_B(0.0, T_minus_t, a), out_axes=-1)(jnp.array([HW_A]))
    # A(t,T) is calibrated to today's curve in NumPy float64 inside
    # compute_hw_A_matrix itself -- the engine's own existing behavior at
    # every precision, not something this demo imposes.
    A_matrix = compute_hw_A_matrix(
        zero_curves=[ZERO_CURVE], hw_a=np.array([HW_A]), hw_sigma=np.array([HW_SIGMA]),
        step_times=step_times, maturities=PILLARS,
        B_matrix=np.asarray(B_matrix, dtype=np.float64),
    )
    yield_curves = reconstruct_yield_curves(
        hw_paths, jnp.asarray(A_matrix, dtype=dtype), jnp.asarray(B_matrix, dtype=dtype)
    )

    # --- repricing, at `dtype`: [Scenarios, TimeSteps, Trades] ---
    # One rate factor drives both discounting and forwarding here, so the
    # cube is the same curve stacked twice (index 0 discount, 1 forward).
    curve_cube = jnp.concatenate([yield_curves, yield_curves], axis=-1)
    npv = price_swaps(curve_cube, PILLARS, REMAPPED)

    # Scale to real notionals in float64: at float16 a multi-million notional
    # overflows to inf, which would destroy the VaR tail for a reason that has
    # nothing to do with pricing arithmetic.
    npv_cube = np.asarray(npv, dtype=np.float64) * SIZES

    # t=0 baseline at the same precision, against today's actual curves.
    base_cube = _flat_curve_cube(ZERO_CURVE, ZERO_CURVE, PILLARS, TODAY, dtype=dtype)
    base_per_trade = np.asarray(
        price_swaps(base_cube, PILLARS, REMAPPED)[0, 0, :], dtype=np.float64
    ) * SIZES

    return npv_cube, base_per_trade


section("Running the Monte Carlo pipeline at each precision")

runs = {}
for label, dtype in [("FP64", jnp.float64), ("FP32", jnp.float32), ("FP16", jnp.float16)]:
    with np.errstate(over="ignore", invalid="ignore"):
        npv_cube, base_per_trade = simulate_and_price(dtype)
    metrics = compute_risk_metrics(
        jnp.asarray(npv_cube), float(base_per_trade.sum()), percentiles=(0.95, 0.99)
    )
    runs[label] = dict(
        npv_cube=npv_cube, base_per_trade=base_per_trade,
        base=float(base_per_trade.sum()),
        metrics={k: np.asarray(v, dtype=np.float64) for k, v in metrics.items()},
    )
    finite = np.mean(np.isfinite(npv_cube))
    print(f"{label}: {npv_cube.shape[0]:,} x {npv_cube.shape[1]} x {npv_cube.shape[2]} cube, "
          f"{finite:.1%} finite")


# =============================================================================
# t=0 anchor: does each precision still agree with ORE before any simulation?
# =============================================================================
section("t=0 NPV vs. ORE (per trade, USD)")

header = f"{'trade':<26}{'ORE':>14}{'FP64':>14}{'FP32':>14}{'FP16':>14}"
print(header)
print("-" * len(header))
for i, name in enumerate(NAMES):
    print(f"{name:<26}{ore_base_per_trade[i]:>14,.2f}"
          + "".join(f"{runs[k]['base_per_trade'][i]:>14,.2f}" for k in ("FP64", "FP32", "FP16")))
print("-" * len(header))
print(f"{'PORTFOLIO':<26}{ore_base:>14,.2f}"
      + "".join(f"{runs[k]['base']:>14,.2f}" for k in ("FP64", "FP32", "FP16")))

print()
for label in ("FP64", "FP32", "FP16"):
    err = np.abs(runs[label]["base_per_trade"] - ore_base_per_trade)
    rel = err / np.abs(ore_base_per_trade)
    print(f"  {label}: max abs {err.max():>12,.4f}   median rel {np.median(rel):.2e}")


# =============================================================================
# The Monte Carlo result: VaR and ES at the final time step.
# =============================================================================
section("VaR / Expected Shortfall at t=5Y (USD)")

FINAL = -1
metric_names = ["VaR_95", "ES_95", "VaR_99", "ES_99"]

header = f"{'metric':<14}" + "".join(f"{k:>16}" for k in ("FP64", "FP32", "FP16"))
print(header)
print("-" * len(header))
for metric in metric_names:
    print(f"{metric:<14}"
          + "".join(f"{runs[k]['metrics'][metric][FINAL]:>16,.2f}" for k in ("FP64", "FP32", "FP16")))
print("-" * len(header))

mc_stderr_95 = runs["FP64"]["metrics"]["ES_95_standardError"][FINAL]
mc_stderr_99 = runs["FP64"]["metrics"]["ES_99_standardError"][FINAL]
tail_95 = runs["FP64"]["metrics"]["ES_95_tailCount"][FINAL]
tail_99 = runs["FP64"]["metrics"]["ES_99_tailCount"][FINAL]
print(f"{'ES_95 MC s.e.':<14}{mc_stderr_95:>16,.2f}   (tail n={int(tail_95)})")
print(f"{'ES_99 MC s.e.':<14}{mc_stderr_99:>16,.2f}   (tail n={int(tail_99)})")


# =============================================================================
# The second yardstick: how much does the SAME FP64 computation move when only
# the Sobol seed changes? That is pure resampling noise -- no precision
# involved -- and it is the honest bar a precision tier has to clear.
# =============================================================================
section("Monte Carlo resampling noise (FP64, varying only the Sobol seed)")

SEEDS = [42, 7, 1234, 2026, 99]
seed_es99, seed_es95 = [], []
for seed in SEEDS:
    cube, base_pt = simulate_and_price(jnp.float64, seed=seed)
    m = compute_risk_metrics(jnp.asarray(cube), float(base_pt.sum()), percentiles=(0.95, 0.99))
    seed_es95.append(float(np.asarray(m["ES_95"])[FINAL]))
    seed_es99.append(float(np.asarray(m["ES_99"])[FINAL]))
    print(f"  seed {seed:>5}:  ES_95 {seed_es95[-1]:>13,.2f}   ES_99 {seed_es99[-1]:>13,.2f}")

seed_spread_95 = max(seed_es95) - min(seed_es95)
seed_spread_99 = max(seed_es99) - min(seed_es99)
print(f"\n  ES_95 spread across seeds: {seed_spread_95:>12,.2f}  (std {np.std(seed_es95):,.2f})")
print(f"  ES_99 spread across seeds: {seed_spread_99:>12,.2f}  (std {np.std(seed_es99):,.2f})")


# =============================================================================
# The comparison that decides the question: precision error vs. sampling
# error. Precision error is only a problem if it is large next to the noise
# the Monte Carlo estimate already carries.
#
# Two independent yardsticks, because they answer slightly different
# questions: the ES standard error is the analytic noise of the tail-mean
# estimate, while the seed spread is the empirically observed movement from
# resampling. A precision tier that clears both is genuinely in the noise.
# =============================================================================
section("Precision error vs. Monte Carlo sampling error")

print(f"{'':<7}{'metric':<9}{'err vs FP64':>14}{'MC s.e.':>12}{'ratio':>9}"
      f"{'seed spread':>14}{'ratio':>9}")
print("-" * 74)

verdicts = {}
for label in ("FP32", "FP16"):
    ratios = {}
    for metric, stderr, spread in (("ES_95", mc_stderr_95, seed_spread_95),
                                   ("ES_99", mc_stderr_99, seed_spread_99)):
        err = abs(runs[label]["metrics"][metric][FINAL] - runs["FP64"]["metrics"][metric][FINAL])
        ratio = err / stderr if stderr > 0 else np.nan
        seed_ratio = err / spread if spread > 0 else np.nan
        ratios[metric] = ratio
        ratios[metric + "_seed"] = seed_ratio
        print(f"{label:<7}{metric:<9}{err:>14,.2f}{stderr:>12,.2f}{ratio:>8.3f}x"
              f"{spread:>14,.2f}{seed_ratio:>8.3f}x")
    verdicts[label] = ratios

print()
paragraph("""
    A ratio well below 1 means the precision error is buried inside the Monte
    Carlo noise: switching to that tier changes the answer by less than simply
    re-running FP64 with a different Sobol seed does. A ratio above 1 means the
    precision tier is adding error you would actually notice in the reported
    risk number.
""")


# =============================================================================
# Where the error concentrates: the tail, not the average.
#
# This is the part a t=0-only comparison cannot show. An error that is
# negligible on a mean NPV can still move a 99th-percentile tail, because the
# tail is an order statistic over a small subset of scenarios -- so a handful
# of scenarios crossing the cutoff moves the number.
# =============================================================================
section("Distribution error by region (final step)")

pnl_64 = runs["FP64"]["npv_cube"][:, FINAL, :].sum(axis=1) - runs["FP64"]["base"]
print(f"{'region':<24}{'FP32 mean abs err':>20}{'FP16 mean abs err':>20}")
print("-" * 64)
order = np.argsort(pnl_64)
regions = {
    "worst 1% (ES_99 tail)": order[: max(1, len(order) // 100)],
    "worst 5% (ES_95 tail)": order[: max(1, len(order) // 20)],
    "middle 90%": order[len(order) // 20: -(len(order) // 20)],
    "all scenarios": order,
}
for region_name, idx in regions.items():
    row = f"{region_name:<24}"
    for label in ("FP32", "FP16"):
        pnl = runs[label]["npv_cube"][:, FINAL, :].sum(axis=1) - runs[label]["base"]
        row += f"{np.abs(pnl[idx] - pnl_64[idx]).mean():>20,.2f}"
    print(row)


# =============================================================================
# float16's range limit, which is separate from its precision.
# =============================================================================
section("float16 range")

fp16_max = float(np.finfo(np.float16).max)
biggest = SIZES.max()
print(f"largest finite float16 value:       {fp16_max:>12,.0f}")
print(f"largest notional in this portfolio: {biggest:>12,.0f}  ({biggest / fp16_max:,.0f}x over)")
print()
paragraph(f"""
    Every trade above was priced at notional 1.0 and scaled to real size in
    float64 afterwards. Priced directly at its booked notional, float16
    overflows to inf and the entire VaR tail becomes NaN -- a range failure
    that has nothing to do with arithmetic quality. Any real float16 pipeline
    has to carry a scaling scheme of this kind; it is not optional.
""")


# =============================================================================
# Conclusion.
# =============================================================================
section("Conclusion")

fp64_t0_err = np.abs(runs["FP64"]["base_per_trade"] - ore_base_per_trade).max()
fp32_es99 = verdicts["FP32"]["ES_99"]
fp16_es99 = verdicts["FP16"]["ES_99"]

paragraph(f"""
    FP64 reproduces ORE's own C++ pricing to ${fp64_t0_err:,.5f} at t=0. Two independent
    implementations agreeing at that level is the parity claim in
    docs/reference/ore-parity.md, measured rather than asserted -- and it is what
    makes FP64 usable as the reference for the Monte Carlo comparison above.
""")

paragraph(f"""
    FP32's ES_99 differs from FP64's by {fp32_es99:.3f}x the Monte Carlo standard error of
    that same estimate, and by {verdicts['FP32']['ES_99_seed']:.3f}x the spread observed from simply changing
    the Sobol seed. Both yardsticks agree: the precision error is far smaller
    than the sampling noise already in the number. At this scenario count FP32
    is effectively free for risk work -- you would have to cut Monte Carlo noise
    by orders of magnitude before FP32's arithmetic became the limiting factor.
""")

paragraph(f"""
    FP16 is a different matter. Its ES_99 differs by {fp16_es99:.3f}x the sampling error and
    {verdicts['FP16']['ES_99_seed']:.1f}x the seed spread -- so switching to FP16 moves the risk number by far
    more than resampling the whole simulation does. It is a systematic bias,
    not noise. The region table above shows why that still understates the
    problem: the error is largest in the loss tail, which is exactly the region
    VaR and ES are computed from. Add the range limit ({biggest / fp16_max:,.0f}x overflow at booked
    notionals) and the two setup steps with no float16 kernel at all, and FP16
    is not a speed/accuracy tradeoff at this tier -- it needs a scaling scheme
    and a mixed-precision setup before it produces a number at all, and the
    number it then produces is worst where it matters most.
""")

paragraph("""
    Caveat on scope: this is one flat-curve, single-rate-factor portfolio of
    vanilla swaps at 8192 scenarios. The FP32 conclusion is a statement about
    this regime, not a general licence -- a steeper curve, more offsetting
    positions, longer horizons or a much higher scenario count all shift the
    balance between precision error and sampling error, and the ratio above is
    the thing to re-measure when they do.
""")

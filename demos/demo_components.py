"""
One component at a time: each section runs a single engine module's public API end to end
on the shared scenarios of demos/demo_scenarios.py, and prints what it returns.

    market_model   simulate the cross-asset cube (`generate_paths`)
    swap           a 2Y payer swap's NPV cube (`price_swaps`)
    european       a 3Y-into-2Y European swaption's NPV cube (`price_swaptions`)
    bermudan       a Bermudan swaption at t=0 and on the paths
    american       an American swaption at t=0 and on the paths
    greeks         swap and European swaption Delta/Gamma/Theta
    var_es         the risk statistics of a swap's NPV cube (`compute_risk_metrics`)

These run on the Hull-White simulation (`SimulationConfig`). For the default market path,
the whole pipeline and market-risk VaR/ES, see demos/demo.py.

Run with: .venv/Scripts/python.exe demos/demo_components.py [section ...]   (default: all)
"""
import sys

import jax.numpy as jnp
import numpy as np
import ORE

from demo_scenarios import (
    EVAL_DATE,
    SWAP_DEMO_MATURITIES,
    cross_asset_demo_config,
    flat_yield_curves,
    single_currency_swap_demo_config,
    swaption_demo_config,
)
from engine.instruments.american_swaption import AmericanSwaptionConfig, price_american_swaptions
from engine.instruments.bermudan_swaption import (
    BermudanSwaptionConfig,
    price_bermudan_swaption_base,
    price_bermudan_swaptions,
)
from engine.instruments.european_swaption import SwaptionConfig, price_swaptions
from engine.instruments.swap import SwapConfig, price_swaps
from engine.risk.greeks import ZeroCurve, swap_delta_gamma, swap_theta, swaption_delta_gamma, swaption_theta
from engine.risk.var_es import compute_risk_metrics
from engine.simulation.market_model import ZeroCurveConfig, generate_paths


def market_model():
    market_cubes = generate_paths(cross_asset_demo_config())

    print("--- Base tensors ---")
    print(f"Equities/FX:  {market_cubes['equities'].shape}")
    print(f"Rates:        {market_cubes['rates'].shape}")
    print(f"Numeraire:    {market_cubes['numeraire'].shape}")
    print(f"Yield curves: {market_cubes['yield_curves'].shape}")

    print("Scenario 0, step 1 (t=0.25), USD discount factors:")
    for i, years in enumerate((1, 2, 5, 10)):
        print(f"  to year {years:>2}: {market_cubes['yield_curves'][0, 0, i, 0]:.4f}")


def swap():
    config = single_currency_swap_demo_config()
    market_cubes = generate_paths(config)
    swap_cfg = SwapConfig(
        notional=1_000_000.0, fixed_rate=0.03, payer=True,
        discount_curve_index=0, forward_curve_index=1,
        swap_tenor="2Y", evaluation_date=EVAL_DATE,
    )
    npv_cube = price_swaps(market_cubes["yield_curves"], SWAP_DEMO_MATURITIES, [swap_cfg])
    print("NPV cube shape:", npv_cube.shape)
    # The cube's time axis is time_grid[1:]: index 0 is the first simulated step, not t=0.
    print(f"Mean NPV across scenarios at t={config.time_grid[1]}:", float(jnp.mean(npv_cube[:, 0, 0])))


def _swaption_model(config):
    """The Hull-White model fields of a swaption on `config`'s first rate factor."""
    return dict(
        rate_factor_index=0,
        hw_a=config.rates.mean_reversion[0],
        hw_sigma=float(np.sqrt(config.joint_covariance[1][1])),
        initial_zero_curve=ZeroCurveConfig(times=[0.0, 1.0, 2.0, 5.0, 10.0, 30.0], rates=[0.03] * 6),
    )


def _print_mean_npv(npv_cube, config):
    print("NPV cube shape:", npv_cube.shape)
    for i, t in enumerate(config.time_grid[1:]):
        print(f"  t={t:.2f}: mean NPV across scenarios = {float(jnp.mean(npv_cube[:, i, 0])):.2f}")


def european():
    config = swaption_demo_config()
    market_cubes = generate_paths(config)
    # Exercise in 3Y into a 2Y swap: live at the early steps, zero after exercise.
    swaption_cfg = SwaptionConfig(
        notional=1_000_000.0, fixed_rate=0.030, payer=True, **_swaption_model(config),
        swap_tenor="2Y", forward_start=ORE.Period(3, ORE.Years), evaluation_date=EVAL_DATE,
    )
    step_times = jnp.array(config.time_grid[1:], dtype=jnp.float64)
    _print_mean_npv(price_swaptions(market_cubes["rates"], step_times, [swaption_cfg]), config)


def bermudan():
    config = swaption_demo_config()
    market_cubes = generate_paths(config)
    bermudan_cfg = BermudanSwaptionConfig(
        notional=1_000_000.0, fixed_rate=0.030, payer=True, **_swaption_model(config),
        exercise_dates=[EVAL_DATE + ORE.Period(years, ORE.Years) for years in (1, 2, 3, 4)],
        swap_tenor="5Y", evaluation_date=EVAL_DATE,
    )
    print("Bermudan t=0 NPV:", price_bermudan_swaption_base(bermudan_cfg))
    step_times = jnp.array(config.time_grid[1:], dtype=jnp.float64)
    _print_mean_npv(price_bermudan_swaptions([bermudan_cfg], market_cubes["rates"], step_times), config)


def american():
    config = swaption_demo_config()
    market_cubes = generate_paths(config)
    american_cfg = AmericanSwaptionConfig(
        notional=1_000_000.0, fixed_rate=0.030, payer=True, **_swaption_model(config),
        first_exercise_date=EVAL_DATE + ORE.Period(1, ORE.Years),
        last_exercise_date=EVAL_DATE + ORE.Period(4, ORE.Years),
        swap_tenor="5Y", evaluation_date=EVAL_DATE,
    )
    print("American exercise opportunities:", len(american_cfg.option_times()))
    print("American t=0 NPV:", price_bermudan_swaption_base(american_cfg))
    step_times = jnp.array(config.time_grid[1:], dtype=jnp.float64)
    _print_mean_npv(price_american_swaptions([american_cfg], market_cubes["rates"], step_times), config)


def greeks():
    pillar_times = [1.0, 2.0, 5.0, 10.0, 30.0]

    disc_curve = ZeroCurve.flat(0.030, pillar_times)
    fwd_curve = ZeroCurve.flat(0.035, pillar_times)
    swap_cfg = SwapConfig(
        notional=1_000_000.0, fixed_rate=0.032, payer=True,
        discount_curve_index=0, forward_curve_index=1,
        swap_tenor="5Y", evaluation_date=EVAL_DATE,
    )
    swap_greeks = swap_delta_gamma(swap_cfg, disc_curve, fwd_curve)
    print("--- Swap Greeks (5Y payer, notional $1MM) ---")
    print("Discount curve Delta ($/1bp):", [round(float(v), 2) for v in swap_greeks["discount_delta"]])
    print("Forward curve Delta ($/1bp): ", [round(float(v), 2) for v in swap_greeks["forward_delta"]])
    print("Discount curve Gamma:        ", [round(float(v), 6) for v in swap_greeks["discount_gamma"]])
    print("Theta (1 day):", round(swap_theta(swap_cfg, disc_curve, fwd_curve), 4))

    swaption_curve = ZeroCurve.flat(0.03, pillar_times)
    swaption_cfg = SwaptionConfig(
        notional=1_000_000.0, fixed_rate=0.030, payer=True, rate_factor_index=0,
        hw_a=0.03, hw_sigma=0.01,
        initial_zero_curve=ZeroCurveConfig(times=pillar_times, rates=[0.03] * len(pillar_times)),
        swap_tenor="5Y", forward_start=ORE.Period(3, ORE.Years), evaluation_date=EVAL_DATE,
    )
    swaption_greeks = swaption_delta_gamma(swaption_cfg, swaption_curve)
    print("--- European swaption Greeks (3Y into 5Y payer, notional $1MM) ---")
    print("Delta ($/1bp):", [round(float(v), 2) for v in swaption_greeks["delta"]])
    print("Gamma:        ", [round(float(v), 6) for v in swaption_greeks["gamma"]])
    print("Theta (1 day):", round(swaption_theta(swaption_cfg, swaption_curve), 4))


def var_es():
    market_cubes = generate_paths(single_currency_swap_demo_config())
    # fixed_rate near the 3.5% forward rate, so the swap starts near fair value and the
    # demo shows two-sided P&L.
    swap_cfg = SwapConfig(
        notional=1_000_000.0, fixed_rate=0.035, payer=True,
        discount_curve_index=0, forward_curve_index=1,
        swap_tenor="2Y",
        # Explicit: SWAP_DEMO_MATURITIES is pinned to EVAL_DATE (I-28).
        evaluation_date=EVAL_DATE,
    )
    npv_cube = price_swaps(market_cubes["yield_curves"], SWAP_DEMO_MATURITIES, [swap_cfg])

    # t=0 baseline: the swap on today's (unshocked) curves.
    base_cube = flat_yield_curves(disc_rate=0.030, fwd_rate=0.035)
    base_npv = float(price_swaps(base_cube, SWAP_DEMO_MATURITIES, [swap_cfg])[0, 0, 0])

    metrics = compute_risk_metrics(npv_cube, base_npv, percentiles=(0.95, 0.99))
    print("Base (t=0) NPV:", base_npv)
    for key, values in metrics.items():
        print(f"{key}: {[round(float(v), 2) for v in values]}")


SECTIONS = {f.__name__: f for f in (market_model, swap, european, bermudan, american, greeks, var_es)}


if __name__ == "__main__":
    names = sys.argv[1:] or list(SECTIONS)
    unknown = [n for n in names if n not in SECTIONS]
    if unknown:
        sys.exit(f"unknown section(s) {unknown}; choose from {list(SECTIONS)}")
    for name in names:
        print(f"\n===== {name} =====")
        SECTIONS[name]()

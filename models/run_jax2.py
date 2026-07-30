import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import ORE


def _default_market_risk_dir() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        cwd / "tests" / "MarketRisk",
        cwd / "MarketRisk",
        Path(__file__).resolve().parent.parent / "tests" / "MarketRisk",
        Path(__file__).resolve().parent.parent / "MarketRisk",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return cwd


def _load_simulation_config(config_file: Path) -> Dict[str, Any]:
    if not config_file.exists():
        raise FileNotFoundError(f"Cannot find simulation config: {config_file}")
    with config_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_market_factors(market_data_file: Path) -> jnp.ndarray:
    if not market_data_file.exists():
        raise FileNotFoundError(f"Cannot find market factor file: {market_data_file}")
    market_df = pd.read_csv(market_data_file)
    if "datumValue" not in market_df.columns:
        raise ValueError(f"Expected column 'datumValue' in {market_data_file}")
    return jnp.asarray(market_df["datumValue"].to_numpy(dtype=float), dtype=jnp.float32)


def _extract_trade_notional_vectors(portfolio_file: Path) -> Tuple[List[str], jnp.ndarray, jnp.ndarray]:
    import xml.etree.ElementTree as ET

    if not portfolio_file.exists():
        raise FileNotFoundError(f"Cannot find portfolio file: {portfolio_file}")

    tree = ET.parse(portfolio_file)
    root = tree.getroot()

    trade_ids: List[str] = []
    notional_values: List[float] = []
    factor_indices: List[int] = []

    for idx, trade in enumerate(root.findall("Trade")):
        trade_id = trade.get("id", f"trade_{idx}")
        trade_ids.append(trade_id)

        notional = 0.0
        for element in trade.iter():
            if element.tag == "Notional":
                try:
                    notional = abs(float(element.text.strip()))
                except (AttributeError, ValueError):
                    notional = 1.0
                break

        if notional == 0.0:
            fx_data = trade.find("FxForwardData")
            if fx_data is not None:
                bought_amount = fx_data.findtext("BoughtAmount", "0")
                sold_amount = fx_data.findtext("SoldAmount", "0")
                try:
                    notional = abs(float(bought_amount)) + abs(float(sold_amount))
                except ValueError:
                    notional = 1.0
            else:
                notional = 1.0

        notional_values.append(notional)
        factor_indices.append(idx % 1)

    if not trade_ids:
        raise ValueError(f"No trades were found in portfolio file {portfolio_file}")

    return trade_ids, jnp.asarray(notional_values, dtype=jnp.float32), jnp.asarray(factor_indices, dtype=jnp.int32)


def _build_cashflow_matrix(notional_values: jnp.ndarray, num_steps: int) -> jnp.ndarray:
    # Create a synthetic single-cashflow structure for each trade.
    return jnp.expand_dims(notional_values, axis=1) * jnp.ones((notional_values.shape[0], num_steps), dtype=jnp.float32)


def _build_time_grid(num_steps: int, horizon_years: float) -> jnp.ndarray:
    return jnp.linspace(0.0, horizon_years, num_steps, dtype=jnp.float32)


def _simulate_market_paths(
    random_key: jnp.ndarray,
    num_scenarios: int,
    num_timesteps: int,
    num_factors: int,
    drift: float,
    vol: float,
) -> jnp.ndarray:
    keys = jax.random.split(random_key, num_factors)
    normals = jax.vmap(
        lambda k: jax.random.normal(k, shape=(num_scenarios, num_timesteps), dtype=jnp.float32)
    )(keys)
    normals = jnp.transpose(normals, axes=(1, 2, 0))
    increments = drift + vol * normals
    return jnp.cumsum(increments, axis=1)


@jax.jit
def _discount_curve(time_grid: jnp.ndarray, rate: float) -> jnp.ndarray:
    return jnp.exp(-rate * time_grid)


@jax.jit
def _price_trade_path(
    trade_notional: float,
    market_path: jnp.ndarray,
    discount_factors: jnp.ndarray,
) -> jnp.ndarray:
    cashflows = trade_notional * (1.0 + 0.001 * market_path)
    return cashflows * discount_factors


@jax.jit
def _build_npv_cube(
    trade_notionals: jnp.ndarray,
    market_paths: jnp.ndarray,
    discount_factors: jnp.ndarray,
) -> jnp.ndarray:
    def price_trade(trade_notional: float) -> jnp.ndarray:
        return _price_trade_path(trade_notional, market_paths, discount_factors)

    return jax.vmap(price_trade, in_axes=0)(trade_notionals)


def run_jax2_batch(
    script_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config_file: str = "Input/ore_montecarlo.xml",
    simulation_config: str = "simulation_config.json",
) -> Dict[str, Any]:
    if script_dir is None:
        script_dir = _default_market_risk_dir()
    else:
        script_dir = Path(script_dir).resolve()

    if output_dir is None:
        output_dir = script_dir / "Output" / "JAX"
    else:
        output_dir = Path(output_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    portfolio_file = script_dir / "Input" / "portfolio.xml"
    market_data_file = script_dir / "Output" / "MonteCarlo" / "marketdata.csv"
    config_path = script_dir / simulation_config

    if not config_path.exists():
        config_path = script_dir / "Input" / simulation_config

    config = {
        "num_scenarios": 100,
        "num_timesteps": 10,
        "horizon_years": 1.0,
        "risk_free_rate": 0.01,
        "drift": 0.0,
        "volatility": 0.1,
        "seed": 42,
    }
    if config_path.exists():
        config.update(_load_simulation_config(config_path))

    trade_ids, trade_notional_values, _factor_indices = _extract_trade_notional_vectors(portfolio_file)
    market_factors = _load_market_factors(market_data_file)

    num_scenarios = int(config["num_scenarios"])
    num_timesteps = int(config["num_timesteps"])
    horizon_years = float(config["horizon_years"])
    rate = float(config["risk_free_rate"])
    drift = float(config["drift"])
    volatility = float(config["volatility"])
    seed = int(config["seed"])

    time_grid = _build_time_grid(num_timesteps, horizon_years)
    discount_factors = _discount_curve(time_grid, rate)

    random_key = jax.random.PRNGKey(seed)
    num_factors = 1
    market_paths = _simulate_market_paths(random_key, num_scenarios, num_timesteps, num_factors, drift, volatility)
    market_paths = market_paths[..., 0]

    cube = _build_npv_cube(trade_notional_values, market_paths, discount_factors)
    cube_host = np.asarray(cube)

    npy_path = output_dir / "npv_cube.npy"
    np.save(npy_path, cube_host)

    summary = {
        "trade_count": len(trade_ids),
        "num_scenarios": num_scenarios,
        "num_timesteps": num_timesteps,
        "npv_cube_path": str(npy_path),
    }

    summary_path = output_dir / "jax2_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    base_npv = np.mean(cube_host[:, :, 0], axis=1).tolist()
    base_summary = pd.DataFrame({"TradeId": trade_ids, "BaseNPV": base_npv})
    base_npv_path = output_dir / "jax2_base_npv.csv"
    base_summary.to_csv(base_npv_path, index=False)

    return {
        "npv_cube": npy_path,
        "summary": summary_path,
        "base_npv": base_npv_path,
        "trade_count": len(trade_ids),
        "trade_ids": trade_ids,
    }


if __name__ == "__main__":
    run_jax2_batch()

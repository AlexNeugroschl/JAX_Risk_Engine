import os
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import jax.numpy as jnp
import pandas as pd

import ORE


def _default_market_risk_dir():
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


def run_hybrid_batch(script_dir=None, output_dir=None, config_file="Input/ore_montecarlo.xml"):
    """
    Run the MarketRisk example through ORE for ingestion and analytics,
    then compute a portfolio NPV in JAX and hand it back to a downstream
    ORE-style output file.
    """

    print("--- Starting Hybrid ORE + JAX Execution ---")

    if script_dir is None:
        script_dir = _default_market_risk_dir()
    else:
        script_dir = Path(script_dir).resolve()

    if output_dir is None:
        output_dir = script_dir / "Output"
    else:
        output_dir = Path(output_dir).resolve()

    ore_config_file = script_dir / config_file
    if not ore_config_file.exists():
        fallback_dir = _default_market_risk_dir()
        if fallback_dir != script_dir:
            script_dir = fallback_dir
            ore_config_file = script_dir / config_file

    ore_output_dir = script_dir / "Output" / "MonteCarlo"

    if not ore_config_file.exists():
        raise FileNotFoundError(
            f"Could not find {ore_config_file}. Make sure the MarketRisk example files are present."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    previous_cwd = Path.cwd()
    try:
        os.chdir(script_dir)
        print(f"Using example directory: {script_dir}")
        print(f"Loading configuration from: {ore_config_file}")

        params = ORE.Parameters()
        params.fromFile(str(ore_config_file))

        app = ORE.OREApp(params)
        print("Running ORE Batch Process for market ingestion and analytics...")
        app.run()
    finally:
        os.chdir(previous_cwd)

    print("--- ORE Execution Complete ---")

    for filename in ("npv.csv", "var.csv"):
        src = ore_output_dir / filename
        dst = output_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
        else:
            raise FileNotFoundError(f"Expected ORE output file {src} was not generated.")

    handoff_result = _compute_jax_valuation_and_handoff(
        script_dir=script_dir,
        ore_output_dir=ore_output_dir,
        output_dir=output_dir,
    )

    print(f"Computed JAX-based portfolio NPV: {handoff_result['jax_total_npv']:.2f}")
    print(f"Wrote ORE outputs to: {output_dir}")
    print(f"Wrote JAX valuation summary to: {handoff_result['jax_npv_csv']}")
    print(f"Wrote ORE handoff file to: {handoff_result['ore_handoff_csv']}")

    return {
        "npv_csv": output_dir / "npv.csv",
        "var_csv": output_dir / "var.csv",
        "jax_npv_csv": handoff_result["jax_npv_csv"],
        "ore_handoff_csv": handoff_result["ore_handoff_csv"],
        "jax_total_npv": handoff_result["jax_total_npv"],
    }


def _compute_jax_valuation_and_handoff(script_dir, ore_output_dir, output_dir):
    portfolio_file = script_dir / "Input" / "portfolio.xml"
    market_data_file = ore_output_dir / "marketdata.csv"
    if not portfolio_file.exists():
        raise FileNotFoundError(f"Could not find {portfolio_file}")
    if not market_data_file.exists():
        raise FileNotFoundError(f"Could not find {market_data_file}")

    trade_ids, trade_types, weights = _extract_trade_weights(portfolio_file)
    market_df = pd.read_csv(market_data_file)

    if market_df.empty:
        market_factors = jnp.ones(len(trade_ids), dtype=jnp.float32)
    else:
        market_factors = jnp.asarray(market_df["datumValue"].to_numpy(dtype=float), dtype=jnp.float32)
        if len(market_factors) < len(trade_ids):
            market_factors = jnp.resize(market_factors, len(trade_ids))
        else:
            market_factors = market_factors[: len(trade_ids)]

    weights_j = jnp.asarray(weights, dtype=jnp.float32)
    factors_j = market_factors

    # The valuation model is intentionally simple but explicit: ORE provides the
    # market context; JAX uses that context to compute a trade-level proxy NPV.
    trade_npv_j = weights_j * (1.0 + factors_j * 0.001)
    total_npv_j = jnp.sum(trade_npv_j)

    trade_rows = []
    for trade_id, trade_type, pv_value in zip(trade_ids, trade_types, trade_npv_j):
        trade_rows.append(
            {
                "TradeId": trade_id,
                "TradeType": trade_type,
                "JaxNPV": float(pv_value),
            }
        )

    jax_npv_csv = output_dir / "jax_npv.csv"
    pd.DataFrame(trade_rows).to_csv(jax_npv_csv, index=False)

    handoff_rows = []
    for trade_id, trade_type, pv_value in zip(trade_ids, trade_types, trade_npv_j):
        handoff_rows.append(
            {
                "TradeId": trade_id,
                "TradeType": trade_type,
                "JaxNPV": float(pv_value),
                "Source": "JAX",
                "AnalysisStage": "PostIngestion",
            }
        )

    handoff_csv = output_dir / "ore_handoff.csv"
    pd.DataFrame(handoff_rows).to_csv(handoff_csv, index=False)

    return {
        "jax_npv_csv": jax_npv_csv,
        "ore_handoff_csv": handoff_csv,
        "jax_total_npv": float(total_npv_j),
    }


def _extract_trade_weights(portfolio_file):
    tree = ET.parse(portfolio_file)
    root = tree.getroot()

    trade_ids = []
    trade_types = []
    weights = []

    for trade in root.findall("Trade"):
        trade_id = trade.get("id", "")
        trade_type = trade.findtext("TradeType", "")
        trade_ids.append(trade_id)
        trade_types.append(trade_type)

        notional = None
        for element in trade.iter():
            if element.tag == "Notional":
                try:
                    notional = abs(float(element.text.strip()))
                except (AttributeError, ValueError):
                    notional = 1.0
                break

        if notional is None:
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

        weights.append(notional)

    return trade_ids, trade_types, weights


if __name__ == "__main__":
    run_hybrid_batch()

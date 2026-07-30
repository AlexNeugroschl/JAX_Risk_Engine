import os
from pathlib import Path

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


def run_pure_ore_batch():
    """
    Executes a complete, pure ORE run using the MarketRisk example.
    The script resolves paths relative to the current working directory and
    the repository layout so it continues to work after moving from tests to models.
    """

    print("--- Starting Pure ORE Execution ---")

    script_dir = _default_market_risk_dir()
    ore_config_file = script_dir / "Input" / "ore_montecarlo.xml"
    output_dir = script_dir / "Output"

    if not ore_config_file.exists():
        raise FileNotFoundError(
            f"Could not find {ore_config_file}. "
            "Make sure the MarketRisk example files are present."
        )

    os.chdir(script_dir)
    print(f"Using example directory: {script_dir}")
    print(f"Loading configuration from: {ore_config_file}")

    params = ORE.Parameters()
    params.fromFile(str(ore_config_file))

    app = ORE.OREApp(params)

    print("Running ORE Batch Process (Pricing & Analytics)... This may take a moment.")
    app.run()

    print("--- ORE Execution Complete ---")

    output_dir.mkdir(exist_ok=True)

    csv_files = sorted(output_dir.rglob("*.csv"))
    if not csv_files:
        print("No CSV output files were generated. Check the ORE log or the XML configuration.")
        return

    print(f"\nGenerated output files under {output_dir}:")
    for csv_file in csv_files:
        print(f"- {csv_file.relative_to(script_dir)}")

    for csv_file in csv_files:
        print(f"\n[{csv_file.relative_to(script_dir)}]")
        df = pd.read_csv(csv_file)
        print(df.head())


if __name__ == "__main__":
    run_pure_ore_batch()
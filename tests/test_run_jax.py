import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "MarketRisk" / "run_jax.py"
SPEC = importlib.util.spec_from_file_location("run_jax", MODULE_PATH)
run_jax = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_jax)
run_hybrid_batch = run_jax.run_hybrid_batch


def test_run_hybrid_batch_creates_expected_outputs(tmp_path):
    script_dir = Path(__file__).resolve().parent / "MarketRisk"
    output_dir = tmp_path / "output"

    result = run_hybrid_batch(script_dir=script_dir, output_dir=output_dir, config_file="Input/ore_montecarlo.xml")

    assert result["npv_csv"].exists()
    assert result["var_csv"].exists()
    assert result["jax_npv_csv"].exists()
    assert result["ore_handoff_csv"].exists()
    assert result["jax_total_npv"] >= 0.0

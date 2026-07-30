import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parent.parent / "models" / "run_jax2.py"
SPEC = importlib.util.spec_from_file_location("run_jax2", MODULE_PATH)
run_jax2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(run_jax2)
run_jax2_batch = run_jax2.run_jax2_batch


class TestRunJax2Batch(unittest.TestCase):

    def test_run_jax2_batch_creates_expected_outputs(self):
        script_dir = Path(__file__).resolve().parent / "MarketRisk"
        output_dir = Path("tmp_run_jax2_output").resolve()
        if output_dir.exists():
            for child in output_dir.iterdir():
                child.unlink()
        output_dir.mkdir(parents=True, exist_ok=True)

        result = run_jax2_batch(script_dir=script_dir, output_dir=output_dir)

        self.assertTrue(result["npv_cube"].exists())
        self.assertTrue(result["summary"].exists())
        self.assertTrue(result["base_npv"].exists())
        self.assertTrue(output_dir.joinpath("npv_cube.npy").exists())
        self.assertTrue(output_dir.joinpath("jax2_summary.json").exists())
        self.assertTrue(output_dir.joinpath("jax2_base_npv.csv").exists())
        self.assertGreater(result["trade_count"], 0)


if __name__ == "__main__":
    unittest.main()

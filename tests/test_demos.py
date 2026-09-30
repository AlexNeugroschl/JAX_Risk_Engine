"""
The component demo (demos/demo_components.py) runs every section to completion, on any date.

I-28: the var_es section's swap once had no `evaluation_date`, so it was scheduled off ORE's
wall-clock today while the pillars stayed at `EVAL_DATE`, and was refused on every day but
2026-07-30. I-65 moved the sections out of `engine` modules' `__main__` blocks.
"""
import subprocess
import sys
from pathlib import Path

import pytest

DEMOS = Path(__file__).resolve().parent.parent / "demos"


class TestComponentDemosRun:
    @pytest.mark.slow
    def test_every_section_runs(self):
        proc = subprocess.run(
            [sys.executable, str(DEMOS / "demo_components.py")],
            capture_output=True, text=True, timeout=600,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        for section in ("market_model", "swap", "european", "bermudan", "american", "greeks", "var_es"):
            assert f"===== {section} =====" in proc.stdout
        assert "Base (t=0) NPV:" in proc.stdout
        assert "VaR_95:" in proc.stdout

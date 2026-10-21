"""ORE failures must release scenario readers before temporary-file cleanup (I-27)."""
import dataclasses
import gc
from pathlib import Path
import weakref

import pytest

from tests.support import ore_market_risk_oracle as oracle
from tests.support.ore_inputs import PortfolioInputs
from tests.support.parity_sweep import case


def test_a_failed_ore_run_releases_reader_owners_and_preserves_its_error(monkeypatch):
    fixture = case(35)
    original = oracle.portfolio_inputs
    readers, paths = [], []

    class Parameters:
        def __init__(self, parameters):
            self.parameters = parameters
        def __getattr__(self, name):
            return getattr(self.parameters, name)
        def setScenarioReader(self, path):
            self.reader = open(path, encoding="utf-8")
            readers.append(weakref.ref(self.reader))
            paths.append(Path(path))

    def inputs(*args, **kwargs):
        built = original(*args, **kwargs)
        return dataclasses.replace(built, parameters=Parameters(built.parameters))

    def fail(self, required):
        # A traceback retains both the owning object and the local reader without cleanup.
        reader = self.parameters.reader
        raise RuntimeError("intentional ORE calibration failure")

    monkeypatch.setattr(oracle, "portfolio_inputs", inputs)
    monkeypatch.setattr(PortfolioInputs, "run", fail)
    risk = fixture.market_risk
    with pytest.raises(RuntimeError, match="intentional ORE calibration failure") as error:
        oracle._run(fixture.market, fixture.trades, risk.history, risk.dates,
                    risk.horizon_days, [.05], fixture.pricing, False)
    gc.collect()
    assert str(error.value) == "intentional ORE calibration failure"
    assert readers and all(reader() is None for reader in readers)
    assert all(not path.exists() for path in paths)

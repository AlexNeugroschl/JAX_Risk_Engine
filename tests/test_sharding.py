"""
The scenario axis across devices (`engine.simulation.sharding`, I-61):

  * the device count is the largest that divides the scenario count, capped by the local
    devices and `JAX_RISK_SCENARIO_DEVICES`; on one device nothing is placed, so a one-device
    run is bit for bit what it was;
  * on four XLA host devices, a portfolio run (LGM, Bermudan included, with a paired float64
    sample), the same run with FP8 storage and stochastic rounding, and a market-risk run
    each hold their cube or P&L on all four devices and equal the one-device run to the
    rounding a smaller batch shape moves.

The device count is fixed when JAX starts, so the four-device runs are a separate process
(`tests/support/sharding_check.py`).
"""
import json
import os
import subprocess
import sys

import jax
import numpy as np
import pytest

from engine.simulation import sharding
from engine.simulation.sharding import SCENARIO_DEVICES_ENV, scenario_device_count, shard_scenarios
from tests.support.worker_stubs import ROOT


@pytest.fixture
def eight_devices(monkeypatch):
    monkeypatch.setattr(sharding.jax, "local_devices", lambda: [object()] * 8)
    monkeypatch.delenv(SCENARIO_DEVICES_ENV, raising=False)


class TestDeviceCount:
    @pytest.mark.parametrize("scenarios, expected", [(4096, 8), (100, 5), (7, 7), (13, 1), (1, 1)])
    def test_the_largest_count_that_divides_the_scenarios(self, eight_devices, scenarios, expected):
        assert scenario_device_count(scenarios) == expected

    def test_the_environment_caps_it(self, eight_devices, monkeypatch):
        monkeypatch.setenv(SCENARIO_DEVICES_ENV, "4")
        assert scenario_device_count(4096) == 4
        monkeypatch.setenv(SCENARIO_DEVICES_ENV, "1")
        assert scenario_device_count(4096) == 1

    def test_a_count_below_one_is_refused(self, eight_devices, monkeypatch):
        monkeypatch.setenv(SCENARIO_DEVICES_ENV, "0")
        with pytest.raises(ValueError, match=SCENARIO_DEVICES_ENV):
            scenario_device_count(64)

    def test_on_one_device_the_array_is_left_where_it_is(self, monkeypatch):
        first = jax.local_devices()[:1]
        monkeypatch.setattr(sharding.jax, "local_devices", lambda: first)
        array = jax.numpy.ones((3, 64, 2))
        assert shard_scenarios(array, axis=1) is array


@pytest.mark.slow
def test_four_devices_hold_the_scenarios_and_give_the_one_device_numbers():
    # Four XLA host devices are CPU devices: on a GPU host JAX would otherwise default to its one GPU.
    env = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=4", "JAX_PLATFORMS": "cpu"}
    env.pop(SCENARIO_DEVICES_ENV, None)
    completed = subprocess.run([sys.executable, "-m", "tests.support.sharding_check"], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=1800)
    assert completed.returncode == 0, completed.stderr[-4000:]
    runs = json.loads(completed.stdout)
    assert runs.pop("device_count") == 4
    for name, run in runs.items():
        one, split = run["one"], run["split"]
        assert len(one["devices"]) == 1 and len(split["devices"]) == 4, name
        held = "pnl_devices" if name == "market-risk" else "cube_devices"
        assert split[held] == 4, f"{name}: the result is on {split[held]} device(s)"

    for name in ("lgm", "fp8-stochastic"):
        one, split = runs[name]["one"], runs[name]["split"]
        cube_one, cube_split = np.asarray(one["cube"]), np.asarray(split["cube"])
        scale = np.max(np.abs(cube_one))
        if name == "lgm":   # float64: the batch shape moves a value by about an ulp
            np.testing.assert_allclose(cube_split, cube_one, rtol=0, atol=1e-13 * scale)
        else:               # an ulp may move a value across an FP8 rounding boundary, rarely
            assert np.mean(cube_split != cube_one) <= 0.01
        for figure in ("epe", "ene"):
            np.testing.assert_allclose(split[figure], one[figure], rtol=0, atol=1e-12 * scale)

    one, split = runs["market-risk"]["one"], runs["market-risk"]["split"]
    pnl_one = np.asarray(one["pnl"])
    np.testing.assert_allclose(split["pnl"], pnl_one, rtol=0, atol=1e-13 * np.max(np.abs(pnl_one)))
    for key, value in one["risk"].items():
        np.testing.assert_allclose(split["risk"][key], value, rtol=1e-12, equal_nan=True, err_msg=key)

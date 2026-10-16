"""I-105: exposure-only runs must not materialize the float64 output cube."""
import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from engine.api.requests import MarketPortfolioRequestSchema
from engine.pricing.cube import PortfolioValuation
from engine.precision import FORMATS, Precision, StagePrecision, load, rounding_key, store
from engine.risk.counterparty.exposure import column_profiles, exposure_profile, netting_set_profile
from engine.run import price_portfolio
from engine.pricing import lgm_grid
from tests.support import portfolio as shared
from tests.support.compiles import count_compiles
from tests.support.exposure_storage_check import _arrays


def test_http_none_reaches_the_engine():
    schema = MarketPortfolioRequestSchema.model_validate({
        "market": shared.market_json(), "trades": [], "scenario_risk": False, "cube_output": "none"})
    assert schema.to_dataclass().include_npv_cube is False


def test_exposure_only_never_reads_the_cube(portfolio_request, monkeypatch):
    def forbidden(self):
        pytest.fail("exposure-only run materialized the full cube")

    monkeypatch.setattr(PortfolioValuation, "cube", property(forbidden))
    monkeypatch.setattr(PortfolioValuation, "float64_cube", forbidden, raising=False)
    # Set on the existing dataclass too, so this proves the eager materialization defect
    # against the pre-fix implementation rather than just a missing constructor field.
    request = dataclasses.replace(portfolio_request)
    request.include_npv_cube = False
    result = price_portfolio(request)
    assert result.npv_cube.shape == (0, 0, 0)
    assert result.exposure is not None and len(result.trade_exposures) == 1
    assert result.scenario_risk_available and result.precision.paths == 64


def _assert_figures(actual, expected, tolerance=1e-12):
    actual, expected = dict(_arrays(actual)), dict(_arrays(expected))
    assert actual.keys() == expected.keys()
    for key in actual:
        finite = np.asarray(expected[key])[np.isfinite(expected[key])]
        scale = max(float(np.max(np.abs(finite))) if finite.size else 0, 1.0)
        np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=tolerance * scale, err_msg=key)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_mixed_columns_equal_cube_profiles_in_every_storage_format(paired, rounding):
    rng = np.random.default_rng(10)
    values = rng.normal(size=(67, 4, len(FORMATS))) * 1000 + np.arange(len(FORMATS)) * 200
    columns = [store(jnp.asarray(values[:, :, i]), name, rounding, rounding_key(10, name))
               for i, name in enumerate(FORMATS)]
    cube = jnp.stack([load(c, jnp.float64) for c in columns], axis=-1)
    high = jnp.asarray(values[:32])
    numeraire = jnp.asarray(rng.uniform(0.95, 1.3, size=(67, 4)))
    high_numeraire = numeraire[:32] * 1.00001
    starts = np.linspace(-100, 100, len(columns))
    common = dict(numeraire=numeraire, discount=jnp.asarray([0.99, 0.98, 0.97, 0.96]), times=[0.5, 1, 1.5, 2],
                  quantiles=(0.5, 0.95, 0.975))
    netting, profiles = column_profiles(columns, starts, **common,
                                        paired=([high[:, :, i] for i in range(len(columns))], high_numeraire)
                                        if paired else None)
    expected = netting_set_profile(cube, starts, **common, paired=(high, high_numeraire) if paired else None)
    _assert_figures(netting, expected)
    for i, profile in enumerate(profiles):
        expected = exposure_profile(cube[:, :, i], starts[i], **common,
                                    paired=(high[:, :, i], high_numeraire) if paired else None)
        _assert_figures(profile, expected)
        # Donation must never invalidate the caller's stored columns.
        np.testing.assert_array_equal(load(columns[i], jnp.float64), cube[:, :, i])


@pytest.mark.parametrize("paired_fraction", [0.0, 0.5])
@pytest.mark.parametrize("storage", ["float64", "float32", "float8_e4m3fn"])
def test_omitting_cube_preserves_every_other_result(portfolio_request, paired_fraction, storage):
    precision = Precision(pricing=StagePrecision(storage=storage), paired_fraction=paired_fraction)
    request = dataclasses.replace(portfolio_request, config=dataclasses.replace(portfolio_request.config,
                                                                              precision=precision))
    full = price_portfolio(request)
    omitted = price_portfolio(dataclasses.replace(request, include_npv_cube=False))
    assert omitted.npv_cube.shape == (0, 0, 0) and full.npv_cube.shape == (64, 4, 1)
    _assert_figures(dataclasses.replace(omitted, npv_cube=full.npv_cube), full, tolerance=0)


def test_empty_columns_and_offsetting_trades():
    values = jnp.arange(64.0).reshape(32, 2) - 30
    common = dict(numeraire=jnp.ones((32, 2)), discount=jnp.ones(2), times=[0.5, 1.0])
    empty, profiles = column_profiles([], [], **common, paired=([], jnp.ones((16, 2))))
    assert profiles == []
    np.testing.assert_array_equal(empty.epe, 0)
    cancelled, profiles = column_profiles([values, -values], [10, -10], **common)
    np.testing.assert_array_equal(cancelled.epe, 0)
    np.testing.assert_array_equal(cancelled.ene, 0)
    for pfe in cancelled.pfe.values():
        np.testing.assert_array_equal(pfe, 0)
    assert np.max(profiles[0].epe) > 0


def test_trade_count_and_values_reuse_reduction_programs():
    first = store(jnp.arange(128.0).reshape(64, 2) - 20, "float8_e4m3fn")
    second = store(jnp.arange(128.0).reshape(64, 2) + 3, "float8_e4m3fn")
    common = dict(numeraire=jnp.ones((64, 2)), discount=jnp.ones(2), times=[0.5, 1.0])
    jax.clear_caches()
    with count_compiles() as cold:
        column_profiles([first], [10.0], **common, paired=([jnp.ones((32, 2))], jnp.ones((32, 2))))
    assert cold["jit__accumulate_column"] == 2  # run and smaller paired shape
    high = [jnp.ones((32, 2))] * 9
    with count_compiles() as warm:
        column_profiles([second] * 9, [-30.0] * 9, **common, paired=(high, jnp.ones((32, 2))))
    assert sum(warm.values()) == 0, dict(warm)


@pytest.mark.parametrize("output, expected", [("none", False), ("inline", True), ("artifact", True)])
def test_http_cube_modes(output, expected):
    schema = MarketPortfolioRequestSchema.model_validate({
        "market": shared.market_json(), "trades": [], "scenario_risk": False, "cube_output": output})
    assert schema.to_dataclass().include_npv_cube is expected


def test_requested_cube_assembly_decodes_mixed_storage():
    columns = [store(jnp.arange(70.0).reshape(35, 2), name) for name in ("float32", "float8_e4m3fn", "float64")]
    valuation = PortfolioValuation([0, 0, 0], columns, 35, 2)
    np.testing.assert_array_equal(valuation.float64_cube(), valuation.cube)
    assert valuation.float64_cube().dtype == jnp.float64


@pytest.mark.parametrize("defect", ["starts", "maturities", "column_shape", "paired_count", "paired_shape"])
def test_malformed_column_inputs_are_refused(defect):
    columns, starts, maturities = [jnp.ones((32, 2))], [0.0], None
    paired = None
    if defect == "starts":
        starts = []
    elif defect == "maturities":
        maturities = []
    elif defect == "column_shape":
        columns = [jnp.ones((31, 2))]
    elif defect == "paired_count":
        paired = ([], jnp.ones((16, 2)))
    else:
        paired = ([jnp.ones((16, 3))], jnp.ones((16, 3)))
    with pytest.raises(ValueError, match="column"):
        column_profiles(columns, starts, jnp.ones((32, 2)), jnp.ones(2), [0.5, 1.0],
                        maturities=maturities, paired=paired)


def test_device_budget_uses_capacity_and_not_live_allocations(monkeypatch):
    stats = {"bytes_limit": 8 * 2 ** 30, "bytes_in_use": 100}
    gpu = SimpleNamespace(platform="gpu", memory_stats=lambda: stats)
    monkeypatch.setattr(lgm_grid.jax, "local_devices", lambda: [gpu])
    assert lgm_grid.induction_memory_budget() == 2 ** 30
    stats["bytes_in_use"] = 7 * 2 ** 30
    assert lgm_grid.induction_memory_budget() == 2 ** 30
    small = SimpleNamespace(platform="tpu", memory_stats=lambda: {"bytes_limit": 4 * 2 ** 30})
    monkeypatch.setattr(lgm_grid.jax, "local_devices", lambda: [gpu, small])
    assert lgm_grid.induction_memory_budget() == 2 ** 29
    monkeypatch.setattr(lgm_grid.jax, "local_devices", lambda: [SimpleNamespace(platform="gpu", memory_stats=lambda: None)])
    assert lgm_grid.induction_memory_budget() == lgm_grid.INDUCTION_MEMORY_BUDGET

"""
W1.5 -- the bond's HTTP surface (`engine.api.schemas`).

Covers the discriminated-union routing, the request/response round trip,
and one bug found while building it: a **scalar** Greek crashed
`GreeksSchema.from_dataclass`, because every pre-W1.5 Greek was a per-pillar
vector and the conversion iterated unconditionally.
"""
import ORE
import numpy as np
import pytest
from pydantic import ValidationError

from engine.api.schemas import (
    BondConfigSchema,
    GreeksSchema,
    PortfolioRequestSchema,
    PortfolioResultSchema,
)
from engine.instruments.treasury import BondConfig, price_bond_base
from engine.portfolio.request import price_portfolio

CURVE = {"times": [0.0, 1.0, 2.0, 5.0, 10.0, 30.0], "rates": [0.03] * 6}
MARKET = {
    "time_grid": [0.0, 0.5, 1.0],
    "scenarios": 32,
    "equities": {"initial_prices": [100.0], "dividend_yields": [0.0],
                 "rate_mapping": [[0.0]]},
    "rates": {"initial_rates": [0.03], "theta": [0.03], "mean_reversion": [0.03],
              "maturities": [0.0, 0.5, 1.0], "initial_zero_curves": [CURVE]},
    "joint_covariance": [[0.04, 0.0], [0.0, 0.0001]],
}

BILL_JSON = {
    "trade_type": "bond",
    "face_amount": 100000.0,
    "maturity_date": "2025-12-15",
    "initial_zero_curve": CURVE,
}

NOTE_JSON = {
    "trade_type": "bond",
    "face_amount": 100000.0,
    "maturity_date": "2026-06-15",
    "initial_zero_curve": CURVE,
    "coupon_rate": 0.04,
    "accrual_day_count": "ACT/ACT (ICMA)",
    "coupon_schedule": [
        {"start_date": "2024-12-15", "end_date": "2025-06-15", "payment_date": "2025-06-15"},
        {"start_date": "2025-06-15", "end_date": "2025-12-15", "payment_date": "2025-12-15"},
        {"start_date": "2025-12-15", "end_date": "2026-06-15", "payment_date": "2026-06-15"},
    ],
}


def request_json(trades, **kwargs) -> dict:
    return {
        "evaluation_date": "2025-06-02",
        "market": MARKET,
        "trades": trades,
        "scenario_risk": False,
        **kwargs,
    }


class TestBondSchemaRouting:
    """The discriminated union must route `trade_type: "bond"`."""

    def test_a_bond_parses_from_json(self):
        parsed = PortfolioRequestSchema.model_validate(request_json([BILL_JSON]))
        assert isinstance(parsed.trades[0], BondConfigSchema)

    def test_a_bond_converts_to_its_dataclass(self):
        parsed = PortfolioRequestSchema.model_validate(request_json([BILL_JSON]))
        dataclass_request = parsed.to_dataclass()
        assert isinstance(dataclass_request.trades[0], BondConfig)

    def test_an_unknown_trade_type_is_still_rejected(self):
        """The union must not have been loosened by adding a member."""
        bad = dict(BILL_JSON, trade_type="collateralized_debt_obligation")
        with pytest.raises(ValidationError):
            PortfolioRequestSchema.model_validate(request_json([bad]))

    def test_the_evaluation_date_default_applies(self):
        """A bond with no `evaluation_date` inherits the request's."""
        parsed = PortfolioRequestSchema.model_validate(request_json([BILL_JSON]))
        bond = parsed.to_dataclass().trades[0]
        assert bond.evaluation_date == ORE.Date(2, 6, 2025)

    def test_a_per_trade_evaluation_date_wins(self):
        dated = dict(BILL_JSON, evaluation_date="2025-07-01")
        parsed = PortfolioRequestSchema.model_validate(request_json([dated]))
        assert parsed.to_dataclass().trades[0].evaluation_date == ORE.Date(1, 7, 2025)


class TestBondSchemaConversionIsFaithful:
    """The JSON must produce the same price as the dataclass directly."""

    def test_a_bill_prices_the_same_over_the_schema(self):
        parsed = PortfolioRequestSchema.model_validate(request_json([BILL_JSON]))
        bond = parsed.to_dataclass().trades[0]
        direct = BondConfig(
            face_amount=100000.0, maturity_date=ORE.Date(15, 12, 2025),
            evaluation_date=ORE.Date(2, 6, 2025),
            initial_zero_curve=parsed.trades[0].initial_zero_curve.to_dataclass(),
        )
        assert price_bond_base(bond) == pytest.approx(price_bond_base(direct))

    def test_a_note_schedule_survives_the_round_trip(self):
        parsed = PortfolioRequestSchema.model_validate(request_json([NOTE_JSON]))
        bond = parsed.to_dataclass().trades[0]
        assert len(bond.coupon_schedule) == 3
        assert bond.coupon_schedule[0].start_date == ORE.Date(15, 12, 2024)
        assert bond.coupon_schedule[-1].end_date == ORE.Date(15, 6, 2026)

    def test_an_absent_payment_date_falls_back_to_the_end_date(self):
        no_payment = dict(NOTE_JSON, coupon_schedule=[
            {"start_date": p["start_date"], "end_date": p["end_date"]}
            for p in NOTE_JSON["coupon_schedule"]
        ])
        parsed = PortfolioRequestSchema.model_validate(request_json([no_payment]))
        bond = parsed.to_dataclass().trades[0]
        assert bond.coupon_schedule[0].payment() == ORE.Date(15, 6, 2025)

    def test_a_malformed_bond_is_refused_at_conversion(self):
        """A matured bond must not survive into a priced result."""
        from engine.instruments.treasury import BondPricingError

        matured = dict(BILL_JSON, maturity_date="2020-01-01")
        parsed = PortfolioRequestSchema.model_validate(request_json([matured]))
        with pytest.raises(BondPricingError):
            parsed.to_dataclass()


class TestBondGreeksSerializeOverHttp:
    """**The scalar-Greek serializer bug, found before shipping.**

    Every pre-W1.5 Greek is a per-pillar VECTOR, so
    `GreeksSchema.from_dataclass` iterated `np.asarray(val).tolist()`
    unconditionally. A bond's delta/gamma are SCALARS (one parallel bump
    against a single curve), and a 0-d array's `.tolist()` returns a bare
    Python float -- so the conversion raised
    `TypeError: 'float' object is not iterable` and the whole response
    failed to serialize.

    Verified: these fail against the pre-fix `from_dataclass`.
    """

    def test_a_scalar_greek_serializes(self):
        schema = GreeksSchema.from_dataclass({"delta": np.asarray(-5.284)})
        assert schema.values["delta"] == pytest.approx([-5.284])

    def test_a_scalar_greek_becomes_a_one_element_list(self):
        """Uniformity: `values` is a list-per-Greek, never sometimes a float."""
        schema = GreeksSchema.from_dataclass({"delta": np.asarray(-5.284)})
        assert isinstance(schema.values["delta"], list)
        assert len(schema.values["delta"]) == 1

    def test_a_vector_greek_is_unchanged(self):
        """The fix must not alter the existing per-pillar case."""
        schema = GreeksSchema.from_dataclass({"delta": np.asarray([1.0, 2.0, 3.0])})
        assert schema.values["delta"] == pytest.approx([1.0, 2.0, 3.0])

    def test_theta_is_still_a_scalar_field(self):
        schema = GreeksSchema.from_dataclass({"theta": np.asarray(8.088)})
        assert schema.theta == pytest.approx(8.088)

    def test_a_full_bond_result_serializes_end_to_end(self):
        """The real path: price a bond, then serialize the whole result."""
        parsed = PortfolioRequestSchema.model_validate(
            request_json([BILL_JSON], compute_greeks=True)
        )
        result = price_portfolio(parsed.to_dataclass())
        serialized = PortfolioResultSchema.from_dataclass(result)
        assert serialized.greeks is not None
        assert serialized.greeks[0].values["delta"][0] < 0
        # And it must actually round-trip to JSON.
        assert "delta" in serialized.model_dump_json()


class TestScenarioRiskOverHttp:
    """The `scenario_risk` flag and its result-side counterpart."""

    def test_scenario_risk_defaults_to_true(self):
        body = {"evaluation_date": "2025-06-02", "market": MARKET, "trades": []}
        assert PortfolioRequestSchema.model_validate(body).scenario_risk is True

    def test_scenario_risk_false_reaches_the_dataclass(self):
        parsed = PortfolioRequestSchema.model_validate(request_json([BILL_JSON]))
        assert parsed.to_dataclass().scenario_risk is False

    def test_the_result_reports_scenario_risk_unavailable(self):
        parsed = PortfolioRequestSchema.model_validate(request_json([BILL_JSON]))
        result = price_portfolio(parsed.to_dataclass())
        serialized = PortfolioResultSchema.from_dataclass(result)
        assert serialized.scenario_risk_available is False
        assert serialized.exposure is None
        assert serialized.trade_exposures == []

    def test_base_npv_is_still_reported_when_risk_is_absent(self):
        """The point of the whole design: a real price, with risk absent.

        If this returned 0.0 the `scenario_risk=False` path would be
        useless -- the bond must still be PRICED.
        """
        parsed = PortfolioRequestSchema.model_validate(request_json([BILL_JSON]))
        result = price_portfolio(parsed.to_dataclass())
        serialized = PortfolioResultSchema.from_dataclass(result)
        assert serialized.base_npv > 90_000.0
        assert len(serialized.base_npv_per_trade) == 1

    def test_a_bond_with_scenario_risk_on_is_refused_over_http(self):
        from engine.instruments.treasury import ScenarioPricingNotSupported

        parsed = PortfolioRequestSchema.model_validate(
            request_json([BILL_JSON], scenario_risk=True)
        )
        with pytest.raises(ScenarioPricingNotSupported):
            price_portfolio(parsed.to_dataclass())


class TestOpenApiSchemaIncludesTheBond:
    """The published contract must advertise the new type."""

    def test_the_bond_appears_in_the_request_schema(self):
        schema = PortfolioRequestSchema.model_json_schema()
        assert "BondConfigSchema" in schema.get("$defs", {})

    def test_scenario_risk_is_documented(self):
        schema = PortfolioRequestSchema.model_json_schema()
        assert "scenario_risk" in schema["properties"]

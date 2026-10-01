"""
Shared demo and test scenarios: today's market and the simulation configuration, with either
interest-rate model per currency. The demos import it as `demo_scenarios` (run as scripts from
demos/), the tests as `demos.demo_scenarios`; the engine never imports it.

The market is sloped (USD 3% -> 5%): a flat curve hides drift and convexity errors, which is
how the Hull-White model's old simulation hid a 4-9% bias (I-42).
"""
import ORE

from engine.market import CurrencyMarket, EquityMarket, Market, SwaptionVolSurface, ZeroCurveConfig, index_name
from engine.simulation.config import CamConfig, HullWhiteConfig, LgmConfig

#: Evaluation date of every scenario here.
EVAL_DATE = ORE.Date(30, 7, 2026)

PILLARS = [0.0, 1.0, 2.0, 5.0, 10.0, 30.0]
USD_DISCOUNT = [0.030, 0.030, 0.034, 0.040, 0.046, 0.050]
USD_INDEX = [0.034, 0.034, 0.038, 0.044, 0.049, 0.052]
EUR_DISCOUNT = [0.020, 0.020, 0.022, 0.026, 0.030, 0.032]
EUR_INDEX = [0.023, 0.023, 0.025, 0.029, 0.033, 0.034]
#: ATM normal swaption volatilities (option tenor x swap tenor), both currencies.
VOLS = SwaptionVolSurface(("1Y", "2Y", "5Y", "10Y"), ("1Y", "5Y", "10Y"),
                          ((0.0080, 0.0088, 0.0090), (0.0085, 0.0091, 0.0093), (0.0090, 0.0093, 0.0094),
                           (0.0092, 0.0094, 0.0096)))
USD_6M, EUR_6M = index_name("USD", 6), index_name("EUR", 6)

#: The simulation grid: quarterly for a year, then yearly to five years.
DATES = tuple(EVAL_DATE + ORE.Period(m, ORE.Months) for m in (3, 6, 9, 12, 24, 36, 48, 60))

#: Each model's parameters per currency: the same reversions; the Hull-White volatility is the
#: short rate's, which at these reversions is close to the LGM's alpha early on.
MODELS = {"HullWhite": HullWhiteConfig, "LGM": LgmConfig}


def demo_market(currencies=("USD", "EUR")) -> Market:
    """USD and (by default) EUR, each with a discount curve, a 6M forwarding curve and swaption
    volatilities; the EURUSD spot and a USD equity (AAPL) when EUR is present."""
    data = {
        "USD": CurrencyMarket(ZeroCurveConfig(PILLARS, USD_DISCOUNT), {USD_6M: ZeroCurveConfig(PILLARS, USD_INDEX)},
                              VOLS),
        "EUR": CurrencyMarket(ZeroCurveConfig(PILLARS, EUR_DISCOUNT), {EUR_6M: ZeroCurveConfig(PILLARS, EUR_INDEX)},
                              VOLS),
    }
    both = "EUR" in currencies
    return Market(EVAL_DATE, {c: data[c] for c in currencies},
                  fx_spots={"EURUSD": 1.10} if both else {},
                  equities={"AAPL": EquityMarket("USD", 150.0)} if both else {})


def demo_simulation(model: str = "HullWhite", samples: int = 4096, dates=DATES, currencies=("USD", "EUR"),
                    calibrated: bool = False) -> CamConfig:
    """The cross-asset simulation on `demo_market(currencies)`: `model` (`"HullWhite"` or
    `"LGM"`) for every currency, with a two-currency run adding the EURUSD rate, the AAPL
    equity and their correlations. `calibrated` bootstraps each currency to a 1Y/2Y/5Y
    co-terminal basket instead of the fixed volatilities."""
    config = MODELS[model]
    basket = (("1Y", "2Y", "5Y"), ("9Y", "8Y", "5Y")) if calibrated else ((), ())
    ir = {"USD": config(0.03, 0.01, *basket), "EUR": config(0.02, 0.008, *basket)}
    ir = {c: ir[c] for c in currencies}
    if "EUR" not in currencies:
        return CamConfig(dates=tuple(dates), base_currency="USD", ir=ir, samples=samples, seed=42)
    return CamConfig(
        dates=tuple(dates), base_currency="USD", ir=ir, fx_volatilities={"EUR": 0.10},
        equity_volatilities={"AAPL": 0.25},
        correlations={("IR:USD", "IR:EUR"): 0.6, ("IR:USD", "FX:EURUSD"): 0.2, ("IR:EUR", "FX:EURUSD"): -0.1,
                      ("EQ:AAPL", "IR:USD"): 0.3},
        samples=samples, seed=42)


def demo_market_json(currencies=("USD",)) -> dict:
    """`demo_market(currencies)` as the HTTP request's `market` (docs/reference/http-api.md)."""
    market = demo_market(currencies)
    curve = lambda c: {"times": list(c.times), "rates": list(c.rates)}  # noqa: E731
    vols = lambda v: {"option_tenors": list(v.option_tenors), "swap_tenors": list(v.swap_tenors),  # noqa: E731
                      "vols": [list(row) for row in v.vols]}
    return {
        "asof": market.asof.ISO(),
        "currencies": {code: {"discount_curve": curve(data.discount_curve),
                              "index_curves": {name: curve(c) for name, c in data.index_curves.items()},
                              "swaption_vols": vols(data.swaption_vols)}
                       for code, data in market.currencies.items()},
        "fx_spots": dict(market.fx_spots),
        "equities": {name: {"currency": eq.currency, "spot": eq.spot} for name, eq in market.equities.items()},
    }


def demo_simulation_json(model: str = "HullWhite", samples: int = 4096, dates=DATES, currencies=("USD",),
                         calibrated: bool = False) -> dict:
    """`demo_simulation(...)` as the HTTP request's `simulation`, one currency's model each."""
    config = demo_simulation(model, samples, dates, currencies, calibrated)
    ir = {code: {"model": model, "reversion": m.reversion, "volatility": m.volatility,
                 "calibration_expiries": list(m.calibration_expiries), "calibration_terms": list(m.calibration_terms)}
          for code, m in config.ir.items()}
    return {"dates": [d.ISO() for d in config.dates], "base_currency": config.base_currency, "ir": ir,
            "fx_volatilities": dict(config.fx_volatilities), "equity_volatilities": dict(config.equity_volatilities),
            "correlations": [{"factor1": a, "factor2": b, "value": v} for (a, b), v in config.correlations.items()],
            "samples": config.samples, "seed": config.seed}

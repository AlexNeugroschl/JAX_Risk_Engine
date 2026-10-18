"""
The precision harness's campaigns of record (`tests/support/precision_harness.py`; I-75, F-07,
docs/planning/details/precision.md §15.4), and the command that runs them:

    .venv/Scripts/python.exe -m tests.support.precision_campaigns CAMPAIGN [--paths N ...]
        [--seeds K] [--cases TEXT ...] [--out DIR]

writes `DIR/CAMPAIGN.json.gz` (every number) and `DIR/CAMPAIGN.md` (the summary) after each path
count, so a long campaign keeps what it has measured. `DIR` defaults to
`compliance/precision-measurements`, where the measurements of record are kept; `--cases`
keeps the cases whose label contains any of the texts. Rerun a campaign after any change to a
kernel it prices.

    linear       the shared portfolio's swaps, Europeans and bond (tests/support/portfolio.py),
                 the 3% -> 5% curves, an LGM calibrated to two coterminal swaptions, six dates
                 to 5 years; 1,024, 4,096 and 16,384 paths, 16 seeds. Every stage stored in
                 every format, rounding and block layout, float32 throughout, and each scaled
                 format with a paired sample (an eighth of the paths) for its coverage
    options      the shared portfolio's Bermudans and American on a coarse grid (8 points per
                 standard deviation, 3 deviations); 1,024, 4,096 and 16,384 paths, 16 seeds,
                 since the rollback's rewrite as matrix products in difference form
                 (2026-10-13, I-83; 256 paths and 8 seeds before, when a path's rollback held
                 a [nodes, nodes] operator per step). The formats that bound the others
                 (float16, both FP8) to nearest in both layouts, and FP8 e4m3 stochastically;
                 each option's flipped exercise decisions (`ExerciseRow`, decision A-24)
    compute_options
                 the `options` campaign's portfolio with the pricing stage, the rollback,
                 computed in each compute format, accumulated and stored in float32, and every
                 stage computed in each format at once (`Precision.throughout`); each format's
                 shared operands in two terms and in its single pass (A-23); 1,024, 4,096 and
                 16,384 paths, 16 seeds
    market_risk  market risk on the market-risk tests' sloped two-curve market
                 (tests/market_risk_support.py): offsetting payer and receiver swaps, a European,
                 a Bermudan (on the coarse grid) and a bond, 10-day Monte Carlo shifts of every
                 pillar; 2,048 and 8,192 scenarios, 16 seeds. The shifts (`simulation`) and the
                 P&L (`pricing`) stored in every format and rounding. Under zero-mean shifts both
                 straddle zero in every block, so a block offset is 0 and both layouts store
                 them alike, bit for bit (a test holds it): the default layout alone is measured
    compute      the `linear` campaign's portfolio with the stages computed below float32 (the
                 kernel rewrite, 2026-10-10): each stage's products in each compute format,
                 accumulated and stored in float32 (float32's own case the floor of the
                 accumulator), and every stage computed in each format at once
                 (`Precision.throughout`, its storage the format's where it has one); each
                 format's shared operands in two terms and in its single pass (`<name>_1x`,
                 decision A-23); 1,024, 4,096 and 16,384 paths, 16 seeds
    compute_market_risk
                 the `market_risk` campaign's revaluation with each trade's products in each
                 compute format, accumulated and stored in float32, the Bermudan's since the
                 rollback's rewrite (2026-10-13); 2,048 and 8,192 scenarios, 16 seeds
    rounding, rounding_options, rounding_market_risk
                 nearest against stochastic product rounding (2026-10-16)
    replicated_storage
                 the `linear` campaign with eight independent scrambles per run (I-111)
    market_risk_calibrated
                 the `market_risk` campaign with the Bermudan calibrated and recalibrated under
                 every scenario (I-112, 2026-10-17), which the fixed-LGM campaigns do not
                 measure (I-114): the `market_risk` storage cases, the `compute_market_risk`
                 compute cases and the `rounding_market_risk` formats rounded stochastically;
                 2,048 and 8,192 scenarios, 16 seeds

Every campaign records its products' pricing settings (`Measurements.settings`, the evidence
table's key, I-114): rows validate a run only under the settings they were measured with.
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import ORE

from engine.instruments.swap import SwapConfig
from engine.market_simulation.config import CamConfig, LgmConfig
from engine.pricing.config import LgmSwaptionEngineConfig, PricingConfig
from tests import market_risk_support as mr
from tests.support import portfolio as shared
from tests.support.precision_harness import (
    COMPUTE, SCALED, Case, MarketRiskCampaign, Measurements, PortfolioCampaign, compute_cases, measure, storage_cases,
    summary, throughout_case,
)
from engine.precision import Precision, StagePrecision

#: Where the measurements of record are kept.
RECORD = Path(__file__).resolve().parents[2] / "compliance" / "precision-measurements"

#: Simulation dates: monthly to 5 years in six steps.
DATES = tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in (3, 6, 12, 24, 36, 60))
#: The LGM of the precision tests, calibrated to the 1Y x 9Y and 2Y x 8Y swaptions.
SIMULATION = CamConfig(dates=DATES, base_currency="USD", ir={"USD": LgmConfig(0.03, 0.01, ("1Y", "2Y"), ("9Y", "8Y"))},
                       samples=1024, seed=1)
#: The Bermudan/American grid of the precision tests (ORE's default is 30 points per standard
#: deviation and 5 deviations).
COARSE_GRID = LgmSwaptionEngineConfig(n_per_std=8, std_devs=3.0)
#: The paired sample's share of the paths: four blocks of 32 at 1,024 paths.
PAIRED = 0.125
#: The markets, as every row measured on them names them (the evidence table's, I-55).
SHARED_MARKET = "USD, the shared portfolio's 3% -> 5% curves (tests/support/portfolio.py)"
MARKET_RISK_MARKET = "USD, the market-risk tests' sloped two-curve market (tests/market_risk_support.py)"


def _shared(*names):
    return tuple(shared.trades()[n] for n in names)


def linear() -> PortfolioCampaign:
    return PortfolioCampaign(
        name="linear",
        description="The shared portfolio's swaps, Europeans and bond (tests/support/portfolio.py) on its 3% -> 5% "
                    "curves, an LGM calibrated to the 1Y x 9Y and 2Y x 8Y swaptions, six dates to 5 years. Each "
                    "case stores one stage in one format (computed in float64), or runs float32 throughout; the "
                    "`:paired` cases add a paired float64 sample of an eighth of the paths.",
        market_name=SHARED_MARKET, market=shared.market(),
        trades=_shared("swap-payer", "swap-receiver-seasoned-icma", "european-payer", "european-receiver-otm-cash",
                       "bond"),
        simulation=SIMULATION, pricing=PricingConfig(),
        cases=(storage_cases() + (throughout_case("float32"),)
               + storage_cases(formats=SCALED, offsets=("block",), paired_fraction=PAIRED)),
        paths=(1024, 4096, 16384), seeds=tuple(range(1, 17)))


def options() -> PortfolioCampaign:
    return PortfolioCampaign(
        name="options",
        description="The shared portfolio's Bermudans and American (tests/support/portfolio.py), each LGM "
                    "calibrated once on the simulation market of the as-of date and kept on every path date, as "
                    "ORE's exposure simulation keeps it (I-49, 2026-10-12), on a coarse grid (8 points per standard "
                    "deviation, 3 deviations); the market and simulation of the `linear` campaign, at ORE's default "
                    "ShiftHorizon (I-32). The rollback as matrix products in difference form (2026-10-13, I-83). "
                    "Each case stores one stage in one format (computed in float64).",
        market_name=SHARED_MARKET, market=shared.market(),
        trades=_shared("bermudan-payer-physical", "bermudan-receiver-cash", "american-payer"),
        simulation=SIMULATION, pricing=PricingConfig(bermudan=COARSE_GRID, american=COARSE_GRID),
        cases=(storage_cases(formats=("float16", "float8_e4m3fn", "float8_e5m2"), roundings=("nearest",))
               + storage_cases(formats=("float8_e4m3fn",), roundings=("stochastic",), offsets=("block",))),
        paths=(1024, 4096, 16384), seeds=tuple(range(1, 17)))


def compute_options() -> PortfolioCampaign:
    return dataclasses.replace(
        options(), name="compute_options",
        description="The `options` campaign's portfolio, market and simulation, computed below float32: each case "
                    "runs the pricing stage's matrix products (the rollback's operator, the cashflows' functions "
                    "on the grid, F-07, 2026-10-13) in one compute format, accumulated and stored in float32, every "
                    "other stage float64 (the `float32` case is the accumulator's floor), or every stage in the "
                    "format (`throughout`). A format below float32 carries the operand every path shares in two "
                    "terms; `<format>_1x` is its single pass (A-23).",
        cases=compute_cases(stages=("pricing",)) + tuple(throughout_case(name) for name in COMPUTE))


def compute() -> PortfolioCampaign:
    return dataclasses.replace(
        linear(), name="compute",
        description="The `linear` campaign's portfolio, market and simulation, computed below float32 (the kernel "
                    "rewrite, F-07): each case runs one stage's matrix products in one compute format, accumulated "
                    "and stored in float32, every other stage float64 (the `float32` case is the accumulator's "
                    "floor), or every stage in the format (`throughout`). A format below float32 carries the "
                    "operand every path shares in two terms; `<format>_1x` is its single pass (A-23).",
        cases=compute_cases() + tuple(throughout_case(name) for name in COMPUTE))


def _market_risk_trades():
    receiver = SwapConfig(notional=9e6, fixed_rate=0.035, payer=False, swap_tenor="9Y", evaluation_date=mr.TODAY,
                          trade_id="swap-receiver")
    return mr.swap(), receiver, mr.european(), mr.bermudan(), mr.bond()


def compute_market_risk() -> MarketRiskCampaign:
    return dataclasses.replace(
        market_risk(), name="compute_market_risk",
        description="The `market_risk` campaign's revaluation, each trade's matrix products in one compute format, "
                    "accumulated and stored in float32 (the `float32` case is the accumulator's floor), the "
                    "Bermudan's induction in difference form (2026-10-13). A format below float32 carries the "
                    "operand every path shares in two terms; `<format>_1x` is its single pass (A-23).",
        cases=compute_cases(stages=("pricing",)))


def market_risk() -> MarketRiskCampaign:
    return MarketRiskCampaign(
        name="market_risk",
        description="Market risk on the market-risk tests' sloped two-curve market (tests/market_risk_support.py): "
                    "a 10Y payer swap and a 9Y receiver that offsets most of it, a European, a Bermudan and a bond, "
                    "revalued under 10-day Monte Carlo shifts of every pillar. Each case stores the shifts "
                    "(`simulation`) or each trade's P&L (`pricing`) in one format, revalued in float64.",
        market_name=MARKET_RISK_MARKET, market=mr.market(), trades=_market_risk_trades(),
        covariance=mr.covariance(), horizon_days=10,
        pricing=dataclasses.replace(mr.PRICING, bermudan=dataclasses.replace(mr.ENGINE, n_per_std=8, std_devs=3.0)),
        cases=storage_cases(stages=("simulation", "pricing"), offsets=("block",)),
        paths=(2048, 8192), seeds=tuple(range(1, 17)))


def market_risk_calibrated() -> MarketRiskCampaign:
    base = market_risk()
    calibrated = dataclasses.replace(mr.CALIBRATED_ENGINE, n_per_std=8, std_devs=3.0)
    return dataclasses.replace(
        base, name="market_risk_calibrated",
        description="The `market_risk` campaign's market, trades, shifts and coarse grid with the Bermudan calibrated "
                    "to the market's swaptions (ORE's CoterminalDealStrike) and recalibrated under every scenario, as "
                    "ORE's historical P&L recalibrates it (I-112): each scenario its own model and induction, where "
                    "the `market_risk` campaigns share one fixed LGM. Each case stores the shifts (`simulation`) or "
                    "each trade's P&L (`pricing`) in one format, revalued in float64; or computes each trade's "
                    "matrix products in one compute format, accumulated and stored in float32 (the `float32` case "
                    "is the accumulator's floor; `<format>_1x` the single pass, A-23), to nearest, and the "
                    "`rounding_market_risk` formats stochastically.",
        pricing=dataclasses.replace(base.pricing, bermudan=calibrated),
        cases=(base.cases + compute_cases(stages=("pricing",))
               + rounding_cases(("pricing",), ("float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"),
                                roundings=("stochastic",))))


def rounding_cases(stages, formats, roundings=("nearest", "stochastic")):
    """Matched product policies in each of `roundings`. Storage stays float32."""
    return tuple(Case(f"{stage}:compute:{name}:{rounding}",
                      Precision(**{stage: StagePrecision("float32", name, "float32", rounding)}))
                 for stage in stages for name in formats for rounding in roundings)


def rounding():
    return dataclasses.replace(compute(), name="rounding",
        description="FP8 product rounding on the linear portfolio, nearest versus stochastic per-path operands; "
                    "shared operands retain two terms. Matched single-scramble draws; stochastic seed varies with the outer seed.",
        cases=rounding_cases(("simulation", "market", "pricing"), ("float8_e4m3fn", "float8_e5m2")))


def rounding_options():
    return dataclasses.replace(compute_options(), name="rounding_options",
        description="Rollback product rounding for Bermudans and Americans, nearest versus stochastic per-path "
                    "operands, with long continuation values projected onto zero before wrapping. Shared operands retain two terms.",
        cases=rounding_cases(("pricing",), ("float16", "tensorfloat32", "bfloat16", "float8_e4m3fn", "float8_e5m2")))


def rounding_market_risk():
    return dataclasses.replace(compute_market_risk(), name="rounding_market_risk",
        description="Market-risk product rounding, nearest versus stochastic per-path operands, matched draws; "
                    "the pooled VaR and ES estimators remain unchanged.",
        cases=rounding_cases(("pricing",), ("float16", "bfloat16", "float8_e4m3fn", "float8_e5m2")))


def replicated_storage():
    return dataclasses.replace(linear(), name="replicated_storage",
        description="Eight independent scrambled Sobol replicates per run; a float64 prefix paired within each. "
                    "Mean errors use independent replicate estimates; Student t intervals (7 degrees of freedom) "
                    "are audited per date as diagnostics, with zero-width misses retained. Nominal confidence "
                    "intervals are unavailable without a validated tail bound (I-111). "
                    "Matched float64 draws. Paired counts round up to whole storage blocks within each replicate.",
        simulation=dataclasses.replace(SIMULATION, replicates=8), seeds=tuple(range(1, 33)),
        cases=(storage_cases(stages=("simulation", "market"), formats=("float16", "bfloat16"),
                             offsets=("block",), paired_fraction=PAIRED)
               + storage_cases(stages=("pricing",), formats=("float8_e4m3fn", "float8_e5m2"),
                               offsets=("block",), paired_fraction=PAIRED)))


CAMPAIGNS = {"linear": linear, "options": options, "market_risk": market_risk, "compute": compute,
             "compute_market_risk": compute_market_risk, "compute_options": compute_options,
             "rounding": rounding, "rounding_options": rounding_options,
             "rounding_market_risk": rounding_market_risk, "replicated_storage": replicated_storage,
             "market_risk_calibrated": market_risk_calibrated}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.support.precision_campaigns",
                                     description="Run a campaign of the precision harness (I-75, F-07).")
    parser.add_argument("campaign", choices=sorted(CAMPAIGNS))
    parser.add_argument("--paths", type=int, nargs="+", help="path counts (default: the campaign's)")
    parser.add_argument("--seeds", type=int, help="the first K of the campaign's seeds (at least 2)")
    parser.add_argument("--cases", nargs="+", help="keep the cases whose label contains any of these")
    parser.add_argument("--out", type=Path, default=RECORD, help=f"directory (default: {RECORD})")
    args = parser.parse_args(argv)

    campaign = CAMPAIGNS[args.campaign]()
    if args.seeds:
        campaign = dataclasses.replace(campaign, seeds=campaign.seeds[:args.seeds])
    if args.cases:
        campaign = dataclasses.replace(campaign, cases=tuple(c for c in campaign.cases
                                                             if any(text in c.label for text in args.cases)))
    args.out.mkdir(parents=True, exist_ok=True)
    measurements = None
    for paths in args.paths or campaign.paths:
        measured = measure(campaign, paths, progress=lambda line: print(line, flush=True))
        measurements = measured if measurements is None else measurements.merge(measured)
        write(measurements, args.out)
    return 0


def write(measurements: Measurements, directory: Path) -> None:
    """`<campaign>.json.gz` and `<campaign>.md` in `directory`."""
    measurements.save(directory / f"{measurements.campaign}.json.gz")
    (directory / f"{measurements.campaign}.md").write_text(summary(measurements) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

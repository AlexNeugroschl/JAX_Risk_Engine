"""
The storage harness's campaigns of record (`tests/support/precision_harness.py`; I-75,
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
                 standard deviation, 3 deviations); 256 paths, 8 seeds, since a path's
                 rollback holds a [nodes, nodes] operator per step until the kernel rewrite
                 (I-83). The formats that bound the others (float16, both FP8) to nearest in
                 both layouts, and FP8 e4m3 stochastically
    market_risk  market risk on the market-risk tests' sloped two-curve market
                 (tests/market_risk_support.py): offsetting payer and receiver swaps, a European,
                 a Bermudan (on the coarse grid) and a bond, 10-day Monte Carlo shifts of every
                 pillar; 2,048 and 8,192 scenarios, 16 seeds. The shifts (`simulation`) and the
                 P&L (`pricing`) stored in every format and rounding. Under zero-mean shifts both
                 straddle zero in every block, so a block offset is 0 and both layouts store
                 them alike, bit for bit (a test holds it): the default layout alone is measured
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
    SCALED, MarketRiskCampaign, Measurements, PortfolioCampaign, measure, storage_cases, summary, throughout_case,
)

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


def _shared(*names):
    return tuple(shared.trades()[n] for n in names)


def linear() -> PortfolioCampaign:
    return PortfolioCampaign(
        name="linear",
        description="The shared portfolio's swaps, Europeans and bond (tests/support/portfolio.py) on its 3% -> 5% "
                    "curves, an LGM calibrated to the 1Y x 9Y and 2Y x 8Y swaptions, six dates to 5 years. Each "
                    "case stores one stage in one format (computed in float64), or runs float32 throughout; the "
                    "`:paired` cases add a paired float64 sample of an eighth of the paths.",
        market=shared.market(),
        trades=_shared("swap-payer", "swap-receiver-seasoned-icma", "european-payer", "european-receiver-otm-cash",
                       "bond"),
        simulation=SIMULATION, pricing=PricingConfig(),
        cases=(storage_cases() + (throughout_case("float32"),)
               + storage_cases(formats=SCALED, offsets=("block",), paired_fraction=PAIRED)),
        paths=(1024, 4096, 16384), seeds=tuple(range(1, 17)))


def options() -> PortfolioCampaign:
    return PortfolioCampaign(
        name="options",
        description="The shared portfolio's Bermudans and American (tests/support/portfolio.py), calibrated on "
                    "every path date, on a coarse grid (8 points per standard deviation, 3 deviations); the market "
                    "and simulation of the `linear` campaign. 256 paths: a path's rollback holds a [nodes, nodes] "
                    "operator per step until the kernel rewrite (I-83). The Bermudan/American kernel moves with "
                    "ShiftHorizon and the recalibration fix (I-32, I-49), after which these rows are measured again.",
        market=shared.market(),
        trades=_shared("bermudan-payer-physical", "bermudan-receiver-cash", "american-payer"),
        simulation=SIMULATION, pricing=PricingConfig(bermudan=COARSE_GRID, american=COARSE_GRID),
        cases=(storage_cases(formats=("float16", "float8_e4m3fn", "float8_e5m2"), roundings=("nearest",))
               + storage_cases(formats=("float8_e4m3fn",), roundings=("stochastic",), offsets=("block",))),
        paths=(256,), seeds=tuple(range(1, 9)))


def market_risk() -> MarketRiskCampaign:
    receiver = SwapConfig(notional=9e6, fixed_rate=0.035, payer=False, swap_tenor="9Y", evaluation_date=mr.TODAY,
                          trade_id="swap-receiver")
    return MarketRiskCampaign(
        name="market_risk",
        description="Market risk on the market-risk tests' sloped two-curve market (tests/market_risk_support.py): "
                    "a 10Y payer swap and a 9Y receiver that offsets most of it, a European, a Bermudan and a bond, "
                    "revalued under 10-day Monte Carlo shifts of every pillar. Each case stores the shifts "
                    "(`simulation`) or each trade's P&L (`pricing`) in one format, revalued in float64.",
        market=mr.market(), trades=(mr.swap(), receiver, mr.european(), mr.bermudan(), mr.bond()),
        covariance=mr.covariance(), horizon_days=10,
        pricing=dataclasses.replace(mr.PRICING, bermudan=dataclasses.replace(mr.ENGINE, n_per_std=8, std_devs=3.0)),
        cases=storage_cases(stages=("simulation", "pricing"), offsets=("block",)),
        paths=(2048, 8192), seeds=tuple(range(1, 17)))


CAMPAIGNS = {"linear": linear, "options": options, "market_risk": market_risk}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.support.precision_campaigns",
                                     description="Run a campaign of the storage harness (I-75).")
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

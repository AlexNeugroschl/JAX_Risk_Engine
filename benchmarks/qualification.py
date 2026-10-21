"""Isolated accuracy campaigns on recorded and independent ORE-sweep fixtures.

python -m benchmarks.qualification --help
Experimental results never replace the shipped precision acceptance records automatically.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import sys

from demos.benchmark_reporting import atomic_json, launch_worker, read_json

TIERS = {"reproduce": ((1024, 4096, 16384), 16),
         "large": ((16384, 65536, 262144), 16),
         "extended": ((262144, 1048576, 4194304), 32)}


def source_digest():
    """Stable across platform line endings, including the tools needed to reproduce a run."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for folder in ("engine", "benchmarks", "tests/support", "demos"):
        for path in sorted((root / folder).rglob("*.py")):
            digest.update(path.relative_to(root).as_posix().encode() + b"\0")
            digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    path = root / "tests/market_risk_support.py"
    digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def build_campaign(name, fixture_seed=None, model="LGM", grid="recorded", cases=None, *, multi_currency=False):
    """Reuse the ORE-parity sweep's market, booked trades and pricing conventions.
    Fixture identities and Sobol seed identities are separate experimental dimensions."""
    import numpy as np
    import ORE
    from engine.market_simulation.config import HullWhiteConfig, LgmConfig
    from engine.risk.market import RateRiskFactors
    from tests.support.precision_campaigns import CAMPAIGNS

    campaign = CAMPAIGNS[name]()
    if fixture_seed is not None:
        from tests.support.parity_sweep import case
        fixture = case(fixture_seed)
        products = {t.product for t in campaign.trades}
        trades = tuple(t for t in fixture.trades if t.product in products)
        # Preserve the campaign's fixed/recalibrated distinction while drawing the fixture's
        # conventions, engines and grid. A calibrated campaign must not silently become fixed.
        pricing = dataclasses.replace(fixture.pricing,
            bermudan=dataclasses.replace(fixture.pricing.bermudan, calibration=campaign.pricing.bermudan.calibration),
            american=dataclasses.replace(fixture.pricing.american, calibration=campaign.pricing.american.calibration))
        campaign = dataclasses.replace(campaign, market=fixture.market, trades=trades, pricing=pricing,
            market_name=f"ORE sweep fixture {fixture_seed}: {fixture.shape}")
        if campaign.kind == "portfolio":
            dates = tuple(fixture.market.asof + ORE.Period(m, ORE.Months) for m in (3, 6, 12, 24, 36, 60))
            ir = {ccy: dataclasses.replace(component, swap_index=fixture.pricing.bermudan.swap_index)
                  for ccy, component in campaign.simulation.ir.items()}
            campaign = dataclasses.replace(campaign, simulation=dataclasses.replace(campaign.simulation, dates=dates, ir=ir))
        else:
            n = RateRiskFactors.from_market(fixture.market).size
            indices = np.arange(n)
            covariance = np.exp(-np.abs(indices[:, None] - indices[None, :]) / 4) * .0008 ** 2 * campaign.horizon_days
            campaign = dataclasses.replace(campaign, covariance=covariance)
    if campaign.kind == "portfolio":
        cls = {"LGM": LgmConfig, "HullWhite": HullWhiteConfig}[model]
        ir = {ccy: cls(**{f.name: getattr(component, f.name) for f in dataclasses.fields(component)})
              for ccy, component in campaign.simulation.ir.items()}
        campaign = dataclasses.replace(campaign, simulation=dataclasses.replace(campaign.simulation, ir=ir))
    if multi_currency:
        from engine.market_data.market import CurrencyMarket, ZeroCurveConfig
        usd = campaign.market.currency("USD")
        def shifted(curve):
            return ZeroCurveConfig(list(curve.times), [r - .012 for r in curve.rates])
        eur = CurrencyMarket(shifted(usd.discount_curve),
            {key.replace("USD-", "EUR-", 1): shifted(curve) for key, curve in usd.index_curves.items()}, usd.swaption_vols)
        market = dataclasses.replace(campaign.market, currencies={"USD": usd, "EUR": eur}, fx_spots={"EURUSD": 1.1})
        trades = campaign.trades + tuple(dataclasses.replace(t, currency="EUR", trade_id=f"EUR-{t.trade_id}")
                                         for t in campaign.trades)
        campaign = dataclasses.replace(campaign, market=market, trades=trades,
                                      market_name=campaign.market_name + "; correlated USD/EUR, EURUSD=1.1")
        if campaign.kind == "portfolio":
            simulation = dataclasses.replace(campaign.simulation,
                ir={"USD": campaign.simulation.ir["USD"], "EUR": campaign.simulation.ir["USD"]},
                fx_volatilities={"EUR": .15}, correlations={("IR:USD", "IR:EUR"): .5,
                                                            ("IR:USD", "FX:EURUSD"): .2})
            campaign = dataclasses.replace(campaign, simulation=simulation)
        else:
            campaign = dataclasses.replace(campaign, covariance=np.kron([[1., .5], [.5, 1.]], campaign.covariance))
    if grid != "recorded":
        nx, sx = {"coarse": (8, 3.), "default": (30, 5.), "refined": (48, 6.)}[grid]
        pricing = dataclasses.replace(campaign.pricing,
            bermudan=dataclasses.replace(campaign.pricing.bermudan, n_per_std=nx, std_devs=sx),
            american=dataclasses.replace(campaign.pricing.american, n_per_std=nx, std_devs=sx))
        campaign = dataclasses.replace(campaign, pricing=pricing)
    if cases:
        campaign = dataclasses.replace(campaign, cases=tuple(c for c in campaign.cases if any(s in c.label for s in cases)))
        if not campaign.cases:
            raise ValueError(f"no matching precision cases for {name}")
    suffix = f"fixture-{fixture_seed if fixture_seed is not None else 'recorded'}-{model}-{grid}"
    return dataclasses.replace(campaign, name=f"{name}-{suffix}" + ("-USD-EUR" if multi_currency else ""))


def worker(point, destination):
    backend = point["backend"]
    os.environ["JAX_PLATFORMS"] = {"cpu": "cpu", "gpu": "cuda,cpu", "tpu": "tpu,cpu"}[backend]
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from engine.api.worker import deterministic_kernels_environment
    os.environ.update(deterministic_kernels_environment(os.environ))
    if backend == "gpu":
        from engine.api.worker import gemm_environment
        os.environ.update(gemm_environment("cublas", os.environ))
    import jax
    from tests.support.precision_harness import measure
    from tests.support.precision_campaigns import write
    from tests.support.precision_timing import execution_environment, workload_settings
    from tests.support.precision_evidence import Standard, rows_of

    devices = jax.devices()
    if devices[0].platform != backend:
        raise ValueError(f"requested {backend}, got {devices[0].platform}")
    if jax.config.jax_disable_jit:
        raise ValueError("qualification requires JIT execution")
    if point.get("expected_gpu") and backend == "gpu" and any(
            point["expected_gpu"].lower() not in d.device_kind.lower() for d in devices):
        raise ValueError(f"expected {point['expected_gpu']}, got {[d.device_kind for d in devices]}")
    if source_digest() != point["source_digest"]:
        raise ValueError("source changed after campaign scheduling")
    campaign = build_campaign(point["campaign"], point["fixture_seed"], point["model"], point["grid"], point["cases"],
                              multi_currency=point["multi_currency"])
    campaign = dataclasses.replace(campaign, seeds=tuple(point["seeds"]))
    with jax.default_device(devices[0]):
        result = measure(campaign, point["paths"], progress=lambda s: print(s, flush=True))
    write(result, destination)
    rows = rows_of(result, Standard.load())
    atomic_json(destination / "result.json", {"status": "invalid" if any(r.recompiles for r in result.runs) else "ok",
        "request": point, "measurement": f"{result.campaign}.json.gz",
        "environment": execution_environment(), "workload": workload_settings(campaign),
        "figures": len(rows), "passing_figures": sum(r.ceiling is not None for r in rows),
        "scope": "matched-path precision acceptance only; sampling/model/discretization accuracy require separate checks"})
    return 0


def report(directory, requests):
    """Retain every failed point and derive contiguous measured ceilings; no promotion."""
    from tests.support.precision_harness import Measurements
    from tests.support.precision_evidence import Standard, generate
    groups, results = {}, []
    for request in requests:
        root = directory / request["directory"]
        result = read_json(root / "result.json") if (root / "result.json").exists() else {
            "status": "pending", "request": request}
        if result["request"] != request:
            raise ValueError(f"result does not match request: {root}")
        results.append(result)
        if result["status"] == "ok":
            path = (root / result["measurement"]).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError("measurement escapes experiment directory")
            measurement = Measurements.load(path)
            key = measurement.campaign
            groups[key] = groups[key].merge(measurement) if key in groups else measurement
    # A failed or pending smaller count cannot disappear from a published ceiling.
    incomplete = {r["request"]["directory"].rsplit("/", 1)[0] for r in results if r["status"] != "ok"}
    eligible = [m for name, m in groups.items() if name not in incomplete]
    if eligible:
        markdown, evidence = generate(eligible, Standard.load())
        (directory / "candidate-evidence.md").write_text(markdown, encoding="utf-8")
        (directory / "candidate-evidence.json").write_text(evidence, encoding="utf-8")
    else:
        for name in ("candidate-evidence.md", "candidate-evidence.json"):
            (directory / name).unlink(missing_ok=True)
    atomic_json(directory / "qualification.json", {"schema_version": 1, "points": results,
        "publication": "experimental candidate evidence, not the shipped acceptance table"})
    lines = ["# Accuracy qualification", "", "Precision acceptance on matched paths; other error sources remain separate.",
             "Candidate evidence is experimental and is never promoted automatically.", "",
             "| Experiment | Status | Passing / measured figure groups |", "|---|---|---|"]
    for r in results:
        lines.append(f"| {r['request']['directory']} | {r['status']} | {r.get('passing_figures', '—')} / {r.get('figures', '—')} |")
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--campaigns", nargs="+", default=["compute", "compute_market_risk"])
    parser.add_argument("--tier", choices=TIERS, default="reproduce")
    parser.add_argument("--paths", type=int, nargs="+")
    parser.add_argument("--seed-ids", type=int, nargs="+")
    parser.add_argument("--fixture-seeds", type=int, nargs="+")
    parser.add_argument("--models", nargs="+", choices=("LGM", "HullWhite"), default=["LGM"])
    parser.add_argument("--grids", nargs="+", choices=("recorded", "coarse", "default", "refined"), default=["recorded"])
    parser.add_argument("--cases", nargs="+", help="case-label substrings; no matches is an error")
    parser.add_argument("--multi-currency", action="store_true", help="correlated USD/EUR portfolio, including simulated FX for exposure")
    parser.add_argument("--backend", choices=("cpu", "gpu", "tpu"), default="cpu")
    parser.add_argument("--expected-gpu", help="require an H100 or other explicit GPU model")
    parser.add_argument("--timeout", type=float, default=3600.)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(json.loads(args.worker), args.out)
    paths = args.paths or TIERS[args.tier][0]
    seeds = args.seed_ids or list(range(1, TIERS[args.tier][1] + 1))
    for values in (paths, seeds, args.campaigns, args.models, args.grids, args.fixture_seeds or [None]):
        if len(values) != len(set(values)):
            parser.error("duplicate experimental selections")
    if any(n < 32 or n & (n - 1) for n in paths) or len(seeds) < 3 or any(s < 0 or s >= 2 ** 32 for s in seeds):
        parser.error("use power-of-two counts >=32 and at least three distinct uint32 seeds")
    if args.fixture_seeds and any(s < 0 or s >= 2 ** 32 for s in args.fixture_seeds):
        parser.error("fixture seed identities must fit uint32")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    from tests.support.precision_campaigns import CAMPAIGNS
    if any(name not in CAMPAIGNS for name in args.campaigns):
        parser.error("unknown campaign")
    digest, requests = source_digest(), []
    for name in args.campaigns:
        for fixture in args.fixture_seeds or [None]:
            for model in (args.models if CAMPAIGNS[name]().kind == "portfolio" else ["LGM"]):
                for grid in args.grids:
                    campaign = build_campaign(name, fixture, model, grid, args.cases, multi_currency=args.multi_currency)
                    for count in sorted(paths):
                        requests.append({"campaign": name, "fixture_seed": fixture, "model": model, "grid": grid,
                            "paths": count, "seeds": list(seeds), "backend": args.backend, "cases": args.cases,
                            "expected_gpu": args.expected_gpu, "source_digest": digest,
                            "multi_currency": args.multi_currency,
                            "directory": f"{campaign.name}/{count}"})
    manifest = {"schema_version": 1, "requests": requests}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "manifest.json"
    if args.resume:
        if not path.exists() or read_json(path) != manifest:
            parser.error("resume requires an identical source and experiment manifest")
    elif any(args.out.iterdir()):
        parser.error("output directory must be empty (or use --resume for the identical experiment)")
    atomic_json(path, manifest)
    for point in requests:
        destination = args.out / point["directory"]
        result_path = destination / "result.json"
        if args.resume and result_path.exists() and read_json(result_path).get("status") == "ok":
            continue
        command = [sys.executable, "-m", "benchmarks.qualification", "--out", str(destination),
                   "--worker", json.dumps(point)]
        code = launch_worker(command, destination, args.timeout)
        if code or not result_path.exists():
            atomic_json(result_path, {"status": "failed", "request": point,
                                     "reason": f"worker exit {code}; see worker.log"})
        report(args.out, requests)
    results = report(args.out, requests)
    return int(any(r["status"] != "ok" for r in results))


if __name__ == "__main__":
    sys.exit(main())

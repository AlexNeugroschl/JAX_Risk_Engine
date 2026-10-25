"""Benchmark-only ORE worker behind the production /runs API and job queue.

ORE executes NPV and trade-level HISTSIM_VAR in disjoint native batches. Portfolio
P&L and RiskStatistics reuse those results; the two-run correctness oracle is not timed.
No production endpoint or pricing implementation is replaced.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path

# ORE's historical analytic indexes scenarios by calendar dates. Disjoint batches
# avoid exhausting QuantLib's supported date range at large scenario counts.
ORE_BATCH_SCENARIOS = 4096


def ore_result(schema):
    import numpy as np
    import ORE

    from engine.precision import Precision
    from engine.risk.market.run import MarketRiskResult
    from engine.run import RunResult, engine_identity
    from tests.support.ore_market_risk_oracle import _rows, _run, business_days_before

    request = schema.check()
    if request.analytics != ("market_risk",):
        raise ValueError("ORE benchmark supports the market_risk analytic alone")
    if request.config.precision != Precision():
        raise ValueError("ORE benchmark supports float64 only")
    source = schema.config.market_risk.scenarios
    if source.source != "historical":
        raise ValueError("ORE benchmark needs historical levels")
    if request.config.base_currency != "USD":
        raise ValueError("ORE benchmark reports in USD only")
    ids = [t.trade_id for t in request.portfolio]
    levels = np.asarray(source.history)
    horizon = source.horizon_days
    count = len(levels) - horizon
    parts, base = [], None
    for start in range(0, count, ORE_BATCH_SCENARIOS):
        stop = min(start + ORE_BATCH_SCENARIOS, count)
        history = levels[start:stop + horizon]
        dates = ([ORE.DateParser.parseISO(d) for d in source.dates[start:stop + horizon]]
                 if source.dates is not None else business_days_before(request.market.asof, len(history)))
        # Each scenario is valued exactly once. NPV supplies the base anchors in
        # the first batch; later batches never rerun the earlier scenarios.
        reports = _run(request.market, request.portfolio, history, dates,
                       horizon, [1 - q for q in request.config.market_risk.quantiles],
                       request.config.pricing, True, include_npv=start == 0)
        rows = _rows(reports["historical_PnL"])
        windows = list(dict.fromkeys((d1, d2) for d1, d2, _, _ in rows))
        expected = [(dates[i], dates[i + horizon]) for i in range(stop - start)]
        if windows != expected:
            raise ValueError("ORE returned different scenario windows")
        by_trade = {(d1, d2, trade): value for d1, d2, trade, value in rows}
        if len(by_trade) != len(windows) * len(ids):
            raise ValueError("ORE returned missing or duplicate trade P&L rows")
        parts.append(np.asarray([[by_trade[(d1, d2, trade)] for trade in ids] for d1, d2 in windows]))
        if start == 0:
            base_by_id = dict(zip(reports["npv"]["TradeId"], reports["npv"]["NPV"]))
            base = [float(base_by_id[trade]) for trade in ids]
    pnl = np.concatenate(parts)
    portfolio = pnl.sum(axis=1)
    stats = ORE.RiskStatistics()
    stats.add(ORE.DoubleVector(portfolio.tolist()))
    risk = {}
    for q in request.config.market_risk.quantiles:
        risk[f"VaR_{q * 100:g}"] = stats.valueAtRisk(q)
        risk[f"ES_{q * 100:g}"] = stats.expectedShortfall(q)
    result = MarketRiskResult(
        base_npv=sum(base), base_npv_per_trade=base, pnl=pnl, portfolio_pnl=portfolio, risk=risk,
        measure=request.config.market_risk.scenarios.measure, source="historical", horizon_days=source.horizon_days,
        num_scenarios=count, risk_factors=list(source.factors), base_currency="USD",
        uncertainty={"method": "not computed by ORE benchmark"})
    identity = dataclasses.replace(engine_identity(), version=f"ORE {ORE.__version__}")
    return RunResult(request=request, engine=identity, market_risk=result)


def ore_job(job):
    from engine.api.requests import RunRequestSchema
    from engine.api.results import run_result_document
    from engine.api.worker import JobResult

    schema = RunRequestSchema.model_validate_json(job.request)
    result = ore_result(schema)
    document, artifacts = run_result_document(result, schema, job.id, f"/runs/{job.id}/artifacts")
    # This identity is explicit even though the HTTP/queue plumbing is shared.
    body = document.model_dump(mode="json")
    body["benchmark_implementation"] = "ore"
    count = result.market_risk.num_scenarios
    body["benchmark_ore_app_runs"] = (count + ORE_BATCH_SCENARIOS - 1) // ORE_BATCH_SCENARIOS
    return JobResult(json.dumps(body, allow_nan=False), artifacts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("jax", "ore"), required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu", "tpu"), required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--expected-gpu")
    args = parser.parse_args(argv)
    from demos.benchmark_reporting import atomic_json
    from engine.api.worker import serve
    from tests.support.precision_timing import execution_environment
    import jax

    try:
        devices = jax.devices()
        if not devices or devices[0].platform != args.backend:
            raise RuntimeError(f"requested {args.backend}, got {devices}")
        if args.expected_gpu and args.backend == "gpu" and any(
                args.expected_gpu.lower() not in d.device_kind.lower() for d in devices):
            raise ValueError(f"expected GPU {args.expected_gpu}, got {devices}")
    except RuntimeError as exc:
        atomic_json(args.metadata, {"status": "unmeasured", "reason": str(exc)})
        return 2
    if jax.config.jax_disable_jit and args.implementation == "jax":
        raise ValueError("API comparison requires JIT; JAX_DISABLE_JIT is set")
    atomic_json(args.metadata, {"status": "ok", "implementation": args.implementation,
                               "backend": args.backend,
                               "devices": [f"{d.platform}:{d.id} ({d.device_kind})" for d in devices],
                               "environment": execution_environment(), "pid": os.getpid()})
    return serve(args.queue, lock_path=args.lock, parent_pid=args.parent_pid,
                 price=ore_job if args.implementation == "ore" else None)


if __name__ == "__main__":
    raise SystemExit(main())

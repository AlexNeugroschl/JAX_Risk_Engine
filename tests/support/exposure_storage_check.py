"""Fresh-process I-105 memory measurements and numeric snapshots.

Run each memory point in a new process; CPU peak RSS includes preparation, compilation,
pricing and reductions. GPU allocator peaks, when available, are reported separately.
Snapshots cover the shared sloped portfolio, Greeks, mixed storage and paired figures.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path

import jax
import numpy as np
import ORE

from engine.precision import Precision, StagePrecision
from engine.run import CamConfig, HullWhiteConfig, PortfolioRequest, RunConfig, price_portfolio
from tests.support import portfolio as shared

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(__file__).resolve().parents[2] / ".jax_cache"))
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")


def _peak_rss():
    if os.name != "nt":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("peak_working_set", "working_set", "peak_paged_pool",
                                               "paged_pool", "peak_nonpaged_pool", "nonpaged_pool",
                                               "pagefile", "peak_pagefile")]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    process = ctypes.windll.kernel32.GetCurrentProcess
    process.restype = wintypes.HANDLE
    measure = ctypes.windll.psapi.GetProcessMemoryInfo
    measure.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not measure(process(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return counters.peak_working_set


def _arrays(value, prefix=""):
    if dataclasses.is_dataclass(value):
        for f in dataclasses.fields(value):
            # Policy and device labels are metadata, not numerical results.
            if f.name not in ("policy", "trades", "products"):
                yield from _arrays(getattr(value, f.name), f"{prefix}/{f.name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _arrays(item, f"{prefix}/{key}")
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            yield from _arrays(item, f"{prefix}/{i}")
    elif isinstance(value, (jax.Array, np.ndarray, float, int)):
        yield prefix, np.asarray(value)


def _request(paths, trades, storage, paired, output=True):
    dates = tuple(shared.ASOF + ORE.Period(m, ORE.Months) for m in range(6, 73, 6))
    simulation = CamConfig(dates=dates, base_currency="USD", ir={"USD": HullWhiteConfig(0.03, 0.01)},
                           samples=paths, seed=11)
    precision = Precision(pricing=StagePrecision(storage=storage), paired_fraction=paired)
    request = PortfolioRequest(shared.market(), trades, RunConfig(simulation=simulation, precision=precision))
    # Also works on pre-I-105 code, where the pipeline ignores this attribute.
    request.include_npv_cube = output
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--paths", type=int, default=16384)
    parser.add_argument("--trades", type=int, default=24)
    parser.add_argument("--storage", default="float64")
    parser.add_argument("--paired", type=float, default=0.0)
    parser.add_argument("--cube", action="store_true")
    args = parser.parse_args()
    if args.snapshot:
        trades = [shared.trades()[n] for n in ("swap-payer", "european-payer", "bermudan-payer-physical",
                                               "american-payer", "bond")]
        arrays = {}
        for name, precision in (
            ("float64", Precision()), ("float32", Precision.throughout("float32")),
            ("mixed-paired", Precision(pricing=StagePrecision(storage="float8_e4m3fn"),
                                       by_trade={"bond": StagePrecision(storage="float32")}, paired_fraction=0.25)),
        ):
            request = _request(64, trades, "float64", 0)
            request.compute_greeks = True
            request.config = dataclasses.replace(request.config, precision=precision)
            result = price_portfolio(request)
            arrays.update(_arrays(result, name))
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.snapshot, **arrays)
        print(json.dumps({"arrays": len(arrays), "snapshot": str(args.snapshot)}))
        return

    trade = shared.trades()["swap-payer"]
    trades = [dataclasses.replace(trade, trade_id=f"swap-{i}", fixed_rate=0.025 + i * 0.0002,
                                  payer=i % 2 == 0) for i in range(args.trades)]
    request = _request(args.paths, trades, args.storage, args.paired, args.cube)
    result = price_portfolio(request)
    list(_arrays(result))  # synchronize all numeric results
    peak = _peak_rss()
    stats = jax.devices()[0].memory_stats() or {}
    print(json.dumps({"paths": args.paths, "trades": args.trades, "storage": args.storage,
                      "paired": args.paired, "cube": args.cube, "peak_rss_bytes": peak,
                      "device_peak_bytes": stats.get("peak_bytes_in_use"),
                      "epe": np.asarray(result.exposure.epe).tolist()}))


if __name__ == "__main__":
    main()

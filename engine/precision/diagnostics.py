"""
Range diagnostics of products below float32 (`ProductOptions.diagnostics="counts"`, F-07): how
many of a stage's operand values the compute format could not hold, counted inside the compiled
kernels and reported per stage in the analytic's `PrecisionReport.product_ranges`.

    PrecisionReport.product_ranges
      {stage: {operand: {count: n}}}
        stage     "simulation", "market", "pricing", or a pricing override's
                  "by_product[<product>]" / "by_trade[<trade id>]"
        operand   "per_path" (the operand each path has its own of) or "shared" (the one every
                  path shares: `engine.precision.products`, decision A-23)
        count     values       operand values rounded to the format (each product call's)
                  nonfinite    of them already NaN or infinite before rounding
                  overflow     finite, but beyond the format's largest after scaling: NaN for
                               E4M3 (it has no infinity), infinite for E5M2 and FP16
                  underflow    nonzero, rounded to zero
                  subnormal    rounded to a nonzero value below the format's smallest normal,
                               where it keeps fewer significant bits

The counts are of the first term of an operand (its own rounding), after the scaling the stage's
product options select (`scaling="power_of_two"` brings each row's largest into range, so it
overflows nothing finite; without it, an E4M3 operand overflows above 448). Residual terms
(`shared_terms > 1`) round what the first leaves and are not counted.

**How.** With diagnostics on, each product rounding operands to a grid sends five integers to
the host (`jax.experimental.io_callback`) once per call; under `vmap` its counts are summed over
the mapped axis first (`jax.custom_batching.custom_vmap`), so a batch of scenarios is one call and
the program is not unrolled. A product inside a scan reports once per step. Off (the default),
nothing is traced and nothing changes; on, the stage's programs are compiled again, and the
transfers cost time: a measurement setting, not a production one.

**One collection at a time.** The callbacks run on XLA's threads, so the counts go to one
collection per process: an analytic with diagnostics on refuses to start while another run in
the process is collecting (`RuntimeError`), rather than mixing their counts. Products outside a
collection (a direct kernel call) are not counted anywhere.
"""
import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import io_callback

#: The counts of one operand, in the order a product sends them.
COUNTS = ("values", "nonfinite", "overflow", "underflow", "subnormal")
#: The operand roles (`engine.precision.products`).
OPERANDS = ("per_path", "shared")

_LOCK = threading.Lock()
_ACTIVE: Optional[Dict[str, Dict[str, np.ndarray]]] = None


class RangeCollection:
    """The counts of one collection (`collecting`): `counts` once it has ended."""

    def __init__(self):
        self.counts: Dict[str, Dict[str, Dict[str, int]]] = {}


@contextmanager
def collecting(enabled: bool) -> Iterator[Optional[RangeCollection]]:
    """Collect the products' range counts while the block runs (None when not `enabled`); the
    collection's `counts` are filled when the block ends, after every callback has run."""
    global _ACTIVE
    if not enabled:
        yield None
        return
    with _LOCK:
        if _ACTIVE is not None:
            raise RuntimeError("another run in this process is collecting product range diagnostics "
                               "(ProductOptions.diagnostics); run them one after another")
        _ACTIVE = {}
    collection = RangeCollection()
    try:
        yield collection
        jax.effects_barrier()
        with _LOCK:
            collection.counts = {stage: {operand: dict(zip(COUNTS, (int(v) for v in totals)))
                                         for operand, totals in operands.items()}
                                 for stage, operands in _ACTIVE.items()}
    finally:
        with _LOCK:
            _ACTIVE = None


def operand_counts(x, rounded, min_exponent: int) -> jax.Array:
    """`[5]` int32: `COUNTS` of operand `x` and its rounding `rounded` (scaled, in the format's
    dtype or a carrier), the format's smallest normal `2^min_exponent`."""
    finite = jnp.isfinite(x)
    term = rounded.astype(jnp.float32)
    held = jnp.isfinite(term)
    smallest = jnp.asarray(2.0 ** min_exponent, jnp.float32)
    return jnp.stack([jnp.asarray(x.size), jnp.sum(~finite), jnp.sum(finite & ~held),
                      jnp.sum(finite & (x != 0) & (term == 0)),
                      jnp.sum(held & (term != 0) & (jnp.abs(term) < smallest))]).astype(jnp.int32)


def report(stage: str, operand: str, counts: jax.Array) -> None:
    """Send `counts` (`operand_counts`) of `stage`'s `operand` to the active collection (a
    product whose arithmetic names no stage, a kernel called directly, as "unlabelled")."""
    _reporter(stage or "unlabelled", operand)(counts)


def _record(stage: str, operand: str, counts) -> None:
    totals = np.asarray(counts, dtype=np.int64).reshape(-1, len(COUNTS)).sum(axis=0)
    with _LOCK:
        if _ACTIVE is None:
            return
        by_operand = _ACTIVE.setdefault(stage, {})
        by_operand[operand] = by_operand.get(operand, np.zeros(len(COUNTS), np.int64)) + totals


_REPORTERS: Dict = {}


def _reporter(stage: str, operand: str):
    """The reporting function of `stage`'s `operand`, one per pair (its `custom_vmap` rule sums a
    batch of counts before the one host call)."""
    key = (stage, operand)
    if key not in _REPORTERS:
        def send(counts):
            io_callback(lambda c: _record(stage, operand, c), None, counts, ordered=False)
            return jnp.zeros((), counts.dtype)

        reporter = jax.custom_batching.custom_vmap(send)

        @reporter.def_vmap
        def _summed(axis_size, in_batched, counts):
            reporter(jnp.sum(counts, axis=0) if in_batched[0] else counts * axis_size)
            return jnp.zeros((axis_size,), counts.dtype), True

        _REPORTERS[key] = reporter
    return _REPORTERS[key]

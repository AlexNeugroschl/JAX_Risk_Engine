"""
The precision report every result carries (docs/planning/details/precision.md §9.5): what a
run's precision was, read from what it did, and how far its figures are from float64.

    PrecisionReport
      policy        Precision                      the policy as run
      trades        {trade id: StagePrecision}     each trade's resolved pricing stage
      realized      {array: format}                the format of every stored array, read from
                                                   the arrays: "shocks", "states", "market",
                                                   "values/<trade id>" (the names `Precision.store`
                                                   gives them; one entry for the whole market)
      devices       ("cpu:0 (cpu)", ...)           the devices holding the run's stored arrays
      backend       "cpu" | "gpu" | "tpu"          the JAX backend of the process that ran it
      jax_version   str
      paths, paired_paths                          the run's paths (or scenarios), and how many
                                                   of them were re-run at float64
      figures       {figure: MeanEstimate | QuantileEstimate}
                                                   with a paired sample (`paired_fraction`),
                                                   per figure: "netting_set/EPE",
                                                   "trades/<trade id>/PFE_95", "portfolio/VaR_99"

`realized` and `devices` are read from the arrays, not the configuration, so the report says
what ran even if a cast point were wrong. Built in the process that ran the job, so an HTTP
job reports its worker's device, not the API's (I-12). An unvalidated combination's warning
per figure joins with roadmap 2.9's evidence table (decision A-11).

Depends on JAX and NumPy only.
"""
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Tuple, Union

import jax

from engine.precision.estimate import MeanEstimate, QuantileEstimate
from engine.precision.formats import name_of
from engine.precision.policy import Precision, StagePrecision
from engine.precision.storage import Stored

Estimate = Union[MeanEstimate, QuantileEstimate]


@dataclass(frozen=True)
class PrecisionReport:
    """See the module docstring."""
    policy: Precision
    trades: Mapping[str, StagePrecision]
    realized: Mapping[str, str]
    devices: Tuple[str, ...]
    backend: str
    jax_version: str
    paths: int
    paired_paths: int = 0
    figures: Mapping[str, Estimate] = field(default_factory=dict)

    @classmethod
    def of(cls, policy: Precision, trades: Iterable, realized: Mapping[str, str], arrays: Iterable,
           paths: int, paired_paths: int = 0, figures: Mapping[str, Estimate] = None) -> "PrecisionReport":
        """The report of a run of `trades` (each with `trade_id` and `product`) under `policy`,
        whose stored `arrays` (plain or `Stored`) give the devices it ran on."""
        return cls(policy=policy, trades={t.trade_id: policy.precision_for(t) for t in trades},
                   realized=dict(realized), devices=devices_of(arrays), backend=jax.default_backend(),
                   jax_version=jax.__version__, paths=paths, paired_paths=paired_paths, figures=dict(figures or {}))


def format_name(x) -> str:
    """The format a stored array (plain or `Stored`) is held in, read from its dtype."""
    return name_of(x.dtype)


def realized_format(arrays: Iterable) -> str:
    """The format of a class of stored arrays (the scenario market's); should they differ,
    every format found, joined with '+', so the report never hides a mixed class."""
    return "+".join(sorted({format_name(a) for a in arrays}))


def devices_of(arrays: Iterable) -> Tuple[str, ...]:
    """The devices holding `arrays`, as `"<platform>:<id> (<kind>)"`, or JAX's default device
    when there are none (a run with no paths computes on it)."""
    found = set()
    for a in arrays:
        found.update((a.values if isinstance(a, Stored) else a).devices())
    found = found or {jax.devices()[0]}
    return tuple(f"{d.platform}:{d.id} ({d.device_kind})" for d in sorted(found, key=lambda d: (d.platform, d.id)))

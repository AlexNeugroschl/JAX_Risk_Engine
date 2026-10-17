"""
The evidence table at run time (I-55; docs/planning/details/precision.md §10): which precision
combinations have passed the acceptance standard (decision A-11) for which figure, up to how
many paths, and the warning a result carries for every figure that has no passing row. The
standard labels results; it never refuses a run (D-9).

The table is generated, never typed: `python -m tests.support.precision_evidence` reads the
precision harness's measurements of record (`compliance/precision-measurements/`) and the
acceptance standard (`compliance/precision-acceptance.json`), and writes the table for people
(`compliance/precision-evidence.md`) and its rows for the engine (`evidence.json`, beside this
module, by case: `evidence_json`). A row:

    campaign, case   the campaign and case of the harness that measured it
    combination      the precision of the stages its figure depends on (`combination`)
    paired           the paired float64 sample's share of the paths, for a figure the sample
                     corrects (`RunFigures.corrected`); 0 otherwise
    scope            a product ("swap", ...: the worst of the campaign's trades of it) or the
                     aggregate ("netting_set", "portfolio")
    figure           the figure's kind: "NPV" (the cube's mean), "EPE", "PFE_95", "VaR_99", "PnL"
    ceiling          the path ceiling: the largest path count measured at which the row passes
                     and passes at every smaller one; None if it fails at the smallest
    products         for an aggregate, the products of the campaign's netting set or portfolio

**Validated.** A figure of a run is validated when the rows of its combination, scope and kind
have a ceiling of at least the run's path count (or scenario count). A trade's figure is its
product's; an aggregate's needs a row whose campaign held every product of the run, and one
pricing stage for every trade (no campaign measures a mix). A figure the paired sample corrects
takes the rows of the largest paired share at most the run's (a larger share only shrinks the
correction's error; with no paired sample, the rows without one); every other figure is the
run's own whatever the sample, and takes the rows without one. Several rows that match (two
campaigns measuring one combination) give their smallest ceiling. Fewer paths than a passing
row's are validated: on Sobol paths the precision error grows against the statistical error
with the path count in every campaign measured (§15.4), so a ceiling bounds from above.

A figure whose stages are all float64 is the reference itself and needs no row. Rows were
measured on the markets and portfolios the campaigns name, with every format below float32
emulated on a CPU, which computes what a device's native units compute (§8.3;
`engine.precision.product_execution` calls a format native only once its device check passes).

Plain Python, no JAX work: the portfolio pipeline and market risk call it once their figures
are computed, so it compiles nothing.
"""
import functools
import json
from dataclasses import dataclass
from importlib import resources
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple

from engine.precision.policy import Precision, StagePrecision

#: Where the table for people is, from the repository's root (each warning names it).
TABLE = "compliance/precision-evidence.md"

#: The rows' file, beside this module.
EVIDENCE_FILE = "evidence.json"


@dataclass(frozen=True)
class RunFigures:
    """The figures a run reports, as the evidence table names them (see the module docstring).

    stages: the adjustable stages the run reads, in pipeline order (market risk has no market
        stage). aggregate: the scope of its whole-portfolio figures. trade, total: the figure
        kinds of each trade and of the aggregate. corrected: the kinds a paired sample corrects.
        unit: what the run counts, "paths" or "scenarios". legend: what a figure's name means
        where a reader could take it for another figure, closing each warning.
    """
    stages: Tuple[str, ...]
    aggregate: str
    trade: Tuple[str, ...]
    total: Tuple[str, ...]
    corrected: FrozenSet[str] = frozenset()
    unit: str = "paths"
    legend: str = ""


@dataclass(frozen=True)
class Row:
    """One row of the table (see the module docstring)."""
    campaign: str
    case: str
    combination: str
    paired: float
    scope: str
    figure: str
    ceiling: Optional[int]
    products: Tuple[str, ...] = ()


def combination(precision: Precision, stages: Sequence[str], pricing: StagePrecision,
                replicates: int = 1) -> Optional[str]:
    """The combination a figure is validated under: each of `stages` as
    `"<stage>=<storage>/<compute>/<accumulate>"` (the pricing stage `pricing`, a trade's own or
    the aggregate's), then the rounding and block layout if one of them stores in a scaled
    format (they change nothing otherwise). Stochastic compute rounding and nondefault
    replicate counts also enter the key. None when every stage is float64: the reference."""
    chosen = [(s, pricing if s == "pricing" else getattr(precision, s)) for s in stages]
    if all(stage == StagePrecision() for _, stage in chosen):
        return None
    parts = [f"{s}={stage.storage}/{stage.compute}/{stage.accumulate}" for s, stage in chosen]
    parts += [f"{s}.compute_rounding={stage.compute_rounding}" for s, stage in chosen
               if stage.compute_rounding != "nearest"]
    if any(stage.scaled_storage for _, stage in chosen):
        parts += [f"rounding={precision.rounding}", f"offset={precision.offset}"]
    if replicates != 1:
        parts.append(f"replicates={replicates}")
    return " ".join(parts)


class Evidence:
    """The table's rows, indexed for `ceiling`; `shipped()` is the engine's own."""

    def __init__(self, rows: Iterable[Row], campaigns: Mapping[str, Mapping[str, str]] = None):
        self.rows = tuple(rows)
        self.campaigns = dict(campaigns or {})
        self._index: Dict[Tuple[str, str, str], List[Row]] = {}
        for row in self.rows:
            self._index.setdefault((row.combination, row.scope, row.figure), []).append(row)

    @classmethod
    def from_json(cls, text: str) -> "Evidence":
        """The rows of an evidence file (`evidence_json`'s layout)."""
        data = json.loads(text)
        campaigns = data["campaigns"]
        rows = []
        for case in data["cases"]:
            campaign = campaigns[case["campaign"]]
            corrected = set(campaign["corrected"])
            for scope, ceilings in case["ceilings"].items():
                products = tuple(campaign["products"]) if scope == campaign["aggregate"] else ()
                for figure, ceiling in ceilings.items():
                    rows.append(Row(case["campaign"], case["case"], case["combination"],
                                    case["paired"] if figure in corrected else 0.0, scope, figure, ceiling, products))
        return cls(rows, campaigns)

    def ceiling(self, combination: str, scope: str, figure: str, paired: float = 0.0,
                products: Optional[Iterable[str]] = None) -> Tuple[bool, Optional[int]]:
        """Whether any row measures `figure` of `scope` under `combination` with the paired
        share `paired` (see the module docstring; 0 for a figure the sample does not correct),
        and the path ceiling they give (None if one of them passes at no path count). With
        `products`, only rows whose campaign held each of them count (an aggregate's)."""
        rows = self._index.get((combination, scope, figure), [])
        if products is not None:
            rows = [r for r in rows if set(products) <= set(r.products)]
        if paired > 0:
            rows = [r for r in rows if 0 < r.paired <= paired]
        else:
            rows = [r for r in rows if r.paired == 0]
        if not rows:
            return False, None
        nearest = max(r.paired for r in rows)
        ceilings = [r.ceiling for r in rows if r.paired == nearest]
        return True, (None if None in ceilings else min(ceilings))


def evidence_json(campaigns: Mapping[str, Mapping], cases: Sequence[Mapping]) -> str:
    """An evidence file: `campaigns`, each campaign's description by name (at least its
    `aggregate` scope, the `products` it priced and the figure kinds a paired sample `corrected`,
    which `Evidence.from_json` reads, and what names its rows: market, revision, device, seeds),
    and `cases`, each `{"campaign", "case", "combination", "paired": the case's paired share,
    "ceilings": {scope: {figure: ceiling}}}`; one case to a line, so a regenerated file's diff
    shows the cases that moved."""
    lines = ['{"generated_by": "python -m tests.support.precision_evidence",',
             f' "campaigns": {json.dumps(campaigns, sort_keys=True)},', ' "cases": [']
    lines.append(",\n".join(f"  {json.dumps(case, sort_keys=True)}" for case in cases))
    return "\n".join(lines) + "\n ]}\n"


@functools.lru_cache(maxsize=1)
def shipped() -> Evidence:
    """The engine's evidence table (`EVIDENCE_FILE`), read once per process."""
    return Evidence.from_json(resources.files(__package__).joinpath(EVIDENCE_FILE).read_text(encoding="utf-8"))


def evidence_warnings(precision: Precision, trades: Sequence, paths: int, figures: RunFigures,
                      evidence: Optional[Evidence] = None, *, replicates: int = 1) -> List[str]:
    """The warnings of a run of `trades` (each with `trade_id` and `product`) at `paths` paths
    or scenarios under `precision`: one for each product, and one for the aggregate, whose
    figures in `figures` have no passing row in `evidence` (the engine's own by default), each
    saying what is validated and up to how many paths. None for a run at float64: it is the
    reference."""
    evidence = shipped() if evidence is None else evidence
    out = []
    by_product: Dict[Tuple[str, Optional[str]], List[str]] = {}
    for trade in trades:
        key = combination(precision, figures.stages, precision.precision_for(trade), replicates)
        if key is not None:
            by_product.setdefault((trade.product, key), []).append(trade.trade_id)
    for (product, key), ids in by_product.items():
        missing = _missing(evidence, key, product, figures.trade, paths, precision, figures)
        if missing:
            out.append(_warning(f"trades {ids} ({product})", key, paths, precision, figures, missing))
    stages = {precision.precision_for(t) for t in trades}
    if len(stages) > 1:
        out.append(f"precision: the {_aggregate(figures)} has no row in the evidence table ({TABLE}): its trades' "
                   f"pricing stages differ ({sorted(_stage_text(s) for s in stages)}), and no campaign measures one "
                   f"of mixed stages (acceptance standard, decision A-11). The run is not refused (D-9).")
    elif stages:
        key = combination(precision, figures.stages, stages.pop(), replicates)
        if key is not None:
            products = sorted({t.product for t in trades})
            missing = _missing(evidence, key, figures.aggregate, figures.total, paths, precision, figures, products)
            if missing:
                out.append(_warning(f"the {_aggregate(figures)} (products {products})", key, paths, precision, figures,
                                    missing))
    return out


def paired_error_warning(precision: Precision, replicates: int = 1) -> Optional[str]:
    """A paired RQMC mean needs independent scrambles to estimate its error (I-111)."""
    if not precision.paired_fraction:
        return None
    if replicates > 1:
        return (f"precision: standard errors use {replicates} independent scrambles as variance diagnostics. "
                "Confidence intervals are unavailable: no validated tail bound, including when the empirical "
                "standard error is zero (I-111); see docs/reference/sampling-uncertainty.md.")
    return ("precision: standard errors are unavailable with one Sobol scramble; set simulation.replicates "
            "to at least 2 independent scrambles (8 or more recommended; I-111).")


def _missing(evidence: Evidence, key: str, scope: str, kinds: Sequence[str], paths: int, precision: Precision,
             figures: RunFigures, products: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """The kinds without a passing row, grouped by what the table says of them."""
    missing: Dict[str, List[str]] = {}
    for kind in kinds:
        paired = precision.paired_fraction if kind in figures.corrected else 0.0
        measured, ceiling = evidence.ceiling(key, scope, kind, paired, products)
        if measured and ceiling is not None and paths <= ceiling:
            continue
        status = ("not measured" if not measured else f"fails at every {figures.unit[:-1]} count measured"
                  if ceiling is None else f"validated up to {ceiling:,} {figures.unit}")
        missing.setdefault(status, []).append(kind)
    return missing


def _warning(who: str, key: str, paths: int, precision: Precision, figures: RunFigures,
             missing: Mapping[str, Sequence[str]]) -> str:
    paired = (f" with a paired sample of {precision.paired_fraction:g} of the paths"
              if precision.paired_fraction and any(k in figures.corrected for ks in missing.values() for k in ks)
              else "")
    statuses = "; ".join(f"{', '.join(kinds)} {status}" for status, kinds in missing.items())
    legend = f" {figures.legend}" if figures.legend else ""
    return (f"precision: no passing row in the evidence table ({TABLE}) for {who} at {paths:,} {figures.unit} under "
            f"{key}{paired}: {statuses} (acceptance standard, decision A-11). The run is not refused (D-9).{legend}")


def _aggregate(figures: RunFigures) -> str:
    return figures.aggregate.replace("_", " ")


def _stage_text(stage: StagePrecision) -> str:
    return f"{stage.storage}/{stage.compute}/{stage.accumulate}"

"""
The precision evidence table (I-55; docs/planning/details/precision.md §10.3), generated from
the precision harness's measurements of record and the acceptance standard, never typed:

    .venv/Scripts/python.exe -m tests.support.precision_evidence [--check]

reads `compliance/precision-measurements/<campaign>.json.gz` for every campaign of record
(`tests/support/precision_campaigns.py`) and `compliance/precision-acceptance.json` (decision
A-11's thresholds, each with its citation), and writes

    compliance/precision-evidence.md    the table for people
    engine/precision/evidence.json      its rows for the engine, which warns of every figure of
                                        a run without a passing row (`engine.precision.evidence`)

`--check` writes nothing and fails if either file differs from what the measurements give;
`tests/test_precision_evidence.py` runs it, so a campaign rerun without regenerating fails.
Rerun this after any campaign.

    Measurements (FigureRow per case, figure and path count; AttributionRow per market-risk case)
      --rows_of--> EvidenceRow per case, scope and figure kind: the combination the case's
                   policy gives the stages its figures depend on (`engine.precision.evidence.
                   combination`), and per path count a Point:
          relative_error  the worst over the scope's figures (a product's trades, or the
                          aggregate) of the root mean square precision error over the seeds
                          in units of the figure's statistical error, on entries float64 does not
                          hold fixed (the harness's `FigureRow.relative_error`)
          fixed_error     the worst error on entries float64 holds fixed, relative to the
                          figure's scale (`FigureRow.fixed_error`)
          green           market risk: the P&L attribution test's green zone at every seed
          passes          relative_error <= the statistical rule, fixed_error <= the fixed
                          rule, and for market risk green (A-11); NaN fails
      --ceiling--> the largest path count at which a row passes and passes at every smaller one

The harness measures a figure's precision error across independent seeds, so no row rests on
the paired estimator's own standard errors (I-111): a paired case is judged as any other.
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from engine.api.requests import PrecisionSchema
from engine.precision.evidence import EVIDENCE_FILE, TABLE, RunFigures, combination, evidence_json
from engine.risk.market.run import run_figures as market_risk_figures
from engine.run.pipeline import run_figures as portfolio_figures
from tests.support.precision_campaigns import CAMPAIGNS, RECORD
from tests.support.precision_harness import FigureRow, Measurements

#: The repository's root.
ROOT = Path(__file__).resolve().parents[2]
#: The acceptance standard (decision A-11), every threshold with its citation.
ACCEPTANCE = ROOT / "compliance" / "precision-acceptance.json"
#: The two files this writes.
MARKDOWN = ROOT / TABLE
EVIDENCE = ROOT / "engine" / "precision" / EVIDENCE_FILE

#: The figures of each kind of campaign, as the engine names them when it warns.
FIGURES = {"portfolio": portfolio_figures(()), "market_risk": market_risk_figures(())}


# ---------------------------------------------------------------------------
# The standard
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Standard:
    """Decision A-11's thresholds (see `compliance/precision-acceptance.json`): the statistical
    rule's share of the statistical error, the fixed rule's share of a figure's scale, and the
    P&L attribution test's green zone (Spearman above, Kolmogorov-Smirnov below)."""
    statistical: float
    fixed: float
    spearman_above: float
    ks_below: float
    text: Mapping[str, Mapping[str, str]]

    @classmethod
    def load(cls, path: Path = ACCEPTANCE) -> "Standard":
        """The standard of `path`, every rule of which must cite its source."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for rule in ("statistical", "fixed", "pla"):
            if not data[rule].get("cite"):
                raise ValueError(f"{path}: the {rule!r} rule has no citation (Basel plan §4, rule 1)")
        return cls(statistical=float(data["statistical"]["value"]), fixed=float(data["fixed"]["value"]),
                   spearman_above=float(data["pla"]["spearman_above"]), ks_below=float(data["pla"]["ks_below"]),
                   text={rule: data[rule] for rule in ("statistical", "fixed", "pla")})

    def green(self, spearman: Sequence[float], kolmogorov_smirnov: Sequence[float]) -> bool:
        """MAR32.42(1)'s green zone at every seed: each correlation above, each statistic below
        (strictly, as the text says; NaN is not green)."""
        return (all(s > self.spearman_above for s in spearman)
                and all(k < self.ks_below for k in kolmogorov_smirnov))


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Point:
    """One row at one path count (see the module docstring); `worst` is the measured figure
    (with its trade) of the largest relative error."""
    paths: int
    relative_error: float
    fixed_error: float
    significance: float
    worst: str
    green: Optional[bool]
    passes: bool


@dataclass(frozen=True)
class EvidenceRow:
    """One figure kind of one scope under one case's combination, at every path count measured."""
    campaign: str
    case: str
    combination: str
    paired: float
    scope: str
    figure: str
    points: Tuple[Point, ...]

    @property
    def ceiling(self) -> Optional[int]:
        """The largest path count at which the row passes and at every smaller one; None if it
        fails at the smallest."""
        ceiling = None
        for point in sorted(self.points, key=lambda p: p.paths):
            if not point.passes:
                break
            ceiling = point.paths
        return ceiling


def rows_of(m: Measurements, standard: Standard) -> List[EvidenceRow]:
    """The evidence rows of one campaign's measurements, in case order, then scope and figure
    in the order measured; a case at float64 everywhere is the reference and gives none."""
    figures = FIGURES[m.kind]
    rows = []
    for label, policy in m.cases.items():
        precision = PrecisionSchema(**policy).to_dataclass()
        if precision.by_product or precision.by_trade:
            raise ValueError(f"{m.campaign}/{label}: a case with overrides has a combination per product; the "
                             f"evidence table reads uniform cases only")
        key = combination(precision, figures.stages, precision.pricing)
        if key is None:
            continue
        groups: Dict[Tuple[str, str], Dict[int, List[FigureRow]]] = {}
        for r in m.figures:
            if r.case == label:
                groups.setdefault(_scope_and_kind(r.figure, m, figures), {}).setdefault(r.paths, []).append(r)
        green = {a.paths: standard.green(a.spearman, a.kolmogorov_smirnov) for a in m.attribution if a.case == label}
        for (scope, kind), by_paths in groups.items():
            points = tuple(_point(paths, by_paths[paths], green.get(paths), m.kind == "market_risk", standard)
                           for paths in sorted(by_paths))
            paired = precision.paired_fraction if kind in figures.corrected else 0.0
            rows.append(EvidenceRow(m.campaign, label, key, paired, scope, kind, points))
    return rows


def _scope_and_kind(figure: str, m: Measurements, figures: RunFigures) -> Tuple[str, str]:
    """A measured figure's scope (its trade's product, or the aggregate) and kind."""
    parts = figure.split("/")
    if parts[0] == "trades":
        return m.products[parts[1]], parts[2]
    if parts[0] != figures.aggregate:
        raise ValueError(f"{m.campaign}: figure {figure!r} is neither a trade's nor the {figures.aggregate}'s")
    return parts[0], parts[1]


def _point(paths: int, rows: Sequence[FigureRow], green: Optional[bool], needs_green: bool,
           standard: Standard) -> Point:
    worst = max(rows, key=lambda r: _sortable(r.relative_error))
    relative = worst.relative_error
    fixed = max((r.fixed_error for r in rows), key=_sortable)
    passes = (relative <= standard.statistical and fixed <= standard.fixed
              and (green is True if needs_green else True))
    return Point(paths=paths, relative_error=relative, fixed_error=fixed,
                 significance=max((r.significance for r in rows), key=_sortable), worst=worst.figure,
                 green=green, passes=passes)


def _sortable(x: float) -> float:
    """NaN sorts first, so a worst case is NaN where any is (the harness's convention)."""
    return math.inf if math.isnan(x) else x


# ---------------------------------------------------------------------------
# The two files
# ---------------------------------------------------------------------------
def load_record(record: Path = RECORD) -> List[Measurements]:
    """Every campaign of record's measurements, in `CAMPAIGNS`' order; a missing one is an
    error, since the table describes them all."""
    return [Measurements.load(record / f"{name}.json.gz") for name in CAMPAIGNS]


def generate(measurements: Sequence[Measurements], standard: Standard) -> Tuple[str, str]:
    """The table for people and the engine's evidence file, as text."""
    rows = {m.campaign: rows_of(m, standard) for m in measurements}
    campaigns = {m.campaign: _campaign(m) for m in measurements}
    cases: Dict[Tuple[str, str], dict] = {}
    for m in measurements:
        for row in rows[m.campaign]:
            entry = cases.setdefault((row.campaign, row.case), {
                "campaign": row.campaign, "case": row.case, "combination": row.combination,
                "paired": float(m.cases[row.case]["paired_fraction"]), "ceilings": {}})
            entry["ceilings"].setdefault(row.scope, {})[row.figure] = row.ceiling
    return markdown(measurements, rows, standard), evidence_json(campaigns, list(cases.values()))


def _campaign(m: Measurements) -> dict:
    """What names a campaign's rows (I-55): its market, code revision, device, seeds and path
    counts; and what `Evidence.from_json` reads."""
    figures, env = FIGURES[m.kind], m.environment
    return {"kind": m.kind, "market": m.market, "aggregate": figures.aggregate,
            "products": sorted(set(m.products.values())), "corrected": sorted(figures.corrected),
            "commit": env.get("commit", "unknown"), "changes": env.get("changes", "unknown"),
            "engine": env.get("engine", "unknown"), "device": f"{env.get('backend', '?')} ({env.get('devices', '?')})",
            "jax": env.get("jax", "unknown"), "seeds": len(m.seeds), "paths": list(m.paths),
            "measured": env.get("measured", "unknown")}


def markdown(measurements: Sequence[Measurements], rows: Mapping[str, Sequence[EvidenceRow]],
             standard: Standard) -> str:
    """The table for people (see the module docstring)."""
    unit = {"portfolio": "paths", "market_risk": "scenarios"}
    t = standard.text
    lines = [
        "# Precision evidence", "",
        "Generated by `python -m tests.support.precision_evidence` from the precision harness's measurements of "
        "record ([precision-measurements/](precision-measurements/)) and the acceptance standard "
        "([precision-acceptance.json](precision-acceptance.json)); do not edit. The engine reads the same rows "
        "(`engine/precision/evidence.json`) and warns of every figure of a run that has no passing row, saying what "
        "is validated and up to how many paths ([I-55](../docs/planning/known-issues.md#i-55), decision D-9: no run "
        "is refused on precision). A passing row is engineering evidence against decision A-11's standard, not a "
        "regulatory approval.", "",
        "## The acceptance standard (A-11)", "",
        "| Rule | Threshold | Applies to | Source |", "|---|---|---|---|",
        f"| Statistical | {t['statistical']['rule']} Share: {standard.statistical:g} | {t['statistical']['applies']} | "
        f"{t['statistical']['cite']} |",
        f"| Fixed entries | {t['fixed']['rule']} Share: {standard.fixed:g} | {t['fixed']['applies']} | "
        f"{t['fixed']['cite']} |",
        f"| P&L attribution | {t['pla']['rule']} | {t['pla']['applies']} | {t['pla']['cite']}; {t['pla']['source']}. "
        f"{t['pla']['transcription']} |", "",
        "## Reading it", "",
        "- **A row** is one figure kind of one scope under one case of a campaign: a product (the worst of the "
        "campaign's trades of it) or the aggregate (the netting set, the market-risk portfolio). Its combination is "
        "the precision of every stage the figure depends on, `<stage>=<storage>/<compute>/<accumulate>`, then the "
        "rounding and block layout where a stage stores in a scaled format; its case label in the campaign's terms "
        "(`tests/support/precision_harness.py`): `<stage>:<format>[:<rounding>:<layout>]` stores one stage, "
        "`<stage>:compute:<format>` computes one stage's products in a format, `throughout:<format>` every stage, "
        "and `:paired` adds a paired float64 sample.",
        "- **It passes at a path count** when, over the campaign's seeds, its root mean square precision error is "
        "within the statistical rule on every entry float64 does not hold fixed and within the fixed rule on every "
        "entry it does, and for market risk the P&L attribution test is green at every seed. The errors are measured "
        "across independent seeds, never from the paired estimator's own standard errors (I-111).",
        "- **The path ceiling** is the largest measured path count at which the row passes and at every smaller one; "
        "none if it fails at the smallest. On Sobol paths the precision error grows against the statistical error "
        "with the path count ([precision.md §15.4](../docs/planning/details/precision.md#154-the-storage-harness-"
        "2026-10-09)), so a run with fewer paths than a ceiling is validated, and one with more is not.",
        "- **A run's figure is validated** by the rows of its combination, scope and kind "
        "(`engine.precision.evidence`): a trade's by its product's rows; the aggregate's by a row whose campaign held "
        "every product of the run, with one pricing stage for all its trades; a figure the paired sample corrects "
        "(EPE, ENE, EE_B, EEE_B, EPE_B, EEPE_B, the Basel EPE and EEPE) by the rows of the largest paired share at "
        "most the run's; the smallest ceiling where rows of two campaigns match.",
        "- **Formats below float32 were emulated** on the CPU named, which computes what native units compute "
        "([precision.md §8.3](../docs/planning/details/precision.md#83-emulation-and-native-speed)). Each campaign's "
        "row names the code it measured: its commit, whether the tree had local changes, and the digest of the "
        "engine's Python sources.", "",
        "## Campaigns", "",
        "| Campaign | Market | Products | Path counts | Seeds | Code | Device | Measured |",
        "|---|---|---|---|---|---|---|---|"]
    for m in measurements:
        c = _campaign(m)
        lines.append(f"| `{m.campaign}` | {m.market} | {', '.join(c['products'])} | "
                     f"{', '.join(f'{p:,}' for p in c['paths'])} {unit[m.kind]} | {c['seeds']} | "
                     f"`{c['commit'][:12]}` (local changes: {c['changes']}), engine `{c['engine']}` | {c['device']}, "
                     f"jax {c['jax']} | {c['measured']} |")
    lines += ["", "## Path ceilings by case", "",
              "Per case and scope, the smallest ceiling over the scope's figures; where a figure passes at no path "
              "count, the first such figure and its relative error at the smallest count (its root mean square "
              "precision error in units of its statistical error; the rule is "
              f"{standard.statistical:g}). Every figure's own ceiling is in the next section and in "
              "`engine/precision/evidence.json`.", ""]
    for m in measurements:
        lines += _ceilings_table(m, rows[m.campaign], unit[m.kind])
    lines += ["## Validated figures", "",
              "Every scope of a case where at least one figure passes, its figures grouped by ceiling.", "",
              "| Campaign | Case | Scope | Figures (ceiling) |", "|---|---|---|---|"]
    for m in measurements:
        lines += _validated(rows[m.campaign], unit[m.kind])
    lines += ["", "## P&L attribution (MAR32.42)", "",
              "Each market-risk case's P&L against the float64 P&L on the same scenarios, over the seeds: the smallest "
              "Spearman correlation and the largest Kolmogorov-Smirnov statistic, and whether every seed is in the "
              f"green zone (Spearman above {standard.spearman_above:g}, KS below {standard.ks_below:g}).", "",
              "| Campaign | Case | Scenarios | Spearman, smallest | KS, largest | Green at every seed |",
              "|---|---|---|---|---|---|"]
    for m in measurements:
        for a in m.attribution:
            lines.append(f"| `{m.campaign}` | {a.case} | {a.paths:,} | {min(a.spearman):.6f} | "
                         f"{max(a.kolmogorov_smirnov):.6f} | "
                         f"{'yes' if standard.green(a.spearman, a.kolmogorov_smirnov) else 'no'} |")
    return "\n".join(lines) + "\n"


def _ceilings_table(m: Measurements, rows: Sequence[EvidenceRow], unit: str) -> List[str]:
    scopes = list(dict.fromkeys(r.scope for r in rows))
    lines = [f"### `{m.campaign}`", "", f"Ceilings in {unit}; {m.market}.", "",
             "| Case | " + " | ".join(scopes) + " |", "|---|" + "---|" * len(scopes)]
    for case in dict.fromkeys(r.case for r in rows):
        cells = []
        for scope in scopes:
            group = [r for r in rows if r.case == case and r.scope == scope]
            failing = next((r for r in group if r.ceiling is None), None)
            if failing is not None:
                first = min(failing.points, key=lambda p: p.paths)
                cells.append(f"none ({failing.figure} {_fmt(first.relative_error)})")
            else:
                cells.append(f"{min(r.ceiling for r in group):,}")
        lines.append(f"| {case} | " + " | ".join(cells) + " |")
    return lines + [""]


def _validated(rows: Sequence[EvidenceRow], unit: str) -> List[str]:
    lines = []
    for case in dict.fromkeys(r.case for r in rows):
        for scope in dict.fromkeys(r.scope for r in rows if r.case == case):
            by_ceiling: Dict[int, List[str]] = {}
            for r in rows:
                if r.case == case and r.scope == scope and r.ceiling is not None:
                    by_ceiling.setdefault(r.ceiling, []).append(r.figure)
            if by_ceiling:
                text = "; ".join(f"{', '.join(figures)} ({ceiling:,} {unit})"
                                 for ceiling, figures in sorted(by_ceiling.items(), reverse=True))
                lines.append(f"| `{rows[0].campaign}` | {case} | {scope} | {text} |")
    return lines


def _fmt(x: float) -> str:
    return "nan" if math.isnan(x) else "inf" if math.isinf(x) else f"{x:.3g}"


# ---------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.support.precision_evidence",
                                     description="Generate the precision evidence table (I-55) from the measurements "
                                                 "of record and the acceptance standard (A-11).")
    parser.add_argument("--check", action="store_true", help="write nothing; fail if either file is out of date")
    args = parser.parse_args(argv)
    texts = dict(zip((MARKDOWN, EVIDENCE), generate(load_record(), Standard.load())))
    if args.check:
        stale = [str(path.relative_to(ROOT)) for path, text in texts.items()
                 if not path.exists() or path.read_text(encoding="utf-8") != text]
        if stale:
            print(f"out of date: {stale}; run python -m tests.support.precision_evidence", file=sys.stderr)
            return 1
        return 0
    for path, text in texts.items():
        path.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

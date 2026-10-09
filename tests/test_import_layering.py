"""
The layout of `engine/` (I-92, docs/planning/details/package-layout.md): nothing at the root of
`engine/` or of `engine/risk/` but `__init__.py`, and the packages in layers, each importing
only the layers below it (audit A-5). An inversion once forced a lazy `__getattr__` into the
run's `__init__.py` to break the resulting cycle.

Checked statically, on each module's import statements (at any depth, inside functions too),
so a violation fails here even if it happens to import cleanly today.
"""
import ast
import pathlib
import re

import pytest

ENGINE = pathlib.Path(__file__).resolve().parents[1] / "engine"

#: The packages of `engine/`, lowest first. A package imports its own modules and those of
#: the layers below it, never a package of its own layer or above.
LAYERS = (
    ("precision", "solvers"),           # JAX and NumPy only: every layer may use them
    ("market_data",),                   # today's market, curves, day counts
    ("models", "instruments", "traderx"),
    ("calibration",),
    ("market_simulation",),
    ("pricing",),
    ("risk",),
    ("run",),
    ("api",),
)
#: `engine/risk/`'s kinds, in layers as `LAYERS`: counterparty exposure borrows the quantile
#: labels of the market-risk estimators, and market risk revalues with the Greeks' price
#: functions.
RISK_LAYERS = (("greeks",), ("market",), ("counterparty",))
#: The one import against the layers, as `(importing package, imported module)`. The Greeks label
#: each trade's region of a profiler trace with the run's phase names (`engine.run.trace`, which
#: imports no engine module: `test_the_exception_is_a_leaf`). The import is made at call time,
#: since importing `engine.run` imports the pipeline, which imports the Greeks.
EXCEPTIONS = {("engine.risk.greeks", "engine.run.trace")}


def _imported_modules(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def _layer_violations(root: pathlib.Path, prefix: str, layers):
    """Every import, by a module of a package of `layers` (directories of `root`, imported as
    `prefix.<package>`), of a package of its own layer or above, as "importer imports module"."""
    rank = {package: i for i, layer in enumerate(layers) for package in layer}
    offenders = []
    for package, level in rank.items():
        for path in sorted((root / package).rglob("*.py")):
            importer = ".".join((prefix, package, *path.relative_to(root / package).with_suffix("").parts))
            for module in _imported_modules(path):
                if not module.startswith(prefix + "."):
                    continue
                target = module[len(prefix) + 1:].split(".")[0]
                if target == package or any(importer.startswith(a) and module == b for a, b in EXCEPTIONS):
                    continue
                if target not in rank or rank[target] >= level:
                    offenders.append(f"{importer} imports {module}")
    return offenders


def test_every_package_has_a_layer():
    """A new package is placed in `LAYERS` (or `RISK_LAYERS`), so its rule is stated."""
    packages = {p.name for p in ENGINE.iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert packages == {p for layer in LAYERS for p in layer}
    kinds = {p.name for p in (ENGINE / "risk").iterdir() if p.is_dir() and p.name != "__pycache__"}
    assert kinds == {p for layer in RISK_LAYERS for p in layer}


@pytest.mark.parametrize("root", ["engine", "engine/risk"])
def test_no_module_but_init_at_the_root(root):
    """`engine/` and `engine/risk/` hold subpackages only: a module at either root is code
    that belongs to no package (I-92). Every package has its `__init__.py`."""
    directory = ENGINE.parent / root
    assert sorted(p.name for p in directory.glob("*.py")) == ["__init__.py"]
    missing = [p.relative_to(ENGINE.parent).as_posix() for p in directory.iterdir()
               if p.is_dir() and p.name != "__pycache__" and not (p / "__init__.py").exists()]
    assert not missing, missing


def test_engine_init_holds_no_code_but_the_x64_flag():
    """`engine/__init__.py` imports no engine module, so importing a leaf such as
    `engine.market_data.day_counts` loads nothing else of the engine (decision A-22)."""
    assert not [m for m in _imported_modules(ENGINE / "__init__.py") if m.startswith("engine")]


def test_packages_import_only_the_layers_below():
    assert _layer_violations(ENGINE, "engine", LAYERS) == []


def test_risk_kinds_import_only_the_kinds_below():
    assert _layer_violations(ENGINE / "risk", "engine.risk", RISK_LAYERS) == []


def test_the_exception_is_a_leaf():
    """What `EXCEPTIONS` lets a lower layer import imports no engine module itself, so the
    exception cannot pull the layers above into the importer."""
    for _, module in EXCEPTIONS:
        path = ENGINE.parent / (module.replace(".", "/") + ".py")
        assert not [m for m in _imported_modules(path) if m.startswith("engine")], module


def test_the_layer_check_sees_an_inversion(tmp_path):
    """The check is not vacuous: a model importing the pricing layer is reported."""
    for package in ("models", "pricing"):
        (tmp_path / package).mkdir()
    (tmp_path / "models" / "lgm.py").write_text("from engine.pricing.cube import value_today\n", encoding="utf-8")
    (tmp_path / "pricing" / "cube.py").write_text("from engine.models.lgm import zeta\n", encoding="utf-8")
    offenders = _layer_violations(tmp_path, "engine", (("models",), ("pricing",)))
    assert offenders == ["engine.models.lgm imports engine.pricing.cube"]


def test_engine_run_imports_eagerly():
    """With the cycle gone, `engine.run` exposes its names as plain attributes."""
    import engine.run as run

    assert "__getattr__" not in vars(run)
    assert callable(run.price_portfolio)


def test_engine_ships_no_demo_or_test_code():
    """I-65: demos live in `demos/` and test tooling in `tests/support/`. No `engine` module
    has a `__main__` demo block or imports either tree."""
    offenders = []
    for path in sorted(ENGINE.rglob("*.py")):
        name = path.relative_to(ENGINE).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
                offenders.append(f"{name} has a __main__ block")
        for module in _imported_modules(path):
            if module.split(".")[0] in ("demos", "tests"):
                offenders.append(f"{name} imports {module}")
    assert not offenders, "; ".join(offenders)


#: A roadmap step number: "roadmap", optionally "step", then n.m, possibly wrapped across a
#: comment line.
ROADMAP_STEP = re.compile(r"roadmap[\s#:]*(?:step[\s#:]*)?\d+\.\d+", re.IGNORECASE)


def test_no_code_cites_a_roadmap_step():
    """Roadmap steps are renumbered whenever the order of work changes, so code, error messages
    and configuration cite the permanent IDs instead: an issue (I-NN), a feature (F-NN), a
    decision (A-n), or a date."""
    root = ENGINE.parent
    paths = [p for tree in ("engine", "tests", "demos") for p in (root / tree).rglob("*.py")]
    paths += [root / "pyproject.toml", root / "constraints.txt", *(root / ".github").rglob("*.yml")]
    offenders = [f"{p.relative_to(root).as_posix()}: {m.group(0)!r}"
                 for p in paths for m in ROADMAP_STEP.finditer(p.read_text(encoding="utf-8"))]
    assert not offenders, "; ".join(offenders)

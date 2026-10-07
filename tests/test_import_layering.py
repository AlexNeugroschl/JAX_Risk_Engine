"""
Import layering (audit A-5): the lower layers of `engine` never import the layers above
them. `engine.portfolio` sits on top of the instruments, which sit on top of the models, so
an instrument or model module importing `engine.portfolio` is an inversion (it once forced a
lazy `__getattr__` into `engine/portfolio/__init__.py` to break the resulting cycle).

Checked statically, on each module's import statements, so a violation fails here even if
it happens to import cleanly today.
"""
import ast
import pathlib

import pytest

ENGINE = pathlib.Path(__file__).resolve().parents[1] / "engine"

#: Package -> packages it must not import.
FORBIDDEN = {
    "models": ("engine.portfolio", "engine.instruments", "engine.api", "engine.risk"),
    "instruments": ("engine.portfolio", "engine.api", "engine.risk", "engine.market_risk"),
    # The shared numerical methods depend on JAX only: every layer imports them.
    "numerics": tuple(f"engine.{p.name}" for p in sorted(ENGINE.iterdir())
                      if p.is_dir() and p.name not in ("numerics", "__pycache__")) + ("engine.market", "engine.day_count"),
}


def _imported_modules(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_package_does_not_import_the_layers_above_it(package):
    offenders = []
    for path in sorted((ENGINE / package).glob("*.py")):
        for module in _imported_modules(path):
            if module.startswith(FORBIDDEN[package]):
                offenders.append(f"{path.name} imports {module}")
    assert not offenders, "; ".join(offenders)


def test_engine_portfolio_imports_eagerly():
    """With the cycle gone, `engine.portfolio` exposes its names as plain attributes."""
    import engine.portfolio as portfolio

    assert "__getattr__" not in vars(portfolio)
    assert callable(portfolio.price_portfolio)


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

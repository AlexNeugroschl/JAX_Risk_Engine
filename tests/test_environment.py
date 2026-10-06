"""
The running environment matches the verified one in constraints.txt.

Every ORE-parity tolerance in this suite (down to 1e-12) was established
against specific jax, jaxlib and ORE builds. A different build can move
results, and the parity test that then fails would point at the pricer, not
at the install. These tests fail first and name the package instead.

They also keep the two dependency files consistent: pyproject.toml's ranges
must admit the locked versions, and every declared dependency must be
locked. See the header of constraints.txt for how to upgrade a pin.
"""
import tomllib
from importlib.metadata import distributions, version
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]

# The packages whose numerics the parity tolerances depend on. Other locked
# packages (pytest, fastapi, ...) may drift locally without invalidating a
# numerical result, so they are not checked here.
NUMERICAL_PACKAGES = ["jax", "jaxlib", "open-source-risk-engine", "numpy", "scipy"]

# JAX's GPU plugin, where the `gpu` extra installed it (roadmap 2.2): its kernels are what a
# GPU run computes with, and JAX refuses to load a plugin whose version is not jaxlib's.
GPU_PLUGINS = sorted(canonicalize_name(d.metadata["Name"]) for d in distributions()
                     if canonicalize_name(d.metadata["Name"]).startswith("jax-cuda"))


def _locked_versions():
    locked = {}
    for line in (ROOT / "constraints.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, _, pinned = line.partition("==")
            assert pinned, f"constraints.txt line is not an exact pin: {line!r}"
            locked[canonicalize_name(name)] = pinned
    return locked


def _declared_requirements():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        declared.extend(extra)
    return [Requirement(r) for r in declared]


LOCKED = _locked_versions()


@pytest.mark.parametrize("package", NUMERICAL_PACKAGES + GPU_PLUGINS)
def test_installed_numerical_package_matches_lock(package):
    assert canonicalize_name(package) in LOCKED, f"{package} {version(package)} is installed but not pinned in constraints.txt"
    assert version(package) == LOCKED[canonicalize_name(package)], (
        f"{package} {version(package)} is installed but constraints.txt pins "
        f"{LOCKED[canonicalize_name(package)]}. Parity tolerances are only "
        f"verified for the pinned version: reinstall with "
        f"`pip install -r requirements.txt`, or upgrade the pin as described "
        f"in constraints.txt."
    )


@pytest.mark.skipif(not GPU_PLUGINS, reason="no GPU plugin installed (the `gpu` extra)")
@pytest.mark.parametrize("plugin", GPU_PLUGINS)
def test_a_gpu_plugin_is_jaxlibs_version(plugin):
    assert version(plugin) == version("jaxlib"), f"{plugin} {version(plugin)} beside jaxlib {version('jaxlib')}"


@pytest.mark.parametrize("requirement", _declared_requirements(), ids=str)
def test_every_declared_dependency_is_locked_within_its_range(requirement):
    name = canonicalize_name(requirement.name)
    assert name in LOCKED, f"{requirement.name} is declared in pyproject.toml but missing from constraints.txt"
    assert requirement.specifier.contains(LOCKED[name]), (
        f"constraints.txt pins {requirement.name}=={LOCKED[name]}, outside "
        f"pyproject.toml's range {requirement.specifier}"
    )

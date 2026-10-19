"""
The engine's identity, which every run result carries so that a figure can be traced to the code
and the libraries that computed it (I-87): the package's version, the git revision of the source
tree (and whether it had uncommitted changes), and the versions of jax, jaxlib and ORE, whose
releases move numbers (the ORE-parity tolerances were verified against `constraints.txt`'s pins).

Read once per process: the libraries cannot change under a running process, and the source
tree is the one it imported. Outside a git checkout (an installed wheel) the revision is None.
"""
import functools
import subprocess
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Optional

#: The distribution names whose versions are recorded.
_ENGINE, _ORE = "jax-risk-engine", "open-source-risk-engine"


@dataclass(frozen=True)
class EngineIdentity:
    """What computed a result. `revision` is the git commit of the engine's source tree, None
    outside a checkout; `modified` whether tracked files differed from it (None without a
    revision), in which case the revision alone does not reproduce the result."""
    version: str
    revision: Optional[str]
    modified: Optional[bool]
    jax: str
    jaxlib: str
    ore: str


@functools.lru_cache(maxsize=1)
def engine_identity() -> EngineIdentity:
    """This process's `EngineIdentity` (see the module docstring)."""
    revision = _git("rev-parse", "HEAD")
    modified = None if revision is None else bool(_git("status", "--porcelain", "--untracked-files=no"))
    return EngineIdentity(version=_version(_ENGINE), revision=revision, modified=modified, jax=_version("jax"),
                          jaxlib=_version("jaxlib"), ore=_version(_ORE))


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _git(*args: str) -> Optional[str]:
    """`git <args>`'s output in the engine's source tree, None if it is not a checkout or git
    is not installed."""
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=10,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None

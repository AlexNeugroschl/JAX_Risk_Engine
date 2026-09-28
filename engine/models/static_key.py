"""By-value hashing for the `_Prepared*` trade-structure dataclasses, so they can be
`jax.jit` static arguments.

Each pricer splits into a CPU "prepare" step (build the ORE trade, resolve schedules into
NumPy arrays) and a jitted "price" step. The prepared object is constant per trade, so it is
passed as a static argument and one kernel is compiled per distinct trade structure.

Static arguments must hash and compare by value. NumPy arrays do neither, and each call
re-prepares a fresh object, so identity-based caching would recompile every time.
"""
import numpy as np


def static_key(obj) -> tuple:
    """A hashable, by-value tuple of `obj`'s dataclass fields.

    NumPy arrays become `(bytes, shape, dtype)`; lists and tuples are normalized
    elementwise into tuples; anything else passes through and hashes on its own terms.
    """
    from dataclasses import fields
    return tuple(_norm(getattr(obj, f.name)) for f in fields(obj))


def _norm(value):
    if isinstance(value, np.ndarray):
        return (value.tobytes(), value.shape, str(value.dtype))
    if isinstance(value, (list, tuple)):
        return tuple(_norm(v) for v in value)
    return value


class StaticKeyMixin:
    """Gives a dataclass `__hash__`/`__eq__` built on `static_key`. Only instances of the
    same class compare equal.

    Declare the dataclass `@dataclass(frozen=True, eq=False)`. With `eq=True` the decorator
    generates its own `__eq__`/`__hash__` on the subclass, which shadow these and fail on
    the NumPy fields ("unhashable type: 'numpy.ndarray'").
    """

    def __hash__(self):
        return hash(static_key(self))

    def __eq__(self, other):
        return type(other) is type(self) and static_key(self) == static_key(other)

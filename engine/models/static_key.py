"""By-value hashing for the `_Prepared*` trade-structure dataclasses, so they
can be passed as `jax.jit` STATIC arguments.

**Why this exists.** Every pricer in this codebase splits into a CPU
"prepare" half (build the ORE trade, resolve schedules/accruals/cashflow
times onto plain NumPy arrays -- see e.g.
`engine.instruments.swap.prepare_swap`) and a GPU "price" half that consumes
those prepared, per-trade-constant arrays together with the simulated paths.
The prepared object is compile-time-constant structure: nothing in it varies
per scenario or per time step. Passing it as a `jax.jit` static argument is
therefore exactly right -- it specializes one compiled kernel per distinct
trade structure, and that kernel is then reused for every subsequent call.

`jax.jit` requires static arguments to be **hashable and comparable by
value**, and a plain `@dataclass` holding NumPy arrays is neither: NumPy
arrays are unhashable, and `==` on them returns an elementwise array (so
`dataclass`'s generated `__eq__`, which compares field tuples, raises "truth
value of an array with more than one element is ambiguous"). `static_key`
below normalizes any such object into a hashable, by-value tuple.

**Why by-value and not by-identity.** Each `price_*` call re-runs its
`prepare_*` helper and gets a FRESH object, so identity- (`id`-) based
caching would miss every single time -- recompiling on every call and, if
memoized on `id()`, leaking entries as ids get recycled. Hashing by value
means two preparations of the same trade against the same market structure
collapse onto one compiled kernel, which is the entire point.
"""
import numpy as np


def static_key(obj) -> tuple:
    """A hashable, by-value view of `obj`'s dataclass fields.

    NumPy arrays become `(bytes, shape, dtype)` triples -- content-based, so
    two independently-built arrays holding the same numbers compare equal.
    Lists/tuples are normalized elementwise (and lists become tuples, since
    lists are unhashable). Anything else -- Python scalars, and objects that
    define their own `__hash__`/`__eq__`, such as `engine.models.lgm.Sigma` --
    passes through untouched and is compared/hashed on its own terms.
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
    """Mixin giving a dataclass `jax.jit`-static-argument-compatible
    `__hash__`/`__eq__` built on `static_key`. Equality is restricted to the
    exact same class, so two different prepared-trade types can never compare
    equal even if their field values coincidentally normalize alike.

    **Declare the dataclass `@dataclass(frozen=True, eq=False)`.** `eq=False`
    is load-bearing, not stylistic: with the default `eq=True` the decorator
    generates `__eq__` (and, under `frozen=True`, `__hash__`) directly ON the
    subclass, and those shadow everything inherited from here -- leaving the
    NumPy-array fields to be hashed raw, which fails with `TypeError:
    unhashable type: 'numpy.ndarray'` the moment `jax.jit` tries to key its
    cache on the argument. `frozen=True` is the ordinary immutability
    guarantee you want from something used as a cache key.
    """

    def __hash__(self):
        return hash(static_key(self))

    def __eq__(self, other):
        return type(other) is type(self) and static_key(self) == static_key(other)

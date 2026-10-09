"""
The root solver of every calibration and exercise boundary (decision A-21): each
bucket of an LGM bootstrap (`engine.calibration.ore_lgm`, `engine.calibration.lgm`) and each
exercise boundary, the LGM's y* (`ore_lgm`), Jamshidian's x* (`engine.pricing.jamshidian`)
and the co-terminal basket's x* (`engine.calibration.basket`).

A problem is `f(x, params)`, monotone in `x` and elementwise over a batch (element i of the
result depends on element i of `x` only: paths, dates). Two solvers, chosen by name:

  * `"Newton"`, the default: a safeguarded Newton method. Each step evaluates f and its
    derivative (`jax.jvp`), narrows the bracket to the side of the root f's sign shows, and
    takes the Newton step where it stays inside the bracket, a bisection step where it would
    leave it (a zero or non-finite derivative included). So it is never less robust than
    bisection; started near the root (`start`) it converges in a few steps.
  * `"Bisection"`: the reference. It halves the bracket a fixed number of times, as the
    engine did before Newton, and reproduces its numbers bit for bit.

Both run a fixed number of steps (`Steps`, per problem) on every backend. A data-dependent stop
would make a GPU report to the host on every step; a fixed count compiles to one loop on the
device.

Brackets. A bounded problem gives its bracket (`bracket=(lo, hi)`): a root beyond an end is that
end, as bisection converges to it (a bootstrap bucket's ceiling and floor). An unbounded one
gives a starting window `[-window, window]`, which is widened, side by side, to `WIDENING` times
itself until it holds the sign change: all the candidates are evaluated at once (one vectorized
evaluation, not a loop), and each side takes the first that holds the root. A root beyond the
last candidate is taken there, as a bounded problem's.

Derivatives. A root is computed on `params` with their tangents stopped and has none itself
(`solve`). Where a price depends on the root and is not stationary in it, `implicit_root`
gives the root its derivative by the implicit function theorem. The ORE helper's price is
stationary in its y*, so its derivatives with y* held fixed are exact.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

#: The solvers, the default first (decision A-21).
SOLVERS = ("Newton", "Bisection")
DEFAULT_SOLVER = SOLVERS[0]

#: The multiples of an unbounded problem's starting window its bracket can be widened to:
#: s_0 = 1 and s_{k+1} = 2 s_k^2, which reaches 2^63 in six steps.
WIDENING = (1.0, 2.0, 8.0, 128.0, 32768.0, 2.0 ** 31, 2.0 ** 63)


@dataclass(frozen=True)
class Steps:
    """A problem's step counts: the reference's halvings and Newton's steps."""
    bisection: int
    newton: int

    def of(self, solver: str) -> int:
        return self.newton if solver == "Newton" else self.bisection


def check_solver(solver: str, field: str = "solver") -> None:
    """Refuse a solver that is not one of `SOLVERS`, naming the setting."""
    if solver not in SOLVERS:
        raise ValueError(f"{field} must be one of {SOLVERS}; got {solver!r}")


def solve(f: Callable, params, start: jax.Array, *, solver: str, steps: Steps, increasing: bool,
          bracket: Optional[Tuple[float, float]] = None, window: Optional[float] = None) -> jax.Array:
    """The root of `x -> f(x, params)` over `start`'s shape, in its dtype, with no derivative.

    `start` is Newton's first point (any finite value; it is moved into the bracket);
    bisection reads only its shape and dtype. `increasing` says how f is monotone. Give either
    `bracket` (bounded) or `window` (unbounded, widened; see the module docstring)."""
    if (bracket is None) == (window is None):
        raise ValueError("give either a bracket or a window")
    check_solver(solver)
    params, start = jax.lax.stop_gradient((params, start))
    g = lambda x: f(x, params)  # noqa: E731
    dtype, shape = start.dtype, start.shape
    edge = lambda value: jnp.broadcast_to(jnp.asarray(value, dtype=dtype), shape)  # noqa: E731
    lo, hi = (edge(bracket[0]), edge(bracket[1])) if bracket is not None else (edge(-window), edge(window))
    n = steps.of(solver)

    if solver == "Bisection":
        if window is not None:
            lo, hi = _widen(g, lo, hi, increasing)[:2]
        root = _bisect(g, lo, hi, n, increasing)
    else:
        lo, hi, f_lo, f_hi = _widen(g, lo, hi, increasing, scales=WIDENING if window is not None else (1.0,))
        x = jnp.where(jnp.isfinite(start), jnp.clip(start, lo, hi), 0.5 * (lo + hi))
        root = _newton(g, x, lo, hi, n, increasing)
        root = jnp.where(_above(f_hi, increasing), hi, jnp.where(_below(f_lo, increasing), lo, root))
    return jax.lax.stop_gradient(root)


def implicit_root(f: Callable, params, start: jax.Array, **options) -> jax.Array:
    """`solve(f, params, start, **options)` with the root's derivative in `params`: at the
    root of f(x, p) = 0, dx = -(df/dp . dp) / (df/dx). The rule is itself differentiable, so
    second derivatives (Gamma) are right too. df/dx is the gradient of the batch sum, exact
    because each element depends on its own inputs only. `params` must be an explicit
    argument: `jax.custom_jvp` attaches tangents to its arguments, not to a closure's tracers."""

    @jax.custom_jvp
    def root(p, x0):
        return solve(f, p, x0, **options)

    @root.defjvp
    def root_jvp(primals, tangents):
        (p, x0), (p_dot, _) = primals, tangents
        x = root(p, x0)
        df_dx = jax.grad(lambda y: jnp.sum(f(y, p)))(x)
        _, df = jax.jvp(lambda q: f(x, q), (p,), (p_dot,))
        return x, -df / df_dx

    return root(params, start)


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------
def _above(value, increasing: bool):
    """Whether the root is above a point where f takes `value` (False where it is NaN)."""
    return value < 0.0 if increasing else value > 0.0


def _below(value, increasing: bool):
    """Whether the root is below a point where f takes `value` (False where it is NaN)."""
    return value > 0.0 if increasing else value < 0.0


def _widen(g, lo, hi, increasing: bool, scales=WIDENING):
    """`(lo, hi, f(lo), f(hi))`: each end moved out to the first of `scales` times itself at
    which the root is no longer beyond it (the last if none), with f there."""
    multiples = jnp.asarray(scales, dtype=lo.dtype).reshape((-1,) + (1,) * lo.ndim)
    k = len(scales)
    # Both sides in one evaluation: f is traced once into the program, not once per side.
    candidates = jnp.concatenate([multiples * lo, multiples * hi])
    values = jax.vmap(g)(candidates)

    def side(candidates, values, beyond):
        held = ~beyond(values, increasing)
        first = jnp.argmax(held.at[-1].set(True), axis=0)[None]
        pick = lambda a: jnp.take_along_axis(a, first, axis=0)[0]  # noqa: E731
        return pick(candidates), pick(values)

    (lo, f_lo), (hi, f_hi) = side(candidates[:k], values[:k], _below), side(candidates[k:], values[k:], _above)
    return lo, hi, f_lo, f_hi


def _bisect(g, lo, hi, n: int, increasing: bool):
    def halve(bounds, _):
        a, b = bounds
        mid = 0.5 * (a + b)
        up = _above(g(mid), increasing)
        return (jnp.where(up, mid, a), jnp.where(up, b, mid)), None

    (a, b), _ = jax.lax.scan(halve, (lo, hi), None, length=n)
    return 0.5 * (a + b)


def _newton(g, x, lo, hi, n: int, increasing: bool):
    """The Newton step where it lands inside the bracket, else a bisection step.

    Where f is monotone, Newton's direction is always the one f's sign gives the bracket, so a
    step from the root's neighbourhood stays inside it and the root, once found, is kept to its
    rounding whatever the count. `rtsafe`'s second test (bisect unless a step halves the one
    before) is not used: it suits a loop that stops at a tolerance, and at a fixed count the
    root's own noise, steps that need not halve, would trigger bisection steps across a bracket
    still wide on one side (measured: a bucket's volatility halved). Where f is not monotone
    across the bracket (a Jamshidian European struck far below the money), Newton can move
    against the bracket and fall back to bisecting; its count covers that.

    Where the slope has the sign f's monotony gives it, the side of x the root is on is the
    Newton step's direction, which is f's sign whenever f is computed once. XLA may compute f
    once for the bracket and again, rounded differently, for the step (seen on the CPU in
    float32 on a batch of 128 paths, even through an optimization barrier: f = 0 moved the
    bracket while f = 6e-8 moved the step); then at the root the step lands just outside the
    bracket, and a bisection step throws the root away. Taking the side from the step keeps the
    two consistent. Where the slope has the other sign (f not monotone there), f's sign gives
    the side, and the step is bisection's."""
    rising = 1.0 if increasing else -1.0

    def step(state, _):
        x, a, b = state
        value, slope = jax.jvp(g, (x,), (jnp.ones_like(x),))
        newton = x - value / slope
        up = jnp.where(rising * slope > 0.0, newton > x, _above(value, increasing))
        a, b = jnp.where(up, x, a), jnp.where(up, b, x)
        inside = (newton >= a) & (newton <= b)   # False for a non-finite step
        return (jnp.where(inside, newton, 0.5 * (a + b)), a, b), None

    (x, _, _), _ = jax.lax.scan(step, (x, lo, hi), None, length=n)
    return x

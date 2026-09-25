# Engine Audit: Open Problems

**Date:** 2026-09-24 · **Scope:** the internal engine (`engine/` except `engine/integration/`
and `engine/api/eod_routes.py`), the test suite, and the demos. The TraderX integration was
out of scope.

This document lists the problems found in the audit that are **not** simple to fix. The
simple ones were fixed in the same pass; they are listed at the end so the two sets can be
told apart. Each finding gives the problem, the evidence, what fixing it takes, and two
ratings:

| Urgency | Meaning |
|---|---|
| **Critical** | Numbers the engine reports today are materially wrong on ordinary inputs. |
| **High** | Wrong numbers on a common path, or a design gap that blocks a stated project goal. |
| **Medium** | Correctness holds, but the design will cost time or produce bugs as the code grows. |
| **Low** | Hygiene. Worth doing when touching the area anyway. |

| Ease | Meaning |
|---|---|
| **Moderate** | A few days. Localized, but changes numbers or public types, so tests move with it. |
| **Hard** | A week or more, or it needs a design decision first. |

Several findings overlap with entries in [known-issues.md](../known-issues.md). Where they
do, this document links the entry and says what the register gets wrong or leaves out.

---

## Summary

| ID | Finding | Urgency | Ease |
|---|---|---|---|
| [M-1](#m-1) | Simulated curves are not arbitrage-free against the input curve | Critical | Hard |
| [M-2](#m-2) | Paid cashflows stay in a swap's NPV forever | High | Moderate |
| [M-3](#m-3) | Options vanish at expiry instead of turning into the swap | High | Moderate |
| [M-4](#m-4) | Trades are defined relative to the evaluation date | High | Hard |
| [M-5](#m-5) | Theta re-rolls the trade instead of ageing it | High | Hard (after M-4) |
| [R-1](#r-1) | The reported VaR/ES is not an end-of-day market-risk VaR | ✅ Resolved 2026-09-24 | — |
| [P-1](#p-1) | Nothing runs on more than one device | High | Hard |
| [P-2](#p-2) | Bermudan/American scenario pricing runs on the host in a Python loop | Medium | Moderate |
| [P-3](#p-3) | The precision study does not exercise the engine's own low-precision path | ✅ Resolved 2026-09-24 | — |
| [A-1](#a-1) | Precision is controlled by toggling a process-global JAX flag | Medium | Hard |
| [A-2](#a-2) | Two short-rate model families, bridged by a state conversion | Medium | Hard (with M-1) |
| [A-3](#a-3) | Trade configs duplicate model parameters, then validate the copies | Medium | Moderate |
| [A-4](#a-4) | ORE global state: implicit evaluation dates and a mutated singleton | Medium | Moderate |
| [A-5](#a-5) | Layering inversion between instruments and portfolio | Low | Moderate |
| [A-6](#a-6) | Demo and test infrastructure lives in the production package | Low | Moderate |
| [A-7](#a-7) | The bond pricer is a separate, plain-Python implementation | Medium | Moderate |
| [Q-1](#q-1) | Source code is mostly narrative prose and project history | Medium | Hard (volume) |
| [Q-2](#q-2) | No lint, type checking, CI, or pinned dependencies | Medium — pins and CI resolved 2026-09-24; lint and types open | Moderate |
| [Q-3](#q-3) | Test suite: slow, unmarked, and coupled to itself | Medium | Moderate |
| [Q-4](#q-4) | Documentation sprawl and stale planning documents | Low | Moderate |

**Decision (2026-09-24, R-1).** The engine reports two different things under two names.
Market-risk VaR/ES is a t=0 full revaluation under Monte Carlo or historical shocks
(`engine.market_risk`, [Market Risk](../risk/market-risk.md)). The multi-step simulation
reports exposure profiles (`engine.risk.exposure`, [Exposure](../risk/exposure.md)),
flagged with a warning wherever M-1, M-2 or M-3 applies. Market risk does not use the
simulated cube, so **M-1 to M-3 now affect only the exposure product**, and they are
scheduled with the counterparty-risk work (Basel plan phase P5), not first. P-3 was
closed by rebuilding the precision study on the market-risk path.

Recommended order from here: **M-4/M-5** (trade dates; theta), then **P-1** (multi-device,
the research goal), then **M-1 to M-3** when exposure or CVA is scheduled. The rest can
follow the work that touches them.

---

## Model correctness

### M-1 — Simulated curves are not arbitrage-free against the input curve {#m-1}

**Urgency: Critical · Ease: Hard**

**Problem.** `_simulate_cross_asset_paths_jit` evolves the short rate as a mean-reverting
process toward a **constant** `theta` from a configured `initial_rates` value
([market_model.py](../../engine/simulation/market_model.py), `step_fn`). The discount factors
built from those paths use the Hull-White `A(t,T)` that is fitted to the input zero curve,
which assumes the **curve-fitted** drift `θ(t)`. The two halves describe different models.
The simulated discount factors are consistent with the curve only when the curve is flat and
`theta` and `initial_rates` happen to equal its level. Even then they miss the convexity term.

**Evidence.** Martingale check at t=2y, 16,384 scenarios, a=3%, σ=1%, `initial_rates` and
`theta` set to the curve's short end. The test is `E[P(t,T)/N(t)] = P(0,T)`:

| Curve | T | Relative error |
|---|---|---|
| flat 3% | 5y | 0.06% |
| flat 3% | 10y | 0.14% |
| upward 3%→5% | 5y | **4.2%** |
| upward 3%→5% | 10y | **8.8%** |

Every demo and nearly every test uses a flat curve, which is why this has never shown up.

**Impact.** Every `npv_cube` value past t=0 on a non-flat curve, and every VaR/ES and
exposure figure derived from it. It also affects the Bermudan/American scenario pricers:
they condition on the simulated short rate via `x_from_r`, so their scenario values
inherit the wrong distribution.

**Fix.** Simulate the zero-mean OU state `x(t)` and set
`r(t) = x(t) + α(t)`, with `α(t) = f(0,t) + σ²/(2a²)(1−e^{−at})²` (Brigo–Mercurio 3.36).
Alternatively, simulate the LGM state directly and build discount factors with
`engine.models.lgm.bond_price`, which is already verified against ORE; that also resolves
[A-2](#a-2). `RatesConfig.theta` and `initial_rates` then become derived quantities, not
inputs, so the config types and every test that sets them change. Add the martingale check
above, on a sloped curve, as a permanent test.

---

### M-2 — Paid cashflows stay in a swap's NPV forever {#m-2}

**Urgency: High · Ease: Moderate**

**Problem.** `_price_one_swap` sums **every** cashflow of the swap at every simulated step.
For a cashflow already paid (`T < t`), `B(t,T)` is clamped to 0 and `A(t,T)` becomes
`P(0,T)/P(0,t) > 1`, so a paid coupon is not just kept but grown forward.

**Evidence.** `demos/demo.py` prices a **3Y** payer swap. Its mean simulated NPV is −9,852
at t=4 and −10,152 at t=5, after the swap has fully matured. It should be exactly 0.

**How this differs from [I-04](../known-issues.md#i-04).** I-04 describes only the
coupon that is *currently accruing*, and says closing it requires historical fixings that
no data source supplies. Neither half of that covers this finding:

- Dropping cashflows **already paid** at a step needs no data at all. It is a mask on
  `payment_time <= t`.
- A coupon that fixes **inside** the simulated horizon can be fixed from the simulated
  path at its fixing step. Only a coupon fixed **before t=0** needs a historical fixing.

I-04's blast-radius figure (~1e-4 to 1e-3 relative) was measured before the first payment
date and understates the error once payments start.

**Fix.** Pass the step times into `price_swaps`. Mask paid cashflows. For the accruing
coupon, use the rate fixed on the path at its fixing date (this needs the fixing dates on
the time grid, or a documented interpolation). Then rewrite I-04 to cover only the
pre-t=0 fixing. `TestAgedSwapKnownLimitation` pins the current behavior and will change.

---

### M-3 — Options vanish at expiry instead of turning into the swap {#m-3}

**Urgency: High · Ease: Moderate**

**Problem.** A European swaption's scenario NPV is set to **0** at every step after expiry
(`_price_one_swaption`: `jnp.where(not_yet_expired, npv_unexpired, 0.0)`). Bermudan and
American NPVs are 0 after the last exercise date. A physically settled option that was
exercised becomes the underlying swap; one that was not exercised is worth 0. The engine
does not track which happened, so every path reports 0.

**Evidence.** In `demos/demo.py`, VaR_99 at t=2, 3, 4 and 5 is 49,820, 44,026, 52,619 and
53,207, roughly the portfolio's base NPV of 43,055 plus drift. The ES standard error is ~0
at those steps. The "risk" is the options disappearing on every path at once, not market
movement.

**Fix.** At each exercise date, record the exercise decision per path: continuation value
against exercise value, both of which the backward induction already computes. After
exercise, carry the underlying swap's value, or the cash-settlement amount for
cash-settled trades. Until then, warn the way I-04 does. This is a silent misstatement of
every VaR/ES step after the first expiry.

---

### M-4 — Trades are defined relative to the evaluation date {#m-4}

**Urgency: High · Ease: Hard**

**Problem.** `SwapConfig`, `SwaptionConfig` and the Bermudan/American configs describe the
underlying swap with a tenor string (`"5Y"`) and optional `forward_start`, both measured
from `evaluation_date`. `build_vanilla_swap` builds a spot-starting schedule from whatever
the evaluation date is. So the same config names a **different trade** on a different
date.

**Evidence.** A 5Y swap config built at the evaluation date matures 2031-08-04; the same
config at evaluation date +30d matures 2031-09-02. A 3Y-into-5Y swaption has expiry
3.008y from its evaluation date, whichever date that is.

**Impact.** A seasoned trade (booked in the past, with a real schedule) cannot be
represented. Multi-day EOD runs reprice a different trade each day. Theta is broken
([M-5](#m-5)). This also underlies I-05 and I-10.

**Fix.** Give trade configs absolute dates: effective date, maturity date, and expiry or
exercise dates for options. Tenor strings can remain as a convenience that is resolved to
dates once, at booking. Every config type, the API schemas, the worker-pool freezing and
most tests change.

---

### M-5 — Theta re-rolls the trade instead of ageing it {#m-5}

**Urgency: High · Ease: Hard (depends on M-4)**

**Problem.** `swap_theta`, `swaption_theta` and `bermudan_theta` compute
`NPV(eval+1d) − NPV(eval) + CF` by rebuilding the config with a new `evaluation_date`.
Because of [M-4](#m-4), that rebuilds a *new* trade one day later: the swap's maturity moves,
and the swaption's expiry stays 3.008y away. The result is the roll-down of a
constant-maturity trade, not the time decay of the booked one. For a European swaption,
theta should be dominated by time value decaying towards expiry; here expiry never
approaches.

The cashflow add-back term (`_swap_cashflows_in_period`) is also inert: the re-rolled swap
never has a payment inside the window, so the term is always 0 for the default 1-day
horizon. (This pass fixed the add-back to include floating coupons, so it will be correct
once M-4 lands; today it has nothing to add.)

**Fix.** Once trades carry absolute dates, reprice the **same** schedule at the new
evaluation date. The tests that pin current theta values will change.

---

## Risk methodology

### R-1 — The reported VaR/ES is not an end-of-day market-risk VaR {#r-1}

**Status: ✅ Resolved 2026-09-24.** `price_portfolio` now reports exposure profiles
(EPE/ENE/EE_B/EEE_B/PFE, ORE's `ExposureCalculator` definitions, numeraire-deflated)
instead of VaR/ES; short-horizon VaR/ES is `engine.market_risk.run_market_risk`, validated
scenario by scenario against ORE. What follows is the finding as originally written.

**Urgency: High · Ease: Hard (needs a decision first)**

**Problem.** `compute_risk_metrics` reports a VaR/ES per simulated step, where P&L is
`NPV(scenario, t) − NPV(0)`:

- over horizons of months to years (the demo grid runs to 5y);
- under the risk-neutral measure (correctly labelled since I-11);
- on **undiscounted** future NPVs, so a t=5y value is compared directly with a t=0 value;
- with no cashflows paid in between and no exercise ([M-2](#m-2), [M-3](#m-3));
- including the risk-neutral drift, by design.

That is an exposure profile, which is useful for PFE/CVA work. It is not the 1-day or
10-day VaR/ES that an end-of-day market-risk process, TraderX, or the
[Basel III plan](basel-iii-compliance-plan.md) needs. The README and demos present it as
"VaR".

**Fix.** Decide what the product reports. For market risk, add a short-horizon mode: one
step at 1d or 10d, with P&L that includes cashflows and uses a real-world or historical
scenario generator. For exposure, rename the outputs (PFE/EPE profiles) and deflate by the
numeraire where the use requires it. Both can coexist, but they should not share the name
"VaR".

---

## Performance and hardware goals

### P-1 — Nothing runs on more than one device {#p-1}

**Urgency: High (against the stated goal) · Ease: Hard**

**Problem.** The README says the engine is "designed to run across multiple TPUs". No code
in `engine/` uses `pmap`, `shard_map`, `jax.sharding`, `device_put` or `jax.devices()`.
Concurrency comes from a process pool (`worker_pool.py`) that runs whole jobs side by side.
That does not split one simulation across devices. The research goal — many
lower-precision simulations run concurrently across devices — has no implementation.

**Fix.** Shard the scenario axis. The Sobol draw, path evolution, pricing and per-step
sorting for VaR are all scenario-parallel. Only the VaR/ES order statistic needs a
cross-device step (a global sort, or a gather of the tail). Sobol points also need a
per-device skip-ahead or independent scrambles, which the `seed` added in this pass
enables. Until this exists, the README should say the engine runs on one device.

### P-2 — Bermudan/American scenario pricing runs on the host in a Python loop {#p-2}

**Urgency: Medium · Ease: Moderate**

**Problem.** `price_bermudan_swaptions` copies `hw_paths` to host memory
(`np.asarray(...)`), loops over time steps in Python, and interpolates with `np.interp`.
The backward induction itself is jitted, but the scenario half, which scales with the
scenario count, never touches the accelerator. The backward induction also compiles once
per distinct trade structure (see [I-21](../known-issues.md#i-21) and
[I-22](../known-issues.md#i-22) for the recompilation problem generally).

**Fix.** Keep the condition grids on device and replace the loop with a vmapped
`jnp.interp` over steps. The interpolation semantics are the same; expect differences at
round-off level only.

**Related, found while building `engine.market_risk` (2026-09-24).** Revaluing a
Bermudan/American per scenario costs ~0.1–0.2s per scenario on CPU at `n_per_std=64`.
Two things are unexplained and worth an XLA profile before any TPU work:
- The first grid revaluation in a process sometimes runs up to 50× faster than any later
  one, with bit-identical results. `jax.clear_caches()` does not restore it, and neither
  does the batch size.
- A plain `jit(vmap(...))` over batches is as slow as the slow case.

Separately, the rollback's per-scenario working set (about 80 MB at `n_per_std=64`) made a
fixed vmap batch of 256 need ~19 GB. `engine.market_risk.revaluation.scenario_batch_size`
now caps it.

### P-3 — The precision study does not exercise the engine's own low-precision path {#p-3}

**Status: ✅ Resolved 2026-09-24.** `demos/demo_precision.py` now runs
`run_market_risk` itself at FP64 and FP32 on a sloped two-curve market with a mixed
portfolio. Measured: the FP32 error in VaR 99% and ES 97.5% is about 4e-4 of the FP64
spread across Sobol seeds. It dropped the FP16 comparison, which no pricer supports. The
finding as originally written:

**Urgency: Medium · Ease: Moderate**

**Problem.** `demos/demo_precision.py` is the project's main evidence that FP32 is "free"
for risk work. It does not call `generate_paths`: it reassembles the pipeline from private
functions (`_simulate_cross_asset_paths_jit`, `compute_hw_A_matrix`, ...), and it builds
`A(t,T)` in float64 at every precision. The engine's own `precision=32` path was never
measured.

That gap hid a real bug. `hull_white.forward_rate` used a 1e-6 finite difference that is
pure cancellation noise in float32: forward rates were off by up to two percentage points.
On the demo configuration the engine's float32 yield cube differed from float64 by
**2.6e-3 relative**. This pass fixed `forward_rate`, and the difference is now **6.8e-7**.
But the demo's conclusion was reached without seeing either number. The demo also uses a
flat curve and one rate factor, so it cannot see [M-1](#m-1).

**Fix.** Measure through `price_portfolio` with `PrecisionConfig`, on a sloped,
multi-factor market. Add float16 to the engine behind the same config, doing setup (Cholesky,
inverse CDF) in float32 as the demo already does, instead of maintaining a fork in the demo.

---

## Architecture

### A-1 — Precision is controlled by toggling a process-global JAX flag {#a-1}

**Urgency: Medium · Ease: Hard**

**Problem.** `market_model.py` enables `jax_enable_x64` **at import** (a library changing
global state as an import side effect). `generate_paths` toggles it per call.
`price_portfolio` forces it back on, and a `_PRICING_LOCK` serializes threads because the
flag is process-global. The worker pool keeps separate 32- and 64-bit process tiers so
each can "fix x64 once at boot and never touch it again". But `price_portfolio` sets it
`True` on every job, so after its first job a 32-bit worker runs with x64 on, and the
isolation the tiers exist for does not hold.

**Fix.** Leave x64 enabled for the whole process and pass `dtype` explicitly everywhere.
Most functions already derive their dtype from their inputs. Then remove the toggle, the
lock and the per-precision pool tiers. The remaining work is to find every array created
without an explicit dtype. `compute_hw_A_matrix`, which hard-codes float64, is one.

### A-2 — Two short-rate model families, bridged by a state conversion {#a-2}

**Urgency: Medium · Ease: Hard (do it with M-1)**

**Problem.** Simulation, swaps and European swaptions use the Hull-White `(A, B, r)` family.
Bermudan/American pricing uses the LGM `(H, ζ, x)` family, which the code documents as a
*different* numerical model for t>0. Scenario pricing of a Bermudan converts the simulated
`r` into an LGM `x` (`x_from_r`), so the option is conditioned on a state from a different
model. [docs/reference/ore-parity.md](../reference/ore-parity.md) calls the two "provably
equivalent"; `engine/models/lgm.py` says they are not. One of them is wrong.

**Fix.** Standardize on LGM (the model ORE's cross-asset model and Bermudan engine use),
as part of [M-1](#m-1). European swaptions can use the LGM analytic formula that
`engine.calibration.basket.price_lgm_swaption` already implements.

### A-3 — Trade configs duplicate model parameters, then validate the copies {#a-3}

**Urgency: Medium · Ease: Moderate**

**Problem.** Every swaption config carries its own `hw_a`, `hw_sigma` and
`initial_zero_curve`, which must equal the simulation's entries for its
`rate_factor_index`. `validate_portfolio_against_simulation` exists mainly to catch the
copies drifting. Swaps carry curve indices instead. Two conventions for the same fact is
the root of several past bugs (I-13, gap item 2).

**Fix.** A trade names its rate factor. Pricers read model parameters and curves from the
market. A calibrated `Sigma` belongs to the market, keyed by factor, not to each trade.

### A-4 — ORE global state: implicit evaluation dates and a mutated singleton {#a-4}

**Urgency: Medium · Ease: Moderate**

**Problem.** Every trade config defaults `evaluation_date` to
`ORE.Settings.instance().evaluationDate`, which is thread-local and defaults to the
wall-clock date. [I-28](../known-issues.md#i-28) was one symptom. `build_vanilla_swap` also
**sets** that global on every call. Results can depend on which thread ran and what ran
before it. This pass added a check that all trades in a portfolio share one evaluation
date, which catches the cross-trade case but not the implicit default.

**Fix.** Make `evaluation_date` a required field. Scope any ORE global that must be set
with a context manager that restores it.

### A-5 — Layering inversion between instruments and portfolio {#a-5}

**Urgency: Low · Ease: Moderate**

**Problem.** Every `engine/instruments/*` module imports its validators from
`engine.portfolio.validation`, a package one layer *above* it. `engine/portfolio/__init__.py`
then needs a PEP 562 lazy `__getattr__` to break the import cycle this creates.

**Fix.** Move the validators to `engine/instruments/_validation.py`, keep a re-export in
the old location for one release, and replace the lazy `__getattr__` with plain imports.

### A-6 — Demo and test infrastructure lives in the production package {#a-6}

**Urgency: Low · Ease: Moderate**

`engine/simulation/demo_scenarios.py` (demo data), `engine/validation/ore_lgm_oracle.py`
(a test oracle that writes ORE XML), `european_swaption.compute_hw_A` and
`market_model._initial_log_discount` (wrappers only tests call) all ship in `engine/`. Move
them to a `tests/support/` or `demos/` module. Tests import them, so paths change.

### A-7 — The bond pricer is a separate, plain-Python implementation {#a-7}

**Urgency: Medium · Ease: Moderate**

`engine/instruments/treasury.py` prices with Python floats and `math.exp`, and has its own
linear-interpolation loop (`_zero_rate_at`) instead of using `hull_white.zero_rate`. So bond
Greeks are bump-and-revalue instead of AD, bonds cannot take a scenario cube (I-24), and a
third interpolation implementation can drift from the other two. Rewrite it on the same
`ZeroCurve` primitives as the other pricers; scenario pricing then comes nearly free.

---

## Code quality, tests and tooling

### Q-1 — Source code is mostly narrative prose and project history {#q-1}

**Urgency: Medium · Ease: Hard (volume, not difficulty)**

Across the engine (excluding the integration package) there are 3,952 docstring lines and
922 comment lines against 4,000 lines of code: **1.2 lines of prose per line of code**.
Much of the prose is history rather than behavior: "used to", "before this existed", "W1.3",
"plan §", "I-14". There are 107 such references. Some docstrings say the opposite of the
code. For example, `_bisect_bucket_sigma` said the bootstrap "should not silently paper
over" an out-of-bracket vol, while doing exactly that (fixed in this pass). A reader has to
get through several paragraphs of incident report to find a function's contract, and the
prose goes stale faster than code because nothing checks it.

**Fix.** Rewrite module and function docstrings to state the contract (inputs, outputs,
units, invariants, the ORE correspondence) in a few lines. Move incident history to the
known-issues register or the commit log, where it already lives. Do it module by module
when touching one, not as a single sweep.

### Q-2 — No lint, type checking, CI, or pinned dependencies {#q-2}

**Urgency: Medium · Ease: Moderate**

There is no `ruff`/`flake8` configuration, no `mypy`/`pyright`, no CI workflow, no
pre-commit hooks, and no dependency in `pyproject.toml` is pinned (only `pydantic` has
a lower bound). That includes `jax`, whose numerics and APIs change between releases. The 1e-12 ORE-parity tests are only as
reproducible as the environment they run in. Add a lock file (or at least lower and upper
bounds on `jax`, `jaxlib` and `open-source-risk-engine`), a linter, and a CI job running the
fast tier of the suite.

**Status: partly resolved 2026-09-24 (pins and CI).**

- `constraints.txt` pins every package to the verified environment, and `requirements.txt`
  applies it. `pyproject.toml` bounds `jax`/`jaxlib` to `>=0.10.2,<0.11` and ORE to
  `>=1.8.16,<1.9`.
- `tests/test_environment.py` fails when the installed numerical packages drift from the
  lock, or when the two files disagree.
- `.github/workflows/ci.yml` runs the fast tier (`-m "not slow"`) on Linux on every push
  and pull request, and the full suite on demand. Every ORE-parity test is in the fast
  tier.

See [User Guide: Pinned versions](../getting-started/user-guide.md#pinned-versions) and
[Running the tests](../getting-started/user-guide.md#running-the-tests).
**Still open:** a linter, type checking and pre-commit hooks. The
[cleanup plan](codebase-cleanup-plan.md) already lists the unused imports a linter would
flag first.

### Q-3 — Test suite: slow, unmarked, and coupled to itself {#q-3}

**Urgency: Medium · Ease: Moderate**

- The full suite takes 20–25 minutes, and there is no fast tier. Nothing is marked `slow`,
  `ore` or `subprocess`, so there is no quick pre-commit run.
  *Resolved with [Q-2](#q-2) (2026-09-24):* a `slow` marker splits off worker-pool tests
  and tests of 5 s or more, and `-m "not slow"` is the fast tier CI runs.
- Test modules import helpers from other test modules
  (`from tests.test_portfolio_entrypoint import _build_trades`) and from `conftest`
  directly (`from conftest import with_scenarios`). Shared helpers belong in a support
  module or in fixtures.
- Several files are named after the project process, not the behavior they test:
  `test_portfolio_gap_fixes.py`, `test_ore_coverage_hardening.py`.
- Many tests reach into private functions (`_price_one_swap`, `_flat_curve_cube`,
  `_resolve_risk_dtype`, ...). This freezes internal structure: refactors like
  [A-3](#a-3) or [A-5](#a-5) break tests without changing behavior.
- Almost every numerical test uses a flat curve. That is why [M-1](#m-1) survived.

### Q-4 — Documentation sprawl and stale planning documents {#q-4}

**Urgency: Low · Ease: Moderate**

`known-issues.md` is 122 KB. [codebase-cleanup-plan.md](codebase-cleanup-plan.md) still says
"Plan only. No code has been changed" and that the suite aborts partway through, although
its Phase 2 is marked implemented and a later full run passed. The docs README lists six
versioned TraderX response documents as peers of the architecture guide. Mark finished plans
as archived, and keep one current status line per plan.

---

## Fixed in this pass

These were simple enough to fix directly. Each fix that changes behavior has a regression
test.

| Fix | Where | Test |
|---|---|---|
| `forward_rate` computed exactly from the interpolant; float32 forward rates were off by up to 2pp, and the engine's float32 yield cube by 2.6e-3 relative (now 6.8e-7) | `engine/models/hull_white.py` | `test_market_model.py::TestForwardRatePrecision` |
| Coupon bonds could not be sent to the worker pool: nested `CouponPeriod` dates were not frozen, so pickling failed on the HTTP path | `engine/portfolio/worker_pool.py` | `test_worker_pool.py::TestTradeFreezingRoundTrip` |
| Swap Delta/Gamma/Theta ignored `accrual_day_count` (an ACT/ACT swap's Greeks were computed on ACT/365 coupons) | `engine/risk/greeks.py` | `test_greeks.py::TestSwapGreeksHonourTheTradesOwnConventions` |
| The theta cashflow add-back ignored floating coupons | `engine/risk/greeks.py` | same class |
| LGM calibration silently returned the 2000bp bracket ceiling as the calibrated sigma when a market vol was out of range (floor saturation on infeasible vol curves stays graceful, as its existing test intends) | `engine/calibration/lgm.py` | `test_calibration_lgm.py::TestUnattainableMarketVolIsRefused` |
| Input validation used `assert` (stripped under `python -O`; also turned bad calibration requests into HTTP 500 instead of 400) | `calibration/basket.py`, `calibration/lgm.py`, `risk/greeks.py` | existing tests, now expecting `ValueError` |
| A portfolio mixing evaluation dates was priced on a shifted time axis | `engine/portfolio/request.py` | `test_portfolio_entrypoint.py::TestSingleEvaluationDate` |
| The Sobol seed was hard-coded; now `SimulationConfig.seed` (default 42, unchanged draw) and exposed in the API schema | `engine/simulation/market_model.py`, `engine/api/schemas.py` | `test_market_model.py::TestSobolSeed` |
| `demo_precision.py` forked the Sobol generator to vary the seed; now uses the engine's | `demos/demo_precision.py` | demo run |
| `reference/ORE` and `reference/traderX` were submodule gitlinks with no `.gitmodules`, so a fresh clone could not fetch them | `.gitmodules` | `git submodule status` |
| Duplicated cashflow extraction in `derive_maturity_pillars` and the aged-swap warning; dead `order` return value; duplicated comment block; stale x64 comment; `x in (inf, -inf)` finite checks | `request.py`, `var_es.py`, `validation.py`, instrument configs | existing tests |
| `.gitignore` duplicates and a typo (`.pytest_cache__/`); no pytest config (`testpaths`); demo `Run with` paths; demo risk-table headers ran together; swap demo labelled step 1 as "t=0" | repo root, `demos/`, `swap.py` | demo runs |
| The multi-step cube's loss quantiles were reported as VaR/ES (R-1); it now reports exposure profiles, and market-risk VaR/ES is a new t=0 revaluation path | `engine/risk/exposure.py`, `engine/market_risk/` | `test_exposure.py`, `test_market_risk.py`, `test_market_risk_ore_parity.py` |
| Swaption t=0 price function promoted float32 to float64 through two dtype-less constants, so `risk=32` swaption Greeks ran partly in float64 | `engine/risk/price_functions.py` | `test_market_risk.py::TestRevaluation::test_swaption_price_function_keeps_float32` |
| VaR/ES keys rounded fractional quantiles: 0.975 was labelled `ES_98`, and 0.995 and 0.999 both `…_100` | `engine/risk/var_es.py` | `test_market_risk.py::TestRun::test_basel_quantile_is_labelled_97_5`, `test_exposure.py` |
| The ORE LGM oracle wrote numpy scalars into ORE market data as `np.float64(...)`, which ORE cannot parse | `engine/validation/ore_lgm_oracle.py` | `test_market_risk_ore_parity.py` (Bermudan case) |
| The bond pricer's cashflow list is now one definition shared by the float pricer and a new JAX price function | `engine/instruments/treasury.py` | `test_market_risk.py::TestRevaluation::test_bond_equals_the_float_pricer_including_a_parallel_shift` |


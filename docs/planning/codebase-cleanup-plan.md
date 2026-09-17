# Codebase Cleanup Plan

**Date:** 2026-09-17 · **Owner:** Alex (JAX Risk Engine side)
**Status:** Plan only. No code has been changed. Every item names the exact file,
line, and verification command.

---

## 0. The constraint, stated precisely

**Test files may be edited. Test *semantics* may not.**

An edit is allowed when it is mechanical and a reviewer can confirm it is
behavior-preserving by reading the diff alone:

| Allowed | Forbidden |
|---|---|
| Changing an import path or module a symbol comes from | Changing an assertion, expected value, or tolerance |
| Deleting a now-duplicated local constant and importing it | Changing what a test *exercises* |
| Renaming a symbol at its import site to follow a rename | Adding/removing/renaming a test function or class |
| Moving a shared helper to `conftest.py` | Changing `@pytest.mark.parametrize` cases |
| Deleting a provably unused import or dead local | Skipping, xfailing, or loosening a test |

**The invariant:** the same test IDs, in the same count, asserting the same
things, producing the same results. `pytest --collect-only -q` output must be
**byte-identical** before and after every phase except where a phase explicitly
adds a test (§3.1 is the only one, and it is additive).

This is looser than the first draft of this plan assumed, and it unlocks Phases 4
and 5 (file moves and module splits), which were previously rejected on
import-churn grounds. Those are now **in scope**.

---

## 1. Baseline — read this before running anything

### 1.1 The suite does not currently complete on this machine

`pytest tests/` **aborts partway through**, at roughly test 105 of 1709, inside
`tests/test_api.py`. This is reproducible and **pre-existing** — it is present in
the repo's own `final_suite_w15.log` from before this analysis began.

The cause is environmental, not a code defect:

```
E0917 04:35:30 contiguous_section_memory_manager.cc:105]
  allocateMappedMemory failed with error:
  The paging file is too small for this operation to complete.
```

followed by `BrokenProcessPool: A process in the process pool was terminated
abruptly`. Several `ProcessPoolExecutor` workers each initialize a full JAX/XLA
runtime ([worker_pool.py:392-406](../../engine/portfolio/worker_pool.py#L392-L406)),
and together they exhaust the Windows pagefile. The observed failures are all in
the subprocess-dispatch tests:

- `test_api.py::TestPortfolioPriceWorkerPoolDispatch::test_two_precisions_submitted_back_to_back_both_complete_correctly`
- `test_api.py::TestGapFixesSurviveTheHttpBoundary::` (3 tests)

**This is not caused by, and will not be fixed by, this cleanup.** But it shapes
the plan, because *"all tests pass the same"* cannot be verified against a suite
that aborts.

**The good news, measured rather than assumed:** the blast radius is exactly the
62 subprocess-dispatch tests in `test_api.py`, `test_worker_pool.py`, and
`test_profiling_and_jit.py`. Deselecting those three files yields **1647 passed,
exit code 0** (§1.2) — so a clean gate over 96.4% of the suite is available
today, and no phase in this plan touches the worker-pool dispatch path that
fails.

### 1.2 Therefore: a two-tier baseline

Capture both, once, before touching anything.

**Tier 1 — the deterministic core (the real regression gate).** Excludes the
three subprocess-heavy files:

```bash
.venv/Scripts/python.exe -m pytest tests/ -q -p no:cacheprovider \
  --deselect tests/test_api.py \
  --deselect tests/test_worker_pool.py \
  --deselect tests/test_profiling_and_jit.py \
  > baseline-core.txt 2>&1
tail -3 baseline-core.txt
```

**This was run for this plan and is known good:**

```
1647 passed, 68 deselected, 75 warnings in 743.13s (0:12:23)     [exit code 0]
```

So **96.4% of the suite (1647 of 1709) is a clean, complete, reproducible gate**
that takes ~12½ minutes. The crash in §1.1 is confined entirely to the 62
subprocess-dispatch tests in the three deselected files. Every phase below can be
verified properly against this tier.

**Tier 2 — collection identity (the structural gate, and the strongest one).**
This is cheap, runs in under a second, and catches *any* accidental change to the
test inventory — a renamed test, a lost parametrize case, a module that stopped
importing:

```bash
.venv/Scripts/python.exe -m pytest tests/ -q --collect-only > baseline-collect.txt 2>&1
tail -1 baseline-collect.txt      # expect: 1709 tests collected
```

**Tier 3 — the subprocess files, individually.** Run one file at a time so pools
do not stack. Record which pass; some may fail on pagefile pressure regardless:

```bash
.venv/Scripts/python.exe -m pytest tests/test_worker_pool.py -q -p no:cacheprovider
.venv/Scripts/python.exe -m pytest tests/test_profiling_and_jit.py -q -p no:cacheprovider
.venv/Scripts/python.exe -m pytest tests/test_api.py -q -p no:cacheprovider
```

### 1.3 After every phase

```bash
diff baseline-collect.txt <(.venv/Scripts/python.exe -m pytest tests/ -q --collect-only 2>&1)
# MUST be empty (except the one added test in §3.1)

diff <(tail -3 baseline-core.txt) <(tail -3 after-core.txt)
# MUST be empty
```

**Stop rule:** any difference that a phase did not explicitly plan for → revert
that phase and re-plan it.

### 1.4 Optional: make the suite completable

Not part of the cleanup, but it would restore a real full-suite gate. Either
raise the Windows pagefile, or make pool size configurable for test runs via an
env var read by
[`_pool_for`'s `pool_size`](../../engine/portfolio/worker_pool.py#L392). The
latter is a code change with behavior implications — **propose separately, do not
fold into this cleanup.** Register as a known issue instead.

---

## 2. What the analysis found

The codebase is **in better shape than "messy" implies**, which is why this plan
is surgical rather than sweeping.

- **1709 tests collect in 0.91s with zero import errors.** The package graph is sound.
- **Zero dead private functions.** An AST sweep of every `engine/` module for
  module-level `_private` symbols never referenced anywhere returned nothing.
- **`engine/integration/__init__.py` has zero export drift** — every imported name
  is in `__all__` and vice versa (AST-verified).
- **The architectural invariants are real and tested** — the `engine.integration`
  → no-simulation-pricer ban
  ([test_integration_pipeline.py:267-287](../../tests/test_integration_pipeline.py#L267-L287)),
  the `engine.portfolio` PEP 562 circular-import break.
- **The heavy docstrings are the documented house style**
  ([coding-style.md](../concepts/coding-style.md)). This plan does not touch them.
  "Bloated" here means *code* doing too much in one place, not prose.

The mess is **localized**: duplicated constants, one dead wrapper, test
boilerplate, two oversized modules, and documentation drift.

**Out of scope:** `reference/ORE/` (~700 vendored third-party files, already
`/reference/`-ignored).

### 2.1 The tree moved during this analysis — W1.5 landed

`engine/instruments/treasury.py` (390 lines) and three test files
(`test_treasury_instrument.py`, `test_api_bond_schemas.py`,
`test_portfolio_bond_wire_through.py`) appeared mid-analysis. Test count went
**1618 → 1709**. All findings below are verified against the **current** tree.

W1.5 also **added a fourth `ORE.Actual365Fixed()`** and a **third `RATE_BUMP`** —
see §3.1 and §3.2. The duplication this plan targets is actively growing, which
is the strongest argument for doing Phase 2 now rather than later.

---

## 3. Phase 2 — Duplicated constants ✅ **IMPLEMENTED 2026-09-17**

> **Status: done.** What landed, and what was deliberately not done:
>
> | Item | Outcome |
> |---|---|
> | §3.1 `TIME_AXIS_DAY_COUNTER` ×3 → ×1 | ✅ Done. `bermudan_swaption` and `greeks` now import the canonical object. |
> | §3.1 added test | ✅ `TestTimeAxisIsOneObject`, 2 tests (identity + an AST guard against re-introduction). |
> | §3.4 dead `_zero_curve_of` wrapper | ✅ Deleted; inlined at all 10 call sites. |
> | §3.3 `NO_TERMS_ARTIFACT` ×2 → ×1 | ✅ Done as a one-line re-export — **no new module**. |
> | §3.3 `vocabulary.py` leaf module | ❌ **Skipped deliberately** — see §3.3. |
> | §3.3 `OK`/`UNAVAILABLE` merge | ❌ **Skipped deliberately** — they are different vocabularies. See §3.3. |
> | §3.2 `RATE_BUMP` ×3 | ❌ Skipped — see §3.2. |
> | §3.5 `_HwZeroCurve` alias rename | ❌ Skipped — cosmetic, 6 files, no correctness value. |
>
> **Verification:** collection went 1716 → 1718, diff showing *only* the two
> added tests; no existing test renamed, removed, or altered. The regression
> guard was proven by injecting a duplicate `ORE.Actual365Fixed()` into
> `greeks.py` and confirming all three relevant tests go red, then reverting.

---

### Original analysis

Phase ordering note: Phase 1 (§7, unused imports) is listed later because it is
trivial. **Phase 2 is the one that matters.** Each item is a live
silent-divergence risk: two definitions of one value that can drift apart with no
test failing.

### 3.1 `ORE.Actual365Fixed()` is constructed in four places

| Location | Name |
|---|---|
| [ore_builders.py:85](../../engine/models/ore_builders.py#L85) | `TIME_AXIS_DAY_COUNTER` (canonical) + `DAY_COUNTER` alias (line 89) |
| [bermudan_swaption.py:176-179](../../engine/instruments/bermudan_swaption.py#L176-L179) | `TIME_AXIS_DAY_COUNTER` + `DAY_COUNTER` — **re-constructed** |
| [greeks.py:141-144](../../engine/risk/greeks.py#L141-L144) | `TIME_AXIS_DAY_COUNTER` + `DAY_COUNTER` — **re-constructed** |
| [treasury.py:83](../../engine/instruments/treasury.py#L83) | `DISCOUNT_DAY_COUNT` — **new in W1.5** |

`bermudan_swaption.py` and `greeks.py` **already import from `ore_builders`**
([line 159](../../engine/instruments/bermudan_swaption.py#L159),
[line 124](../../engine/risk/greeks.py#L124)) — there is no import-cycle
justification. It is copy-paste.

**Why this is not pedantry.** The entire purpose of
[engine/day_count.py](../../engine/day_count.py) is that the *time axis* role is
permanently ACT/365 and **not configurable**, in deliberate contrast to the
per-instrument accrual role. That invariant is asserted by
[test_day_count_roles.py:245-259](../../tests/test_day_count_roles.py#L245-L259) —
**but that test only checks `ore_builders`' copy.** The other three are
unasserted. Change the time axis and three sites silently keep the old value
while the test stays green.

**Fix (for the two re-constructions):**

```python
# bermudan_swaption.py and greeks.py — replace the local assignments with:
from engine.models.ore_builders import TIME_AXIS_DAY_COUNTER, DAY_COUNTER  # noqa: F401
```

Keep `DAY_COUNTER` re-exported from both — it is public surface that
`test_day_count_roles.py` reaches for.

**`treasury.py`'s `DISCOUNT_DAY_COUNT` — ✅ decided: left as its own object.**
It is a **third role**, and its own docstring says so explicitly: *"This is the
*discounting* convention and is distinct from the instrument's own *accrual*
convention (the W1.1 split)."* It is used once
([treasury.py:280](../../engine/instruments/treasury.py#L280)) to discount a bond's
cashflows, never to index the simulated cube — W1.1's whole point is that roles
with different reasons to change get different names, so collapsing a third role
into role 1 because both are currently ACT/365 would undo that split rather than
honor it.

The AST guard added in this phase checks only `TIME_AXIS_DAY_COUNTER` and
`DAY_COUNTER` by name, so it correctly leaves this constant alone (verified).

**Then add one test** — the only addition in this plan, closing the gap that
allowed the drift. Identity (`is`), not equality, because two distinct
`Actual365Fixed()` objects compare equal but are not one source of truth:

```python
def test_time_axis_day_counter_is_one_object_everywhere(self):
    from engine.models import ore_builders
    from engine.instruments import bermudan_swaption
    from engine.risk import greeks
    assert bermudan_swaption.TIME_AXIS_DAY_COUNTER is ore_builders.TIME_AXIS_DAY_COUNTER
    assert greeks.TIME_AXIS_DAY_COUNTER is ore_builders.TIME_AXIS_DAY_COUNTER
```

Baseline goes 1709 → 1710. **This is the only expected collection change in the
entire plan.**

### 3.2 `RATE_BUMP = 1e-4` is defined three times under two names

| Location | Name |
|---|---|
| [greeks.py:131](../../engine/risk/greeks.py#L131) | `DEFAULT_RATE_BUMP = 0.0001` |
| [note.py:134](../../engine/integration/note.py#L134) | `RATE_BUMP = 1e-4` |
| [treasury.py:86](../../engine/instruments/treasury.py#L86) | `RATE_BUMP = 1e-4` — **new in W1.5** |

All three are the same 1bp bump, sourced from the same ORE sensitivity-config
default. `greeks.py` and `note.py` cannot share (integration must not import
risk), but the *value and its provenance* could live in a leaf.

**❌ Skipped, deliberately.** A bump size that drifts produces a differently-
*scaled* sensitivity, not a wrong one, and each of the three is independently
tested. More importantly the three cannot share a home without inventing one:
`greeks.py` (risk layer) and `note.py` (integration layer) are separated by the
tested import ban, and `treasury.py`'s copy is part of the deliberate
instrument-layer independence described in §6.3. Since §3.3's `vocabulary.py` was
not created either, there is nowhere natural for it to land — and creating a
module solely to host `1e-4` would be ceremony. The three stay as they are.

### 3.3 Status and reason-code strings defined twice

| Constant | Definition A | Definition B |
|---|---|---|
| `OK = "ok"` | [normalize.py:58](../../engine/integration/normalize.py#L58) | [result.py:78](../../engine/integration/result.py#L78) |
| `UNAVAILABLE = "unavailable"` | [normalize.py:59](../../engine/integration/normalize.py#L59) | [result.py:80](../../engine/integration/result.py#L80) |
| `NO_TERMS_ARTIFACT` | [normalize.py:70](../../engine/integration/normalize.py#L70) | [terms.py:78](../../engine/integration/terms.py#L78) |

The symptom is visible at
[pipeline.py:88-97](../../engine/integration/pipeline.py#L88-L97), which must
disambiguate with aliased imports (`NO_TERMS_ARTIFACT as
NORMALIZE_NO_TERMS_ARTIFACT`) — the importer has to remember *which module's*
copy of the same string it holds.

### ✅ What was actually done — and why it is much smaller than proposed

The original proposal here was a new `engine/integration/vocabulary.py` leaf
holding all of these. **On inspection that was over-engineering, and the two
halves of the problem turned out to be genuinely different.**

**`NO_TERMS_ARTIFACT` — fixed, in one line.** It really is one fact ("the bundle
shipped no terms artifact"), used as a reason code by both modules with identical
meaning. `normalize.py` **already imported from `terms.py`**
([normalize.py:49](../../engine/integration/normalize.py#L49)), so no cycle and no
new module was needed — `normalize` now re-exports it from `terms`, which is where
the join that discovers the condition lives. The
`NO_TERMS_ARTIFACT as NORMALIZE_NO_TERMS_ARTIFACT` alias in `pipeline.py` was
dropped, since the ambiguity it worked around no longer exists. Creating a whole
module to rehome a single string would have been ceremony, not engineering.

**`OK` / `UNAVAILABLE` — deliberately NOT merged.** The hazard flagged below was
real, and the source settles it.
[normalize.py:56-57](../../engine/integration/normalize.py#L56-L57) describes its
constants as *"`Quantity.status` values. A deliberately small vocabulary
**mirroring** the per-calculation statuses in `engine.integration.result`."*
Mirroring — not the same thing. `Quantity.status` has **two** values (a
normalization outcome); `result.STATUSES` has **five** (a per-calculation
reporting status, including `unsupported`/`failed`/`not-applicable`, which are
meaningless for a unit conversion). They coincide on two strings today by
design, and are free to diverge.

Merging them would have fused two layers' vocabularies because two string
literals happened to match — *precisely the "worse bug than the duplication"*
warned about below. They stay separate.

With both halves resolved this way, `vocabulary.py` had nothing left to hold, so
it was never created.

> ### ⚠ Two hazards, both load-bearing
>
> **1. A new file in `engine/integration/` is scanned by an existing test.**
> [test_integration_pipeline.py:267](../../tests/test_integration_pipeline.py#L267)
> globs `engine/integration/*.py` and AST-walks every file for banned imports
> (`jax`, `engine.instruments`, `engine.models`, `engine.risk`,
> `engine.simulation`, `engine.portfolio`). `vocabulary.py` must import
> **nothing**. If it does, that test fails — which is the system working
> correctly.
>
> **2. `normalize.OK` and `result.OK` may be different concepts.** They are the
> same *string* today, but one is a normalization provenance and the other a
> calculation status. **Confirm they are genuinely one concept before merging.**
> If they merely coincide, give them distinct names (`NORMALIZE_OK` /
> `STATUS_OK`). *Fusing two distinct ideas because their values collide is a
> worse bug than the duplication being fixed.*

### 3.4 A dead wrapper, and a name collision

[engine/portfolio/request.py:609](../../engine/portfolio/request.py#L609):

```python
def _zero_curve_of(cfg, curve_config, dtype=jnp.float64) -> _HwZeroCurve:
    return _HwZeroCurve.from_config(curve_config, dtype=dtype)
```

**`cfg` is never used.** All nine call sites pass a config the function ignores.
It is a pure pass-through with a vestigial first argument — *and* it collides in
name with [bermudan_swaption.py:445](../../engine/instruments/bermudan_swaption.py#L445)'s
`_zero_curve_of`, a genuinely different function with real dtype-preservation
logic that is reached directly by
[test_bermudan_swaption.py:560](../../tests/test_bermudan_swaption.py#L560).

Two same-named private helpers, one a no-op, is exactly the "unclear logic" the
request describes.

**Fix:** delete `request._zero_curve_of`; inline
`_HwZeroCurve.from_config(curve_config, dtype=...)` at its nine call sites (lines
784, 1040, 1041, 1045, 1046, 1051, 1052, 1059, 1060, 1071). **Leave
`bermudan_swaption._zero_curve_of` untouched** — it earns its existence, and a
test imports it by name.

### 3.5 The `_HwZeroCurve` alias — optional

`engine.models.hull_white.ZeroCurve` is imported as `_HwZeroCurve` in six modules.
The alias made sense when competing `ZeroCurve` types existed; there is now one,
and [greeks.py:125](../../engine/risk/greeks.py#L125) already imports it
unaliased. Cosmetic. If done, do it as one mechanical rename in its own commit.
**Check [test_greeks.py:442](../../tests/test_greeks.py#L442) first** — it imports
`ZeroCurve as GreeksZeroCurve` and would need its import line updated (an allowed
mechanical edit).

### Verification

```bash
grep -rn "ORE.Actual365Fixed()" engine/     # once in ore_builders, plus treasury if kept
diff baseline-collect.txt <(pytest tests/ -q --collect-only 2>&1)   # +1 test only
```

---

## 4. Phase 3 — Test-suite boilerplate

Two constants duplicated across the integration tests, with an existing home:
[tests/conftest.py](../../tests/conftest.py).

- **`FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"`** — verbatim in
  **15** files.
- **`MARKET = {"mode": "assumed-profile", "assumedProfileId": "flat-3pct-v1"}`** — in
  **6** files.

**Fix.** Define both in `conftest.py`; replace each local definition with
`from conftest import FIXTURES, MARKET`.

**Why constants, not fixtures.** `MARKET` is used at *module scope* as a default
argument
([test_integration_accrual_source.py:38](../../tests/test_integration_accrual_source.py#L38)),
and `FIXTURES` appears in `@pytest.mark.parametrize` decorators, which evaluate at
collection time. A pytest fixture is unreachable from either position.

> **Verify the import style on ONE file before doing all 15.** `from conftest
> import X` depends on pytest's rootdir/`prepend` import mode, and this repo has
> **no pytest configuration at all** (§4.1). If it does not resolve, use a plain
> `tests/_fixtures.py` helper module — guaranteed to work, costs one extra file.

This is a pure delete-one-line/add-one-import per file: **21 lines removed, 15
added, zero assertions touched.**

### 4.1 There is no pytest configuration

No `[tool.pytest.ini_options]` in `pyproject.toml`, and no `pytest.ini`,
`setup.cfg`, or `tox.ini`. Everything runs on defaults and rootdir inference.

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
# `reference/` holds vendored ORE source with its OWN conftest.py and hundreds of
# run.py files; a bare `pytest .` from the root can wander into it.
norecursedirs = ["reference", ".venv", ".claude", "*.egg-info"]
```

Worth doing precisely because
[reference/ORE/Examples/conftest.py](../../reference/ORE/Examples/conftest.py)
exists — a bare `pytest` is one careless invocation from collecting vendored
third-party tests.

**Do NOT add `-W error`, `--strict-markers`, or coverage gates.** Those change
which tests pass, violating the core constraint.

---

## 5. Phase 1 — Unused imports and stray artifacts *(trivial)*

### 5.1 Genuinely unused imports

Found via `pyflakes`, **each hand-checked** against the module that declares it;
intentional re-exports excluded.

| File | Line | Symbol |
|---|---|---|
| [api/routes.py](../../engine/api/routes.py#L49) | 49 | `price_portfolio` — leftover from the pre-worker-pool architecture |
| [api/schemas.py](../../engine/api/schemas.py#L22) | 22 | `typing.Sequence` |
| [integration/bundle.py](../../engine/integration/bundle.py#L62) | 62 | `dataclasses.field` |
| [integration/market_inputs.py](../../engine/integration/market_inputs.py#L58) | 58 | `dataclasses.field` |
| [integration/result.py](../../engine/integration/result.py#L54) | 54 | `dataclasses.field` |
| [integration/terms.py](../../engine/integration/terms.py#L70) | 70 | `dataclasses.field` |
| [integration/pipeline.py](../../engine/integration/pipeline.py#L73) | 73, 87 | `typing.Sequence`, `MarketInputsNotSupplied` |
| [instruments/swap.py](../../engine/instruments/swap.py#L53) | 53 | `typing.Any`, `typing.Dict` |
| [instruments/bermudan_swaption.py](../../engine/instruments/bermudan_swaption.py#L162) | 162 | `_H`, `_H_prime`, `_lgm_x_from_r` |
| [portfolio/worker_pool.py](../../engine/portfolio/worker_pool.py#L83) | 83 | `typing.Optional` |
| [risk/greeks.py](../../engine/risk/greeks.py#L101) | 101, 118 | `typing.Union`, `_PreparedBermudan` |

**Re-verify `engine/integration/terms.py` and `engine/api/schemas.py` before
editing** — both are modified in the working tree and may have changed since.

### 5.2 DO NOT delete — deliberate re-exports

Already `# noqa: F401`-marked; all load-bearing:

- [portfolio/\_\_init\_\_.py:39](../../engine/portfolio/__init__.py#L39) — the PEP 562 circular-import break
- [portfolio/request.py:107](../../engine/portfolio/request.py#L107) — makes `engine.portfolio._validate_common_fields` resolve where the design names it
- [models/ore_builders.py:107](../../engine/models/ore_builders.py#L107) — re-exports `engine.day_count`'s table

[instruments/swap.py:61](../../engine/instruments/swap.py#L61)'s
`TIME_AXIS_DAY_COUNTER`/`LegCashflows` are flagged but **verify first** —
`grep -rn "from engine.instruments.swap import" engine/ tests/` — other modules
may pull them through.

### 5.3 Leave alone

`american_swaption.py:166/171` and `bermudan_swaption.py:1027` are inside
`if __name__ == "__main__":` demo blocks. Unused only because the block does not
reference them; deleting makes the demo blocks inconsistent with each other.

### 5.4 Test-file dead locals — **read before deleting**

A dead local is sometimes a *dropped assertion*, not clutter. Inspect each:

- [test_diverse_portfolio_e2e.py:291](../../tests/test_diverse_portfolio_e2e.py#L291) `base_cube`, [:303](../../tests/test_diverse_portfolio_e2e.py#L303) `r0_path`
- [test_european_swaption.py:51](../../tests/test_european_swaption.py#L51) `first_accrual_start`, [:359](../../tests/test_european_swaption.py#L359) `prepared`

If one turns out to be a dropped assertion, **that is a bug to report, not to
fix here** — restoring it changes test semantics.

### 5.5 Stray artifacts

- **`__pycache__/_jitpatch_plugin.cpython-311-pytest-9.1.1.pyc`** — a compiled
  pytest plugin **whose source exists nowhere in the repo**. A stale `.pyc` with
  no source can still be imported under some path configurations. Delete the root
  `__pycache__/`.
- **`__pycache__/demo.cpython-311.pyc`** — orphaned; `demo.py` moved to `demos/`.
- **`.claude/worktrees/agent-a1f1a12a4992cfd92/`** — 1.2 MB stale copy of an
  *older* tree (still has `engine/trades/` and root `demo.py`). Git-ignored, but a
  working-directory trap: `grep -r` from the root returns matches from code that
  no longer exists. **Delete.**
- **`final_suite_w15.log`** — untracked 100+ KB crash log in the repo root. Move to
  a scratch dir or add `*.log` to [.gitignore](../../.gitignore).
- **`.profile-out/`, `.profile-out-small/`** — correctly ignored; safe to delete,
  they regenerate.

---

## 6. Phase 4 — Misplaced files

Each move uses the same safe pattern: **move, then leave a re-export shim at the
old path**, so nothing breaks and the move is verifiable independently of
call-site cleanup.

### 6.1 `engine/api/schemas.py` holds orchestration, not schemas

**The clearest genuine misplacement in the codebase.**

`PortfolioRequestSchema.to_dataclass()`
([schemas.py:322-359](../../engine/api/schemas.py#L322-L359)) does not translate a
schema. It:

1. scans trades for the first uncalibrated Bermudan/American;
2. raises a domain error if none is found;
3. builds a `ZeroCurve`;
4. **calls `build_coterminal_basket(...)`**
   ([calibration/basket.py:96](../../engine/calibration/basket.py#L96)) — a real
   calibration entry point.

That is pricing-orchestration policy in a file whose contract is "Pydantic mirrors
of the engine dataclasses." Its own comment admits the coupling: *"`_fill_
calibrated_sigma` already assumes a single shared basket applies uniformly … so
this mirrors that."* **A policy hand-mirrored in two modules will drift.**

The symptom from the other side: [routes.py:59](../../engine/api/routes.py#L59)
imports `_parse_ore_date` — a **private** name — out of `schemas.py`.

**Fix, in two safe steps:**

1. Create `engine/api/converters.py`; move `_parse_ore_date`/`_parse_ore_period`
   there as **public** `parse_ore_date`/`parse_ore_period`. Re-export the old
   private names from `schemas.py` so nothing breaks. *A name another module
   imports is public whatever it is called.*
2. Extract the calibration-basket construction into
   `converters.py::build_calibration_targets(trades, basket_schema)`.
   `to_dataclass` calls it and returns to being a translation. Pure
   extract-function, no logic change.

**Deferred deliberately:** unifying the duplicated "first uncalibrated trade"
policy with [request.py::_fill_calibrated_sigma](../../engine/portfolio/request.py#L753)
is a **behavior** question, not cleanup. Log it in `known-issues.md`.

⚠ `engine/api/schemas.py` is **modified in the working tree** — re-read before
editing.

### 6.2 `engine/simulation/demo_scenarios.py`

Example/test scenario data inside `engine.simulation`, a package `pyproject.toml`
ships to users. Its docstring calls it *"Canonical demo/reference scenarios."*

[architecture.md:428](../concepts/architecture.md#L428) claims nothing depends on
it *"except demo code and tests — it is never required."* **That is now false** —
[conftest.py:23](../../tests/conftest.py#L23) imports it at module scope, and six
`engine/` `__main__` blocks reach for it.

**Still recommend NOT moving it**, even under the relaxed constraint. The blocker
is not test churn — it is that six `engine/` modules import it, and `engine/`
importing from `tests/` would be strictly worse than today. Instead:

1. **Fix the false claim** in architecture.md.
2. **Add a note** to the module docstring: it lives in `engine/` because
   `engine/*/__main__` blocks depend on it, and `engine/` must never import from
   `tests/`.

*A documented deliberate choice is not mess. An undocumented one the docs actively
contradict is.*

### 6.3 `engine/instruments/treasury.py` duplicates `integration/bill|note` — **leave it**

W1.5's module restates the discounting convention (`_zero_rate_at`,
`_discount_factor`, ACT/365 continuously compounded) that
`engine.integration.bill`/`note` already implement.

**This is deliberate and correct.** Its docstring
([treasury.py:57-68](../../engine/instruments/treasury.py#L57-L68)) explains:
`engine.integration` sits *above* `engine.instruments`, and reversing that to
share ~20 lines of `exp(-r*t)` would couple the instrument layer to the TraderX
bundle format. Critically, **the duplication is pinned by a cross-check test** —
`TestAgreesWithTheIntegrationPricers` prices the same instrument through both
paths and asserts agreement to the cent.

Duplication with a test that fails on divergence is not the same risk as §3.1's
unasserted copies. **Listed so it is not "tidied" into a layering violation.**

### 6.4 `engine/day_count.py` at package root — **leave it**

The only module directly under `engine/`. Placement is deliberate: importable by
both `engine.models.ore_builders` and `engine.integration.note` without either
reaching the other, asserted by
[test_integration_pipeline.py:302-310](../../tests/test_integration_pipeline.py#L302-L310).

Listed so a future reader does not "tidy" it into `engine/models/` and break the
invariant. Consider documenting it in `engine/__init__.py` (currently 0 bytes).

---

## 7. Phase 5 — Oversized modules

Only where size reflects **multiple responsibilities**, not thorough docs.

> **Before any split, read §7.3.** Tests import private symbols directly from both
> target modules. Splits must preserve those import paths.

### 7.1 `engine/portfolio/request.py` — 1080 lines, four responsibilities

| Lines | Concern |
|---|---|
| 126-246 | Precision config (`PricingPrecisionOverride`, `RiskPrecisionOverride`, `PrecisionConfig`, `_dtype_of`, `_resolve_pricing_dtype`, `_resolve_risk_dtype`) |
| 250-505 | Validation (`validate_portfolio_against_simulation`, `_validate_swap_curve_indices`, `_warn_if_aged_swap_exposure`, `_warn_if_not_reset_aligned`) |
| 506-608 | Request/result dataclasses + `derive_maturity_pillars` |
| 613-1080 | Orchestration (`price_portfolio` + nine private helpers) |

**The precision block is the clean extraction** — no dependency on any instrument
config or on `price_portfolio`. Move to `engine/portfolio/precision.py`,
re-export from `request.py`.

Validation second, to `engine/portfolio/validators.py` — **a different file from
the existing leaf `validation.py`**. `validate_portfolio_against_simulation`
imports instrument configs, which `validation.py`'s docstring says it must never
do, so a separate file is **required**, not stylistic.

Keep `engine/portfolio/__init__.py`'s `_REQUEST_EXPORTS` frozenset resolving every
public name — `test_portfolio*.py` and `engine/api/schemas.py` both import
`PrecisionConfig`/`PricingPrecisionOverride` from `engine.portfolio`.

⚠ **`request.py` is modified in the working tree** — re-read before editing.

### 7.2 `engine/integration/pipeline.py` — 720 lines, wiring plus policy

Its docstring claims *"Everything here is wiring; each step's judgement lives in
its own module."* **No longer accurate.** It holds four per-instrument outcome
builders — `_bill_outcomes` (353), `_note_outcomes` (392), `_equity_outcomes`
(458), `_accrued_outcome` (527) — plus parsing helpers.

**Optional:** extract the four builders to `engine/integration/outcomes.py`,
leaving `pipeline.py` the dispatcher it claims to be.

> ⚠ **`_priced_outcomes` encodes a load-bearing dispatch ORDER.**
> ([pipeline.py:307-352](../../engine/integration/pipeline.py#L307-L352)) Equity is
> checked *before* the market-input check and *before* `_parse_signed_face`,
> because an equity's `quantity` is a signed share count, not a currency face.
> Documented in the module docstring and pinned by tests. **Keep
> `_priced_outcomes` intact as the single dispatcher; move only leaf builders. If
> that cannot be done cleanly, skip this item** — the cost of getting it wrong
> exceeds the tidiness gained.
>
> Also: a new file in `engine/integration/` is AST-scanned by
> [test_integration_pipeline.py:267](../../tests/test_integration_pipeline.py#L267)
> for banned imports.

### 7.3 ⚠ Tests import private symbols from both split targets

These import lines must keep resolving. **Re-exporting from the original module
is the safest approach** — then zero test edits are needed:

**From `engine.portfolio.request`:**
- [test_portfolio_entrypoint.py:481](../../tests/test_portfolio_entrypoint.py#L481) `_resolve_risk_dtype`
- [test_portfolio_entrypoint.py:537](../../tests/test_portfolio_entrypoint.py#L537) `_fill_calibrated_sigma`
- [test_portfolio_entrypoint.py:568](../../tests/test_portfolio_entrypoint.py#L568) `_resolve_pricing_dtype`, `_resolve_risk_dtype`
- [test_portfolio_gap_fixes.py:51](../../tests/test_portfolio_gap_fixes.py#L51) `_swap_curve_configs`
- [test_portfolio_bond_wire_through.py:24](../../tests/test_portfolio_bond_wire_through.py#L24) (new, W1.5)

**From `engine.integration.pipeline`:**
- [test_integration_accrual_source.py:31](../../tests/test_integration_accrual_source.py#L31) `_accrual_source`
- [test_integration_equity.py:413](../../tests/test_integration_equity.py#L413) `_equity_outcomes`
- [test_integration_note.py:903](../../tests/test_integration_note.py#L903) `_note_outcomes`
- [test_integration_pipeline.py:162](../../tests/test_integration_pipeline.py#L162) `_build_item`

Updating these import lines *is* an allowed mechanical edit — but re-exporting
costs nothing and keeps the diff smaller.

### 7.4 Large but correctly organized — leave alone

- **`bermudan_swaption.py` (1051)** — banner-sectioned; the LGM backward induction
  is one algorithm, and splitting scatters a single derivation.
- **`greeks.py` (900)** — cleanly sectioned by instrument. Already right.
- **`market_model.py` (762)** — 211 lines are the `__main__` demo; the module is ~550.
- **`note.py` (739)**, **`treasury.py` (390)** — one instrument, one pricer each.

---

## 8. Phase 6 — Documentation drift

### 8.1 The root `README.md` contradicts the roadmap

| Component | README says | Reality |
|---|---|---|
| Live API (TraderX integration) | 🔜 Planned | ✅ **Shipped** — `engine/api/`, phase 11 ✅, `test_api.py` |
| Greeks | *absent* | ✅ Shipped — phase 7 ✅ |
| LGM calibration | *absent* | ✅ Shipped — phase 9 ✅ |
| Precision research | *absent* | ✅ Shipped — phase 12 ✅ |
| EOD integration boundary | *absent* | ✅ Shipped — 15 test files |
| Treasury bills/notes | *absent* | ✅ Shipped — W1.5, `treasury.py` |

The README is the **first thing a reader sees**, and it understates the project by
five subsystems while marking a shipped one "Planned." Rebuild from
[roadmap-and-history.md](roadmap-and-history.md)'s status column.

⚠ `docs/README.md` is modified in the working tree — re-read first.

### 8.2 `known-issues.md` — scrambled ID ordering

1132+ lines, IDs running:

```
I-01 I-02 I-03 | I-04 I-06 | I-05 I-07 I-16 I-18 I-08 I-09 I-10 I-11 I-12 I-21 I-22 | I-13 I-14 I-15 I-17 I-19 I-20
```

**Two separate `## FIXED` sections** (lines 93 and 781), with I-16/I-18 wedged
between I-07 and I-08. Finding an issue by ID means scanning the file. W1.5 adds
**I-24**, so this is still growing.

**Fix (presentation only, no content rewrite):** add a numeric ID index table at
the top (every issue already has an `{#i-NN}` anchor, so it is mechanical); merge
the two FIXED sections, keeping "found during the TraderX EOD exchange" as a
`###` subheading; sort within each status section by ID.

### 8.3 `docs/planning/` — six versioned response docs, no index

These are **dated correspondence with a counterparty**, cited by
[traderx-integration-plan.md](traderx-integration-plan.md) as *"the negotiation, in
order."* **Do not delete or merge them.**

But the directory mixes three kinds of document with no signposting:
correspondence, live plans, and historical/implemented ones.

**Fix:** add `docs/planning/README.md` — an index naming each document's kind,
date, and status. No files move, no links break. *(The alternative — moving the
six into `eod-correspondence/` — breaks every relative link between them, and
`.md` links are checked by no test.)*

### 8.4 `requirements.txt` omits the `profiling` extra

[requirements.txt](../../requirements.txt) is `-e .[api,dev]`; `pyproject.toml`
declares a third extra, `profiling` (xprof), required by
[demo_profile_small.py](../../demos/demo_profile_small.py) and
[profiling.md](../concepts/profiling.md).

Do **not** add it to the default install — xprof is heavy and only needed for
trace capture. Add a comment naming it and how to get it
(`pip install -e .[api,dev,profiling]`), so the omission reads as a decision.

### 8.5 `architecture.md` layout section is stale

[architecture.md:108](../concepts/architecture.md#L108)'s repository tree predates
`engine/integration/`'s newest modules (`schema.py`, `schema_version.py`,
`workload.py`), `engine/api/eod_routes.py`, and W1.5's
`engine/instruments/treasury.py`. Regenerate the tree block.

---

## 9. Explicitly rejected — and why

| Proposal | Why rejected |
|---|---|
| Trim the long module docstrings | Documented house style; they encode *why* decisions were made — the most expensive knowledge to recover. "Bloated" refers to logic. |
| Split `greeks.py` / `bermudan_swaption.py` | Already banner-sectioned. Splitting scatters one algorithm. |
| Deduplicate `treasury.py` against `integration/bill|note` | Deliberate: reversing it violates layering. Pinned by `TestAgreesWithTheIntegrationPricers`. (§6.3) |
| Move `demo_scenarios.py` out of `engine/` | Six `engine/__main__` blocks import it; `engine/` importing `tests/` is worse. Document instead. (§6.2) |
| Move `engine/day_count.py` into `engine/models/` | Breaks the tested integration-must-not-import-models invariant. |
| Delete `_PRICING_LOCK` | Deliberate defense-in-depth after the worker-pool rewrite. Not dead. |
| Rename test files to match modules | Named for *why they were written* (a bug class) — useful provenance. Renaming churns 1709 test IDs, violating collection identity. |
| Merge the six EOD response docs | Dated correspondence; the chain is cited as evidence. |
| Add `-W error` / coverage gates | Changes which tests pass. Violates the core constraint. |
| Unify the duplicated calibration policy (§6.1) | A behavior change, not cleanup. Log as a known issue. |
| Fix the pagefile/pool-size crash (§1.1) | Pre-existing and environmental. Real, but a separate change with behavior implications. |

---

## 10. Summary

| Phase | Engine files | Test files | Risk | Collection change |
|---|---|---|---|---|
| 2. Duplicated constants | ~8 | 0–1 | Low | **+1** (§3.1) |
| 3. Test boilerplate → conftest | 1 | 15 (one line each) | Low | none |
| 1. Unused imports + artifacts | ~13 | 4 (dead locals) | None | none |
| 6. Documentation | ~6 docs | 0 | None | none |
| 4. Misplaced files | ~4 | 0 | Medium | none |
| 5. Module splits | ~4 | 0 (if re-exported) | Medium | none |

**Recommended order:** 2 → 3 → 1 → 6, then optionally 4 → 5. Commit after each.

**Highest value:** §3.1 (the day-counter, now duplicated **four** ways after
W1.5), §3.4 (the dead wrapper and name collision), §8.1 (the README), §8.2 (the
issue index). Those four remove everything *actively misleading*: a constant that
can silently diverge from its only tested copy, two different functions sharing
one name, a README understating the project by five subsystems, and an
unnavigable issue register.

**Phases 2, 3, 1, and 6 are near-zero-risk** and can be done in one sitting.
Stopping there leaves the codebase meaningfully cleaner with almost no chance of
breakage. **Phases 4 and 5 are genuinely optional** — worth doing only if the
oversized modules are actively causing friction.

**Stop rule:** if `--collect-only` output differs from baseline by anything other
than §3.1's single added test, or if the Tier-1 core summary changes at all →
revert that phase and re-plan. The constraint is not "tests still pass" — it is
"**the same tests pass the same way**."

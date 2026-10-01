# Precision

Design and plan for adjustable numeric precision, from float64 down to FP8, and for the
execution architecture that runs it. It implements [I-55](../known-issues.md#i-55),
[F-07](../features.md#f-07), [I-12](../known-issues.md#i-12) and
[I-72](../known-issues.md#i-72), and the owner decisions A-9 and A-10 to A-16 and the revised
D-9 ([compliance/decisions.md](../../../compliance/decisions.md)). It replaces
`sub-fp32-precision.md`; its findings are kept, corrected, in [§15](#15-measurements).

**Written:** 2026-10-01 · **Steps:** roadmap [1.4 to 1.8](../roadmap.md#stage-1--structure),
[2.7, 2.8](../roadmap.md#stage-2--correctness-and-precision),
[3.2, 3.4](../roadmap.md#stage-3--performance), [6.3](../roadmap.md#stage-6--features)

---

## 1. Purpose

The project's research question (root [README](../../../README.md)): can many
lower-precision simulations, run across TPU devices, match the figures of fewer
double-precision simulations in the same wall-clock time? Answering it needs an engine in
which:

1. precision is set per stage, per product and per trade, from float64 down to FP8;
2. every reduced-precision result says how far it is from float64, measured on the run itself;
3. the default stays float64 and keeps ORE parity;
4. low precision is actually faster where the hardware has native support for it, and the
   speed is measured, not projected.

Precision never decides whether a run happens (D-9). It decides what the result is labelled
with: validated for that figure, or a warning naming the evidence that is missing.

## 2. Principles

These are not negotiable. Every step's exit criterion checks them, and a change that
conflicts with one is redesigned rather than excused.

### 2.1 ORE parity

- **The default is float64 everywhere,** and every ORE parity suite runs on the default. Parity
  at the default is therefore never traded for precision features.
- **Steps 1.4 to 1.8 keep the default bit for bit.** Their cast points are no-ops at float64
  (`astype` to the same dtype, which XLA removes), and a golden snapshot proves it
  ([§13.1](#131-bit-for-bit-and-ore-parity)).
- **Step 2.8 is the one step that moves float64 numbers.** It rewrites the kernels for
  low-precision compute and uses the same kernels at float64 (A-16), so float64 changes at
  rounding level. Every ORE parity suite must pass at its existing tolerance (1e-14 to 1e-8)
  before the golden snapshot is re-baselined, and the re-baseline is recorded with the
  measured size of the change.
- **Low-precision results are compared with the engine's own float64 result,** not with ORE.
  The float64 result carries the ORE parity; the precision report carries the distance from it.
  The chain is ORE ⇄ engine float64 ⇄ engine low precision, and each link has its own evidence.

### 2.2 Accuracy

- **Measured, not assumed.** Every reduced-precision run can carry a paired float64 sample
  and report its error per figure ([§9](#9-estimation-and-the-precision-report)).
- **Bias is the criterion, not error per value.** Random rounding errors average out like
  Monte Carlo noise; a systematic shift does not. Formats are judged by the bias of each
  output figure against its Monte Carlo standard error.
- **Unbiased where possible.** Means (NPV, EPE, ENE) are corrected by a two-level estimator,
  so precision error becomes variance, not bias (A-13).
- **Fragile stages stay float64.** Calibration, t=0 prices, Greeks and every reduction over
  paths are fixed at float64 (A-10): they cost little, and low precision breaks them (bump
  gamma in float32 is noise; the bisection cannot converge).
- **Red first, on sloped curves.** Flat curves cancel the drift and convexity terms that
  rounding errors also hide in. Every accuracy test runs on a 3% → 5% curve as well as flat.

### 2.3 Speed

- **float64 must not get slower.** Each step's benchmark is compared with the step before
  ([§13.9](#139-performance)).
- **Low precision pays in two ways only:** fewer bytes moved and stored (any hardware), and
  native low-precision matrix units (FP8 on Ironwood and H100). Elementwise arithmetic below
  float32 is emulated on every chip. The design puts the low precision where these pay.
- **Speed is measured on the target hardware** (Ironwood, H100; the owner has access),
  step 3.4. CPU numbers are accuracy evidence only.
- **One compile cache per host.** The worker process ([§11](#11-execution-architecture))
  compiles a job shape once; the per-process pools compiled it once per worker.

### 2.4 Design: DRY, KISS, modularity

- **One table of formats** (`engine/precision/formats.py`) serves validation, the HTTP
  schema, storage and the report. No other module maps names or bit counts to dtypes. Both
  of today's `64 → float64 else float32` helpers go (`engine/portfolio/config.py:116`,
  `engine/market_risk/run.py:96`); the second silently maps any value to float32.
- **One resolver** (`precision_for(trade)`) decides a trade's precision; pricing and market
  risk both call it.
- **One storage mechanism** (`store`/`load`) for every class and both pipelines.
- **Few cast points.** Only the lines in [§6.3](#63-cast-points) read the configuration.
  Inside a stage, code follows the dtype of its inputs and never takes a dtype argument.
- **One kernel implementation for every precision** (A-16). A second, low-precision copy of
  each pricer would drift from the one ORE parity is proven on.
- **Refuse, never coerce.** An unknown format, a compute narrower than allowed, or the old
  configuration shape is refused by field name before any work (A-12).
- **Modular.** `engine/precision/` depends on JAX and NumPy only. The pipeline imports it;
  it imports nothing from the pipeline.

## 3. Concepts

### 3.1 Three precisions

| Precision | What it is | Rounding happens | What it buys |
|---|---|---|---|
| **storage** | The format a value is kept in between steps | Once per value, when stored | Memory and bandwidth: the cube in FP8 is 8× smaller than in float64 |
| **compute** | The format each arithmetic result is rounded to | At every operation, so errors build up over a scan, a sum of flows, a rollback | Time, on hardware with native units for it |
| **accumulate** | The format sums and matrix products accumulate in | At every addition into the sum | Keeps long sums from drifting or overflowing |

Low-precision hardware works this way: FP8 inputs to a matrix multiply, with sums
accumulated in float16 or float32. No accelerator sums in FP8. Keeping the three apart is
what lets the study say which one limits accuracy.

### 3.2 Stages: adjustable and fixed

| Stage | Adjustable | Per product and trade | Why |
|---|---|---|---|
| Simulation (shocks, states) | Yes | No | One set of paths serves every trade |
| Scenario market (curves, fixings, FX) | Yes | Storage per run; each trade reads it at its own compute | Shared input |
| Pricing on paths (every pricer, incl. the Bermudan per-path recalibration) | Yes | **Yes** | Trades are priced one at a time |
| Values (the cube column of a trade) | Yes | **Yes** | Every reduction goes through float64, so mixed formats never meet in arithmetic |
| Market-risk revaluation and P&L | Yes | **Yes** | Same resolver |
| Calibration (cross-asset model, each trade at t=0) | No, float64 | — | Cheap; bisection to 1e-12 needs float64 |
| t=0 NPVs, step moments | No, float64 | — | Cheap |
| Greeks (bump and AD) | No, float64 | — | Bump gamma is a second difference; float32 makes it noise |
| Reductions over paths or scenarios (EPE, PFE, netting sums, VaR/ES) | No, float64 | — | Where precision error would become bias; free to do in float64 |

Fixed stages are built on the same mechanism, so a later study can open one by adding a
field, not by changing the pipeline (A-10).

### 3.3 Array classes

The cast points name the arrays they convert. One name per kind of array, shared by both
pipelines:

| Class | Portfolio run | Market-risk run |
|---|---|---|
| `shocks` | Sobol normals `[T, S, d]` | Scenario shifts `[N, f]` |
| `states` | Model states `[S, T, d]` | — |
| `market` | Scenario curves `[S, D, K+1]`, fixings, FX | Shocked pillar rates |
| `values` | Cube `[S, D, trades]` | NPVs and P&L per scenario |

## 4. Configuration

### 4.1 The types

```python
@dataclass(frozen=True)
class StagePrecision:
    storage: str = "float64"
    compute: str = "float64"
    accumulate: str = "float64"

@dataclass(frozen=True)
class Precision:
    simulation: StagePrecision = StagePrecision()   # shocks and states
    market: StagePrecision = StagePrecision()       # the scenario market
    pricing: StagePrecision = StagePrecision()      # path pricing and the values it stores
    by_product: Mapping[str, StagePrecision] = field(default_factory=dict)
                                                    # "swap", "european_swaption", "bermudan_swaption",
                                                    # "american_swaption", "bond": overrides pricing
    by_trade: Mapping[str, StagePrecision] = field(default_factory=dict)
                                                    # trade_id: overrides by_product and pricing
    rounding: str = "nearest"                       # nearest | stochastic (sub-32-bit storage)
    paired_fraction: float = 0.0                    # share of paths re-run at float64 (§9)
```

`Precision()` is float64 everywhere. It replaces `PrecisionConfig`,
`PricingPrecisionOverride`, `RiskPrecisionOverride` and `MarketRiskRequest.precision`. The old
shapes are refused with an error naming the replacement, in Python and over HTTP (a 422), as
roadmap 1.3 did with the Hull-White request (A-12). `RunConfig.precision` and
`MarketRiskRequest.precision` take the same type; a market-risk run reads `simulation` for
its shifts and `pricing` for its revaluation.

### 4.2 Resolution

`precision_for(trade)` returns `by_trade[trade_id]`, else `by_product[product]`, else
`pricing` (A-15). It is the only lookup; a misspelt trade id or product is refused, not
ignored.

### 4.3 Validation

Checked before any work, by `engine/precision/policy.py`, each refusal naming its field:

- every name is in the format table;
- `accumulate` is at least as wide as `compute`, and no narrower than float32 below 32-bit
  compute (no accelerator accumulates narrower);
- `storage` is no wider than `compute` (storing wider gains nothing);
- `compute` below float32, and `accumulate` different from `compute`, are refused until step
  2.8 enables them, by name, citing the step;
- every `by_trade` key is a trade in the request, every `by_product` key a product.

Everything else may be run (D-9). An unvalidated combination gets a warning
([§10](#10-acceptance-standard-and-evidence)), never a refusal.

### 4.4 Examples

```python
Precision()                                                   # the default: float64

Precision(simulation=StagePrecision("float32", "float32", "float32"),
          market=StagePrecision("float32", "float32", "float32"),
          pricing=StagePrecision("float32", "float32", "float32"))   # float32 throughout

Precision(simulation=StagePrecision(storage="float8_e4m3fn", compute="float32", accumulate="float32"),
          market=StagePrecision(storage="float16", compute="float32", accumulate="float32"),
          pricing=StagePrecision(storage="float8_e4m3fn", compute="float32", accumulate="float32"),
          by_product={"bermudan_swaption": StagePrecision("float32", "float32", "float32")},
          rounding="stochastic", paired_fraction=0.02)               # FP8 storage, Bermudans kept wider
```

## 5. Modules

```
engine/precision/
  formats.py   # the format table: name -> dtype, bits, mantissa bits, max, scaled?
  policy.py    # StagePrecision, Precision, validation, precision_for
  storage.py   # Stored (pytree: values, scales | None, format); store(), load()
  report.py    # PrecisionReport: the policy as run, realized dtypes, device, paired errors
  estimate.py  # two-level estimator for means; paired differences for quantiles
```

The pipeline imports these; they import nothing from it. Each module has its own unit tests
that need no market, trade or ORE ([§13.4](#134-storage-properties)).

## 6. Mechanism

### 6.1 The format table

| Name | Bits | Mantissa bits | Max | Scaled when stored | Enabled at step |
|---|---|---|---|---|---|
| `float64` | 64 | 52 | 1.8e308 | No | 1.4 |
| `float32` | 32 | 23 | 3.4e38 | No | 1.4 |
| `float16` | 16 | 10 | 65,504 | Yes | 1.6 |
| `bfloat16` | 16 | 7 | 3.4e38 | Yes | 1.6 |
| `float8_e4m3fn` | 8 | 3 | 448 | Yes | 1.6 |
| `float8_e5m2` | 8 | 2 | 57,344 | Yes | 1.6 |

FP4 (`float4_e2m1fn`) is added at step 6.3. A name outside the table is refused.

### 6.2 Storage: `store` and `load`

`store(x, format, rounding, key) -> Stored` and `load(stored, dtype) -> array`.

- **float64 and float32:** a plain `astype`. `Stored` holds no scales.
- **Every sub-32-bit format is stored with block scales** (A-10's single rule, rather than
  deciding per class which formats need range help). Blocks are 32 consecutive entries along
  the **scenario axis**, each with a power-of-two scale, kept as float32, that brings the
  block's largest magnitude to the format's maximum. Power-of-two scales are exact (the idea
  behind the OCP MX formats), so scaling adds no rounding of its own. The scenario axis is
  the axis step 3.2 shards, so blocks fall inside shards. Overhead: 4 bytes per 32 values.
- **Rounding:** `nearest`, or `stochastic` (up or down at random, in proportion to the
  distance, so each value's error averages to zero). Stochastic draws come from the run's
  seed, so runs reproduce.
- A scenario count that is not a multiple of 32 gets a short last block; zeros and NaN pass
  through unchanged.

### 6.3 Cast points

The only places that read the configuration:

| # | Where | What it does |
|---|---|---|
| 1 | `engine/simulation/random.py` | Normals generated at `simulation.compute` (the clip epsilon from the compute dtype, not the storage dtype), then `store(shocks)` |
| 2 | `engine.simulation.cam.evolve_states` | `load(shocks)`; the scan's carried state stays at `simulation.compute`; only the emitted states are stored, `store(states)` |
| 3 | Scenario market build (`engine/simulation/scenario_market.py`) | `load(states)`; curves computed at `market.compute` (the z-independent terms keep coming from float64); `store(market)` |
| 4 | Valuation (`engine/valuation/portfolio.py`) and market-risk revaluation | Per trade, `precision_for(trade)`: `load(market)` at its compute, price, `store` its column as values |
| 5 | Exposure (`engine/risk/exposure.py`) and VaR/ES (`engine/risk/var_es.py`) | `load(values, float64)`, then reduce |

The cube becomes a sequence of per-trade `Stored` columns (one array when every trade
shares a format). `PortfolioResult.npv_cube` stays an array, loaded at float64, so
consumers see no change.

### 6.4 Inputs follow dtype

Inside a stage, every array takes the dtype of the arrays it is computed from. About 46
`jnp.asarray/zeros/...` calls in `engine/valuation`, `engine/calibration`, `engine/models`
and `engine/risk` name no dtype today and so become float64 under the x64 flag; each becomes
"the input's dtype". No kernel signature gains a dtype argument.

Strict promotion (`jax_numpy_dtype_promotion="strict"`) in CI turns any accidental
float32/float64 mix into a failing test ([§13.3](#133-dtype-discipline)).

### 6.5 Constants that depend on dtype

Found while designing; fixed or derived from the dtype in step 1.4:

- `hi * (1.0 - 1e-9)` (`engine/calibration/ore_lgm.py:354`) rounds to `hi` in float32. The same
  bisection runs per path in the Bermudan recalibration.
- `jnp.finfo(dtype).eps` in the Sobol clip (`engine/simulation/random.py:42`) must use the
  compute dtype once storage is narrower.
- `scenario_batch_size(itemsize)` (`engine/market_risk/revaluation.py`) must use the compute
  itemsize.

Step 1.4 also searches every path kernel for literal tolerances (`1e-`) and either derives
each from the dtype or shows it holds at float32.

### 6.6 What goes

- `check_run`'s refusal of `pricing`, `risk` and `calibration` below 64.
- `run_market_risk`'s `jax.config.update("jax_enable_x64", True)`. x64 stays on, set once
  when `engine` is imported, as since roadmap 1.3.
- `_PRICING_LOCK` (`engine/portfolio/request.py`), after the thread-safety audit of step 1.4
  (no module-level caches and no ORE globals were found; a concurrency test confirms it).
- The per-precision pool tiers (`engine/portfolio/worker_pool.py`): one pool from step 1.4;
  the pool itself goes with step 1.8.

## 7. Low-precision storage

Storage below 32 bits needs no kernel changes: values are loaded to the compute precision,
which stays at float32 or above until step 2.8. It is the first low-precision capability
(step 1.6), and the first measurement campaign (step 2.7) runs on it while stage 2 continues,
because it never changes float64 numbers.

What is known (measured 2026-10-01, [§15](#15-measurements)):

- **FP8 is viable for shocks.** Round-to-nearest storage of 4M standard normals biases a call
  payoff by about one Monte Carlo standard error at 4M paths; stochastic rounding by about
  0.15. The earlier verdict (useful only to ~1,400 paths) compared error per value with Monte
  Carlo error, which is the wrong criterion ([§15.2](#152-bias-not-error-per-value)).
- **FP4 is not viable stored naively,** with either rounding (25 and 45 standard errors).
  Stochastic rounding keeps each value's mean but inflates the variance by 4.7%, which
  inflates volatility. A variance correction through the block scales is the candidate fix
  (step 6.3).
- **States, curves and values are unmeasured.** States are sums of many shocks and are
  reused at every step, so FP8 is expected to fail for them; the measurement will show it.

## 8. Compute below float32

### 8.1 What blocks it

JAX already runs `+`, `*`, `exp`, `log`, `max`, `sum` and `cumsum` in bfloat16, float16 and
both FP8 formats; on CPU, XLA computes each operation wider and rounds the result to the low
format, so the numerics are those of low-precision arithmetic. Only `cholesky` and
`norm.ppf` have no kernel. The obstacles are numerical:

| Obstacle | Example | Remedy |
|---|---|---|
| **Range** | FP8 e4m3 tops at 448, float16 at 65,504; NPVs reach 1e6 and more | Scaled units: per unit notional, block scales, values relative to their t=0 value |
| **Accumulation** | An FP8 `cumsum` of 10,000 × 0.1 returns NaN (it passes 448); bfloat16 returns 1008 | The `accumulate` precision: low-precision operands, float32 sums |
| **Cancellation** | A swap NPV is two legs of ~1e8 that nearly cancel; FP8 keeps about one significant digit | Difference form: deterministic part in float64, path deviation in low precision (§8.2) |
| **Missing or unstable kernels** | `norm.ppf`, `cholesky`; the inverse normal CDF overflows in FP8 | Computed at float32 and rounded: they are inputs, not where the time goes. Cholesky already runs once on the host |

### 8.2 The difference form

Each kernel splits a value into a part that does not depend on the path, computed once at
float64, and the path's deviation from it, which is small and well conditioned. Example: a
swap's path NPV uses `K − F(0)` from float64 and only the deviation `F(path) − F(0)` at low
precision, so the near-cancellation of the two legs happens in float64. The scenario market
already does this for its z-independent terms (`engine/simulation/scenario_market.py`,
`implied_log_discounts`).

Kernel families, rewritten one at a time, each measured by emulation before the next: the
simulation scan, scenario curves, legs, the Bachelier and Jamshidian Europeans, the Bermudan
rollback and its per-path recalibration, exposure. One implementation serves every precision
(A-16); [§2.1](#21-ore-parity) gives the re-baseline protocol.

### 8.3 Emulation and native speed

- **Accuracy** is studied on CPU by emulation: the numbers are those of the low format.
- **Speed** needs native units, and they exist only for matrix products: FP8 on Ironwood and
  H100, FP4 on TPU 8t/8i and Blackwell. Elementwise FP8 is emulated on every chip.
- So the wall-clock half of the research question depends on expressing the heavy kernels as
  matrix products where they can be: leg pricing (amounts per flow × discount factors per
  path and flow), the Bermudan rollback (a fixed Gaussian kernel matrix × values), the
  correlation mixing (already a product, but small). Step 3.4.

## 9. Estimation and the precision report

### 9.1 Reductions

Every reduction over paths or scenarios loads the values at float64 first. Summing adds no
error of its own, and the rounding errors of individual values average out as Monte Carlo
noise when they are unbiased.

### 9.2 The paired sample

With `paired_fraction > 0`, that share of paths (the first paths of the same scrambled Sobol
sequence, so the same random numbers) is also run at float64 throughout. The differences
measure the precision error on this run.

### 9.3 Means: the two-level estimator (A-13)

For a mean figure F (NPV, EPE, ENE, EE_B, EPE_B), with N paths at the configured precision and
n paired paths:

    F = mean over N of f_low  +  mean over n of (f_64 − f_low)

The second term corrects the first, so F is an unbiased estimate of the float64 figure.
Precision error becomes variance: it shrinks as n grows instead of staying as bias. If the
correction has low variance, few float64 paths suffice and most of the work runs at low
precision, which is the research question in estimator form. This is two-level Monte Carlo
in the sense Giles and Sheridan-Methven use for reduced-precision and approximate random
variables. The corrected value is the reported figure; the uncorrected value and both terms'
standard errors are reported beside it.

### 9.4 Quantiles: measured, not corrected

PFE, VaR and ES are not means, so the estimator above does not apply directly. They are
computed from the float64-loaded values, and the paired sample reports the difference
between the low-precision and float64 quantile on the paired paths. Multilevel quantile
estimation (Giles and Haji-Ali) is a later research item (step 6.3).

### 9.5 The report

`PortfolioResult.precision` and `MarketRiskResult.precision`, a `PrecisionReport`:

- the policy as run, including every resolved per-trade precision;
- the realized dtype of every stored class, read from the arrays, not the configuration;
- the device and backend the job ran on (closes [I-12](../known-issues.md#i-12));
- per figure: the corrected and uncorrected value, the paired mean difference, its standard
  error, the largest paired difference, the path counts;
- the evidence verdict per figure: validated, or the warning ([§10](#10-acceptance-standard-and-evidence)).

## 10. Acceptance standard and evidence

The standard labels results; it never stops a run (D-9).

### 10.1 Figures with a Basel III test (A-11)

Basel III does not regulate numerical precision; it judges model outputs through
backtesting and P&L attribution (MAR32). For market-risk P&L, VaR and ES:

1. **The P&L attribution test between the low-precision and the float64 P&L:** Spearman
   correlation and the Kolmogorov–Smirnov statistic, in the green zone. The thresholds are
   transcribed from MAR32 into the regulatory profile under the Basel plan's double-entry
   rule ([basel-iii.md](basel-iii.md) §4); they are not taken from memory. This test is loose
   (it was built for front-office vs risk-model differences), so it is a floor.
2. **The Basel plan's P6.2 rule:** the precision error below 1% of the figure's statistical
   error (ES, VaR), and below 1e-6 relative for the standardised figures, which have no
   randomness.

### 10.2 Figures Basel does not cover (A-11)

NPV, EPE, ENE, EE_B, EEE_B, EPE_B, EEPE_B, PFE: the P6.2 statistical rule, **labelled as an
engineering rule**, not a regulatory one. Basel's counterparty rules (CRE53) require EPE
backtesting but set no numeric tolerance.

### 10.3 The evidence table

One table, shared with the Basel plan's P6 gate: per figure, per precision combination
(storage, compute and accumulate per stage, product overrides), the validation run, how it
was done, at how many paths, the measured bias and its standard error, the verdict, and the
**path ceiling** (the path count above which the bias exceeds the rule). It lives in
`compliance/precision-evidence.md` and is produced by a script from the measurement runs, not
typed. A result whose combination has no passing row for a figure carries a warning naming
the row that is missing.

## 11. Execution architecture

### 11.1 Decision (A-14)

The HTTP API and the engine are separate processes. The API validates a request, writes it
to a durable job queue (a SQLite file) and returns `202`. **One engine worker process** per
host takes jobs from the queue one at a time, parses the request JSON itself, prices it and
writes the result back. The worker is single-threaded and owns every device on its host.

### 11.2 Why

| Reason the process pools existed | Now |
|---|---|
| A different x64 flag per tier | Gone since 1.3; x64 is always on |
| Keep the HTTP server responsive during minutes-long jobs | The separate worker process does it |
| Run jobs in parallel | XLA uses the whole device per job; on TPU one process owns a chip, so several processes need pinning and fight step 3.2's sharding |
| Crash isolation | Today a dead worker likely breaks the pool (no `BrokenProcessPool` handling); a separate worker is restarted by its supervisor without touching the API |

What it removes: the freeze/thaw of ORE objects (the worker reads JSON, not pickled
dataclasses), the spawn setup (I-33), the per-worker compile, `_PRICING_LOCK`, and threads
altogether. `price_portfolio` called from Python never involved any of this and is unchanged.

### 11.3 Multi-device and multi-host

- **One host:** the worker builds a `jax.sharding.Mesh` over its devices once and shards the
  scenario axis inside each job (step 3.2). No threads, no pinning.
- **Several hosts (a TPU pod slice):** one worker per host, all running the same program
  (`jax.distributed.initialize`), coordinated through the queue.

### 11.4 The queue (A-14; not a main priority)

SQLite rows: request JSON, status (`pending`, `running`, `done`, `failed`, `interrupted`),
failure class, result reference, timestamps. It survives restarts and closes the portfolio
half of [I-08](../known-issues.md#i-08); the EOD path keeps its publication store. A worker
that restarts marks its `running` row `interrupted`. Step 1.8 is placed last in stage 1 and
nothing in the precision work depends on it; step 3.2 does.

## 12. Plan

| Step | Work | Exit criterion | Size |
|---|---|---|---|
| **1.4** | `engine/precision/` (formats, policy, storage at float64/float32, report skeleton); `Precision` replaces the old types in `RunConfig`, `MarketRiskRequest` and the HTTP schema, the old shape refused (A-12); the five cast points; inputs follow dtype; float64 reductions; the constants of §6.5; market risk on the same module; remove `check_run`'s refusal, the flag set, the lock and the tiers; strict promotion in CI | Default: golden snapshot bit for bit, every ORE parity suite unchanged. float32 throughout: cube bit-identical to today's `simulation=32` cube; exposure differs only by the float64 reductions. Fast tier green under strict promotion | M |
| **1.5** | Per-product and per-trade precision (A-15): `by_product`, `by_trade`, `precision_for`; per-trade stored columns; market risk per trade | Bit for bit at the default; a mixed run (Bermudan float32, swaps float64) equals each trade run alone at its precision, column for column | S |
| **1.6** | Sub-32-bit storage: block scales, both roundings; `float16`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2` enabled for storage | The storage properties of §13.4; bit for bit at the default | M |
| **1.7** | Paired sample, two-level estimator for means, `PrecisionReport` with realized dtypes and device (closes I-12) | §13.6; bit for bit at the default | M |
| **1.8** | Engine worker process and the SQLite queue (A-14); delete `worker_pool.py`'s pool and freeze/thaw | §13.8, including the Linux run; float64 job time and compile count no worse | M |
| **2.7** | *Parallel with stage 2.* Measurement campaign for storage formats per class and product, at several path counts, fixed seeds; the evidence table and the warnings (§10) | Table complete for every figure × class × format; thresholds fixed before measuring; a verdict and path ceiling per row | M |
| **2.8** | Difference-form kernels (§8.2), one family at a time; compute below float32 enabled; `accumulate` honoured | Per family: ORE parity suites at their tolerances, then the re-baseline of §2.1; emulated FP8/bfloat16 compute measured into the evidence table | L |
| **3.2** | Shard the scenario axis in the worker (I-61) | Results equal the one-device run within reduction-order rounding; scaling measured | L |
| **3.4** | Timing on Ironwood and H100: storage formats, then matrix-product forms of the heavy kernels with native FP8 | Wall time per figure and precision against float64 at equal accuracy (the evidence table's path ceilings) | M |
| **6.3** | FP4 storage (variance correction through the block scales), FP4 compute on TPU 8t/8i, multilevel quantile estimation | Evidence rows for FP4 | L |

**Order (decided, K):** the structure first (1.4 to 1.8), stage 2's correctness work next,
with the storage measurement (2.7) alongside because it changes no float64 number; the kernel
rewrite (2.8) after stage 2's own kernel changes (2.4 recalibration, 2.5 `ShiftHorizon`, 2.6
the volatility strike axis), so no kernel is rewritten twice; then performance, which
requires frozen numbers.

## 13. Testing

Verification rules of [README.md](../README.md#verification-rules) apply, notably: the
virtualenv's interpreter, counts from `--collect-only`, a summary line printed, and Linux for
process changes.

### 13.1 Bit for bit and ORE parity

- Before step 1.4, save a golden snapshot from `main` in a separate worktree: the shared
  portfolio's t=0 NPVs, cube, exposure and per-trade EPE at float64 and with a float32
  simulation, bump Greeks, AD Greeks, a market-risk run at 64 and 32. Compare with
  `np.array_equal` (value, dtype, shape) after every step from 1.4 to 1.8. The Greeks run
  takes about 16 minutes and 10 GB.
- Every ORE parity suite runs unchanged at the default after every step.
- Step 2.8: parity suites pass at their tolerances first; then the snapshot is re-baselined,
  with the largest change per array recorded in the commit and in known-issues' verification
  status.

### 13.2 Configuration

- Unknown formats, compute below the enabled floor, storage wider than compute, accumulate
  narrower than compute, unknown trade ids and products: refused, each naming its field.
- The old `PrecisionConfig` shape: refused in Python and with a 422 over HTTP, naming the
  replacement.
- A completeness test: every field of `Precision` and `StagePrecision` exists in the HTTP
  schema, and every format in the table is accepted by it.

### 13.3 Dtype discipline

- A CI job runs the fast tier with `jax_numpy_dtype_promotion="strict"`.
- A parametrized test over every class × enabled format runs a small portfolio and compares
  the realized dtypes in `PrecisionReport` with the policy.
- A test that the resolver is the only lookup: per-trade overrides change only that trade's
  column.

### 13.4 Storage properties

Fast, no market or ORE needed:

- values representable in the format round-trip exactly;
- round-to-nearest error is at most half a step of the format times the block scale;
- stochastic rounding is unbiased (mean over many draws within three standard errors of the
  input) and repeats exactly for the same seed;
- no block overflows; zeros and NaN pass through; a scenario count not divisible by 32;
- `store`/`load` at float64 and float32 is the identity.

### 13.5 Low-precision sanity on sloped curves

At float32 throughout, on the 3% → 5% curve: no NaN, no calibration at its ceiling, the cube
within a stated float32 tolerance of float64. These tests are written red first against the
constants of §6.5.

### 13.6 The estimator and the report

- float64 against float64: every paired difference exactly 0, the corrected figure equal to
  the uncorrected one.
- A test-only format with a known bias: the report flags it, and the two-level estimate
  removes it within its standard error.
- Coverage: over independent seeds, the corrected estimate's confidence interval contains
  the float64 figure at the stated rate.

### 13.7 Statistical acceptance (slow tier)

The evidence table's rows become slow-tier tests with fixed seeds and the thresholds fixed
before measuring: bias against the rule of §10 at the stated path counts.

### 13.8 Execution

- Jobs queued together give the same bits as run one after another.
- A failing job fails only its own row, with a failure class; the worker survives.
- A worker killed mid-job leaves an `interrupted` row; a restarted worker picks up the next.
- The existing API tests pass unchanged (`202`, poll, result).
- The full suite on Linux (Docker `python:3.11`), as process changes require.

### 13.9 Performance

- float64 wall time and compile count of the shared portfolio against the previous step,
  median of five warm runs: no regression beyond the run-to-run spread.
- float32 timing recorded.
- After 1.8: a second identical job compiles nothing.
- Step 3.4 on Ironwood and H100: wall time per figure at equal accuracy.

### 13.10 Demos

`demos/demo_precision.py` moves to `Precision`, adds FP16 and FP8 storage runs, and prints
each run's report.

## 14. Scope

**In:** float64/float32 compute and storage; float16, bfloat16 and FP8 storage; compute below
float32 after 2.8; per-stage, per-product and per-trade precision; block scales and both
roundings; the paired sample, the two-level estimator for means and the report; the evidence
table and warnings; the worker process and queue; timing on Ironwood and H100.

**Out, with the reason:**

| Item | Why | Where |
|---|---|---|
| Low-precision calibration, t=0 and Greeks | Cheap and fragile (A-10); openable later by a field | — |
| FP4 | Needs TPU 8t/8i or Blackwell | 6.3 |
| Variance correction of stored shocks | Only if 2.7 shows bias | 6.3 |
| Multilevel quantile estimation | A research project of its own | 6.3 |
| Low-precision speed on CPU | Emulated; CPU is accuracy evidence only | — |

## 15. Measurements

All on CPU, `jax==0.10.2`, `ml_dtypes==0.5.4`.

### 15.1 Kernel coverage

Measured 2026-09-17 (commit `e4ca50b`), extended 2026-10-01.

| dtype | `cholesky` | `norm.ppf` | `matmul` | elementwise (`+ * exp log max sum cumsum`) |
|---|---|---|---|---|
| float64, float32 | OK | OK | OK | OK |
| bfloat16, float16, FP8 (e4m3fn, e5m2) | `NotImplementedError` | `TypeError` | OK | OK, result in the low dtype |
| FP4 (e2m1fn) | `NotImplementedError` | `TypeError` | refused by XLA's CPU backend | — |

On this build FP4 takes a full byte per value, so it saves no memory until packed.

Accumulation, 10,000 × 0.1: bfloat16 `cumsum` 1008; FP8 e4m3 `sum` 320, `cumsum` NaN.

Computing in the low format overflows: Acklam's inverse normal CDF over 20,000 uniforms gives
float64 error 3.9e-9, float32 3.0e-4, and NaN in bfloat16, float16 and FP8 (its
coefficients reach ±276 and the intermediates overflow). Computing at float32 and storing low
works at every format (max / rms error of the stored normals: float16 9.8e-4 / 2.1e-4,
bfloat16 7.8e-3 / 1.7e-3, FP8 e4m3 1.2e-1 / 2.6e-2, FP8 e5m2 2.5e-1 / 5.3e-2). A
matmul-only matrix square root (Newton–Schulz) runs below float32 (8×8, error float32
1.1e-6, float16 3.4e-3, bfloat16 2.5e-2) but returns a symmetric, not triangular, factor.

### 15.2 Bias, not error per value

Measured 2026-10-01. The earlier analysis set the rms error per stored value equal to the Monte Carlo error and
read off a path ceiling (FP8 about 1,400 paths). That is the wrong criterion: unbiased
errors average out with the paths. Measured instead, the bias of a call payoff
`max(x − 0.3, 0)` on 4M stored standard normals, against its Monte Carlo standard error
2.4e-4:

| Format | Nearest | Stochastic | Variance bias, nearest / stochastic |
|---|---|---|---|
| FP8 e4m3 | −2.4e-4 (≈1 standard error) | +3.4e-5 (≈0.15) | −7.9e-4 / +1.3e-3 |
| FP4 e2m1 | +6.2e-3 (≈25) | +1.1e-2 (≈45) | +1.1e-2 / +4.7e-2 |

This is one payoff on raw shocks, not pricing through the pipeline; step 2.7 measures the
pipeline's figures.

## 16. Decisions this document implements

Recorded in [compliance/decisions.md](../../../compliance/decisions.md), 2026-10-01.

| # | Decision |
|---|---|
| D-9 (revised) | Precision is a free choice for the adjustable stages, per run, product and trade; a run is never refused on precision; unvalidated combinations carry a warning |
| A-9 | Adjustable precision, the old mechanism removed once the replacement works (steps 1.4, 1.8) |
| A-10 | Only the heavy scenario-axis stages are adjustable; calibration, t=0, Greeks and reductions stay float64, built on the same mechanism so a stage can be opened later |
| A-11 | Acceptance standard: Basel III tests where they exist (P&L attribution, plus the Basel plan's P6.2 rule); the P6.2 statistical rule, labelled as engineering, for figures Basel does not cover |
| A-12 | The old precision configuration is refused, not translated |
| A-13 | Means are corrected by the two-level estimator; quantiles are measured |
| A-14 | Execution: API and one single-threaded engine worker process per host, through a durable SQLite queue |
| A-15 | Precision per stage, overridable per product and per trade (trade over product over stage) |
| A-16 | One kernel implementation for every precision; float64 is re-baselined once, after ORE parity passes |

Engineering defaults, changeable without a decision: block size 32 along the scenario axis;
the rounding default chosen by step 2.7; `paired_fraction` 0 by default, 0.02 suggested for
reduced-precision runs; the evidence table's location.

# Precision

Design and plan for adjustable numeric precision, from float64 down to FP8, and for the
execution architecture that runs it. It implements [I-55](../known-issues.md#i-55),
[F-07](../features.md#f-07), [I-12](../known-issues.md#i-12) and
[I-72](../known-issues.md#i-72), and the owner decisions A-9, A-10 to A-16, A-22 and the revised
D-9 ([compliance/decisions.md](../../../compliance/decisions.md)). It replaces
`sub-fp32-precision.md`; its findings are kept, corrected, in [§15](#15-measurements).

**Written:** 2026-10-01 · **Steps:** roadmap [1.4 to 1.8](../roadmap.md#stage-1--structure), 2.3,
[3.6, 3.7, 3.8](../roadmap.md#stage-3--foundations),
[5.1 to 5.3](../roadmap.md#stage-5--precision-research)

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
  ([§13.1](#131-bit-for-bit-and-ore-parity)). One change between 1.5 and 1.6 was allowed to
  move float64 at rounding level, by decision of 2026-10-02: jitting the pricers with the
  trade as a traced argument (I-21, I-22). It followed step 3.7's procedure below, and the
  snapshot was re-baselined after it (§13.1); 1.6 to 1.8 are bit for bit against that.
  Step 2.4 re-baselined the AD Greeks alone in the same way, each now one compiled program
  (§13.1); every other array stayed bit for bit.
- **Step 3.7 is the one step that moves float64 numbers.** It rewrites the kernels for
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
  gamma in float32 is noise; a root solve cannot converge past float32's resolution).
- **Red first, on sloped curves.** Flat curves cancel the drift and convexity terms that
  rounding errors also hide in. Every accuracy test runs on a 3% → 5% curve as well as flat.

### 2.3 Speed

- **float64 must not get slower.** Each step's benchmark is compared with the step before
  ([§13.9](#139-performance)).
- **Low precision pays in two ways only:** fewer bytes moved and stored (any hardware), and
  native low-precision matrix units (FP8 on Ironwood and H100). Elementwise arithmetic below
  float32 is emulated on every chip. The design puts the low precision where these pay.
- **Speed is measured on the target hardware** (Ironwood, H100; the owner has access),
  step 5.2. CPU numbers are accuracy evidence only.
- **One compile cache per host.** The worker process ([§11](#11-execution-architecture))
  compiles a job shape once (measured: a second identical job builds no program); the
  per-process pools compiled it once per worker.

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
| Calibration (cross-asset model, each trade at t=0) | No, float64 | — | Cheap; a root solve to 1e-12 needs float64 |
| t=0 NPVs, step moments | No, float64 | — | Cheap |
| Greeks (bump and AD) | No, float64 | — | Bump gamma is a second difference; float32 makes it noise |
| Reductions over paths or scenarios (EPE, PFE, netting sums, VaR/ES) | No, float64 | — | Where precision error would become bias; free to do in float64 |

Fixed stages are built on the same mechanism, so a later study can open one by adding a
field, not by changing the pipeline (A-10).

Market risk has no float64 t=0 stage of its own: its base values are the revaluation of the
unshocked curves at `pricing.compute`, the anchor every P&L is measured from, so a zero shift
is exactly zero P&L at every precision (a float64 base under a float32 revaluation would bias
every P&L by the base's rounding). Each trade's P&L is computed at `pricing.compute`, stored
at `pricing.storage` (storing the difference, not the two NPVs, keeps the cancellation in the
compute precision) and loaded at float64 for VaR/ES. Settled in step 1.4.

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
    rounding_seed: int = 0                          # the stochastic rounding's seed (1.6)
    paired_fraction: float = 0.0                    # share of paths re-run at float64 (§9)
```

`rounding` and `rounding_seed` were added in step 1.6 (§6.2), `paired_fraction` in step 1.7
(§9.6).

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

Built in step 1.5 (2026-10-02):

- `Precision.precision_for(trade)` is a method of the policy and reads only `trade.trade_id`
  and `trade.product`, so `engine.precision` still imports nothing from the pipeline.
- A product is a class constant of each trade config (`SwapConfig.product == "swap"`,
  `"european_swaption"`, `"bermudan_swaption"`, `"american_swaption"`, `BondConfig.product ==
  "bond"`), the same names as the HTTP `trade_type`s (a test holds them equal).
  `engine.valuation.portfolio.PRODUCTS` collects them from the `Trade` union, so a new trade
  type adds its product by existing.
- `Precision.check_overrides(trades, PRODUCTS)` refuses a `by_product` key outside `PRODUCTS`
  and a `by_trade` key that is no trade's id. `validate_trades` calls it, so the portfolio
  run (`validate_request`, synchronously on the HTTP route: a 400), `value_portfolio` and
  `run_market_risk` all check before any work. Over HTTP an unknown product is already a 422
  (the schema's keys are the products).
- Overrides are keyed by trade id, so ids must be unique. `PortfolioRequest` already required
  it; `run_market_risk`, which accepted repeated ids, now refuses them too
  (`require_unique_ids`, shared). Below the requests (`value_portfolio` called directly) a
  `by_trade` override applies to every trade of its id.
- The overrides are kept in an immutable mapping (`Overrides`), so `Precision` stays a frozen,
  hashable, picklable value; a dict given is copied.

### 4.3 Validation

Checked before any work, by `engine/precision/policy.py`, each refusal naming its field:

- every name is in the format table;
- `accumulate` is at least as wide as `compute`, and no narrower than float32 below 32-bit
  compute (no accelerator accumulates narrower);
- `storage` is no wider than `compute` (storing wider gains nothing);
- `compute` below float32, and `accumulate` different from `compute`, are refused until step
  3.7 enables them, by name, citing the step;
- every `by_trade` key is a trade in the request, every `by_product` key a product;
- `rounding` is `nearest` or `stochastic`, and `stochastic` only when some stage or override
  stores in a scaled format (otherwise it would be accepted and ignored); `rounding_seed` is a
  non-negative integer (1.6);
- `paired_fraction` is a number in [0, 1] (1.7). At float64 everywhere it is accepted: it
  measures zero, a check of the pairing, not a setting that does nothing.

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
  formats.py   # the format table: name -> dtype, bits, mantissa bits, max, scaled?   (1.4)
  policy.py    # StagePrecision, Precision, validation (1.4); by_product, by_trade, precision_for (1.5)
  storage.py   # store(), load() (1.4); Stored (pytree: values, scales, format, axis), rounding (1.6)
  report.py    # PrecisionReport: the policy as run, realized dtypes, device, paired errors (1.7)
  estimate.py  # two-level estimator for means; paired differences for quantiles (1.7)
  products.py  # matmul(): every matrix product, at its compute format's precision (2.3)
```

Step 1.4 built the first three at float64/float32; step 1.6 added `Stored` and the scaled
formats; step 1.7 added `estimate.py` and `report.py` (§9.6). At float64 and float32 a stored
value is still the array itself (§6.2): a wrapper would carry no scales and only change every
consumer's types, and it keeps the default bit for bit.

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

FP4 (`float4_e2m1fn`) is added at step 5.3. A name outside the table is refused.

### 6.2 Storage: `store` and `load`

`store(x, format, rounding, key) -> Stored` and `load(stored, dtype) -> array`.

- **float64 and float32:** a plain `astype`, returning the array (step 1.4). Storing or
  loading at an array's own dtype returns it unchanged, so the float64 default moves no bit.
- **Every sub-32-bit format is stored with block scales** (A-10's single rule, rather than
  deciding per class which formats need range help). Blocks are 32 consecutive entries along
  the **scenario axis**, each with a power-of-two scale, kept as float32, that brings the
  block's largest magnitude to the format's maximum. Power-of-two scales are exact (the idea
  behind the OCP MX formats), so scaling adds no rounding of its own. The scenario axis is
  the axis step 3.8 shards, so blocks fall inside shards. Overhead: 4 bytes per 32 values.
- **Rounding:** `nearest`, or `stochastic` (up or down at random, in proportion to the
  distance, so each value's error averages to zero). Stochastic draws come from the run's
  seed, so runs reproduce.
- A scenario count that is not a multiple of 32 gets a short last block; zeros and NaN pass
  through unchanged.

As built (step 1.6, `engine/precision/storage.py`):

- `store(x, format, rounding, key, axis)` returns the array itself at float64 and float32
  (rounded to nearest whatever `rounding` says, so 1.4's behaviour is unchanged) and a `Stored`
  at a scaled format: `values` in the format with the array's shape, `scales` float32 with the
  scenario axis cut to ceil(S / 32) blocks, the format name and the axis. `load` accepts both.
  `Stored` is a pytree, so it passes through `jit` and sits inside a `ScenarioMarket`.
- **The scale** of a block is 2^-k for the largest k with `amax · 2^k ≤ max` (from the
  mantissas and exponents of `amax` and the format's max, exactly), so the block's largest
  finite magnitude lands in (max/2, max]. k is clamped to [-126, 126], so every scale and its
  inverse is a normal float32; only bfloat16 blocks below 2 in magnitude reach the clamp, and
  they stay normal bfloat16 numbers. Non-finite entries do not count towards `amax`.
- **Rounding is done on the format's grid, not by the dtype conversion.** The spacing at a
  scaled value v is `2^(max(binade(v), min_exponent) - mantissa_bits)`; `v / spacing` is
  rounded to an integer, half to even (`nearest`), or down or up with probability equal to the
  fractional part (`stochastic`, uniform draws in the compute dtype); the result is cast to the
  format exactly, since it lies on its grid. Every step is a power-of-two multiply. The
  conversion is not trusted because XLA converts float64 to FP8 and float16 through float32,
  rounding twice (measured: 2 of 2M values differ from a correct rounding for e4m3, 118 for
  float16). A test holds the nearest rounding equal to ml_dtypes' conversion of the scaled
  values. The spacing is kept a normal number of the compute dtype, since XLA's CPU flushes
  subnormals; that coarsens only bfloat16's subnormals under float32 compute, values at least
  2^119 below their block's largest (a scaled block's largest is at least 1).
- NaN, zeros (signed) and infinities pass through; `float8_e4m3fn` has no infinity, so one
  becomes NaN there. Magnitudes beyond float32's range cannot be scaled and overflow, as they
  would in float32 storage.
- **The stochastic draws** come from `Precision.rounding_seed` and the array's name in the run,
  not the simulation's seed: a market-risk run has none (historical scenarios), and naming
  each array makes arrays round independently and a trade's column round the same in a mixed
  run as alone (the exit criterion of 1.5, kept under stochastic rounding). `Precision.store(x,
  storage, stream, axis)` is the one place the pipeline stores; the names are `"shocks"`,
  `"states"`, `"market/<i>"` (the market's path arrays in `ScenarioMarket.map_arrays` order)
  and `"values/<trade id>"` (cube columns and market-risk P&L).
- **No recompiles.** The quantizer and the loader are module-level `jit` programs with the
  format, axis and dtype static: one compile per array shape, format and rounding, reused by
  every trade, date and run (a test counts the compiled programs).
- Overhead: FP8 holds an array in an eighth of float64 plus 4 bytes per 32 values.

### 6.3 Cast points

The only places that read the configuration:

| # | Where | What it does |
|---|---|---|
| 1 | `engine.simulation.config.simulate` | Normals generated (`engine/simulation/random.py`, the clip epsilon from the compute dtype) and bridged at `simulation.compute`, then `store(shocks)` |
| 2 | `simulate` | `load(shocks)`; `evolve_states` runs at `simulation.compute` (it follows the shocks' dtype; the moments are cast to it); `store(states)` |
| 3 | `simulate` | `load(states)` at `market.compute`; `build_scenario_market` follows the states' dtype (the z-independent terms keep coming from float64); every path array of the market stored at `market.storage` (`ScenarioMarket.map_arrays`); the tenor grid, which has no scenario axis, kept at `market.compute` (since 1.6) |
| 4 | `engine.valuation.portfolio.value_portfolio`; `engine.market_risk.run_market_risk` | Per trade, at `stage = precision_for(trade)` (1.5): `load(market)` at `stage.compute` (once per dtype, with its path fixings), price the trade, `store` its cube column at `stage.storage`. Market risk: shifts rounded to `simulation.compute` and stored once; per trade loaded at `stage.compute`, revalued (`revalue_trade` follows the shifts' dtype), the P&L stored at `stage.storage` |
| 5 | `engine.portfolio.market_path` (exposure); `run_market_risk` (VaR/ES) | `load(values, float64)` and the numeraire at float64, then reduce |

Every store goes through `Precision.store`, which adds the policy's rounding (§6.2). The
shocks `[T, S, d]` have their scenario axis at 1, every other class at 0.

The tenor grid (`ScenarioCurves.tenor_times`, `[D, K+1]`) is the curves' coordinates, not
scenario data: block scales need a scenario axis, and an FP8 grid would move every pillar.
Until 1.6 it was stored at `market.storage` like the curves, so a float32 market storage under
float64 compute rounded it to float32; since 1.6 it stays at `market.compute`, which changes
that one combination's numbers at float32 rounding level (no default and no snapshot run).

The first three live in `simulate`, the orchestrator of the simulation and market stages;
the kernels it calls (`generate_sobol_normals`, `apply_brownian_bridge`, `evolve_states`,
`build_scenario_market`) follow the dtype they are given. `generate_sobol_normals` keeps its
`dtype` argument: it is where the shocks are created, so it has no input to follow.

The cube is a sequence of per-trade columns, each in its own storage format
(`PortfolioValuation.columns`, step 1.5; `Stored` columns from 1.6).
`PortfolioValuation.cube` sets them side by side in their shared format when it is float64 or
float32, as before, or loaded at float64 when the formats differ or are scaled (exact: float64
holds every format's values times their power-of-two scales).
`PortfolioResult.npv_cube` stays an array, loaded at float64, so consumers see no change. A
trade priced at its precision inside a mixed run gives exactly the column it gives priced
alone with that `pricing` stage: trades share only the loaded scenario market and fixings,
which depend on the dtype alone.

### 6.4 Inputs follow dtype

Inside a stage, every array takes the dtype of the arrays it is computed from. About 46
`jnp.asarray/zeros/...` calls in `engine/valuation`, `engine/calibration`, `engine/models`
and `engine/risk` name no dtype today and so become float64 under the x64 flag; each becomes
"the input's dtype". No kernel signature gains a dtype argument.

Strict promotion (`jax_numpy_dtype_promotion="strict"`) in CI turns any accidental
float32/float64 mix into a failing test ([§13.3](#133-dtype-discipline)).

Step 1.4 did it by running float32 portfolios under strict promotion and fixing each mix where
it arose, at the kernel's entry: `legs_cube`, `european_cube` and `jamshidian_cube` cast their
coupon tables and variances to the curves' dtype (`Legs.astype`, `EuropeanTerms.astype`); the
per-path recalibration casts its basket (`BasketInstrument.astype`), volatilities and bracket
to the curves' dtype, keeps its bucket widths as Python floats (a NumPy float64 scalar is not
weakly typed) and its root bracket in that dtype; the path volatility (`Sigma.astype`) follows
the path curves. `engine.models.curves.curve_dtype` is the one helper that reads a curve's
dtype. Each cast is a no-op at float64. Before 1.4 the "float32" path was in fact mixed: the
float64 constants promoted the float32 curves, so pricing on paths ran mostly in float64 and
the cube came out float64, while path fixings ran in float32.

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

Outcome (step 1.4):

- The ceiling: `hi * (1 - ceiling_tolerance(dtype))`, with `ceiling_tolerance = max(1e-9,
  4 eps)` (`engine.calibration.ore_lgm`), unchanged at float64. Red first: with the fixed 1e-9 a
  float32 bisection stuck below the top was not flagged
  (`tests/test_precision.py::TestRecalibrationInFloat32`). On a path the flag is still
  dropped: [I-73](../known-issues.md#i-73).
- The Sobol clip: the normals are generated at `simulation.compute`, so `jnp.finfo(dtype).eps`
  is the compute dtype's.
- `scenario_batch_size`: `revalue` passes the itemsize of the shifts' dtype, which is the
  pricing compute dtype.
- Literal tolerances in the path kernels hold at float32: `jnp.maximum(t, 1e-12)` in the LGM
  bonds (1e-12 is a normal float32), the discount floor `log(1e-5)`, the variance floor 1e-6
  (computed on the host in float64), and the Jamshidian and per-path bisections, which run a
  fixed number of halvings and so converge to the dtype's resolution. The other `1e-`
  literals are host-side float64 validation (correlation checks, the scenario covariance,
  the curve tolerance of market risk).

### 6.6 What goes (gone since step 1.4)

- `check_run`'s refusal of `pricing`, `risk` and `calibration` below 64.
- `run_market_risk`'s `jax.config.update("jax_enable_x64", True)`. x64 stays on, set once
  when `engine` is imported, as since roadmap 1.3.
- `_PRICING_LOCK` (`engine/portfolio/request.py`), after the thread-safety audit of step 1.4
  (no module-level caches and no ORE globals were found; a concurrency test confirms it).
- The per-precision pool tiers (`engine/portfolio/worker_pool.py`): one pool from step 1.4;
  the pool itself went with step 1.8.

Removing the lock rests on the audit: the pipeline keeps no module-level caches or mutable
state, and never reads ORE's global evaluation date (trades carry theirs, I-64; only the test
oracle sets it). `tests/test_portfolio_entrypoint.py::TestPricePortfolioConcurrency` runs a
float64 and a float32 request on two threads at once, repeatedly, each bit for bit equal to
its sequential run.

### 6.7 Matrix products (step 2.3)

A matrix product's precision is part of the compute format, so the policy states it, never
the device (decision A-22). `engine.precision.matmul(a, b)` (the semantics of `a @ b`) asks
`product_precision(dtype)` for the operands' format and passes it to `jnp.matmul`; the
precision is then in the program itself (`dot_general`'s `precision`), and AD's transposes
of the product carry it too. float64 and float32 map to `lax.Precision.HIGHEST`; a format not
yet enabled for compute is refused, naming step 3.7, which maps TensorFloat-32, bfloat16
passes and FP8 products to formats of the policy (§8.3).

Every matrix product of the engine's JAX code goes through it; at step 2.3 there are nine:
the simulation step (`evolve_states`, 2), the Brownian bridge (1, as one `[T, T] x [T, S*d]`
product), the Bermudan rollback's bookkeeping (4) and the calibration's spread correction
(`_corrections`, 2). NumPy products on the host are exact in their dtype and are left alone.
JAX's process-wide `jax_default_matmul_precision`, which 2.2 set on import, is neither set
nor read: anyone could override it, and the report would then be false.
`tests/test_accelerator_defaults.py::TestEveryMatrixProductStatesItsPrecision` records every
`dot_general` JAX binds while the pipelines run from cleared caches (the simulation under
both models, the scenario market, every product's path pricing with the per-path
recalibration, today's values, AD Greeks, market risk; float64 and float32) and fails on any
without its format's precision or outside the engine's code, naming the line. The coding
rule is in [coding-style.md](../../concepts/coding-style.md#core-constraints).

On a CPU the precision does nothing (XLA's CPU backend computes a float32 product in float32
whatever it says), so step 2.3 moved no bit (§13.1); on a GPU it is what 2.2's process
default did, product by product.

## 7. Low-precision storage

Storage below 32 bits needs no kernel changes: values are loaded to the compute precision,
which stays at float32 or above until step 3.7. It is the first low-precision capability
(step 1.6, done 2026-10-02), and the first measurement campaign (step 3.6) runs on it while
stage 2 continues, because it never changes float64 numbers.

What is known (measured 2026-10-01, [§15](#15-measurements)):

- **FP8 is viable for shocks.** Round-to-nearest storage of 4M standard normals biases a call
  payoff by about one Monte Carlo standard error at 4M paths; stochastic rounding by about
  0.15. The earlier verdict (useful only to ~1,400 paths) compared error per value with Monte
  Carlo error, which is the wrong criterion ([§15.2](#152-bias-not-error-per-value)).
- **FP4 is not viable stored naively,** with either rounding (25 and 45 standard errors).
  Stochastic rounding keeps each value's mean but inflates the variance by 4.7%, which
  inflates volatility. A variance correction through the block scales is the candidate fix
  (step 5.3).
- **Through the pipeline** (1.6, [§15.3](#153-storage-through-the-pipeline)): one stage at a
  time on the shared portfolio, float16 is within 1e-5 of notional in bias and 7e-6 in EPE;
  FP8 moves EPE by 0.04% (shocks), 2.6% (curves, e4m3) and 0.6% (cube). Two mechanisms limit
  it, both recorded as [I-75](../known-issues.md#i-75): a column whose paths sit close
  together against their level (a bond's cube, a long log discount factor) keeps only the
  format's few bits of its spread, and rounded to nearest every path of a block moves the same
  way, a bias; stochastic rounding turns that into noise (the cube's FP8 bias 3 to 4 times
  smaller at 256 paths). Storing the deviation from a level (the difference form of §8.2, or a
  block offset) is the remedy; step 3.6 measures, per class, which formats need it.

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
  correlation mixing (already a product, but small). These forms are written in step 3.7,
  in the same rewrite as the difference form, so no kernel is rewritten twice; step 5.2 times
  them.
- **A product's precision is the policy's, not the device's.** XLA's default runs a float32
  matrix product in TensorFloat-32 on an NVIDIA GPU (a 10-bit mantissa) and in bfloat16
  passes on a TPU. Measured on the RTX 5060 (roadmap 2.2): a float32 simulation's cube was
  1.5% off the float64 one under that default, and within float32 rounding at full
  precision. Since step 2.3 (decision A-22) each of the engine's matrix products states its
  precision, taken from its operands' compute format (`engine.precision.matmul`, §6.7), so a
  float32 compute policy is float32 on every device and no process setting can change it
  (2.2 had set JAX's process-wide default instead). TensorFloat-32, bfloat16 passes and FP8
  products are compute formats in their own right, which step 3.7 makes selectable in the
  policy (and the report names) through the same helper, and 5.2 times; the device's default
  is never the silent choice.
- The rollback's matrix form also fixes its memory. Today each column (the option, the
  underlying, each cached cashflow) is interpolated at `[nodes, quadrature nodes]` points,
  vmapped over the columns, although the interpolation weights depend only on the grids:
  measured 2026-10-05, one Bermudan on a 384-per-std, 10-std grid peaked at 11.5 GB (under
  1 MB per column at ORE's default 30 and 5). One `[nodes, nodes]` operator per step, applied
  to all columns at once, holds one such buffer and runs on the matrix units. Rolling the
  columns back one at a time (`lax.map`) also cut the peak, to 5.6 GB, bit for bit, but trades
  away the batched kernel on an accelerator, so it was not adopted.

## 9. Estimation and the precision report

### 9.1 Reductions

Every reduction over paths or scenarios loads the values at float64 first. Summing adds no
error of its own, and the rounding errors of individual values average out as Monte Carlo
noise when they are unbiased.

### 9.2 The paired sample

With `paired_fraction > 0`, that share of paths (the first paths of the same scrambled Sobol
sequence, so the same random numbers) is also run at float64 throughout. The differences
measure the precision error on this run.

On a CPU the paired paths are the run's own bit for bit, since every kernel is per path, so a
float64 run measures exactly zero. A GPU chooses its kernels by batch shape, and the paired
sample is a smaller batch than the run, so there the paired paths equal the run's own to
about an ulp (measured on an RTX 5060, roadmap 2.2: a float64 EPE's correction of 1e-11 on
values near 1e6). That noise is far below any precision error the sample measures. Removing
it would mean pricing every path at float64.

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
estimation (Giles and Haji-Ali) is a later research item (step 5.3).

### 9.5 The report

`PortfolioResult.precision` and `MarketRiskResult.precision`, a `PrecisionReport`:

- the policy as run, including every resolved per-trade precision;
- the realized dtype of every stored class, read from the arrays, not the configuration;
- the device and backend the job ran on (closes [I-12](../known-issues.md#i-12));
- per figure: the corrected and uncorrected value, the paired mean difference, its standard
  error, the largest paired difference, the path counts;
- the evidence verdict per figure: validated, or the warning ([§10](#10-acceptance-standard-and-evidence)).

The verdict comes with step 5.1's evidence table; the rest was built in step 1.7 (§9.6).

### 9.6 As built (step 1.7, 2026-10-02)

**The paired sample.** `paired_paths(N, f)` (`engine/precision/estimate.py`) is 0 at `f = 0`,
else `f·N` rounded up to whole blocks of 32 paths, at least one block and at most every path.
Whole blocks keep the paired paths' block scales the run's own, and keep the number of
distinct paired shapes small (each compiles once). The portfolio pipeline
(`engine.portfolio.market_path._paired_sample`) simulates `n` paths with the same `CamConfig`
(its `samples` replaced) on the same calibrated model, at `Precision()`, and prices them with
`value_paths`, which prices paths only: t=0 values, calibrations and Greeks are float64 already
and are not repeated. Market risk revalues the first `n` scenarios' shifts at float64, in the
vmapped batches a float64 run uses (whole batches from the first scenario, then cut to `n`):
in a batch of their own shape the values differed from the float64 run's by an ulp (an ES by
1.5e-16 relative; XLA vectorizes by shape), and the run's batches reuse its compiled programs.

That the paired paths are the run's own was measured before building on it: a float64 run of
96 paths and runs of 32 and 64 with the same seed agree on their common paths bit for bit, in
the cube and the numeraire, under the LGM and the Hull-White model, the per-path Bermudan
recalibration included. scipy's scrambled Sobol sequence gives the same first points whatever
the sample size, and every kernel is per path. A test holds it: at float64 every paired
difference is exactly 0, and the figures equal the run without a paired sample bit for bit.

**The estimator.** `two_level_mean(low [N, ...], high [n, ...])` returns a `MeanEstimate`:
`value = mean_N(low) + mean_n(high - low)`, `uncorrected`, `correction`, `max_difference` and
three standard errors, all float64 and one jitted program per pair of shapes. The paired
paths are among the N, so the two terms are correlated and the corrected figure's variance is

    Var(F) = Var(f) / N + Var(d) / n + 2 Cov(f, d) / N,        d = g − f on the paired paths,

each moment the sample's (`ddof=1`; NaN with fewer than two paths). Without the covariance
term a 95% interval covers 99.9% or 87% of the time when `d` is correlated with `f` at −0.9 or
0.9 (the coverage test). `paired_quantile(statistic, low, high)` returns a `QuantileEstimate`:
the statistic on every path at the run's precision (the reported figure), on the paired paths
at the run's precision and at float64, and their difference.

**The figures.** Portfolio: EPE and ENE are two-level estimates per simulation date
(`ExposureProfile.estimates`, t=0 excluded since it has no paths), and the profile's EE_B,
EEE_B, EPE_B, EEPE_B and Basel figures are computed from the corrected EPE; PFE per quantile
is measured. Both for the netting set (`"netting_set/EPE"`) and each trade
(`"trades/<trade id>/PFE_95"`). The t=0 NPV is float64 and needs no estimate. Market risk:
each VaR and ES measured (`"portfolio/VaR_99"`); the reported VaR/ES are the run's own. A
paired sample too small for a quantile's tail gives NaN, as the ES of an empty tail is (ORE
refuses it): 64 paired scenarios hold no observation beyond a 99% VaR.

**The report.** `PrecisionReport` (`engine/precision/report.py`) on `PortfolioResult.precision`
and `MarketRiskResult.precision`, and over HTTP on the job's result (`PrecisionReportSchema`):
`policy`, `trades` (each trade's resolved stage), `realized` (`"shocks"`, `"states"`,
`"market"`, `"values/<trade id>"`, the names `Precision.store` gives the arrays; read from the
arrays: `simulate` records the formats of the shocks and states it does not keep, in
`ScenarioMarket.simulation_formats`), `devices` (`"cpu:0 (cpu)"`, read from the stored
arrays), `backend`, `jax_version`, `paths`, `paired_paths`, `figures`. It is built in the
process that ran the job, so an HTTP job reports its worker's device (I-12). A run without
scenario risk reports its policy, no stored array, and JAX's default device.

**What a paired sample costs.** The paired run is a float64 run of `n` paths: about
`n / N` of a float64 run's path work, plus a first compile of its shapes (a repeated run
compiles nothing). Measured 2026-10-02 on CPU, five cheap trades (swaps, Europeans, a bond),
4,096 paths, the cube in FP8 e4m3, warm medians: 45 ms without a paired sample, 90 ms at 2%
(96 paths), 100 ms at 25%, 154 ms with every path paired. On a portfolio this cheap the second
simulation's and pricing's per-call dispatch dominates, not path work; on one with
Bermudans the path work does. The same runs show what the report is for: the FP8 cube's EPE
correction is 11 (2%) to 79 (100%) of its standard errors from zero, the nearest-rounding bias
of [I-75](../known-issues.md#i-75), measured on the run itself.

**Limits.** The standard errors treat paths as independent. Sobol paths are not, and the
rounding errors of the 32 paths of a block share a scale, so on the pipeline they are
nominal until step 3.6 measures their coverage there (the synthetic coverage tests and three
pipeline seeds pass). A quantile's paired measurement needs a tail on the paired paths: ES at
99% on 64 scenarios is NaN.

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

### 11.1 Decision (A-14), built in step 1.8 (2026-10-04)

The HTTP API and the engine are separate processes. The API validates a request, writes it
to a durable job queue (a SQLite file) and returns `202`. **One engine worker process** per
host takes jobs from the queue one at a time, parses the request JSON itself, prices it and
writes the result back. The worker is single-threaded and owns every device on its host.

As built: `engine/api/job_queue.py` (the queue and the worker lock), `engine/api/worker.py`
(the worker, command `jax-risk-worker`), `engine/api/supervisor.py` (the API's supervision);
the user-facing description is [HTTP API: Jobs](../../reference/http-api.md#jobs-the-queue-and-the-engine-worker).

### 11.2 Why

| Reason the process pools existed | Now |
|---|---|
| A different x64 flag per tier | Gone since 1.3; x64 is always on |
| Keep the HTTP server responsive during minutes-long jobs | The separate worker process does it |
| Run jobs in parallel | XLA uses the whole device per job; on TPU one process owns a chip, so several processes need pinning and fight step 3.8's sharding |
| Crash isolation | A dead pool worker likely broke the pool (no `BrokenProcessPool` handling); the separate worker is restarted by its supervisor without touching the API |

What it removed: the freeze/thaw of ORE objects (the worker reads JSON, not pickled
dataclasses), the `multiprocessing` spawn setup (I-33), the per-worker compile, and every
thread on the engine side (`_PRICING_LOCK` went in 1.4). `price_portfolio` called from Python
never involved any of this and is unchanged. The API process keeps the server's own request
thread pool, and the EOD path its locks (`engine/integration/`), which 1.8 does not touch.

### 11.3 Multi-device and multi-host

- **One host (done 2026-10-04):** the scenario draws of each job (the Sobol normals, market
  risk's shifts) are placed on a one-axis `jax.sharding.Mesh` of the host's devices, split
  along the scenario axis, and XLA's sharding propagation carries the split through the rest
  of the job (`engine/simulation/sharding.py`). As many devices as divide the scenario count,
  capped by `JAX_RISK_SCENARIO_DEVICES`; one device places nothing, so one-device runs are
  unchanged bit for bit. Four CPU host devices equal one to about 6e-16 relative
  (`tests/test_sharding.py`). No threads, no pinning.
- **Several hosts (a TPU pod slice):** one worker per host, all running the same program
  (`jax.distributed.initialize`). Under SPMD every host must run *the same job* at the same
  time, so the hosts cannot each claim from the queue as one host's worker does: process 0
  claims and the others receive the body from it (`jax.experimental.multihost_utils`, or the
  distributed client's key-value store), and only process 0 writes the row. SQLite on a
  shared filesystem is not a safe multi-host lock. Step 3.8 designs this; 1.8 built the
  one-host case.

### 11.4 The queue (A-14; not a main priority)

SQLite rows: the request body as received, status (`pending`, `running`, `done`, `failed`,
`interrupted`), failure class and error, the result document, the claiming worker, the XLA
programs the job built, timestamps. It survives restarts and closes the portfolio half of
[I-08](../known-issues.md#i-08); the EOD path keeps its publication store.

Decisions taken while building it (1.8):

- **One worker per queue, by an OS file lock** (`<queue>.worker.lock`, `flock`/`msvcrt`). A
  starting worker takes it before importing JAX, so a redundant one exits in milliseconds,
  and the OS frees it however the holder dies. Holding it proves no other worker runs, so a
  starting worker marks every `running` row `interrupted` (its predecessor's job; never
  retried, so a job that kills the worker cannot crash-loop it).
- **The result is stored in the row**, not as a reference to a file: one atomic write, and
  the poll route splices the stored document into its response without parsing it. SQLite's
  1 GB limit fails such a job as `infrastructure`; cubes that large are I-09's problem first.
- **Supervision without a thread.** In `spawn` mode the API checks its child on every
  submission and poll and restarts it if dead; a dead worker matters only then. With several
  API processes (`uvicorn --workers`), a supervisor starts no worker while another process's
  worker holds the lock. The worker watches the process that started it and exits when it
  is gone (between jobs), so a killed server or test run leaves no engine process (the pool's
  orphans held tens of GB for days). `external` mode leaves supervision to systemd or a
  container.
- **The body as received** is what is queued and parsed again (`json.loads`, then the
  route's schema), so the worker prices the dataclass the route validated, by construction.
- **`running` is reported** (the pool could not distinguish it from `pending`), and
  `JobStatusSchema` gained `interrupted` and `failure_class`.
- **Failure classes by exception type**: `KeyError`/`MissingFixingError` →
  `missing-market-data`; `NotImplementedError`/`UnsupportedDayCountError` →
  `unsupported-product`; `ArithmeticError` → `numerical-failure`; other `ValueError`/
  `TypeError` → `bad-terms`; anything else → `infrastructure`.

## 12. Plan

| Step | Work | Exit criterion | Size |
|---|---|---|---|
| **1.4** (done 2026-10-01) | `engine/precision/` (formats, policy, storage at float64/float32; the report skeleton moved to 1.7, `Stored` to 1.6, §5); `Precision` replaces the old types in `RunConfig`, `MarketRiskRequest` and the HTTP schema, the old shape refused (A-12); the cast points; inputs follow dtype; float64 reductions; the constants of §6.5; market risk on the same module; remove `check_run`'s refusal, the flag set, the lock and the tiers; strict promotion in CI | Default: golden snapshot bit for bit, every ORE parity suite unchanged. float32: the scenario market and the market-risk revaluation bit for bit as before; the portfolio cube is not, and cannot be (§13.1); exposure and VaR/ES differ by the float64 reductions. Fast tier green under strict promotion. Met: §13.1 | M |
| **1.5** (done 2026-10-02) | Per-product and per-trade precision (A-15): `by_product`, `by_trade`, `precision_for`; per-trade stored columns; market risk per trade | Bit for bit at the default; a mixed run (Bermudan float32, swaps float64) equals each trade run alone at its precision, column for column. Met: §13.1 | S |
| **1.6** (done 2026-10-02) | Sub-32-bit storage: block scales, both roundings; `float16`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2` enabled for storage | The storage properties of §13.4; bit for bit at the default. Met: §13.1, §13.4 | M |
| **1.7** (done 2026-10-02) | Paired sample, two-level estimator for means, `PrecisionReport` with realized dtypes and device (closes I-12) | §13.6; bit for bit at the default. Met: §13.1, §13.6 | M |
| **1.8** (done 2026-10-04) | Engine worker process and the SQLite queue (A-14); delete `worker_pool.py`'s pool and freeze/thaw | §13.8, including the Linux run; float64 job time and compile count no worse. Met: §13.8, §13.9 | M |
| **3.6** | *Parallel with stage 3.* Storage relative to a level for classes whose level swamps their spread (I-75); a measurement harness for storage formats per class and product, at several path counts, fixed seeds, rerunnable on any kernel change | Harness covers every figure × class × format; thresholds fixed before measuring; I-75's FP8 bond column unbiased within its Monte Carlo standard error | M |
| **3.7** | Difference-form kernels (§8.2), one family at a time, as matrix products where a family can be one (§8.3); compute below float32 enabled; `accumulate` honoured | Per family: ORE parity suites at their tolerances, then the re-baseline of §2.1; emulated FP8/bfloat16 compute measured by the harness | L |
| **3.8** | Shard the scenario axis in the worker (I-61): one host done 2026-10-04; several hosts next | Results equal the one-device run within reduction-order rounding (one host: met, `tests/test_sharding.py`); scaling measured on TPU | M |
| **5.1** | The evidence table and the warnings (§10), from 3.6's harness on the kernels after 3.4, 3.5 and 3.7 | Table complete; a verdict and path ceiling per row; a warning on every result without a passing row | S |
| **5.2** | After 3.8 and 5.1. Timing across devices on Ironwood and H100: storage formats, then 3.7's matrix-product kernels with native FP8 | The research result: wall time per figure and precision against float64 at equal accuracy (the evidence table's path ceilings), many low-precision paths against fewer float64 ones | M |
| **5.3** | FP4 storage (variance correction through the block scales), FP4 compute on TPU 8t/8i, multilevel quantile estimation | Evidence rows for FP4 | L |

**Order (decided: A-19, 2026-10-05, which replaced the earlier structure-first order):** the structure first
(1.4 to 1.8, done); then the roadmap's near-term milestone (stage 2: the profiling demo on a
local GPU), which moves no float64 number; then the foundations (stage 3), where the storage
measurement (3.6) runs alongside because it changes no float64 number, and the kernel rewrite
(3.7), difference form and matrix products together, comes after stage 3's own kernel changes
(3.3 strike axis, 3.4 `ShiftHorizon`, 3.5 recalibration), so no kernel is rewritten twice, and
before stage 4 adds any kernel; multi-host (3.8) moves no arithmetic and is parallel. The
evidence table (5.1) is measured on the kernels that ship, starting once 3.7 is done; timing
(5.2) and FP4 (5.3) follow it.

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
- **Step 1.4's result (2026-10-01).** 164 arrays at the default precision (today's values under
  two engines; the shared portfolio's cube, exposure and per-trade EPE under the LGM and the
  Hull-White model; bump and AD Greeks; market risk at float64), plus the float32 scenario
  market and the raw float32 market-risk revaluation: all 164 identical in value, dtype and
  shape. Market risk at float32: the P&L identical in value (now float64-loaded), VaR/ES within
  1e-7 relative (float64 reductions). The portfolio cube at float32 is not bit for bit with
  the old `simulation=32` cube, and the criterion as first written could not hold: that cube
  was float64, priced in mixed precision (float64 coupon tables promoted the float32 curves,
  path fixings ran in float32; §6.4). Float32 throughout differs from it by at most 1.5e-6
  relative (both are about 2e-6 from float64); a float32 simulation and market priced in
  float64 is 5.6e-7 from float64.
- **Step 1.5's result (2026-10-02).** The same snapshot script, 232 arrays from a worktree of
  `2b70d15` (the default runs and, as before, the float32 runs): all 232 identical in value,
  dtype and shape, float32 included, since a policy without overrides prices exactly as
  before. Every ORE parity suite passed unchanged (full suite 2,337 passed). The exit
  criterion's tests (`TestPerTradePortfolio`, `TestPerTradeMarketRisk`) are red when every
  trade is priced at `pricing.compute`. float64 speed (§13.9), old and new trees interleaved:
  market risk on four trades, median of ten warm runs, 5.95/6.07 s before and 5.73/5.84 s
  after (no regression); the shared portfolio's run varies 2.5× from run to run (per-path
  recalibration, I-53), and the fastest of five was 118.7 s before and 120.3 s after.
- **Re-baseline after the jit change (2026-10-02, between 1.5 and 1.6).** The pricers,
  the LGM bootstrap and the grid engine became module-level jitted programs taking the trade
  as a traced argument (I-21, I-22; [profiling §3.7](../../concepts/profiling.md#37-trade-data-as-traced-arguments-2026-10-02)),
  which moves float64 at rounding level: XLA fuses whole programs where values were computed
  op by op, and some old results depended on which operands XLA folded as constants. The same
  snapshot script, 232 arrays, old tree (a worktree of `a6c63bf`) against new: 40 identical,
  among them the float32 scenario market (simulation is untouched); the rest moved by at most,
  relative to each array's largest magnitude: today's values 2.3e-16; the LGM and Hull-White
  cubes 5.1e-15 and 5.0e-15; market-risk P&L 9.8e-15; every bump and AD Greek by at most
  5.1e-14 of its trade's NPV (a Gamma's own relative change looks larger, up to 2e-7, since it
  is a second difference of NPVs). float32 runs, at float32 rounding: the cube 2.7e-6, the
  raw shocked revaluation 1.6e-6, market-risk statistics 7.5e-6. Every ORE parity suite
  passes at its tolerance. Steps 1.6 to 1.8 compare against a snapshot of the commit that
  made this change, not of `a6c63bf`. The
  whole snapshot took 14 minutes instead of 69 (bump Greeks 21 s instead of 29 minutes),
  both measured with other jobs running.
- **Step 1.6's result (2026-10-02).** The same snapshot script, 232 arrays, from a worktree of
  `f51227c` (the commit of the jit change) against the 1.6 tree: all 232 identical in value,
  dtype and shape, the float32 runs included. float64 and float32 storage are the same plain
  cast as before, and the one float32 behaviour 1.6 changes, the tenor grid under a float32
  market storage with float64 compute (§6.3), is in no snapshot run.
- **Step 1.7's result (2026-10-02).** The same snapshot script, 232 arrays, from a worktree of
  `f0a438c` (1.6) against the 1.7 tree: all 232 identical in value, dtype and shape, the
  float32 runs included. The default path changed shape (t=0 values computed apart from the
  path pricing, `value_paths`; one pricing context for every trade's t=0 value; the report
  built from the arrays) but not its arithmetic. A paired sample is off by default, and at
  float64 it measures exactly zero (`tests/test_precision_report.py`). float64 speed (§13.9),
  old and new trees interleaved, three passes of five warm runs each: market risk on four
  trades, median 2.19 s before and after (fastest 1.87 and 1.65 s); a five-trade, 1,024-path
  portfolio, median 39 ms before and 37 ms after.
- **Step 1.8's result (2026-10-04).** No file of the pipeline changed (`git diff` against
  `0e44f1d` touches `engine/api/`, a docstring of `engine/portfolio/profiling.py` and the
  deleted `engine/portfolio/worker_pool.py`), so the snapshot was not rerun: it would compare
  the same code. What 1.8 changed is the path from the HTTP body to `price_portfolio` and
  back, and that is shown bit for bit: float64 and float32 jobs through the real worker equal
  the direct call in the test process, cube, base NPV and EPE
  (`tests/test_engine_worker.py::TestEngineWorkerPricing`), and the HTTP tests of
  `tests/test_api.py` pass unchanged in what they assert. float64 speed (§13.9), old and new
  trees interleaved, three passes of 15 warm jobs each, the whole HTTP round trip (submit,
  poll, parse) on a five-trade, 1,024-path portfolio: median 56.8 ms before and 54.2 ms after
  (p10 46/44 ms, p90 117/118 ms); the cold first job 5.39 s before and 5.22 s after; the
  results identical across trees. A first attempt cost 20 ms more per job: the queue opened
  an SQLite connection per call, and closing a WAL database's last connection checkpoints
  it; one connection per thread fixed it without giving up `synchronous=FULL`.
- **Step 2.2's result (2026-10-05).** 2.2 changes no kernel, but it sets three process
  defaults in `engine/__init__.py` (no GPU preallocation, matrix products at their operands'
  precision, deterministic GPU kernels), and the last two change every program's HLO or
  compile options. The same snapshot script, 232 arrays, from a worktree of `1a533f3` against
  the 2.2 tree, on CPU: all 232 identical in value, dtype and shape, the float32 runs included
  (XLA's CPU backend computes a float32 product in float32 whatever the precision
  configuration says). On the GPU (RTX 5060, WSL2) float64 is not bit for bit with the CPU,
  and is held to the parity tolerances, which every ORE parity suite meets: the demo's job
  equals the CPU's today's value exactly, the cube and EPE to 9e-16 of their scale, the AD
  Greeks to 2.3e-11 relative.
- **Step 2.3's result (2026-10-06).** 2.3 moves 2.2's three process defaults to the processes
  that own them and states every matrix product's precision in the program
  (`engine.precision.matmul`, §6.7), so the products' HLO carries the same precision 2.2's
  process default gave it; the Brownian bridge's `tensordot` became a reshaped matrix product
  (checked bit for bit against it on its own first, at float64 and float32, up to
  `[120, 2048, 4]`). The same snapshot script, 232 arrays, from a worktree of `7b51e19` (2.2)
  against the 2.3 tree, on CPU: all 232 identical in value, dtype and shape, the float32 runs
  included.
- **Step 2.4's result (2026-10-06): the AD Greeks re-baselined.** 2.4 makes each trade's AD
  Greeks one jitted program per product (Delta and Gamma from one linearization of the
  gradient over all its curves; Vega another), where they were eager derivatives of the
  jitted pricers ([profiling §3.8](../../concepts/profiling.md#38-the-ad-greeks-as-one-program-per-product-2026-10-06-roadmap-24)).
  The roadmap expected 2.4 to keep every number; it cannot for the AD Greeks: the repeated
  job's recompiles came from the eager derivatives' programs living only in JAX's bounded
  internal caches, and once a derivative is one XLA program, XLA fuses its forward and
  backward passes and reorders their reductions. The snapshot gained the demo's own job
  (`demos/demo_profile_small.py`'s request through the API's schema, run twice in the
  process: Hull-White calibrated, options recalibrated on every path date, AD Greeks), 100
  arrays, so 332 in all, from a worktree of `2df78cb` (2.3) against the 2.4 tree, on CPU:
  286 identical in value, dtype and shape, every cube, exposure, today's value, bump Greek,
  market-risk figure and float32 run among them. The 46 that moved are AD Greeks, by at most,
  relative to each array's largest magnitude: the demo Bermudan's discount Gamma 3.4e-14 and
  Delta 1.7e-14, the demo American's discount Delta 1.2e-14, every other below 1e-14 (the
  shared portfolio's at most 9.5e-15, the demo swap's and European's at most 5.8e-16). A pillar
  whose true sensitivity is zero changed between rounding noises of order 1e-18. A Jamshidian
  European's AD Greeks, in no snapshot run, moved by at most 2.7e-16 of their scale. On the GPU
  the demo's cube, EPE and today's value are identical to the old code's, the AD Greeks
  within 6.6e-16 of their scale, and two runs of each tree are identical. Every ORE parity
  suite passes at its tolerance (none compares an AD Greek with ORE: ORE's are bump
  sensitivities, I-51). Steps from 2.5 on compare against a snapshot of the 2.4 commit.
- **Step 2.5's result (2026-10-07): the solver re-baselined.** 2.5 puts every root of the
  engine on one solver (`engine.numerics.roots`, decision A-21): a safeguarded Newton method
  by default, the bisections as the reference (`"Bisection"`), and batches a Bermudan's or
  American's path dates of one basket shape into one calibration under Newton. The snapshot
  gained the Jamshidian engine (the shared portfolio's European on the paths, its AD Greeks)
  and the standalone bootstrap (`POST /calibration/lgm`'s, on a sloped curve, its Vega through
  x*): 359 arrays, from a worktree of `eadbcc6` (2.4) against the 2.5 tree, on CPU.
  - With every solver set to `"Bisection"`: all 359 identical in value, dtype and shape. The
    reference keeps one path date per calibration call: batched over dates, even the
    bisection moved 43 arrays (float64 at most 3.9e-16 of their scale, the cubes and
    exposures through the recalibrated options; float32 at most 1.2e-7), since XLA vectorizes
    another shape and the bisection's last comparisons follow the residual's last bit.
  - With the `"Newton"` default: 149 identical (every market-risk figure, whose engines do not
    calibrate, the raw float32 scenario market, the standalone Vega); 210 moved. In float64,
    relative to each array's largest magnitude: today's values at most 7.1e-17; the cubes and
    exposures at most 3.3e-15 (shared portfolio, LGM), 5.6e-15 (Hull-White), 5.3e-15
    (Jamshidian), 7.0e-15 (the demo's job); AD Greeks at most 2.2e-14 (Delta, Gamma, Vega);
    the standalone calibration's volatilities 4.4e-14 and model values 1.0e-14. Larger, and
    still rounding: a difference of recalibrated prices amplifies it, bump Delta to 2.3e-11,
    bump Vega to 9.3e-13 and bump Gamma to 6.4e-8 of their scale (second differences at a 1bp
    shift), Theta to 1.1e-11 of itself (1e-16 of the values it is the difference of); the
    standalone calibration's `rmse`, itself 4e-11 of rounding, by 70% of itself; float32 runs
    at most 1.5e-7 (float32's rounding on the recalibrated paths).
  Every ORE parity suite passes at its tolerance under the default, and under the reference
  too (the 15 modules that compare with ORE or QuantLib, 509 tests, with the default set to
  `"Bisection"` for the run). Steps from 2.5 on compare against a
  snapshot of the 2.5 commit.
- Step 3.7: parity suites pass at their tolerances first; then the snapshot is re-baselined,
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
  the realized dtypes in `PrecisionReport` with the policy. Done in step 1.7:
  `tests/test_precision_report.py::TestReport::test_the_realized_format_of_every_class_is_the_policys`
  (each stage in each of the six formats, one trade overridden), and market risk's shifts and
  P&L (`TestMarketRisk`).
- A test that the resolver is the only lookup: per-trade overrides change only that trade's
  column. Done in step 1.5: `tests/test_precision.py::TestPerTradePortfolio` and
  `TestPerTradeMarketRisk` (a mixed run equals each trade alone at its precision, bit for bit,
  in the cube and the market-risk P&L; red when the market is loaded at `pricing.compute`
  for every trade), `TestOverrides` (trade over product over stage) and a mixed run under
  strict promotion.

### 13.4 Storage properties

Fast, no market or ORE needed:

- values representable in the format round-trip exactly;
- round-to-nearest error is at most half a step of the format times the block scale;
- stochastic rounding is unbiased (mean over many draws within three standard errors of the
  input) and repeats exactly for the same seed;
- no block overflows; zeros and NaN pass through; a scenario count not divisible by 32;
- `store`/`load` at float64 and float32 is the identity.

Done in step 1.6 (`tests/test_precision.py::TestScaledStorage`), for every scaled format, from
float64 and float32: values of the format round-trip exactly under both roundings; nearest
equals ml_dtypes' correctly rounded conversion and is within half a step times the scale;
each scale is a float32 power of two that brings its block's largest magnitude into
(max/2, max]; stochastic rounding moves a value to one of its two neighbours, its mean within
three standard errors where nearest is biased, repeats for its key and differs for another;
zeros (signed), NaN and infinities pass through (NaN in `float8_e4m3fn`); short last blocks on
any axis; blocks independent of each other; FP8's size; a pytree loading inside `jit`; misuse
refused; each kernel compiled once per shape. Through the pipeline
(`TestScaledStoragePipeline`): every stage stored in every scaled format (realized formats read
from the arrays), end to end under strict promotion with stochastic rounding, a stored cube
column within a step of its float64 values, a stochastic run reproducing bit for bit with no
new compile and moving with its seed, market risk's P&L within a step and exactly zero for a
zero shift; per trade, a mixed run with scaled formats and stochastic rounding equals each
trade alone, column for column, in the cube and the P&L. Slow tier: every stage × format on
the shared portfolio within measured bounds (§15.3), and stochastic rounding's smaller bias of
an FP8 cube.

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

Done in step 1.7, `tests/test_precision_report.py`: the estimator on synthetic samples
(identical samples give the plain mean bit for bit and a zero correction; a "format" adding 0.3
and noise is corrected to within three standard errors while its correction is 50 of its own
from zero; with every path paired the estimate is the float64 mean; 95% coverage over 400 seeds
at correlations −0.9, 0 and 0.9 between the difference and the value); the portfolio pipeline
(float64 against float64 under both models: every paired difference exactly 0 and every
figure the plain run's bit for bit; FP8 cube storage with every path paired: EPE/ENE equal the
float64 run's to 1e-12 of the peak while the uncorrected means are off; FP8 e5m2 storage of a
bond's concentrated cube, three seeds: the corrected EPE within three standard errors of the
float64 run on the same paths; a repeated paired run compiles nothing; a float32 paired run
under strict promotion); market risk (float64 against float64 exactly 0; with every scenario
paired the float64 VaR/ES measured exactly); the report (realized formats, devices, a run
without paths, the wire form); over HTTP, the worker's report on a job's result
(`tests/test_api_market_path.py`, I-12). The quantiles' measurement is exact by construction;
their correction is step 5.3.

### 13.7 Statistical acceptance (slow tier)

The evidence table's rows become slow-tier tests with fixed seeds and the thresholds fixed
before measuring: bias against the rule of §10 at the stated path counts.

### 13.8 Execution

- Jobs queued together give the same bits as run one after another.
- A failing job fails only its own row, with a failure class; the worker survives.
- A worker killed mid-job leaves an `interrupted` row; a restarted worker picks up the next.
- The existing API tests pass unchanged (`202`, poll, result).
- The full suite on Linux (Docker `python:3.11`), as process changes require.

Met by step 1.8 (2026-10-04), in `tests/test_engine_worker.py`:
`test_jobs_queued_together_give_the_bits_of_jobs_run_one_after_another`,
`test_a_failing_job_fails_only_its_own_row` with the failure-class table,
`test_a_worker_killed_mid_job_leaves_it_interrupted`. `tests/test_api.py` passes with two
changes that follow from the design, not from a result: the module fixture that shut the pool
down went (the session's worker is stopped by `tests/conftest.py`), and the test that a
fresh job is not priced eagerly accepts `running`, which the pool could not report. Red
first on the code before 1.8: a pool worker killed while idle made the next submission fail
with `BrokenProcessPool` (the new worker: the killed job `interrupted`, the next `done`), and
a job id was a `404` from a second app process (now served from the queue). The full suite
ran on Linux ([verification status](../known-issues.md#verification-status)).

### 13.9 Performance

- float64 wall time and compile count of the shared portfolio against the previous step,
  median of five warm runs: no regression beyond the run-to-run spread.
- float32 timing recorded.
- After 1.8: a second identical job compiles nothing. Met: each job's row records the XLA
  programs it built (JAX's `backend_compile_duration` event); the benchmark portfolio's first
  job built 122, each of the 18 repeats 0
  (`test_a_second_identical_job_compiles_nothing`). The pool built the same programs once per
  worker (two by default).
- Step 5.2 on Ironwood and H100, across devices (after 3.8): wall time per figure at equal accuracy.

### 13.10 Demos

`demos/demo_precision.py` moves to `Precision` (done in 1.4, with a storage-only float32 run
beside float32 throughout), adds FP16, BF16 and FP8 storage runs, FP8 with both roundings
(done in 1.6), and prints a run's report with a paired sample (done in 1.7).

## 14. Scope

**In:** float64/float32 compute and storage; float16, bfloat16 and FP8 storage; compute below
float32 after 3.7; per-stage, per-product and per-trade precision; block scales and both
roundings; the paired sample, the two-level estimator for means and the report; the evidence
table and warnings; the worker process and queue; timing on Ironwood and H100.

**Out, with the reason:**

| Item | Why | Where |
|---|---|---|
| Low-precision calibration, t=0 and Greeks | Cheap and fragile (A-10); openable later by a field | — |
| FP4 | Needs TPU 8t/8i or Blackwell | 5.3 |
| Variance correction of stored shocks | Only if 3.6 shows bias | 5.3 |
| Multilevel quantile estimation | A research project of its own | 5.3 |
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

This is one payoff on raw shocks, not pricing through the pipeline; step 3.6 measures the
pipeline's figures.

### 15.3 Storage through the pipeline

Measured 2026-10-02 on the step 1.6 code: the shared portfolio (8 trades, 3% -> 5% curves),
256 paths, 3 dates, one stage stored in the format and the others float64 (compute float32 in
the stored stage, float64 for pricing, so only storage differs). Per unit notional: the
largest error of a cube entry, the largest bias (the mean over the paths of a trade on a date),
and the largest change of the netting set's EPE relative to its peak. One seed: a screening,
not the evidence of step 3.6.

| Format | Stage | Nearest: max error / bias / EPE | Stochastic: max error / bias / EPE |
|---|---|---|---|
| float16 | shocks and states | 7.6e-5 / 6.2e-7 / 1.4e-6 | 8.3e-5 / 2.3e-6 / 1.1e-5 |
| float16 | market | 9.8e-5 / 3.0e-6 / 1.0e-5 | 1.8e-4 / 3.1e-6 / 3.6e-5 |
| float16 | cube | 4.5e-4 / 9.5e-6 / 6.7e-6 | 5.0e-4 / 3.3e-5 / 3.0e-5 |
| bfloat16 | shocks and states | 5.5e-4 / 2.6e-6 / 9.8e-6 | 1.8e-2 / 7.1e-5 / 5.6e-5 |
| bfloat16 | market | 1.8e-2 / 6.6e-5 / 1.1e-4 | 1.5e-3 / 2.7e-5 / 4.2e-4 |
| bfloat16 | cube | 3.5e-3 / 1.1e-4 / 1.0e-4 | 5.5e-3 / 1.5e-4 / 1.1e-4 |
| FP8 e4m3 | shocks and states | 1.7e-2 / 8.0e-5 / 3.6e-4 | 1.7e-2 / 1.9e-4 / 8.8e-4 |
| FP8 e4m3 | market | 1.7e-2 / 4.1e-4 / 2.6e-2 | 2.7e-2 / 1.2e-3 / 4.4e-3 |
| FP8 e4m3 | cube | 3.3e-2 / 6.8e-3 / 5.9e-3 | 6.4e-2 / 2.7e-3 / 2.0e-3 |
| FP8 e5m2 | shocks and states | 1.5e-2 / 3.1e-4 / 1.3e-3 | 3.0e-2 / 4.3e-4 / 2.4e-3 |
| FP8 e5m2 | market | 2.5e-2 / 1.2e-3 / 5.1e-2 | 5.1e-2 / 2.3e-3 / 1.1e-2 |
| FP8 e5m2 | cube | 6.5e-2 / 3.7e-2 / 3.2e-2 | 1.2e-1 / 8.5e-3 / 5.7e-3 |

Reading it:

- **The largest errors are exercise decisions** that flip on one path: the cash-settled
  Bermudan, worth 0 at float64 and 1.7% of its notional with a bfloat16 market or FP8 shocks
  (it exercises in one run and not the other). The median entry error is 10 to 100 times
  smaller. Bias, not the largest error, is the criterion (§15.2).
- **Nearest rounding of a concentrated column is a bias.** The bond's cube sits near 1e6 with a
  path spread near 1e4; FP8 keeps 3 or 2 mantissa bits of the level, so the 32 paths of a block
  round alike. Stochastic rounding removes the bias, leaving noise that shrinks with the paths:
  the FP8 cube's bias falls from 6.8e-3 to 2.7e-3 (e4m3) and 3.7e-2 to 8.5e-3 (e5m2). The
  market's curves are concentrated the same way (a long log discount factor varies by a few
  percent of itself across paths), and an FP8 market moves EPE by 2.6% (e4m3, nearest).
  [I-75](../known-issues.md#i-75).
- **float16 is accurate in every stage** at this path count: bias at most 1e-5 of notional.
  bfloat16 keeps 7 mantissa bits to float16's 10 and is about ten times worse; its range is
  not needed with block scales.
- Each run took 12 to 24 s against 98 s for the float64 reference, which includes its first
  compiles; CPU time is not evidence of speed (§2.3).

**Market risk** (`demos/demo_precision.py`, 2026-10-02: five trades with offsetting swaps,
8,192 scenarios, five seeds; the shifts and each trade's P&L stored in the format, revaluation
in float64). The largest change of a figure from float64, against the spread of float64
across seeds:

| Storage | VaR 99% | ES 97.5% |
|---|---|---|
| float16 | 0.044 | 0.005 |
| bfloat16 | 0.25 | 0.13 |
| FP8 e4m3, nearest | 1.8 | 1.8 |
| FP8 e4m3, stochastic | 2.2 | 4.5 |

- **Netting amplifies storage error.** Each trade's P&L is stored at its own scale, and the
  payer and receiver swaps' P&Ls nearly cancel in the portfolio's, so the portfolio P&L's error
  is large against the portfolio P&L itself (FP8: 25% of its largest value).
- **Stochastic rounding biases quantiles.** It keeps each value's mean but adds variance, and
  a wider P&L distribution has a larger VaR and ES: unbiased noise becomes a bias of a tail
  figure, here larger than nearest's. Means are corrected by the two-level estimator (A-13);
  quantiles are not (§9.4), so for VaR/ES the storage itself must be precise enough. Step 3.6
  measures it per figure; [I-75](../known-issues.md#i-75).

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
| A-22 (2026-10-06) | A matrix product's precision is its compute format's, stated by the product (`engine.precision.matmul`, §6.7), never a process-wide default (step 2.3) |

Engineering defaults, changeable without a decision: block size 32 along the scenario axis;
the rounding default chosen by step 3.6; `paired_fraction` 0 by default, 0.02 suggested for
reduced-precision runs; the evidence table's location.

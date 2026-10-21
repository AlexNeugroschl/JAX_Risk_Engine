# Precision

Design and plan for adjustable numeric precision, from float64 down to FP8, and for the
execution architecture that runs it. It implements [I-55](../known-issues.md#i-55),
[F-07](../features.md#f-07), [I-12](../known-issues.md#i-12) and
[I-72](../known-issues.md#i-72), and the owner decisions A-9, A-10 to A-16, A-22 and the revised
D-9 ([compliance/decisions.md](../../../compliance/decisions.md)). It replaces
`sub-fp32-precision.md`; its findings are kept, corrected, in [§15](#15-measurements).

**Written:** 2026-10-01 · **Items:** [F-07](../features.md#f-07), [I-55](../known-issues.md#i-55),
[I-61](../known-issues.md#i-61), in the roadmap's [stage 1](../roadmap.md#stage-1) (accurate
and fast on TPU and GPU clusters) and [stage 2](../roadmap.md#stage-2) (the research result).
The structure stage (2026-10-01 to 10-04, done) built the mechanism, the storage harness (2026-10-09) closed
[I-75](../known-issues.md#i-75), and the kernel rewrite enabled compute below float32 for every
path kernel ([§8](#8-compute-below-float32)): 2026-10-10, and the Bermudan/American rollback on
2026-10-13

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
- **The structure stage's precision work (2026-10-01 to 10-04) kept the default bit for bit.** Their cast points are no-ops at float64
  (`astype` to the same dtype, which XLA removes), and a golden snapshot proves it
  ([§13.1](#131-bit-for-bit-and-ore-parity)). One change between per-trade precision and sub-32-bit storage was allowed to
  move float64 at rounding level, by decision of 2026-10-02: jitting the pricers with the
  trade as a traced argument (I-21, I-22). It followed the kernel rewrite's procedure below, and the
  snapshot was re-baselined after it (§13.1); the rest of the structure stage is bit for bit against that.
  The compiled Greeks (2026-10-06) re-baselined the AD Greeks alone in the same way, each now one compiled program
  (§13.1); every other array stayed bit for bit.
- **The kernel rewrite (F-07) moves float64 numbers at rounding level.** It rewrites the kernels
  for low-precision compute and uses the same kernels at float64 (A-16). Every ORE parity suite
  passes at its existing tolerance (1e-14 to 1e-8), and the golden snapshot stays within the
  margin of A-19's 2026-10-10 revision (the first families, 2026-10-10: float64 within 2.2e-14
  of each array's scale, float32 within 8.2e-6; §13.1). The Bermudan/American rollback's rewrite
  follows the same procedure, its model ORE's on the paths since 2026-10-12 (`ShiftHorizon`,
  I-32; the calibration kept as ORE built the trade, I-49).
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
  float32 is emulated on every chip. The design puts the low precision where these pay: the
  storage format between stages, the compute format in the matrix products, and everything
  else at the accumulate format ([§3.1](#31-three-precisions)).
- **Speed is measured on the target hardware** (Ironwood, H100; the owner has access),
  the research result (F-07). CPU numbers are accuracy evidence only.
- **One compile cache per host.** The worker process ([§11](#11-execution-architecture))
  compiles a job shape once (measured: a second identical job builds no program); the
  per-process pools compiled it once per worker.

### 2.4 Design: DRY, KISS, modularity

- **One table of formats** (`engine/precision/formats.py`) serves validation, the HTTP
  schema, storage and the report. No other module maps names or bit counts to dtypes. Both
  of today's `64 → float64 else float32` helpers go (`engine/run/config.py:116`,
  `engine/risk/market/run.py:96`); the second silently maps any value to float32.
- **One resolver** (`precision_for(trade)`) decides a trade's precision; pricing and market
  risk both call it.
- **One storage mechanism** (`store`/`load`) for every class and both pipelines.
- **Few cast points.** Only the lines in [§6.3](#63-cast-points) read the configuration.
  Inside a stage, code follows the dtype of its inputs and never takes a dtype argument; a
  kernel with matrix products takes the stage's `Arithmetic` (its compute and accumulate
  formats) as one static argument, since a product format such as TensorFloat-32 has no dtype
  of its own to follow ([§8](#8-compute-below-float32)).
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
| **compute** | The format of a stage's matrix products: their operands, and how the product runs (TensorFloat-32, bfloat16 passes, FP8 units); below float32 the operand every path shares in two terms of the format (A-23, §8.3) | Once per operand, at every product, so errors build up over a scan's steps | Time, on hardware with native units for it |
| **accumulate** | The format the products accumulate in, and every other operation of the stage is evaluated in (sums, the scan's carry, a level plus a path's deviation, payoff formulas) | At every operation | Keeps long sums from drifting or overflowing, and the levels the deviations are measured from |

Low-precision hardware works this way: FP8 inputs to a matrix multiply, with sums
accumulated in float16 or float32, and the elementwise work around the products in float32
(no device runs it faster below float32, §2.3). No accelerator sums in FP8. Keeping the three
apart is what lets the study say which one limits accuracy.

**As built (the kernel rewrite, 2026-10-10).** Levels are float64 and meet each path's
deviation in two terms of the accumulate format (§8.2). The plan's first reading of compute was "the
format each arithmetic result is rounded to, at every operation". It was not built that way:
elementwise arithmetic below float32 models no device (every chip emulates it, §2.3), several
of its kernels do not exist below float32 (`norm.cdf`, `norm.ppf`, a root solve, §8.1), and an
FP8 exponential of a level-sized number is noise. So compute is the format of the products,
where native units pay, and accumulate is the format of the rest, as a native low-precision
kernel runs it; the kernels are in difference form so that what a product multiplies is a
path's deviation, never a level ([§8.2](#82-the-difference-form)). Recorded in A-16's row of
decisions.md, the reading confirmed by the owner on 2026-10-11.

### 3.2 Stages: adjustable and fixed

| Stage | Adjustable | Per product and trade | Why |
|---|---|---|---|
| Simulation (shocks, states) | Yes | No | One set of paths serves every trade |
| Scenario market (curves, fixings, FX) | Yes | Storage per run; each trade reads it at its own compute | Shared input |
| Pricing on paths (every pricer) | Yes | **Yes** | Trades are priced one at a time |
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
compute precision) and loaded at float64 for VaR/ES. Settled in the precision mechanism (2026-10-01).

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
    rounding_seed: int = 0                          # the stochastic rounding's seed (2026-10-02)
    offset: str = "block"                           # block | none: a scaled format's block layout (2026-10-09)
    paired_fraction: float = 0.0                    # share of paths re-run at float64 (§9)
```

`rounding` and `rounding_seed` were added in sub-32-bit storage (2026-10-02) (§6.2), `paired_fraction` in the precision report (2026-10-02)
(§9.6), `offset` with the storage harness (2026-10-09, §6.2).

`Precision()` is float64 everywhere. It replaces `PrecisionConfig`,
`PricingPrecisionOverride`, `RiskPrecisionOverride` and `MarketRiskRequest.precision`. The old
shapes are refused with an error naming the replacement, in Python and over HTTP (a 422), as
the shared pipeline did with the Hull-White request (A-12). `RunConfig.precision` and
`MarketRiskRequest.precision` take the same type; a market-risk run reads `simulation` for
its shifts (their storage, and the accumulate format they are rounded to: the scenarios are
drawn before the run, so its simulation stage has no product, and its compute format is not
read, [I-95](../known-issues.md#i-95)) and `pricing` for its revaluation.

`StagePrecision.arithmetic` is the stage's compute and accumulate formats as its kernels take
them (`engine.precision.Arithmetic`, frozen and hashable, a static argument: one compile per
pair of formats), since the kernel rewrite (2026-10-10).

### 4.2 Resolution

`precision_for(trade)` returns `by_trade[trade_id]`, else `by_product[product]`, else
`pricing` (A-15). It is the only lookup; a misspelt trade id or product is refused, not
ignored.

Built with per-trade precision (2026-10-02):

- `Precision.precision_for(trade)` is a method of the policy and reads only `trade.trade_id`
  and `trade.product`, so `engine.precision` still imports nothing from the pipeline.
- A product is a class constant of each trade config (`SwapConfig.product == "swap"`,
  `"european_swaption"`, `"bermudan_swaption"`, `"american_swaption"`, `BondConfig.product ==
  "bond"`), the same names as the HTTP `trade_type`s (a test holds them equal).
  `engine.pricing.cube.PRODUCTS` collects them from the `Trade` union, so a new trade
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
- `accumulate` is an array format (not `tensorfloat32`, `bfloat16_3x` or a single pass
  `<name>_1x`), float32 or float64
  (no matrix unit accumulates narrower), at least as wide as `compute`'s values, and float32
  under a compute format below float32, whose matrix units accumulate in float32
  (`engine.precision.arithmetic.arithmetic_refusal`, shared by `Arithmetic`);
- `storage` is an array format no wider than `accumulate` (storing wider than the stage
  computes gains nothing; until the kernel rewrite, no wider than `compute`, which accumulated
  in itself);
- a compute format the table marks pending is refused naming its item (none since the kernel
  rewrite, 2026-10-10; FP4 will be until F-07's FP4 part);
- every `by_trade` key is a trade in the request, every `by_product` key a product;
- `rounding` is `nearest` or `stochastic`, and `stochastic` only when some stage or override
  stores in a scaled format (otherwise it would be accepted and ignored); `rounding_seed` is a
  non-negative integer (2026-10-02);
- `offset` is `block` or `none`, and `none` only when some stage or override stores in a scaled
  format, for the same reason (2026-10-09);
- `paired_fraction` is a number in [0, 1] (2026-10-02). At float64 everywhere it is accepted: it
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

fp8 = StagePrecision(storage="float8_e4m3fn", compute="float8_e4m3fn", accumulate="float32")
Precision(simulation=fp8, market=fp8, pricing=fp8,
          by_product={"bermudan_swaption": StagePrecision("float32", "float32", "float32")},
          paired_fraction=0.02)                   # FP8 products and storage (2026-10-10), Bermudans in float32

Precision.throughout("tensorfloat32")             # TensorFloat-32 products, float32 everywhere else

Precision.throughout("bfloat16_1x")               # bfloat16 products in one pass, the shared operands
                                                  # rounded once (biased, A-23): for measuring it
```

## 5. Modules

```
engine/precision/
  formats.py   # the format table: name -> dtype, bits, mantissa bits, max, scaled?   (2026-10-01);
               # array or product format, the kind of product (2026-10-10); the shared operand's
               # terms, the base, each format's single pass <name>_1x (2026-10-11, A-23)
  arithmetic.py  # Arithmetic: a stage's compute and accumulate formats, as kernels take them (2026-10-10)
  policy.py    # StagePrecision, Precision, validation (2026-10-01); by_product, by_trade, precision_for (2026-10-02)
  storage.py   # store(), load() (2026-10-01); Stored (pytree: values, scales, format, axis), rounding (2026-10-02);
               # block offsets (Stored.offsets, 2026-10-09)
  grid.py      # exact rounding onto a format's grid, power-of-two scales: storage's and the products' (2026-10-10)
  report.py    # PrecisionReport: the policy as run, realized dtypes, device, paired errors (2026-10-02);
               # each compute format's execution, native or emulated (2026-10-10)
  estimate.py  # two-level estimator for means; paired differences for quantiles (2026-10-02)
  products.py  # matmul(): every matrix product, at its compute format's precision (2026-10-06); in every
               # compute format, accumulated in the accumulate format; product_execution (2026-10-10);
               # the shared operand named and carried in its terms (2026-10-11, A-23)
```

The precision mechanism (2026-10-01) built the first three at float64/float32; sub-32-bit storage (2026-10-02) added `Stored` and the scaled
formats; the precision report (2026-10-02) added `estimate.py` and `report.py` (§9.6). At float64 and float32 a stored
value is still the array itself (§6.2): a wrapper would carry no scales and only change every
consumer's types, and it keeps the default bit for bit.

The pipeline imports these; they import nothing from it. Each module has its own unit tests
that need no market, trade or ORE ([§13.4](#134-storage-properties)).

## 6. Mechanism

### 6.1 The format table

| Name | Bits | Mantissa bits | Max | Array format | Scaled when stored | Product | Shared terms | Storage enabled | Compute enabled |
|---|---|---|---|---|---|---|---|---|---|
| `float64` | 64 | 52 | 1.8e308 | Yes | No | its dtype | 1 | Yes | Yes |
| `float32` | 32 | 23 | 3.4e38 | Yes | No | its dtype | 1 | Yes | Yes |
| `float16` | 16 | 10 | 65,504 | Yes | Yes | its dtype | 2 | Yes | Yes (2026-10-10) |
| `bfloat16` | 16 | 7 | 3.4e38 | Yes | Yes | its dtype | 2 | Yes | Yes (2026-10-10) |
| `float8_e4m3fn` | 8 | 3 | 448 | Yes | Yes | its dtype | 2 | Yes | Yes (2026-10-10) |
| `float8_e5m2` | 8 | 2 | 57,344 | Yes | Yes | its dtype | 2 | Yes | Yes (2026-10-10) |
| `tensorfloat32` | 32 | 10 | 3.4e38 | No | No | TensorFloat-32 | 2 | No | Yes (2026-10-10) |
| `bfloat16_3x` | 32 | 16 | 3.4e38 | No | No | three bfloat16 passes | 1 | No | Yes (2026-10-10) |
| `<name>_1x` | as `<name>` | as `<name>` | as `<name>` | No | No | as `<name>` | 1 | No | Yes (2026-10-11) |

An **array format** has a dtype of its own, so it stores and accumulates. A **product format**
(2026-10-10) is a way of running a matrix product, a compute format only: the TensorFloat-32 of
an NVIDIA GPU (float32 operands rounded to a 10-bit mantissa) and the three bfloat16 passes of
a TPU (each operand split into a bfloat16 and the bfloat16 of its remainder, the remainders'
product dropped: about 16 bits), whose values are float32 arrays. These are the product
precisions XLA picks by default for a float32 product on those devices, made choices of the
policy (A-22). **Shared terms** (2026-10-11, A-23, §8.3): every format below float32 whose
product is one pass carries the operand every path shares in two terms of the format; its
single pass is a product format of its own, `<name>_1x` (`float16_1x`, `bfloat16_1x`,
`float8_e4m3fn_1x`, `float8_e5m2_1x`, `tensorfloat32_1x`), the row of `<name>` under another
name, whose values are `<name>`'s (`dtype_of`) but which is neither stored nor accumulated in.
`bfloat16_3x` carries each operand in two bfloat16 terms already. Each row names its **base**,
the row whose grid, dtype and units its products use (itself, or the format a single pass is
of). FP4 (`float4_e2m1fn`) is added at FP4 (F-07). A name outside the table is refused.

### 6.2 Storage: `store` and `load`

`store(x, format, rounding, key) -> Stored` and `load(stored, dtype) -> array`.

- **float64 and float32:** a plain `astype`, returning the array (the precision mechanism (2026-10-01)). Storing or
  loading at an array's own dtype returns it unchanged, so the float64 default moves no bit.
- **Every sub-32-bit format is stored with block scales** (A-10's single rule, rather than
  deciding per class which formats need range help). Blocks are 32 consecutive entries along
  the **scenario axis**, each with a power-of-two scale, kept as float32, that brings the
  block's largest magnitude to the format's maximum. Power-of-two scales are exact (the idea
  behind the OCP MX formats), so scaling adds no rounding of its own. The scenario axis is
  the axis multi-host (I-61) shards, so blocks fall inside shards. Overhead: 4 bytes per 32 values.
- **Rounding:** `nearest`, or `stochastic` (up or down at random, in proportion to the
  distance, so each value's error averages to zero). Stochastic draws come from the run's
  seed, so runs reproduce.
- A scenario count that is not a multiple of 32 gets a short last block; zeros and NaN pass
  through unchanged.

As built (sub-32-bit storage (2026-10-02), `engine/precision/storage.py`):

- `store(x, format, rounding, key, axis)` returns the array itself at float64 and float32
  (rounded to nearest whatever `rounding` says, so the precision mechanism's behaviour is unchanged) and a `Stored`
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
  run as alone (an exit criterion of per-trade precision, kept under stochastic rounding). `Precision.store(x,
  storage, stream, axis)` is the one place the pipeline stores; the names are `"shocks"`,
  `"states"`, `"market/<i>"` (the market's path arrays in `ScenarioMarket.map_arrays` order)
  and `"values/<trade id>"` (cube columns and market-risk P&L).
- **No recompiles.** The quantizer and the loader are module-level `jit` programs with the
  format, axis and dtype static: one compile per array shape, format and rounding, reused by
  every trade, date and run (a test counts the compiled programs).
- Overhead: FP8 holds an array in an eighth of float64 plus 4 bytes per 32 values.

**Block offsets** (2026-10-09, I-75; `Precision.offset`, `"block"` by default). A block whose
entries sit close together against their level spends the format's few mantissa bits on the
level: a bond's cube column (near 1e6, spread near 1e4) keeps about one bit of its spread in
FP8, and rounded to nearest every path of a block moves the same way, a bias (§15.3). With
offsets, a concentrated block, one whose finite entries all have one sign and whose farthest
from zero is at most three times its nearest, is stored as their deviations from its midpoint
(the mean of its largest and smallest finite entries), rounded to float32 and kept beside the
scale (`Stored.offsets`, the shape of `scales`); the scale is then set by the deviations, at
most half the block's range. Every other block has offset 0. The midpoint is rounded on
float32's grid in the input's dtype, not by a cast alone, so that the deviations are taken from
the offset kept: XLA on a GPU removed the float64 -> float32 -> float64 round trip of the cast
inside a fusion, and a one-entry block then stored its deviation from the unrounded midpoint,
missing its value by float32's rounding of it (2026-10-11, I-99; red on an RTX 5060 in
`tests/test_precision.py::TestScaledStorage::test_any_scenario_count_and_axis_short_last_block`).

- **Which offset.** Three were weighed. The level the plan named first, the cube's t=0 value
  and the curves' path-independent part (§8.2), needs each class's own level in the pipeline,
  and a t=0 value does not follow the level's drift over dates; an offset per block and date
  follows it, needs nothing from the pipeline, and serves every class through one mechanism
  (§2.4). Of the per-block offsets, three were tried in turn. The midpoint of every one-signed
  block let a small entry of a positive block (a deep out-of-the-money option on one path) load
  negative: on the first trial runs an option's ENE, exactly 0 in float64, came out positive.
  The entry nearest zero keeps every sign, but its deviations lie in [0, range], skewed, so
  nearest rounding no longer cancels: on the first record run of the `linear` campaign the
  bond's FP8 column kept a bias of 18 to 36 (e4m3) and 75 to 120 (e5m2) on values near 1e6,
  4.7 to 28 of its standard errors at 16 seeds, which failed threshold 1 of §15.4 (a thousandth
  of the bias without offsets, but detectable; the run is summarized there). The midpoint of a
  concentrated block only does both: its deviations are symmetric, and a rounding error, at most
  a spacing of the format at the largest deviation (an eighth of the range in e5m2, the
  coarsest), is below a concentrated block's nearest magnitude, at least half its range, so no
  entry changes sign. A one-signed block that reaches nearer zero than half its range would
  gain less than a bit from any offset (its largest magnitude is less than 1.5 times its
  range), and keeps offset 0.
- **Never worse.** Half a block's range is no larger than its largest magnitude, so a
  concentrated block's scale, and the format's step, is never coarser than without an offset.
  Every other block keeps offset 0 and stores exactly what it stores without one, bit for bit,
  so zeros (signed) pass through, a zero shift's P&L stays exactly zero, and shocks, states and
  P&L, which straddle zero, are stored alike in both layouts.
- **Exactness.** The deviations are computed from the stored offset in the input's dtype (exact
  where an entry is within a factor two of it), and loading adds the offset back with one
  rounding in the loaded dtype, where an offset is non-zero.
- **Cost.** 4 more bytes per 32 values: FP8 holds an array in an eighth of float64 plus 8 bytes
  per 32. `offset="none"` keeps the OCP MX layout (scales only), which FP8 matrix units read
  directly; a stored array reports its layout (`Stored.offset`, read from the arrays).
- **Bit for bit.** With `offset="none"` scaled storage is the code before it bit for bit: a
  snapshot of every scaled format, both roundings, from float64 and float32, and a portfolio and
  a market-risk run with every stage stored scaled, 168 arrays, all identical (§13.1).

### 6.3 Cast points

The only places that read the configuration:

| # | Where | What it does |
|---|---|---|
| 1 | `engine.market_simulation.config.simulate` | Normals generated in the simulation's accumulate format (`engine/market_simulation/sobol.py`, the clip epsilon from it) and bridged by one product in `simulation.arithmetic`, then `store(shocks)` |
| 2 | `simulate` | `load(shocks)` at the accumulate format; `evolve_states` in `simulation.arithmetic` (§8.2); `store(states)` |
| 3 | `simulate` | `load(states)` at the market's accumulate format; `build_scenario_market` in `market.arithmetic` (the level from float64, one product for the deviations, §8.2); every path array of the market stored at `market.storage` (`ScenarioMarket.map_arrays`); the tenor grid and the curves' levels, which have no scenario axis, kept in float64, as every level is (§8.2) |
| 4 | `engine.pricing.cube.value_portfolio`; `engine.risk.market.run_market_risk` | Per trade, at `stage = precision_for(trade)` (2026-10-02): `load(market)` at the stage's accumulate format (once per format; path fixings once per arithmetic), price the trade in `stage.arithmetic`, `store` its cube column at `stage.storage`. Market risk: shifts rounded to the simulation's accumulate format and stored once; per trade loaded at its stage's accumulate format, revalued in `stage.arithmetic`, the P&L stored at `stage.storage` |
| 5 | `engine.run.pipeline` (exposure); `run_market_risk` (VaR/ES) | `load(values, float64)` and the numeraire at float64, then reduce; exposure loads one trade column at a time, retaining a float64 netting sum and sorting it for exact PFE (I-105, §17) |

Before the kernel rewrite (2026-10-10) the stages loaded at, and computed in, their compute
format, which equalled their accumulate format; since then each stage loads at its accumulate
format and runs its products in its compute format (`StagePrecision.arithmetic`).

Every store goes through `Precision.store`, which adds the policy's rounding (§6.2). The
shocks `[T, S, d]` have their scenario axis at 1, every other class at 0.

The tenor grid (`ScenarioCurves.tenor_times`, `[D, K+1]`) is the curves' coordinates, not
scenario data: block scales need a scenario axis, and an FP8 grid would move every pillar.
Until sub-32-bit storage it was stored at `market.storage` like the curves, so a float32 market storage under
float64 compute rounded it to float32; since sub-32-bit storage it stays at `market.compute`, which changes
that one combination's numbers at float32 rounding level (no default and no snapshot run). Since
the kernel rewrite (2026-10-10) the grid and the curves' levels (`ScenarioCurves.level`) are
float64 in every run, as every level is (§8.2); the precision policy neither stores nor loads
them, and a kernel reads a path's deviation from a curve's level at the deviation's format
(`log_discount_deviation`, which casts the times to it).

The first three live in `simulate`, the orchestrator of the simulation and market stages;
the kernels it calls (`generate_sobol_normals`, `apply_brownian_bridge`, `evolve_states`,
`build_scenario_market`) follow the dtype they are given and take the stage's arithmetic for
their products. `generate_sobol_normals` keeps its `dtype` argument: it is where the shocks are
created, so it has no input to follow.

The cube is a sequence of per-trade columns, each in its own storage format
(`PortfolioValuation.columns`, per-trade precision (2026-10-02); `Stored` columns from sub-32-bit storage).
`PortfolioValuation.cube` sets them side by side in their shared format when it is float64 or
float32, as before, or loaded at float64 when the formats differ or are scaled (exact: float64
holds every format's values times their power-of-two scales).
`PortfolioResult.npv_cube` stays an array, loaded at float64 when requested (the default).
`PortfolioRequest.include_npv_cube=False` (HTTP `outputs.npv_cube: "none"`) leaves it empty without
changing exposure or precision figures; reductions read stored columns directly (§17). A
trade priced at its precision inside a mixed run gives exactly the column it gives priced
alone with that `pricing` stage: trades share only the loaded scenario market and fixings,
which depend on the accumulate format and the arithmetic alone.

### 6.4 Inputs follow dtype

Inside a stage, every array takes the dtype of the arrays it is computed from. About 46
`jnp.asarray/zeros/...` calls in `engine/pricing`, `engine/calibration`, `engine/models`
and `engine/risk` name no dtype today and so become float64 under the x64 flag; each becomes
"the input's dtype". No kernel signature gains a dtype argument. Since the kernel rewrite
(2026-10-10) a kernel with matrix products takes the stage's `Arithmetic` as one static
argument, whose accumulate format is its inputs' dtype: the compute format of a product cannot
be read from an array (TensorFloat-32's operands are float32 arrays), and a process-wide or
traced-context setting would be a cache key JAX does not see (and the report could not vouch
for it, A-22).

Strict promotion (`jax_numpy_dtype_promotion="strict"`) in CI turns any accidental
float32/float64 mix into a failing test ([§13.3](#133-dtype-discipline)).

The precision mechanism (2026-10-01) did it by running float32 portfolios under strict promotion and fixing each mix where
it arose, at the kernel's entry: `legs_cube`, `european_cube` and `jamshidian_cube` cast their
coupon tables and variances to the curves' dtype (`Legs.astype`, `EuropeanTerms.astype`); the
per-path recalibration casts its basket (`BasketInstrument.astype`), volatilities and bracket
to the curves' dtype, keeps its bucket widths as Python floats (a NumPy float64 scalar is not
weakly typed) and its root bracket in that dtype; the path volatility (`Sigma.astype`) follows
the path curves. `engine.market_data.curves.curve_dtype` is the one helper that reads a curve's
dtype. Each cast is a no-op at float64. Before the precision mechanism the "float32" path was in fact mixed: the
float64 constants promoted the float32 curves, so pricing on paths ran mostly in float64 and
the cube came out float64, while path fixings ran in float32.

### 6.5 Constants that depend on dtype

Found while designing; fixed or derived from the dtype in the precision mechanism (2026-10-01):

- `hi * (1.0 - 1e-9)` (in the standalone bootstrap, `engine/calibration/lgm.py` until 2026-10-18)
  rounds to `hi` in float32. The same bisection ran per path in the Bermudan recalibration.
- `jnp.finfo(dtype).eps` in the Sobol clip (`engine/market_simulation/sobol.py:42`) must use the
  compute dtype once storage is narrower.
- `scenario_batch_size(itemsize)` (`engine/risk/greeks/scenario_pnl.py`) must use the compute
  itemsize.

The precision mechanism (2026-10-01) also searches every path kernel for literal tolerances (`1e-`) and either derives
each from the dtype or shows it holds at float32.

Outcome (the precision mechanism (2026-10-01)):

- The ceiling: `hi * (1 - ceiling_tolerance(dtype))`, with `ceiling_tolerance = max(1e-9,
  4 eps)` (`engine.calibration.lgm`), unchanged at float64. Red first: with the fixed 1e-9 a
  float32 bisection stuck below the top was not flagged
  (`tests/test_precision.py::TestRecalibrationInFloat32`). Since 2026-10-12 the flag is part of
  ORE's acceptance on every bootstrap ([I-73](../known-issues.md#i-73)), and every calibration is
  float64: the per-path recalibration, the bootstrap's one float32 use, went with A-24.
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

### 6.6 What went (2026-10-01)

- `check_run`'s refusal of `pricing`, `risk` and `calibration` below 64.
- `run_market_risk`'s `jax.config.update("jax_enable_x64", True)`. x64 stays on, set once
  when `engine` is imported, as since 2026-10-01.
- `_PRICING_LOCK` (`engine/run/request.py`), after the thread-safety audit of the precision mechanism (2026-10-01)
  (no module-level caches and no ORE globals were found; a concurrency test confirms it).
- The per-precision pool tiers (`engine/portfolio/worker_pool.py`, as it then was): one pool from the precision mechanism (2026-10-01);
  the pool itself went with the engine worker (2026-10-04).

Removing the lock rests on the audit: the pipeline keeps no module-level caches or mutable
state, and never reads ORE's global evaluation date (trades carry theirs, I-64; only the test
oracle sets it). `tests/test_portfolio_entrypoint.py::TestPricePortfolioConcurrency` runs a
float64 and a float32 request on two threads at once, repeatedly, each bit for bit equal to
its sequential run.

### 6.7 Matrix products (2026-10-06)

A matrix product's precision is part of the compute format, so the policy states it, never
the device (decision A-22). `engine.precision.matmul(a, b)` (the semantics of `a @ b`) asks
`product_precision(dtype)` for the operands' format and passes it to `jnp.matmul`; the
precision is then in the program itself (`dot_general`'s `precision`), and AD's transposes
of the product carry it too. float64 and float32 map to `lax.Precision.HIGHEST`; since the
kernel rewrite (2026-10-10) every compute format of the table does, its operands rounded first,
but for a product format run natively, which states its algorithm (`TF32_TF32_F32`,
`BF16_BF16_F32_X3`), and `matmul(a, b, arithmetic, shared)` takes the stage's compute and
accumulate formats (§8.3) and, since 2026-10-11, which operand every path shares (`"left"` or
`"right"`, A-23), refusing a product in a stage's arithmetic that does not say. Without an arithmetic it multiplies float64 or float32 operands in their own
format, as the stages fixed at float64 (A-10) and the kernels not yet rewritten do; operands
below float32 without one are refused, since their format does not say what they accumulate
in.

Every matrix product of the engine's JAX code goes through it; at the stated product precision (2026-10-06) there were nine:
the simulation step (`evolve_states`, 2), the Brownian bridge (1, as one `[T, T] x [T, S*d]`
product), the Bermudan rollback's bookkeeping (4) and the calibration's spread correction
(`_corrections`, 2). The kernel rewrite (2026-10-10) made the simulation step one product and
added the scenario market's (1), the legs' flow sums (one per block of columns: the fixed flows,
the coupons, the coupons' basis points when asked for) and the Jamshidian engine's option sum
(1) (§8.2). The rollback's rewrite (2026-10-13) replaced the four with the induction's own per
step (§8.2): the rollback by its operator and the cashflows' values by their functions on the
grid, each on the level (float64) and on the paths' deviations (the stage's arithmetic). NumPy
products on the host are exact in their dtype and are left alone.
JAX's process-wide `jax_default_matmul_precision`, which the GPU run set on import, is neither set
nor read: anyone could override it, and the report would then be false.
`tests/test_accelerator_defaults.py::TestEveryMatrixProductStatesItsPrecision` records every
`dot_general` JAX binds while the pipelines run from cleared caches (the simulation under
both models, the scenario market, every product's path pricing with the per-path
recalibration, today's values, AD Greeks, market risk; float64 and float32) and fails on any
without its format's precision or outside the engine's code, naming the line. The coding
rule is in [coding-style.md](../../concepts/coding-style.md#core-constraints).

On a CPU the precision does nothing (XLA's CPU backend computes a float32 product in float32
whatever it says), so the stated product precision (2026-10-06) moved no bit (§13.1); on a GPU it is what the GPU run's process
default did, product by product.

## 7. Low-precision storage

Storage below 32 bits needs no kernel changes: values are loaded to the stage's accumulate
format, float32 or float64 (its compute format before the kernel rewrite, which equalled it).
It is the first low-precision capability
(sub-32-bit storage, done 2026-10-02), and the first measurement campaign (the storage harness, I-75) runs on it
alongside the kernel work, because it never changes float64 numbers.

What is known (measured 2026-10-01, [§15](#15-measurements)):

- **FP8 is viable for shocks.** Round-to-nearest storage of 4M standard normals biases a call
  payoff by about one Monte Carlo standard error at 4M paths; stochastic rounding by about
  0.15. The earlier verdict (useful only to ~1,400 paths) compared error per value with Monte
  Carlo error, which is the wrong criterion ([§15.2](#152-bias-not-error-per-value)).
- **FP4 is not viable stored naively,** with either rounding (25 and 45 standard errors).
  Stochastic rounding keeps each value's mean but inflates the variance by 4.7%, which
  inflates volatility. A variance correction through the block scales is the candidate fix
  (FP4 (F-07)).
- **Through the pipeline** (2026-10-02, [§15.3](#153-storage-through-the-pipeline)): one stage at a
  time on the shared portfolio, float16 is within 1e-5 of notional in bias and 7e-6 in EPE;
  FP8 moves EPE by 0.04% (shocks), 2.6% (curves, e4m3) and 0.6% (cube). Two mechanisms limit
  it, both recorded as [I-75](../known-issues.md#i-75): a column whose paths sit close
  together against their level (a bond's cube, a long log discount factor) keeps only the
  format's few bits of its spread, and rounded to nearest every path of a block moves the same
  way, a bias; stochastic rounding turns that into noise (the cube's FP8 bias 3 to 4 times
  smaller at 256 paths). Block offsets (§6.2, 2026-10-09) are the remedy, and the storage
  harness measures every format per stage, figure and path count (§15.4).

## 8. Compute below float32

Built by the kernel rewrite (F-07) for every path kernel: on 2026-10-10 for all but the
Bermudan/American rollback, and on 2026-10-13 for the rollback, once its model was ORE's on the
paths (2026-10-12: `ShiftHorizon`, I-32; the calibration kept as ORE built the trade, I-49), so
that no kernel was rewritten twice. Until then a Bermudan's or American's pricing stage computed
and accumulated in one format, float32 or float64, and was refused otherwise by name; since,
every product's pricing stage takes any compute format.

### 8.1 What blocks it

JAX already runs `+`, `*`, `exp`, `log`, `max`, `sum` and `cumsum` in bfloat16, float16 and
both FP8 formats; on CPU, XLA computes each operation wider and rounds the result to the low
format, so the numerics are those of low-precision arithmetic. Only `cholesky` and
`norm.ppf` have no kernel. The obstacles are numerical:

| Obstacle | Example | Remedy (as built, 2026-10-10) |
|---|---|---|
| **Range** | FP8 e4m3 tops at 448, float16 at 65,504; NPVs reach 1e6 and more | A product's operands are scaled into the format's range by a power of two per row (a path) and per column (§8.3); the arrays around the products are in the accumulate format |
| **Accumulation** | An FP8 `cumsum` of 10,000 × 0.1 returns NaN (it passes 448); bfloat16 returns 1008 | The `accumulate` format: every product accumulates in float32 (or float64 under float32 compute), every sum and carry is in it |
| **Cancellation** | A swap NPV is two legs of ~1e8 that nearly cancel; FP8 keeps about one significant digit | Difference form (§8.2): the level, which no path depends on, in float64, added to each path in two terms of the accumulate format; a product multiplies only each path's deviation from it |
| **Missing or unstable kernels** | `norm.ppf`, `cholesky`; the inverse normal CDF overflows in FP8 | Everything but the products runs at the accumulate format, float32 or wider (§3.1): the inverse normal CDF, the Bachelier and Black formulas, the Jamshidian root. Cholesky runs once on the host |

### 8.2 The difference form

Each kernel splits a value into a level that does not depend on the path, computed in
float64, and the path's deviation from it, small and well conditioned, which is all a product
multiplies. Any level gives the same values, up to rounding (a test values the legs at the
model's level and at none); a near one keeps the deviations small.

**A level meets a path in two terms** (`engine.precision.add_level`, `deviation_from`): the
level in float64 (it has no path axis, so this is cheap even on a TPU, which emulates float64),
then `hi + (lo + deviation)` in the accumulate format, `hi` the level on the format's grid and
`lo` what it leaves (rounded on the grid in float64, not through a round trip XLA may remove).
Each path's value is then rounded once, by itself. Rounding the level once to the accumulate
format and adding it to every path, as the first build of the rewrite did, shares that rounding
with every path: a bias more paths do not remove, measured at 0.04 on a float32 bond's mean
near 1e6 (significance 287 over 8 seeds, ten times the code before the rewrite); with two terms
it is at most 3e-4, within 2 of its standard errors, below the code before (0.003). At float64
the low term is absent and nothing changes. The curves' levels and tenor times
(`ScenarioCurves`), the mean path, the legs' coefficients and their path fixings' levels are
float64 in every run for the same reason.

As built (2026-10-10), family by family:

- **The Brownian bridge** (`engine.market_simulation.sobol`): the bridge, the differencing of
  its path values into increments and their standardization by sqrt(dt) are one `[T, T]` matrix
  built on the host in float64 (`bridge_increments_matrix`), applied by one product. Differencing
  on the device subtracted path values that grow as sqrt(t) to get increments that do not.
- **The simulation scan** (`engine.market_simulation.paths`, `engine.models.cam.path_moments`):
  a state is its mean path (float64, host) plus a deviation in units of a power of two near its
  standard deviation; the deviation's step `y <- [y, z] @ [M~, L~]'` is one product, its carry
  the accumulator; the drift and the mean cancel; the mean is added in two terms. A log equity
  spot near 4.6 deviates by 0.06 to 0.5: the same scan on the states themselves errs on the
  level, several times more (a test holds it). Every operand is near 1, and the scales are exact powers of two.
- **The scenario market** (`engine.market_simulation.scenario_market`): every path array, each
  curve's log discount factors, the log numeraire, log FX and equity spots, is linear in the
  state, so the market is a level (the model's at the mean state, float64) plus one product of
  each path's standardized state deviation with the market's loadings on the state (`_Layout`,
  `[D, d, C]`: one column per curve tenor, the numeraire, each spot), then the discount floor and
  the exponentials in the accumulate format. A multi-factor model's curves are the same product
  with more rows. The curves keep their level (`ScenarioCurves.level`).
- **The legs** (`engine.pricing.legs.leg_values`): the level values every flow off the level
  curves with ORE's arithmetic (it is the t=0 value when there are no deviations, so today's
  price is the same code); a path's deviation is `[e, h] @ C`, with `e = expm1(dL)` at every
  fixed payment, `dL` the curve's node deviations carried to the time by the curve's own
  two-node interpolation (a curve's log discount factor is linear in its node values for both
  curve types, `log_discount_deviation`: a stencil in the accumulate format, not a product, a
  dense interpolation product having cost the float64 kernel twice its time for 14 nodes of
  which each time reads 2), `h = (F0 + m) e + dF (1 + e)` per coupon, `dF` its rate's deviation
  (the forecast ratio's, or a path fixing's), and `C` the level's present value of each fixed
  flow and of a unit rate on each coupon, since `(F + m) P - (F0 + m) P0 = P0 [(F0 + m) e + dF
  (1 + e)]`. Every term is a sensitivity times a deviation, so the legs' cancellation stays in
  the level. The outputs are a payer swap's value, the annuity and the floating basis points, as
  many as the engine reads; each block of columns is its own product (§13.9). FixingManager's
  path fixings are a level and deviations alike (`PathFixings`).
- **The Europeans** (`engine.pricing.european`, `engine.pricing.jamshidian`): Bachelier reads the
  exercise-into swap's value V, annuity A and floating basis points from the legs; the forward
  less the effective strike is `V / A`, a swap's value whose legs cancel in the level; the formula
  and the cash annuity at the accumulate format. Jamshidian reads its discount factors as the
  level times `1 + e`; x* and the bond options at the accumulate format, and their sum against
  the coupons' amounts a product in the compute format.
- **Market risk** (`engine.risk.market.revaluation`): a scenario's P&L is the deviation of a
  trade's value under its shift of the pillar rates (a zero curve's log discount factor is linear
  in its rates too), so the legs' P&L is the product alone and a scenario that moves nothing has
  exactly zero P&L by construction; a European's P&L is its formula on the shifted values less on
  the level, and exactly 0 where nothing moved, not left to two evaluations of one value agreeing
  (XLA may round them apart). A Bermudan's or American's is the induction's deviation (below),
  exactly 0 where nothing moved, every deviation then being 0 (revalued in full, in one format,
  until 2026-10-13). One calibrated to the market is recalibrated under each scenario (I-112,
  2026-10-17), so its model, the state grid and the rollback operator are the scenario's own:
  the calibration in float64 (A-10), a first pass; then per scenario its own induction in
  difference form, the level the scenario's model on the base curves (in the curves' format, the
  accumulate format, as market risk's levels are), the deviation the curves' move in the compute
  format, `P&L = (level − base) + deviation` added in two terms (`add_level`), exactly 0 where
  nothing moved (`engine.risk.greeks.price_functions.recalibrated_pnl`). The level is no longer
  shared by every scenario, so its operator is per scenario too (`working_bytes_per_model`
  sizes the batch); the evidence rows of the `market_risk` campaigns were measured on a fixed
  LGM and do not cover this kernel: the evidence keys it apart (`model=recalibrated`, I-114,
  2026-10-18) and the `market_risk_calibrated` campaign measures it (§10.4).
- **The Bermudan/American induction** (`engine.pricing.lgm_grid`, 2026-10-13, I-83): ORE's grid
  engine in the numeraire's units, where neither the state grid nor the rollback depends on the
  curves. Each step's rollback is one `[nodes, nodes]` operator (`_rollback_operator`, ORE's
  interpolated Gaussian convolution as a matrix, built by a scatter of its weights), shared by
  every path and every column (the option, the underlying, the coupons cached in progress), one
  product `[S, columns, nodes] x [nodes, nodes]`. A cashflow's deflated value at a node is a
  coefficient that depends on the path (its discount factor at the payment, `P(0, T)`; an Ibor
  coupon's also its forecast's ratio `P_idx(0, T1) / P_idx(0, T2)`) times a function of the state
  that is the model's alone (ORE's `reducedDiscountBond`, `exp(-H_T^2 zeta/2 - H_T x)`; an Ibor
  coupon two such functions, `_basis`), so a row's cashflow actions (ORE's bookkeeping, replayed
  on the host as weights, `_Induction`) are one product of the paths' coefficients `[S, J]` with
  the functions weighted into the row's targets `[J, targets x nodes]`. The induction runs on the
  level curves (no path axis, float64 on the paths) and, given each path's `Deviations`, on each
  path's deviation, the rollback and the bookkeeping, being linear, on the deviations alone. The
  coefficients' deviations are the legs' (`e = expm1(dL)` at a payment; a coupon's `(F0 + m) e +
  dF (1 + e)` at its known rate's deviation; a forecast's `expm1(dL_pay + dL_idx(T1) -
  dL_idx(T2))`). At an exercise, with `c` the level's exercise value less its continuation, a
  path's deviation is `max(d_option - max(c, 0), d_exercise + min(c, 0))`, exactly the path's max
  less the level's, the level entering in two terms of the accumulate format. A coupon cached in
  progress (an American's broken period) takes a column only while it is cached, a column reused
  once it has moved (`_cache_slots`: a Bermudan caches none, an American one or two per leg);
  rows that roll nothing back skip their products (`lax.cond`), so a cube's dates, padded to the
  trade's largest shape (`one_shape`), share one program. At float64 each path's value is the
  induction on its own curves to rounding (a test holds it to 1e-12; ORE's engine to 1e-11 as
  before), and float32 is 2 to 6 times closer to float64 than when the rollback ran in float32
  throughout, 5 to 20 times less biased (the level is float64).
- **Exposure** (`engine.risk.counterparty.exposure`) is a reduction over paths, float64 by
  decision (A-10): its inputs, the cube columns and the numeraire, are loaded at float64. It was
  already in its final form.

One implementation serves every precision (A-16): at float64 the difference form moves the
numbers at rounding level (§13.1), and every ORE parity suite passes at its tolerance. Below
float32 each product's path-independent operand (the bridge's matrix, the step moments, the
loadings, the flows' present values) would be one rounding shared by every path, a bias: since
2026-10-11 it is carried in two terms of the format (A-23, §8.3), and the single pass that rounds
it once is a format of its own (§6.1, §15.5).

### 8.3 Emulation and native speed

- **Accuracy** is studied on CPU by emulation: the numbers are those of the low format.
- **Speed** needs native units, and they exist only for matrix products: FP8 on Ironwood and
  H100, FP4 on TPU 8t/8i and Blackwell. Elementwise FP8 is emulated on every chip.
- So the wall-clock half of the research question depends on expressing the heavy kernels as
  matrix products where they can be: leg pricing (amounts per flow × discount factors per
  path and flow), the Bermudan rollback (a fixed Gaussian kernel matrix × values), the
  correlation mixing (already a product, but small). The kernel rewrite wrote the simulation,
  the scenario market, the legs and the Europeans this way (2026-10-10) and the rollback
  (2026-10-13, §8.2), each in the same rewrite as its difference form, so no kernel is rewritten
  twice; the research result (F-07) times them.
- **A product's precision is the policy's, not the device's.** XLA's default runs a float32
  matrix product in TensorFloat-32 on an NVIDIA GPU (a 10-bit mantissa) and in bfloat16
  passes on a TPU. Measured on the RTX 5060 (2026-10-06): a float32 simulation's cube was
  1.5% off the float64 one under that default, and within float32 rounding at full
  precision. Since the stated product precision (2026-10-06) (decision A-22) each of the engine's matrix products states its
  precision, taken from its operands' compute format (`engine.precision.matmul`, §6.7), so a
  float32 compute policy is float32 on every device and no process setting can change it
  (the GPU run had set JAX's process-wide default instead). Since the kernel rewrite
  (2026-10-10) TensorFloat-32 (`tensorfloat32`), bfloat16 passes (`bfloat16_3x`) and FP8 products
  are compute formats in their own right, selectable in the policy through the same helper; the
  device's default is never the silent choice.
- **As built (2026-10-10, the shared operand 2026-10-11).** `matmul(a, b, arithmetic, shared)`
  multiplies in the compute format and accumulates in the accumulate format:
  - float64, float32: the operands cast to the format, full precision (`HIGHEST`), accumulated
    in the accumulate format. float32 operands accumulated in float64 are rounded onto float32's
    grid in float64 and multiplied as float64 (every product of two is exact there): asked for
    float32 operands with a float64 result (`preferred_element_type`), an RTX 5060 summed in
    float32 (7.6e-7 off, 2026-10-11; the CPU summed in float64, so only the GPU run showed it);
  - float16, bfloat16, FP8, tensorfloat32: each row of the left operand (a path) and each column
    of the right one scaled by the power of two that brings its largest magnitude to the format's
    top (bfloat16 and TensorFloat-32 have float32's range and are not scaled: scaling two operands
    to the top would overflow the accumulator), rounded to the format's grid exactly
    (`engine.precision.grid`, shared with storage: XLA converts float64 to FP8 through float32,
    rounding twice), cast to it, multiplied into float32 (TensorFloat-32 by the device's
    algorithm, `TF32_TF32_F32`, where native, on operands already on its grid), and the scales
    taken back out. A per-row scale keeps each path's rounding independent of every other path's
    operands, whatever its batch, sample or device (XLA may still sum another shape's product in
    another order, an ulp apart, as at float32); per-tensor scales (cuBLASLt's FP8 default) would
    have tied every path's rounding to the largest path. A product of two values of these formats
    is exact in float32;
  - **the shared operand in two terms** (2026-10-11, A-23, [I-96](../known-issues.md#i-96)):
    every product of a stage's kernels names the operand every path shares (`shared="left"`, the
    bridge's matrix; `"right"`, the step moments, the market's loadings, the legs' and the
    Jamshidian engine's flow values at the level). Rounded once, its error is the same on every
    path: a bias that more paths do not remove (§15.5). The formats of the bullet above carry it
    in two terms of the format (`Format.shared_terms`): its rounding, and the rounding of what
    that leaves, each scaled into the range by its own powers of two (the left terms per row,
    the right per column, as any operand). The terms are stacked along the shared operand's free
    axis, so the product is still one product, twice as wide on that side (`[S, K] x [K, 2N]`,
    `[2T, T] x [T, S]`), and the two halves are summed in float32, the smaller first. The
    shared operand's error is then the second term's rounding, about the square of the format's
    unit roundoff, and what is left is each path's own rounding, which averages out. The single
    pass is `<name>_1x` (§6.1), one term, what one native product does, kept so that its bias and
    its speed are measured. Every term is a value of the format, so a native unit runs the
    product as before, on its own operands;
  - bfloat16_3x: each float32 operand split into its bfloat16 and the bfloat16 of its remainder,
    `lo·hi + hi·lo + hi·hi` (`BF16_BF16_F32_X3` where native), float32 accumulation. The split
    rounds on bfloat16's grid in float32 arithmetic, never through a float32 -> bfloat16 ->
    float32 round trip, which XLA may remove (`xla_allow_excess_precision`, on by default): on a
    GPU it did, the remainder came out 0 and the three passes one (2026-10-10).

  `product_execution(format, device)` says which ran: native where the device has units for the
  format and they were shown to give its numbers (`tests/support/precision_devices.py`, every
  arithmetic a stage accepts against its emulation on four shapes): an NVIDIA GPU's float16 from
  compute capability 7.0, bfloat16, TensorFloat-32 and bfloat16 passes from 8.0, FP8 from 8.9 where
  the process sends its products to cuBLAS; a TPU's bfloat16 and its passes (XLA's own float32
  there; to be checked on the device). Else emulated: a CPU emulates every format below float32, a
  TPU FP8 until Ironwood's is shown right. Emulated and native give the same numbers up to the
  order of the float32 sums: the engine rounds the operands as the unit would, and an emulated FP8
  product multiplies them as float16 values, which hold every FP8 value, with exact products. The
  report names each compute format the run used and its execution (`PrecisionReport.products`).
  Measured on CPU (2026-10-10, 6 × 40 by 40 × 5 operands spanning six decades, against references
  built with ml_dtypes): every format within its unit roundoff plus float32 accumulation
  (`tests/test_compute_precision.py::TestProducts`).

  **On a GPU** ([I-97](../known-issues.md#i-97)). On an RTX 5060 (compute capability 12.0, jax
  0.10.2, 2026-10-10) the check found XLA's FP8 product 70% off for some shapes and precisions, and
  its `BF16_BF16_F32_X3` a single pass's error for some shapes. The faults are XLA's, not the card's
  (located 2026-10-11, reduced to plain JAX in [xla-gpu-reproductions.md](xla-gpu-reproductions.md)):
  its Triton GEMM emitter computes every FP8 product on a matrix shape 0.54 to 0.79 of `|a|·|b|`
  off under the worker's determinism flag (without it XLA's autotuner picks cuBLASLt for most
  shapes, and only the smallest is wrong), while the same products through cuBLASLt (`--xla_gpu_enable_triton_gemm=false`) are exact on
  its FP8 path (`__cublas$lt$matmul$f8`); and it compiles a matrix-vector product as a fusion that
  ignores the bfloat16-passes algorithm (FP8's matrix-vector fusion is exact, off the FP8 units).
  Since 2026-10-21, native execution follows backend/device capability independently of
  the emitter. FP8 retains its scaled operands, E5M2/E4M3 shared-operand layout and FP32
  accumulation. The blanket cuBLAS prerequisite and BF16 vector fallback are removed.
  Normal usage requests native operations; qualification checks their actual arithmetic
  and lowering separately. A compiler failure remains a failure.

  The affected local process explicitly selects `--gpu-gemm cublas` (or the corresponding
  XLA flag) for FP8, and `JAX_RISK_EMULATE_PRODUCTS=bfloat16_3x` for the vector fusion.
  The latter uses explicit FP32 products over the same rounded operands and reports
  emulated dispatch. Neither setting is inferred from a GPU name or imposed on H100.
  Process settings must precede tracing in a fresh interpreter; cached executables do not
  change when environment variables change. Reports retain the overrides and measurements
  with different execution settings cannot merge.

  The 2026-10-19/20 measurements and source snapshots retain the prior dispatch rules as
  historical evidence. The [2026-10-21 routing record](../../../compliance/precision-research/2026-10-21/README.md)
  records this change, default numerical failures and passing explicit workarounds. The
  qualification tool recognizes both cuBLAS FP8 calls and Triton FP8/FP32 dot fusions;
  actual matrix instructions still require trace inspection. H100 remains unmeasured.
- The rollback's matrix form also fixed its memory (2026-10-13, I-83). Each column (the option,
  the underlying, each cached cashflow) was interpolated at `[nodes, quadrature nodes]` points
  per path and step, vmapped over the columns and the paths, although the interpolation weights
  depend only on the grids: measured 2026-10-05, one Bermudan on a 384-per-std, 10-std grid
  peaked at 11.5 GB (under 1 MB per column at ORE's default 30 and 5), and the demo's job asked
  for 277 GB at 65,536 paths. One `[nodes, nodes]` operator per step, applied to every column of
  every path at once, holds one such buffer, shared, and runs on the matrix units: a path holds
  3 to 5 copies of its columns `[columns, nodes]` (XLA's memory analysis, `working_bytes_per_path`;
  10 to 41 KB per path at float64 on the demo's and ORE's grids), each date's paths priced in
  blocks within a budget (`path_blocks`, `blocked`: strided, so every device of the scenario
  axis's split works on every block), and the demo's job runs at 262,144 paths on the CPU and on
  an 8 GB GPU (§15.7).

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
about an ulp (measured on an RTX 5060, 2026-10-06: a float64 EPE's correction of 1e-11 on
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
estimation (Giles and Haji-Ali) is a later research item (FP4 (F-07)).

### 9.5 The report

`PortfolioResult.precision` and `MarketRiskResult.precision`, a `PrecisionReport`:

- the policy as run, including every resolved per-trade precision;
- the realized dtype of every stored class, read from the arrays, not the configuration;
- the device and backend the job ran on (closes [I-12](../known-issues.md#i-12));
- per figure: the corrected and uncorrected value, the paired mean difference, its standard
  error, the largest paired difference, the path counts;
- the evidence verdict per figure: validated, or the warning ([§10](#10-acceptance-standard-and-evidence)).

The rest was built with the precision report (2026-10-02, §9.6). The verdict came with the
evidence table (2026-10-11, §10.3) and is carried in the result's `warnings`, not the report:
a figure validated says nothing, one without a passing row is named in a warning with what is
validated for it and up to how many paths, beside the run's other warnings.

### 9.6 As built (2026-10-02)

**The paired sample.** `paired_paths(N, f)` (`engine/precision/estimate.py`) is 0 at `f = 0`,
else `f·N` rounded up to whole blocks of 32 paths, at least one block and at most every path.
Whole blocks keep the paired paths' block scales the run's own, and keep the number of
distinct paired shapes small (each compiles once). The portfolio pipeline
(`engine.run.pipeline._paired_sample`) simulates `n` paths with the same `CamConfig`
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

**Uncertainty (2026-10-16).** `CamConfig.replicates` divides the total path budget into
independently scrambled Sobol draws. A paired sample takes a whole-block prefix within
each draw. Mean standard errors use the spread of complete replicate estimates over
sqrt(R), including correction covariance. One scramble reports unavailable errors;
`PrecisionReport.replicates` and `uncertainty_method` identify the design. Market-risk
ES errors delete whole scrambles and recompute the tail, or use a conditional moving-block
bootstrap for historical windows. Point estimates and pooled quantile rules remain ORE's.

Nominal intervals are explicitly unavailable under I-111's conservative rule:
`PrecisionReport.interval` states `no_validated_tail_bound`, with null confidence level
and endpoints, for all figures and entries. Raw standard errors remain variance
diagnostics. Neither zero empirical error nor additional scrambles certifies a tail
bound. The [implementation and experiments](precision-accuracy.md) retain the failed
criteria without changing them; [Sampling uncertainty](../../reference/sampling-uncertainty.md)
defines the public contract. A quantile's paired measurement still needs a tail:
ES at 99% on 64 scenarios is NaN.

## 10. Acceptance standard and evidence

The standard labels results; it never stops a run (D-9).

### 10.1 Figures with a Basel III test (A-11)

Basel III does not regulate numerical precision; it judges model outputs through
backtesting and P&L attribution (MAR32). For market-risk P&L, VaR and ES:

1. **The P&L attribution test between the low-precision and the float64 P&L:** Spearman
   correlation and the Kolmogorov–Smirnov statistic, in the green zone. The thresholds are
   transcribed from MAR32 into the regulatory profile under the Basel plan's double-entry
   rule ([basel-iii.md](basel-iii.md) §4); they are not taken from memory. This test is loose
   (it was built for front-office vs risk-model differences), so it is a floor. Until the
   profile exists (Basel P0.3), the evidence table reads them from
   `compliance/precision-acceptance.json`, transcribed once from MAR32.42(1) as published at
   bis.org (2026-10-11: green if the correlation is above 0.80 and the statistic below 0.09),
   the second, independent entry being P0.3's.
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

### 10.4 As built (2026-10-11)

**The standard as data.** `compliance/precision-acceptance.json` holds A-11's thresholds, each
with its citation, read by the generator (a rule without one is refused; no threshold is a
literal in code, the Basel plan's §4 rule 1):

| Rule | Threshold | For |
|---|---|---|
| Statistical (P6.2) | root mean square precision error over the seeds at most 0.01 of the figure's statistical error, on every entry float64 does not hold fixed | every figure; Basel's for P&L, VaR and ES, engineering for the rest |
| Fixed entries | root mean square precision error at most 1e-6 of the figure's largest float64 value, on every entry float64 holds fixed | P6.2's rule for deterministic figures, applied where a figure has no statistical error (I-55's finding: a netting set's ENE where every path is in the money, an EEE_B at its t=0 value, a matured trade's) |
| P&L attribution (MAR32.42(1)) | Spearman above 0.80 and Kolmogorov–Smirnov below 0.09 at every seed | P&L, VaR and ES, with the statistical rule |

**The generator.** `python -m tests.support.precision_evidence` reads every campaign of record
(`compliance/precision-measurements/`) and the standard, and writes the table
(`compliance/precision-evidence.md`) and the engine's rows (`engine/precision/evidence.json`);
`--check` fails if either is out of date, and `tests/test_precision_evidence.py` runs it, so a
campaign rerun without regenerating fails the suite. A row is one figure kind of one scope
under one case: a product (the worst of the campaign's trades of it) or the aggregate (the
netting set, the market-risk portfolio); its combination is the precision of every stage the
figure depends on (`engine.precision.evidence.combination`: market risk has no market stage;
the rounding and block layout count only where a stage stores in a scaled format). It passes
at a path count when every rule holds; NaN fails. Its **path ceiling** is the largest measured
path count at which it passes and at every smaller one, none if it fails at the smallest. Each
row names its campaign, whose market, products, path counts, seeds, device, jax and code are
in the table: the commit, whether the tree had local changes, and a digest of the engine's
Python sources (`tests.support.precision_harness.engine_digest`), exact either way.

**The warnings.** `engine.precision.evidence.evidence_warnings`, called by the portfolio
pipeline (`engine.run.pipeline.precision_warnings`) and market risk (`run_market_risk`) once
their figures exist: plain Python, no JAX work, nothing to compile. For each product of the
run, and for the aggregate, the figures (`run_figures`: the cube's mean per date, EPE, ENE,
EE_B, EEE_B, EPE_B, EEPE_B, PFE at the run's quantiles and the Basel EPE and EEPE; or each
trade's mean P&L and the portfolio's with its VaR and ES) without a passing row whose ceiling
is at least the run's paths, in one warning: the trades, the combination, the paths, and per
figure "validated up to N paths", "passes at no path count measured" or "not measured". The
matching rules, each the conservative reading:

- a trade's figures by its product's rows, its pricing stage its own (`precision_for`); a trade
  whose stages are all float64 is the reference and is never warned of;
- every row under its campaign's **pricing settings** (since 2026-10-18, I-114): per product, the
  settings that change its kernel's arithmetic (`PricingConfig.kernel_settings`), as the
  campaign's analytic ran them (`pricing_settings` of the portfolio pipeline and of market
  risk): the European engine (Bachelier's formula, or Jamshidian's x* root by its solver in the
  stage's format), the grid engine's state grid (`nx`, `sx`: the operator's size), an
  American's exercise steps per year (the rollback's length), and `model=fixed` or
  `model=recalibrated` (market risk recalibrates a calibrated option under each scenario,
  I-112, each its own operator; the exposure simulation keeps one, A-24). A trade's figure
  takes the rows measured under the run's settings for its product, the aggregate's those whose
  campaign held each of the run's products under the run's settings; rows of other settings
  validate nothing, and the warning names what they were measured with ("not measured with
  these pricing settings (measured with ...)"). The model's parameters (reversion, volatility,
  calibration strategy, tolerance, shift horizon, the calibration's float64 solver) change the
  numbers a kernel rounds, not the kernel, and are named by the campaign as its market is;
- the aggregate's by a row whose campaign held every product of the run, and only when every
  trade has one pricing stage (no campaign measures a mix: such a run's aggregate is warned of);
- fewer paths than a ceiling validate: in every campaign the precision error grows against the
  statistical error with the path count (§15.4), so a ceiling bounds from above;
- a figure the paired sample corrects (`CORRECTED_FIGURES`: EPE, ENE and everything computed
  from the corrected EPE) by the rows of the largest paired share at most the run's, a larger
  share only shrinking the correction's error; any other figure is the run's own, by the rows
  without a sample;
- rows of two campaigns for one combination give the smaller ceiling.

A paired run reports unavailable uncertainty with one scramble, or independent-replicate
variance estimates with an explicit rare-event interval warning ([I-111](../known-issues.md#i-111)).
Market-risk ES identifies its replicate jackknife or conditional historical block bootstrap.

**What a row does not say.** A row is evidence for the market and portfolio its campaign
names, measured on a CPU, where every format below float32 is emulated with the numbers a
native unit computes (§8.3); a format runs natively on a device only once its device check
passes (`product_execution`, I-97). Another market, model or product mix may need its own
campaign; the warning says what was validated, not that a run is right. A passing row is not a
regulatory approval.

**The pricing settings measured** (2026-10-18). European rows are Bachelier's alone, and
every option row is on the coarse grid (`nx=8 sx=3`; an American's 24 steps per year): a
Jamshidian European, or a Bermudan or American on ORE's default grid (30 by 5) below float64, is
warned of as unmeasured. Market risk's Bermudan rows are of the fixed LGM (`market_risk`,
`compute_market_risk`, `rounding_market_risk`) and, since 2026-10-18, recalibrated under every
scenario (`market_risk_calibrated`: the same trades, market, shifts and grid, every combination
the three measure). The records measured before then had their settings recorded from their
definitions, unchanged from each record's commit; no row moved.

The recalibrated kernel's rows differ from the fixed LGM's where the float32 accumulator shows:
in float32 compute (and `bfloat16_3x`, as close to it) the Bermudan's mean P&L is biased (its
significance 21 at 2,048 scenarios, 37 at 8,192), its error 1.8% of the statistical error at
8,192 against the fixed model's 0.9%, so its ceiling is 2,048 scenarios where the fixed rows
give 8,192. It is the accumulator's floor (these cases accumulate in float32), larger where each
scenario's level, its model on the base curves, is held in it rather than one level shared:
with the products in float32 and a float64 accumulator the bias is 2e-5 for either model, in a
float32 accumulator 0.016 for the fixed model and 0.035 recalibrated (8 seeds at 8,192
scenarios, against a statistical error of 2). Formats below float32 are dominated by their own
rounding and differ little (float16 compute 6.9% against 7.5% at 8,192). Of the portfolio's 170
figure ceilings, 9 differ, its VaR and ES at a ceiling's edge in both directions.

## 11. Execution architecture

### 11.1 Decision (A-14), built 2026-10-04

The HTTP API and the engine are separate processes. The API validates a request, writes it
to a durable job queue (a SQLite file) and returns `202`. **One engine worker process** per
host takes jobs from the queue one at a time, parses the request JSON itself, prices it and
writes the result back. The worker is single-threaded and owns every device on its host.

As built: `engine/api/job_queue.py` (the queue and the worker lock), `engine/api/worker.py`
(the worker, command `jax-risk-worker`), `engine/api/supervisor.py` (the API's supervision);
the user-facing description is [HTTP API: Jobs](../../reference/http-api.md#runs-the-queue-and-the-engine-worker).

### 11.2 Why

| Reason the process pools existed | Now |
|---|---|
| A different x64 flag per tier | Gone since the shared pipeline; x64 is always on |
| Keep the HTTP server responsive during minutes-long jobs | The separate worker process does it |
| Run jobs in parallel | XLA uses the whole device per job; on TPU one process owns a chip, so several processes need pinning and fight multi-host's sharding |
| Crash isolation | A dead pool worker likely broke the pool (no `BrokenProcessPool` handling); the separate worker is restarted by its supervisor without touching the API |

What it removed: the freeze/thaw of ORE objects (the worker reads JSON, not pickled
dataclasses), the `multiprocessing` spawn setup (I-33), the per-worker compile, and every
thread on the engine side (`_PRICING_LOCK` went in the precision mechanism). `price_portfolio` called from Python
never involved any of this and is unchanged. The API process keeps the server's own request
thread pool, and the TraderX path its locks (`engine/traderx/`), which the engine worker did not touch.

### 11.3 Multi-device and multi-host

- **One host (done 2026-10-04):** the scenario draws of each job (the Sobol normals, market
  risk's shifts) are placed on a one-axis `jax.sharding.Mesh` of the host's devices, split
  along the scenario axis, and XLA's sharding propagation carries the split through the rest
  of the job (`engine/market_simulation/sharding.py`). As many devices as divide the scenario count,
  capped by `JAX_RISK_SCENARIO_DEVICES`; one device places nothing, so one-device runs are
  unchanged bit for bit. Four CPU host devices equal one to about 6e-16 relative
  (`tests/test_sharding.py`). No threads, no pinning.
- **Several hosts (a TPU pod slice; built 2026-10-19, `engine/api/hosts.py`):** one worker per
  host, all running the same program (`jax-risk-worker --distributed`: `jax.distributed.initialize`,
  the cluster detected by JAX or given by `JAX_RISK_COORDINATOR`, `JAX_RISK_NUM_PROCESSES` and
  `JAX_RISK_PROCESS_ID`). Under SPMD every host must run *the same job* at the same time, so the
  hosts cannot each claim from the queue as one host's worker does: the primary (process 0)
  alone opens the queue (SQLite on a shared filesystem is not a safe multi-host lock), claims
  each job and broadcasts it, as fixed-shape chunks so the broadcast compiles once
  (`jax.experimental.multihost_utils.broadcast_one_to_all`); at each phase boundary it decides
  for every host whether a cancel has reached the job (one broadcast), so all stop at the same
  boundary; after the job every host's outcome is gathered (`process_allgather`) and the
  primary writes the row, a job the hosts ended differently failing as `infrastructure`; the
  primary alone purges finished jobs. A failure the request determines (`bad-terms`,
  `missing-market-data`, `unsupported-product`, `numerical-failure`) is raised alike on every
  host and reported by all; any other may be one host's while the others wait in a collective,
  so that host exits (`fail_stop`), and JAX's distributed service stops the rest (measured:
  "Terminating process because the JAX distributed service detected fatal errors"). The
  restarted primary puts the job back once and runs it again whole (`JobQueue.requeue_running`,
  the queue's `attempts`, schema version 4); a job claimed twice is `interrupted`. The scenario
  mesh spans every host's devices, the same number from each (`scenario_device_count`); an
  array split across hosts is read with a gather (`host_array`). A run on fewer devices than it
  has (a scenario count without a fitting divisor) says so in its warnings. Tested on CPU: two
  processes of two devices each, JAX's gloo collectives (Linux): the runs of the four-device
  check give the one-device numbers (`tests/test_sharding.py`), and a stub worker set serves,
  cancels, fails and retries jobs (`tests/test_multihost_worker.py`). Its speed against device
  count is measured on TPU and H100 (I-61).

### 11.4 The queue (A-14; not a main priority)

SQLite rows: the request body as received, status (`pending`, `running`, `done`, `failed`,
`interrupted`), failure class and error, the result document, the claiming worker, the XLA
programs the job built, timestamps. It survives restarts and closes the portfolio half of
[I-08](../known-issues.md#i-08); the TraderX path keeps its publication store.

Decisions taken while building it (2026-10-04):

- **One worker per queue, by an OS file lock** (`<queue>.worker.lock`, `flock`/`msvcrt`). A
  starting worker takes it before importing JAX, so a redundant one exits in milliseconds,
  and the OS frees it however the holder dies. Holding it proves no other worker runs, so a
  starting worker marks every `running` row `interrupted` (its predecessor's job; never
  retried, so a job that kills the worker cannot crash-loop it).
- **The result is stored in the row**, not as a reference to a file: one atomic write, and
  the poll route splices the stored document into its response without parsing it. SQLite's
  1 GB limit fails such a job as `infrastructure`; a cube that large is asked for by reference
  instead (`outputs.npv_cube: "artifact"`, the default since 2026-10-18), stored as chunks of at
  most 8 MiB.
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

Revised with the run request (2026-10-18; schema version 3):

- **One kind of job, a run request** (I-87); the rows of the per-route kinds a version-2 file
  held are kept, the unrun ones marked `interrupted`. The migration is one `BEGIN IMMEDIATE`
  transaction, the version read inside it, so a crash or a second process mid-migration leaves
  the file at its old version (I-106).
- **One worker per host, not per queue file** (I-106): the lock is `JAX_RISK_WORKER_LOCK`, by
  default one fixed path in the temp directory, and its holder records the queue it serves; an
  API whose queue is not that one refuses runs (`503`) rather than queueing runs no worker will
  take.
- **A retried submission is the same job** (I-89): the key unique in the table and the body's
  canonical hash beside it, so two API processes racing on one key get one job.
- **Cancelling** (I-91): a pending job at once; a running one flagged, the worker looking at the
  flag at every phase boundary of the run (`engine.run.trace.at_phase_boundaries`) and stopping
  with a `BaseException`, so no handler on its way out turns it into a failure; the worker and
  its compiled programs go on.
- **Retention** (I-76): the worker purges finished jobs past `JAX_RISK_RETENTION_DAYS` (30 by
  default) between jobs, never one submitted with `keep`.
- **A worker that cannot start is reported** (I-77): the supervisor sends its child's stderr to
  `<queue>.worker.log`, records each exit, and reports two failed starts running (an error exit
  having served no job), or one it could not launch, on a pending run's status and `/health`.

## 12. Plan

| Step | Work | Exit criterion | Size |
|---|---|---|---|
| **The precision mechanism** (done 2026-10-01) | `engine/precision/` (formats, policy, storage at float64/float32; the report skeleton moved to the precision report, `Stored` to sub-32-bit storage, §5); `Precision` replaces the old types in `RunConfig`, `MarketRiskRequest` and the HTTP schema, the old shape refused (A-12); the cast points; inputs follow dtype; float64 reductions; the constants of §6.5; market risk on the same module; remove `check_run`'s refusal, the flag set, the lock and the tiers; strict promotion in CI | Default: golden snapshot bit for bit, every ORE parity suite unchanged. float32: the scenario market and the market-risk revaluation bit for bit as before; the portfolio cube is not, and cannot be (§13.1); exposure and VaR/ES differ by the float64 reductions. Fast tier green under strict promotion. Met: §13.1 | M |
| **Per-trade precision** (done 2026-10-02) | Per-product and per-trade precision (A-15): `by_product`, `by_trade`, `precision_for`; per-trade stored columns; market risk per trade | Bit for bit at the default; a mixed run (Bermudan float32, swaps float64) equals each trade run alone at its precision, column for column. Met: §13.1 | S |
| **Sub-32-bit storage** (done 2026-10-02) | Sub-32-bit storage: block scales, both roundings; `float16`, `bfloat16`, `float8_e4m3fn`, `float8_e5m2` enabled for storage | The storage properties of §13.4; bit for bit at the default. Met: §13.1, §13.4 | M |
| **The precision report** (done 2026-10-02) | Paired sample, two-level estimator for means, `PrecisionReport` with realized dtypes and device (closes I-12) | §13.6; bit for bit at the default. Met: §13.1, §13.6 | M |
| **The engine worker** (done 2026-10-04) | Engine worker process and the SQLite queue (A-14); delete `worker_pool.py`'s pool and freeze/thaw | §13.8, including the Linux run; float64 job time and compile count no worse. Met: §13.8, §13.9 | M |
| **The storage harness** (I-75, done 2026-10-09) | Storage relative to a level for classes whose level swamps their spread (block offsets, §6.2); a measurement harness for storage formats per class and product, at several path counts, fixed seeds, rerunnable on any kernel change (§15.4) | Harness covers every figure × class × format; thresholds fixed before measuring; I-75's FP8 bond column unbiased within its Monte Carlo standard error. Met: §15.4 | M |
| **The kernel rewrite** (F-07; done: the first families 2026-10-10, the rollback 2026-10-13) | Difference-form kernels (§8.2), one family at a time (those the recalibration fix did not change first, the Bermudan/American rollback after it; `ShiftHorizon` and the recalibration fix done 2026-10-12), as matrix products where a family can be one (§8.3); compute below float32 enabled; `accumulate` honoured | Per family: ORE parity suites at their tolerances, then the snapshot within the margin (§2.1); emulated FP8/bfloat16 compute measured by the harness. Met for the simulation, the scenario market, the legs, the Europeans and market risk (§13.1, §15.5), and for the rollback, its memory per path a few columns on the grid (I-83, §13.1, §15.7) | L |
| **Multi-host** (I-61) | Shard the scenario axis in the worker (I-61): one host done 2026-10-04; several hosts done 2026-10-19 (§11.3) | Results equal the one-device run within reduction-order rounding (met on one host and on two CPU processes, `tests/test_sharding.py`, `tests/test_multihost_worker.py`); scaling measured on TPU | M |
| **The evidence table** (I-55, done 2026-10-11) | The evidence table and the warnings (§10), from the storage harness on each kernel family as the kernel rewrite lands it (the Bermudan/American rows measured again after `ShiftHorizon`, I-32, and after the recalibration fix, I-49, both 2026-10-12, and after the rollback's rewrite, 2026-10-13, at the linear campaign's path counts, with their compute formats and each option's flipped exercise decisions, §15.7) | Table complete; a verdict and path ceiling per row; a warning on every result without a passing row. Met: §10.4, §15.6 | S |
| **The research result** (F-07) | After multi-host and the evidence table. Timing across devices on Ironwood and H100: storage formats, then the kernel rewrite's matrix-product kernels with native FP8; the harness built 2026-10-19 (§19) | The research result: wall time per figure and precision against float64 at equal accuracy (measured over seeds, §19), many low-precision paths against fewer float64 ones | M |
| **FP4** (F-07) | FP4 storage (variance correction through the block scales), FP4 compute on TPU 8t/8i, multilevel quantile estimation | Evidence rows for FP4 | L |

**Order (decided: A-19, deadline scope clarified 2026-10-19):** the structure, local GPU
profiling, existing low-precision kernels, evidence table and multi-host worker set are built.
First extend the existing comparison tools (F-07), alongside target-device correctness and
native FP8 qualification (I-121, I-97). Publish the first verified target-device result without
waiting for the other platform or cluster scaling; qualify scaling (I-61) before making
multi-device/host claims. Basel III follows the cluster research result. FP4, INT8, additional
multi-pass/Qwix methods and reduced-precision today's PV/Greeks are outside the first result;
FP4 and quantile correction remain separate later experiments. The current evidence table
continues to hold per kernel and pricing setting (I-114). See the roadmap and §19 for the
bounded deadline comparison; new capabilities do not gate it.

## 13. Testing

Verification rules of [README.md](../README.md#verification-rules) apply, notably: the
virtualenv's interpreter, counts from `--collect-only`, a summary line printed, and Linux for
process changes.

### 13.1 Bit for bit and ORE parity

**From 2026-10-10 (A-19 revised) the snapshot is compared within a margin, not bit for bit:**
every array relative to its largest magnitude (a Greek to its trade's NPV), float64 to 1e-10
and float32 to 1e-4; formats below float32 are judged by the storage harness (§15.4). A move
beyond the margin is a planned model change, stated per array and re-baselined, or a bug.
Every ORE parity suite still passes at its tolerance. The bit-for-bit results below are the
record of stages 1 and 2.

- Before the precision mechanism (2026-10-01), save a golden snapshot from `main` in a separate worktree: the shared
  portfolio's t=0 NPVs, cube, exposure and per-trade EPE at float64 and with a float32
  simulation, bump Greeks, AD Greeks, a market-risk run at 64 and 32. Compare with
  `np.array_equal` (value, dtype, shape) after every change of the structure stage. The Greeks run
  takes about 16 minutes and 10 GB.
- Every ORE parity suite runs unchanged at the default after every step.
- **Result of the precision mechanism (2026-10-01).** 164 arrays at the default precision (today's values under
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
- **Result of per-trade precision (2026-10-02).** The same snapshot script, 232 arrays from a worktree of
  `2b70d15` (the default runs and, as before, the float32 runs): all 232 identical in value,
  dtype and shape, float32 included, since a policy without overrides prices exactly as
  before. Every ORE parity suite passed unchanged (full suite 2,337 passed). The exit
  criterion's tests (`TestPerTradePortfolio`, `TestPerTradeMarketRisk`) are red when every
  trade is priced at `pricing.compute`. float64 speed (§13.9), old and new trees interleaved:
  market risk on four trades, median of ten warm runs, 5.95/6.07 s before and 5.73/5.84 s
  after (no regression); the shared portfolio's run varies 2.5× from run to run (per-path
  recalibration, I-53), and the fastest of five was 118.7 s before and 120.3 s after.
- **Re-baseline after the jit change (2026-10-02, between per-trade precision and sub-32-bit storage).** The pricers,
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
  passes at its tolerance. The rest of the structure stage compares against a snapshot of the commit that
  made this change, not of `a6c63bf`. The
  whole snapshot took 14 minutes instead of 69 (bump Greeks 21 s instead of 29 minutes),
  both measured with other jobs running.
- **Result of sub-32-bit storage (2026-10-02).** The same snapshot script, 232 arrays, from a worktree of
  `f51227c` (the commit of the jit change) against the new tree: all 232 identical in value,
  dtype and shape, the float32 runs included. float64 and float32 storage are the same plain
  cast as before, and the one float32 behaviour it changes, the tenor grid under a float32
  market storage with float64 compute (§6.3), is in no snapshot run.
- **Result of the precision report (2026-10-02).** The same snapshot script, 232 arrays, from a worktree of
  `f0a438c` (sub-32-bit storage) against the new tree: all 232 identical in value, dtype and shape, the
  float32 runs included. The default path changed shape (t=0 values computed apart from the
  path pricing, `value_paths`; one pricing context for every trade's t=0 value; the report
  built from the arrays) but not its arithmetic. A paired sample is off by default, and at
  float64 it measures exactly zero (`tests/test_precision_report.py`). float64 speed (§13.9),
  old and new trees interleaved, three passes of five warm runs each: market risk on four
  trades, median 2.19 s before and after (fastest 1.87 and 1.65 s); a five-trade, 1,024-path
  portfolio, median 39 ms before and 37 ms after.
- **Result of the engine worker (2026-10-04).** No file of the pipeline changed (`git diff` against
  `0e44f1d` touches `engine/api/`, a docstring of `engine/run/trace.py` and the
  deleted `engine/portfolio/worker_pool.py`), so the snapshot was not rerun: it would compare
  the same code. What the engine worker changed is the path from the HTTP body to `price_portfolio` and
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
- **Result of the GPU run (2026-10-05).** It changes no kernel, but it sets three process
  defaults in `engine/__init__.py` (no GPU preallocation, matrix products at their operands'
  precision, deterministic GPU kernels), and the last two change every program's HLO or
  compile options. The same snapshot script, 232 arrays, from a worktree of `1a533f3` against
  the new tree, on CPU: all 232 identical in value, dtype and shape, the float32 runs included
  (XLA's CPU backend computes a float32 product in float32 whatever the precision
  configuration says). On the GPU (RTX 5060, WSL2) float64 is not bit for bit with the CPU,
  and is held to the parity tolerances, which every ORE parity suite meets: the demo's job
  equals the CPU's today's value exactly, the cube and EPE to 9e-16 of their scale, the AD
  Greeks to 2.3e-11 relative.
- **Result of the stated product precision (2026-10-06).** It moves the GPU run's three process defaults to the processes
  that own them and states every matrix product's precision in the program
  (`engine.precision.matmul`, §6.7), so the products' HLO carries the same precision the GPU run's
  process default gave it; the Brownian bridge's `tensordot` became a reshaped matrix product
  (checked bit for bit against it on its own first, at float64 and float32, up to
  `[120, 2048, 4]`). The same snapshot script, 232 arrays, from a worktree of `7b51e19` (the GPU run)
  against the new tree, on CPU: all 232 identical in value, dtype and shape, the float32 runs
  included.
- **Result of the compiled Greeks (2026-10-06): the AD Greeks re-baselined.** It makes each trade's AD
  Greeks one jitted program per product (Delta and Gamma from one linearization of the
  gradient over all its curves; Vega another), where they were eager derivatives of the
  jitted pricers ([profiling §3.8](../../concepts/profiling.md#38-the-ad-greeks-as-one-program-per-product-2026-10-06)).
  The roadmap expected it to keep every number; it cannot for the AD Greeks: the repeated
  job's recompiles came from the eager derivatives' programs living only in JAX's bounded
  internal caches, and once a derivative is one XLA program, XLA fuses its forward and
  backward passes and reorders their reductions. The snapshot gained the demo's own job
  (`demos/demo_profile_small.py`'s request through the API's schema, run twice in the
  process: Hull-White calibrated, options recalibrated on every path date, AD Greeks), 100
  arrays, so 332 in all, from a worktree of `2df78cb` (the stated product precision) against the new tree, on CPU:
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
  sensitivities, I-51). Later changes compare against a snapshot of its commit.
- **Result of the Newton solver (2026-10-07): the solver re-baselined.** It puts every root of the
  engine on one solver (`engine.solvers.roots`, decision A-21): a safeguarded Newton method
  by default, the bisections as the reference (`"Bisection"`), and batches a Bermudan's or
  American's path dates of one basket shape into one calibration under Newton. The snapshot
  gained the Jamshidian engine (the shared portfolio's European on the paths, its AD Greeks)
  and the standalone bootstrap (`POST /calibration/lgm`'s, on a sloped curve, its Vega through
  x*): 359 arrays, from a worktree of `eadbcc6` (the compiled Greeks) against the new tree, on CPU.
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
  `"Bisection"` for the run). Later changes compare against a
  snapshot of its commit.
- **Result of the API reaching every setting and the ORE simulation oracle (2026-10-07): the exposure re-baselined towards ORE.** The API
  changes no number (the HTTP layer, and market risk's validation split without its
  arithmetic). The ORE simulation oracle found three differences from ORE in the assembled
  pipeline, each fixed: the simulation market holds its tenor points at the times from the
  as-of date (I-84), the exposure starts from the simulation market's t=0 values (I-85), and a
  cash-settled option's maturity is its last exercise date (I-86). The same script, 359 arrays,
  from a worktree of `c8e21ea` (the Newton solver) against the new tree, on CPU: 198 identical in value,
  dtype and shape (today's values, both Greeks methods, the Jamshidian Greeks, every market-risk
  figure and its raw float32 revaluation, the standalone bootstrap, the float32 scenario
  market's discount factors); 161 moved, as intended, relative to each array's largest
  magnitude: the cubes up to 1.8e-3 (shared portfolio, LGM and Hull-White), 2.8e-3
  (Jamshidian), 1.6e-2 (the demo's job, monthly dates to 5 years); the exposure profiles up to
  3.2e-2 (a cash-settled European's EPE, its start moved to the simulation market's value), the
  netting set's up to 2.5e-2 (EPE_B); the scenario market's tenor times by 2.7e-4 of their
  scale (3 days on the 30Y point of a later date). The float32 runs moved with the float64
  ones. Every ORE parity suite passes at its tolerance, and swaps, Europeans and bonds now
  equal ORE's own simulation on its paths (`tests/test_ore_xva_parity.py`). Later changes
  compare against a snapshot of its commit.
- **Result of the package layout (2026-10-08, I-92).** Renames and moves only
  ([package-layout.md](package-layout.md)). The same script, made layout-adaptive (each import
  tried under its new name, then its old), 359 arrays, from a worktree of `7d52e17` against
  the new tree, on CPU: all 359 identical in value, dtype and shape, the Greeks of both
  methods, the float32 runs and the demo's job included. Later changes compare against a
  snapshot of its commit.
- **Result of the storage harness (2026-10-09, I-75).** It adds block offsets to scaled storage,
  the default for every scaled format, and changes no float64 or float32 arithmetic. The same
  script, 359 arrays, from a worktree of `d19f9c5` against the new tree, on CPU: all 359
  identical in value, dtype and shape. A second snapshot covers scaled storage, which the first
  does not: `store`/`load` of three arrays (normals, a concentrated column, zeros beside NaN) in
  every scaled format under both roundings from float64 and float32, and a portfolio and a
  market-risk run with every stage stored scaled under both roundings, 168 arrays; with
  `offset="none"` all 168 are identical to the code before, so the layout without offsets is
  the storage of before, bit for bit (the NaN padding of a short block, which replaced zeros,
  counts towards no scale either way). The default layout moves the scaled runs' numbers, as
  intended (§15.4). Later changes compare against a snapshot of its commit.
- **Result of the kernel rewrite's first families (2026-10-10).** The simulation scan and the
  Brownian bridge, the scenario market, the legs and path fixings, the Bachelier and Jamshidian
  Europeans and market-risk revaluation in difference form (§8.2), compute below float32 enabled
  for them. Every ORE parity suite passes at its tolerance (the full suite, the verification
  status). The same script, 359 arrays, from a worktree of `53f85b9` against the new tree, on
  CPU, compared within the margin of A-19's 2026-10-10 revision: 136 identical, none beyond the
  margin. Largest change relative to each array's scale (a Greek's to its trade's NPV): float64
  cubes and exposures 2.2e-14 (LGM), 2.1e-14 (Hull-White), 3.7e-15 (Jamshidian), 7.0e-15 (the
  demo's job); market risk 1.5e-14; today's values 1.6e-16; AD Greeks 2.2e-15, bump Greeks
  7.6e-15; the standalone bootstrap and the Jamshidian Greeks identical. float32: the cube
  2.2e-6, market risk 8.3e-6, the raw shocked revaluation 2.5e-6, the scenario market 3.6e-7
  (margin 1e-4); before the levels met the paths in two terms (§8.2) the float32 cube had moved
  5.1e-6. This snapshot is the reference for the Bermudan/American rollback's rewrite (§2.1).
- **Result of the rollback's rewrite (2026-10-13).** The Bermudan/American induction as matrix
  products in difference form (§8.2), on the paths and in market risk's P&L, compute below
  float32 enabled for it. Every ORE parity suite passes at its tolerance (the full suite, the
  verification status). The same script with `--greeks`, 359 arrays, from a worktree of
  `6e85d09` against the new tree, on CPU: 188 identical, none beyond the margin. Largest change
  relative to each array's scale: float64 1.2e-14 (the demo's Bermudan's exposure; the shared
  portfolio's option cubes 8e-15), a Greek 2.2e-14 of its trade's NPV (bump Gamma, through each
  bump's recalibration and the induction); float32 1.2e-6 (an American's EPE; market risk's
  ES_99 standard error), the float32 option cubes now 2 to 6 times closer to float64 than the
  code before, which ran the rollback in float32 throughout, and 5 to 20 times less biased.

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
  the realized dtypes in `PrecisionReport` with the policy. Done in the precision report (2026-10-02):
  `tests/test_precision_report.py::TestReport::test_the_realized_format_of_every_class_is_the_policys`
  (each stage in each of the six formats, one trade overridden), and market risk's shifts and
  P&L (`TestMarketRisk`).
- A test that the resolver is the only lookup: per-trade overrides change only that trade's
  column. Done in per-trade precision (2026-10-02): `tests/test_precision.py::TestPerTradePortfolio` and
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

Done in sub-32-bit storage (2026-10-02) (`tests/test_precision.py::TestScaledStorage`), for every scaled format, from
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
an FP8 cube without offsets.

Block offsets (2026-10-09): every property above that holds in both layouts is tested in both
(nearest equal to ml_dtypes' rounding of the deviations, the scale set by the largest
deviation, stochastic rounding to a neighbour, signed zeros, NaN and infinities, short blocks,
independent blocks, the pytree, bytes held), and the round trip of the format's own values,
which an offset breaks, without them. `TestBlockOffsets`: a concentrated column (1e6 ± 1e4)
loses more than 7/8 of its largest error and its mean error is noise, where without offsets
FP8 moves the mean by more than ten of its standard errors; the offset is a concentrated
block's midpoint in float32, and 0 for a block of both signs, a zero, one reaching nearer zero
than half its range, or no finite entry; the scale is never coarser and the error never above
the bound without an offset; a constant block is held to float32's precision of it; no entry
changes sign, at the rule's edge (the farthest three times the nearest) in e5m2 and
stochastically too;
no dtype mix under strict promotion. The policy's `offset` reaches `Precision.store` and is
refused where it would change nothing; with `offset="none"` scaled storage is the code before
it bit for bit (§6.2).

### 13.5 Low-precision sanity on sloped curves

At float32 throughout, on the 3% → 5% curve: no NaN, no calibration at its ceiling, the cube
within a stated float32 tolerance of float64. These tests are written red first against the
constants of §6.5.

Below 32 bits (slow, `tests/test_precision.py::test_scaled_storage_on_the_sloped_shared_market_is_sane`):
every trade of the shared portfolio with one stage stored in each scaled format, its cube's
largest error and its bias over 256 paths within bounds about four times those measured; since
2026-10-13 on the cells whose exercise decision the format did not flip (decision A-24 (5): a
cash-settled option's flipped exercise moves it by its whole value, which the harness measures as
its own result), the flipped share of the options' cells at most 5%. Judged so, the bounds came
down: bfloat16's simulation and market storage from 7e-2 to 2.3e-3 and 3.3e-3 of notional, which
had held one cash Bermudan's flip (1.68e-2) where their rounding is 30 times smaller.

### 13.5a Compute below float32 (2026-10-10)

`tests/test_compute_precision.py`, fast tier:

- **The products**: every format below float32 whose product is one pass, and its single pass,
  against an independent reference (ml_dtypes' correctly rounded conversion of operands scaled
  per row and column, or a 10-bit mantissa rounding for TensorFloat-32, the shared operand in
  its terms, each the reference rounding of what the ones before leave, an exact product, the
  float32 accumulation's bound), with either operand shared; bfloat16 passes against their
  pieces; every compute format within its unit roundoff on operands spanning six decades; a
  path's rounding independent of every other path (scales per path), the left or the right
  operand shared; vectors, batches, zeros, an empty contraction; the program one product of
  operands of the format into float32, as wide as the shared operand's terms; float32
  accumulated in float64 (on any device: red first on the GPU, 2026-10-11); the native/emulated
  table for CPU, three GPU generations and two TPU kinds, a single pass on its format's units, a
  native product format stating its algorithm; a product in a stage's arithmetic that does not
  name its shared operand refused (A-23).
- **The shared operand's bias** (2026-10-11, A-23, I-96): over 4,096 paths of a per-path operand
  whose mean is far from 0, the mean error of a single pass is 20 or more of its standard errors
  in every format, with either operand shared, and of two terms below 4 (10-bit grids,
  bfloat16's) or below an eighth of the single pass's (FP8, whose rest is each path's own
  rounding, §15.5). Through the pipeline, every stage in float16, bfloat16 or TensorFloat-32 at
  2,048 paths: every trade's mean error on every date below 4 standard errors with two terms,
  above 6 with the single pass; red first, the code before failing it at 14 to 24.
- **The arithmetic**: its refusals, naming the field.
- **Each family at float64 is the direct formula to rounding**: the scan against `x <- M x + b +
  L z`, the bridge against differencing the path values, the market against the model's
  formulas (its level the curve at the mean state), the legs against ORE's arithmetic on each
  path's own curve at the model's level and at a level of 0, a path on the level exactly the
  level, the P&L against `f(base + shift) - f(base)`; the path moments against a 200,000-path
  sample, the curves' interpolation of deviations exactly linear.
- **Below float32 the error is relative to the deviation**: the scan's error within a few unit
  roundoffs of each step's scale and, for a log equity spot, below a third of the same scan's on
  the states themselves; the curves' within four unit roundoffs of their deviation from the
  level; a swap's cube within thirty of its paths' spread; every trade's mean value within eight
  of the paths' spread (plus float32's rounding of the level) with every stage in the format,
  under strict promotion; market-risk P&L within four of each trade's largest.
- **The pipeline**: the report names every compute format of a run and its execution; a
  repeated run compiles nothing; a Bermudan or American computes in any format (refused below
  float32, and computing and accumulating in two formats, until 2026-10-13), and can be kept
  wider by product beside a low-precision run; the paired sample corrects an FP8 run to float64;
  a scenario that moves nothing has exactly zero P&L at every compute format, both European
  engines and the Bermudan's; the Bermudan's P&L within four unit roundoffs of its largest, as
  the legs'.
- **The induction** (2026-10-13, `tests/test_rollback_difference_form.py`): the rollback's
  operator is ORE's interpolated convolution (and interpolation alone without variance), to
  1e-14; the cashflows' coefficients times their functions are the LGM's bond prices, every
  kind of coupon (forecast, known, fixed today, before and after its index start), to 1e-12; a
  Bermudan caches no coupon and an American a few columns; each scenario of shifted curves and
  each path of a cube is the induction on its own curves and fixings to 1e-12; a padded trade
  keeps its value and a cube compiles its induction once for every date, another trade of the
  shape not at all; its memory per path within `working_bytes_per_path` and a twentieth of the
  old operator per step; every compute format prices the options under strict promotion within
  its bound (slow).

### 13.6 The estimator and the report

- float64 against float64: every paired difference exactly 0, the corrected figure equal to
  the uncorrected one.
- A test-only format with a known bias: the report flags it, and the two-level estimate
  removes it within its standard error.
- Coverage: over independent seeds, the corrected estimate's confidence interval contains
  the float64 figure at the stated rate.

Done in the precision report (2026-10-02), `tests/test_precision_report.py`: the estimator on synthetic samples
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
(`tests/test_api_portfolio.py`, I-12). The quantiles' measurement is exact by construction;
their correction is FP4 (F-07).

### 13.7 Statistical acceptance (slow tier)

The thresholds are fixed before measuring. The storage harness's first such test (2026-10-09):
`tests/test_precision_harness.py::test_a_bonds_fp8_cube_column_is_unbiased_with_block_offsets`
(§15.4, threshold 1). The evidence table's rows are not rerun by the suite, its campaigns
taking about 80 minutes: they are the measurements of record, and
`tests/test_precision_evidence.py` holds the table and the engine's rows to them and to the
standard (`--check`), and the rules, the lookup and a reduced run's warnings to their
definitions (2026-10-11, §10.4).

### 13.8 Execution

- Jobs queued together give the same bits as run one after another.
- A failing job fails only its own row, with a failure class; the worker survives.
- A worker killed mid-job leaves an `interrupted` row; a restarted worker picks up the next.
- The existing API tests pass unchanged (`202`, poll, result).
- The full suite on Linux (Docker `python:3.11`), as process changes require.

Met by the engine worker (2026-10-04), in `tests/test_engine_worker.py`:
`test_jobs_queued_together_give_the_bits_of_jobs_run_one_after_another`,
`test_a_failing_job_fails_only_its_own_row` with the failure-class table,
`test_a_worker_killed_mid_job_leaves_it_interrupted`. `tests/test_api.py` passes with two
changes that follow from the design, not from a result: the module fixture that shut the pool
down went (the session's worker is stopped by `tests/conftest.py`), and the test that a
fresh job is not priced eagerly accepts `running`, which the pool could not report. Red
first on the code before the engine worker: a pool worker killed while idle made the next submission fail
with `BrokenProcessPool` (the new worker: the killed job `interrupted`, the next `done`), and
a job id was a `404` from a second app process (now served from the queue). The full suite
ran on Linux ([verification status](../known-issues.md#verification-status)).

### 13.9 Performance

- float64 wall time and compile count of the shared portfolio against the previous step,
  median of five warm runs: no regression beyond the run-to-run spread.
- float32 timing recorded.
- After the engine worker: a second identical job compiles nothing. Met: each job's row records the XLA
  programs it built (JAX's `backend_compile_duration` event); the benchmark portfolio's first
  job built 122, each of the 18 repeats 0
  (`test_a_second_identical_job_compiles_nothing`). The pool built the same programs once per
  worker (two by default).
- The research result (F-07) on Ironwood and H100, across devices (after multi-host): wall time per figure at equal accuracy.
- **The kernel rewrite's first families (2026-10-10)**, old and new trees interleaved, three
  passes of five warm runs, float64 on CPU: a five-trade linear portfolio (4,096 paths, 20 dates)
  0.370 s against 0.352 s (median; within the run-to-run spread, 5%), market risk on a swap, a
  European and a bond (8,192 scenarios) 0.075 s against 0.23 s, three times faster (the P&L is
  one product, not a revaluation less a base). The leg kernel alone is a third slower on a CPU
  ([I-53](../known-issues.md#i-53)): the same operations (11.6 ms against 11.7 ms in a trace)
  scheduled less concurrently. Two layouts measured and discarded on the way: dense
  interpolation weights as a product (2.1 times the old kernel; a two-node stencil in the
  accumulate format replaced it) and two columns per coupon (twice the operand's bytes).

### 13.10 Demos

`demos/demo_precision.py` moves to `Precision` (done in the precision mechanism, with a storage-only float32 run
beside float32 throughout), adds FP16, BF16 and FP8 storage runs, FP8 with both roundings
(done in sub-32-bit storage), and prints a run's report with a paired sample (done in the precision report).
The kernel rewrite (2026-10-10) adds bfloat16 and FP8 compute runs, the Bermudan kept in
float32 by product, and prints the report's product executions.

## 14. Scope

**In:** float64/float32 compute and storage; float16, bfloat16 and FP8 storage; compute below
float32 (float16, bfloat16, FP8, TensorFloat-32, bfloat16 passes; for every path kernel, since
2026-10-10 and the Bermudan/American rollback since 2026-10-13); per-stage, per-product and per-trade precision; block scales and both
roundings; the paired sample, the two-level estimator for means and the report; the evidence
table and warnings; the worker process and queue; timing on Ironwood and H100.

**Out, with the reason:**

| Item | Why | Where |
|---|---|---|
| Low-precision calibration, t=0 and Greeks | Cheap and fragile (A-10); openable later by a field | — |
| FP4 | Needs TPU 8t/8i or Blackwell | F-07 (FP4) |
| Variance correction of stored shocks | Only if the storage harness shows bias | F-07 (FP4) |
| Multilevel quantile estimation | A research project of its own | F-07 (FP4) |
| Low-precision speed on CPU | Emulated; CPU is accuracy evidence only | — |
| Elementwise arithmetic below float32 | No device runs it faster; several of its kernels do not exist; an FP8 exponential of a level is noise (§3.1) | — |

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

This is one payoff on raw shocks, not pricing through the pipeline; the storage harness (I-75) measures the
pipeline's figures.

### 15.3 Storage through the pipeline

Measured 2026-10-02 on the sub-32-bit storage (2026-10-02) code: the shared portfolio (8 trades, 3% -> 5% curves),
256 paths, 3 dates, one stage stored in the format and the others float64 (compute float32 in
the stored stage, float64 for pricing, so only storage differs). Per unit notional: the
largest error of a cube entry, the largest bias (the mean over the paths of a trade on a date),
and the largest change of the netting set's EPE relative to its peak. One seed: a screening,
not the evidence of the storage harness (I-75).

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
  quantiles are not (§9.4), so for VaR/ES the storage itself must be precise enough. The storage harness (I-75)
  measures it per figure; [I-75](../known-issues.md#i-75).

### 15.4 The storage harness (2026-10-09)

The harness (`tests/support/precision_harness.py`) runs a campaign's cases through the public
entries (`price_portfolio`, `run_market_risk`) at each path count, over independent Sobol
seeds, against the float64 run on the same paths; the campaigns of record and the command that
runs them are `tests/support/precision_campaigns.py`, and their measurements are kept in
[compliance/precision-measurements/](../../../compliance/precision-measurements/) (JSON, with a
generated markdown summary). Per case, figure and path count it records the precision error's
mean (the bias), its standard error and root mean square, and the figure's statistical error
(its float64 spread over the seeds); per paired case the coverage of the paired estimator's
two standard errors; per market-risk case Spearman's correlation and the Kolmogorov–Smirnov
statistic of its P&L against float64's; per case the programs compiled after its first seed.

**Thresholds, fixed before the campaigns of record were measured** (the trial runs that
debugged the harness used two to four seeds on slices of the cases):

1. **I-75 closes** if, rounded to nearest in FP8 (e4m3, e5m2), the bond's cube column of the
   `linear` campaign at 1,024 and 4,096 paths and 16 seeds, with block offsets, has at every
   date a bias within three of its standard errors of zero and below the Monte Carlo standard
   error of the column's mean (its float64 spread over the paths over sqrt(paths)); and without
   them is beyond both at some date (I-75's bias, so the test can fail).
   `tests/test_precision_harness.py::test_a_bonds_fp8_cube_column_is_unbiased_with_block_offsets`.
2. **The block layout default stays `"block"`** if, for every stage, scaled format, rounding and
   path count of the `linear` and `options` campaigns, the worst relative error over the figures
   with offsets is at most 1.1 times the one without (the 10% for the seeds' noise in a worst
   case over many figures). Otherwise the default is `"none"`, and the classes where offsets
   help are recorded.
3. **The rounding default stays `"nearest"`** unless stochastic rounding, in the default layout,
   has the smaller worst relative error for every stage, scaled format and path count of the
   `linear` campaign, and for VaR and ES in every case of the `market_risk` campaign (§15.3
   found stochastic rounding's added variance biases quantiles).
4. **The paired estimator's standard errors are adequate** where each interval's pooled coverage
   (nominal 95%) is at least 90% for every paired case and path count of the `linear` campaign.
   A case below it is a finding for the evidence table (I-55), recorded there.
5. **No case compiles after its first seed**, in any campaign.

**The first record run failed threshold 1** (2026-10-09). Its block offset was each one-signed
block's entry nearest zero, chosen on the trial runs to keep every entry's sign (§6.2). On the
`linear` campaign it cut the bond column's FP8 error a thousandfold, but its deviations,
measured from a block's minimum, are skewed, and nearest rounding of them left a bias of 18 to
36 (e4m3) and 75 to 120 (e5m2) on values near 1e6: 4.7 to 28 of its standard errors. The layout
was changed to the midpoint of concentrated blocks (§6.2), the thresholds left as they were,
and every campaign measured again. Of that run, thresholds 2, 4 and 5 passed but for one
comparison of threshold 2, which the zero statistical error of a fixed figure (below) made
infinite, and that the harness has since left out of its ratios.

**Results of the record run** (2026-10-09, CPU, 16 seeds; [linear.md](../../../compliance/precision-measurements/linear.md),
[options.md](../../../compliance/precision-measurements/options.md),
[market_risk.md](../../../compliance/precision-measurements/market_risk.md)):

1. **I-75 is closed.** With block offsets the bond's FP8 column has no detectable bias at any
   path count: significance at most 0.97, 1.9 and 2.3 (e4m3) and 0.75, 1.3 and 2.8 (e5m2) at
   1,024, 4,096 and 16,384 paths, its bias at most 4 on values near 1e6, a hundredth (1,024
   paths) to a thirtieth (16,384) of the column's Monte Carlo standard error; without them 370
   to 1,700 (e4m3) and 300,000 to 10 million (e5m2) of its standard errors.
   The test passes, and its half without offsets fails as it must.
2. **The block layout stays `"block"`**: in every comparison of every campaign its worst
   relative error is at most the layout's without, and for a concentrated class much smaller.
   The simulation stage's arrays (shocks, states) straddle zero and are stored alike,
   bit for bit.
3. **The rounding stays `"nearest"`**: stochastic rounding has the smaller worst relative error
   in 8 of 36 comparisons of the `linear` campaign (FP8 cube storage at 16,384 paths among
   them), not in all; and in market risk (below).
4. **The paired estimator's standard errors.** Its precision interval covers 0.950 (pricing),
   0.934 to 0.946 (simulation, market) of the checks, nominal 0.95, every case at least 0.91,
   so threshold 4 passes. But the standardized errors' root mean square is 1.00 to 1.01 for
   storage of the cube and 1.5 to 3.0 for storage of the shocks, states and curves: there the
   correction's standard error is too small in a heavy tail, the ENE of an option, non-zero on
   few paths (single checks at 17 to 29 standard errors). Its total interval covers 0.99 to 1.0
   with a root mean square of 0.30 to 0.40: on Sobol paths a figure's Monte Carlo error is about
   a third of what the independent-paths formula states. Both were recorded on I-55, and are
   [I-111](../known-issues.md#i-111) since the evidence table closed it (§15.6).
5. **No recompiles**: every case, in every campaign, compiles nothing after its first seed.

The relative error, worst over every figure and date of the `linear` campaign, the root mean
square precision error in units of the figure's statistical error (nearest rounding; the
summary has every case, rounding and paired run):

| Stage, storage | 1,024 | 4,096 | 16,384 | Worst figure at 16,384 | Bias significance at 16,384 |
|---|---|---|---|---|---|
| `simulation:float32` | 9.8e-06 | 2.7e-05 | 5.8e-05 | trades/swap-payer/PFE_95 | 4.1 |
| `simulation:float16` | 0.063 | 0.16 | 0.41 | trades/swap-payer/PFE_95 | 2.8 |
| `simulation:bfloat16` | 0.63 | 1.1 | 2.1 | trades/bond/NPV | 3.2 |
| `simulation:float8_e4m3fn` | 5.9 | 15 | 45 | trades/european-payer/PFE_95 | 103 |
| `simulation:float8_e5m2` | 11 | 37 | 130 | trades/swap-payer/EPE | 301 |
| `market:float32` | 0.00024 | 0.0003 | 0.0016 | trades/bond/EPE | 4.1 |
| `market:float16`, block / none | 0.33 / 1.4 | 0.54 / 2.9 | 1.2 / 6.7 | trades/swap-payer/PFE_95 | 2.7 / 115 |
| `market:bfloat16`, block / none | 0.98 / 39 | 1.5 / 154 | 2.9 / 1,380 | trades/swap-payer/NPV | 10 / 588 |
| `market:float8_e4m3fn`, block / none | 9.8 / 2,140 | 64 / 9,670 | 215 / 52,700 | trades/swap-payer/EPE | 117 / 7,550 |
| `market:float8_e5m2`, block / none | 50 / 2,420 | 153 / 16,700 | 569 / 95,600 | trades/swap-receiver-seasoned-icma/NPV | 572 / 16,300 |
| `pricing:float32` | 0.001 | 0.0041 | 0.011 | trades/bond/NPV | 3.2 |
| `pricing:float16`, block / none | 0.042 / 11 | 0.076 / 37 | 0.21 / 98 | netting_set/PFE_95 | 3.0 / 186 |
| `pricing:bfloat16`, block / none | 0.35 / 1,570 | 0.62 / 9,190 | 2.9 / 53,300 | netting_set/EE_B | 3.8 / 2,160 |
| `pricing:float8_e4m3fn`, block / none | 7.3 / 60,300 | 16 / 354,000 | 83 / 2,060,000 | trades/european-payer/EPE | 162 / 26,600 |
| `pricing:float8_e5m2`, block / none | 19 / 73,800 | 56 / 435,000 | 330 / 2,530,000 | trades/european-payer/EPE | 192 / 10,100,000 |
| `throughout:float32` | 0.003 | 0.014 | 0.069 | trades/bond/NPV | 158 |

Reading it:

- **Block offsets remove I-75's mechanism wherever it acts**: the cube (bond) and the curves,
  by factors of 100 to 30,000. What remains in FP8 is the format itself: its few bits on
  blocks that are not concentrated (an option's values, from zero up) and on the shocks, and the
  nonlinear figures (EPE, PFE) that turn rounding noise into a bias. These are the significant
  biases the table shows, and the evidence table's rows for FP8.
- **The relative error grows with the path count**, nearly as sqrt(N) or faster: on Sobol paths
  a smooth figure's statistical error falls almost as 1/N, while rounding noise that differs
  from path to path averages out only as 1/sqrt(N). Every format below float32 thus has a path
  count beyond which its error passes any fixed share of the statistical error (the evidence
  table's path ceiling, I-55); below 1% (the P6.2 rule), only float32 storage and compute, at
  these path counts.
- **float16 cube storage** stays below a fifth of the statistical error to 16,384 paths, and
  float16 curves near one; bfloat16 is three to ten times worse; FP8 is many statistical errors
  off, but without a bias in a concentrated mean (I-75), and its cube in an eighth of float64's
  memory plus a quarter for scales and offsets.
- **A paired sample of an eighth of the paths raises the relative error** of smooth figures here
  (FP8 cube: 17, 31 and 119 against 7.3, 16 and 83): it removes the bias of a mean, but adds the
  correction's variance, and on Sobol paths the statistical error it is measured against is
  small. It pays where a bias dominates, which block offsets have removed for concentrated
  columns.
- **Figures float64 holds fixed** (a netting set's ENE where every path is in the money, an
  EEE_B or EEPE_B at its t=0 value) are moved by at most 3.9e-5 of their scale in any case with
  offsets; they have no statistical error to measure a rule against, so the evidence table needs
  an absolute rule for them (I-55; since 2026-10-11 the fixed rule, 1e-6 of the scale, §10.4).

**Options** (`options`, 256 paths, 8 seeds; measured again on 2026-10-12 twice: at ORE's default
`ShiftHorizon` with the American's as-of window, I-32, then with each LGM calibrated once where
ORE builds the trade and kept on every path, I-49, engine digest `eec51de554711f73`). An option's
cube column reaches zero, so its blocks keep offset 0 and both layouts store it alike up to the
noise (FP8 e4m3 cube storage 3.9 with offsets, 4.1 without; float16 0.042 and 0.043); the curves
gain as in `linear` (FP8 e4m3 market 4.2 against 148, e5m2 23 against 506). float16 is below a
fifth of the statistical error in every stage (0.042 to 0.17); FP8 is 3.6 to 23 statistical
errors off with block offsets, the largest on the cash Bermudan's PFE and EEPE_B. Against the
rows measured with the LGM recalibrated on every path the same day: float16 in the simulation and
market stages went from 0.03-0.04 to 0.16 (the American's EPE_B), FP8 market storage without
offsets from 72 and 126 to 148 and 506 (the cash Bermudan's EEPE_B), the rest within the noise,
and each case runs in 4.6-7.2 s where it took 10-11 s. A model kept on every path does not follow
a path's rounded curves as a recalibrated one does, so rounding moves more exercise decisions
(a cash Bermudan exercised is 0 thereafter): the options' errors are exercise decisions as much
as values. Stochastic rounding is worse in all three stages.

**Market risk** (`market_risk`, 2,048 and 8,192 scenarios, 16 seeds). VaR and ES at 8,192
scenarios, worst relative error, nearest rounding: shifts stored in float16 0.028, bfloat16
0.12, FP8 e4m3 0.55, e5m2 0.91; P&L stored in float16 0.096, bfloat16 0.18, FP8 e4m3 1.95,
e5m2 5.4 (§15.3's one-portfolio measurement, FP8 e4m3 P&L 1.8 times the seed spread, agrees).
Stochastic rounding is worse for VaR and ES in 12 of 16 comparisons, its added variance
widening the tail. P&L attribution against float64, worst over the seeds: Spearman 1.000000 and
Kolmogorov–Smirnov 0.00098 for float16 P&L, 0.9924 and 0.0081 for FP8 e4m3, 0.9706 and 0.0166
for e5m2; the evidence table judges them against MAR32's thresholds as transcribed (A-11). The P&L's
mean is where a format's error is largest in units of its statistical error, being the
smallest of these errors on Sobol scenarios.

### 15.5 Compute below float32 (2026-10-10)

The kernel rewrite's two campaigns of the precision harness (`tests/support/precision_campaigns.py`,
`compute` and `compute_market_risk`; [compute.md](../../../compliance/precision-measurements/compute.md),
[compute_market_risk.md](../../../compliance/precision-measurements/compute_market_risk.md)), on CPU
(every format below float32 emulated, with the numbers a native unit computes, §8.3), 16 seeds:
the `linear` campaign's portfolio with one stage's products in each compute format, accumulated
and stored in float32 (the `float32` case is the floor of the accumulator), and every stage in the
format at once (`throughout`, stored in the format where it has one); market risk's revaluation
the same, its trades but the Bermudan. The storage campaigns were measured again on the same code:
every case computed in float64 equals its record of 2026-10-09 (§15.4) to three significant
figures, in `linear`, `options` and `market_risk` alike (282 rows), and `throughout:float32` fell
to 0.29 to 0.65 of its record.

**Rerun on 2026-10-11 with the shared operands in two terms** (A-23, [I-96](../known-issues.md#i-96)):
each format below float32 whose product is one pass measured twice, by its name (two terms) and
as its single pass (`<name>_1x`, the first run's numbers: every single-pass row equals the first
run's row of its format to the digits recorded, and `float32` and `bfloat16_3x` theirs). The
worst relative error over every figure and date, means and quantiles apart (the root mean square
precision error in units of the figure's statistical error), two terms, then the single pass:

| Case | 1,024 means / quantiles | 4,096 | 16,384 |
|---|---|---|---|
| `throughout:float32` | 0.0015 / 9e-5 | 0.0041 / 2e-4 | 0.045 / 6e-4 |
| `throughout:bfloat16_3x` | 0.0040 / 1e-3 | 0.019 / 4e-3 | 0.095 / 0.010 |
| `throughout:tensorfloat32` | 0.15 / 0.32 | 0.15 / 0.53 | 0.47 / 0.99 |
| `throughout:tensorfloat32_1x` | 0.57 / 0.20 | 3.5 / 0.41 | 12 / 1.8 |
| `throughout:float16` | 0.16 / 0.32 | 0.30 / 0.53 | 0.50 / 1.9 |
| `throughout:float16_1x` | 0.59 / 0.32 | 3.6 / 0.74 | 12 / 3.3 |
| `throughout:bfloat16` | 0.63 / 1.0 | 2.4 / 1.7 | 17 / 5.5 |
| `throughout:bfloat16_1x` | 2.2 / 1.5 | 11 / 3.4 | 45 / 3.8 |
| `throughout:float8_e4m3fn` | 21 / 6.3 | 133 / 13 | 677 / 40 |
| `throughout:float8_e4m3fn_1x` | 81 / 17 | 500 / 47 | 1,700 / 135 |
| `throughout:float8_e5m2` | 55 / 32 | 300 / 90 | 1,860 / 258 |
| `throughout:float8_e5m2_1x` | 120 / 42 | 650 / 118 | 3,000 / 341 |
| `pricing:compute:float32` | 0.0010 / 4e-5 | 0.0038 / 1e-4 | 0.011 / 3e-4 |
| `pricing:compute:float16` | 0.024 / 0.059 | 0.065 / 0.16 | 0.15 / 0.55 |
| `pricing:compute:float16_1x` | 0.31 / 0.091 | 1.9 / 0.23 | 6.4 / 1.0 |
| `pricing:compute:bfloat16` | 0.17 / 0.41 | 0.58 / 0.97 | 0.95 / 2.2 |
| `pricing:compute:bfloat16_1x` | 2.0 / 0.42 | 9.9 / 0.99 | 39 / 3.0 |
| `pricing:compute:float8_e4m3fn` | 2.6 / 2.8 | 9.5 / 5.4 | 30 / 14 |
| `pricing:compute:float8_e4m3fn_1x` | 45 / 15 | 281 / 42 | 948 / 121 |
| `pricing:compute:float8_e5m2` | 5.7 / 26 | 33 / 73 | 115 / 213 |
| `pricing:compute:float8_e5m2_1x` | 44 / 14 | 274 / 41 | 928 / 120 |

Market risk at 8,192 scenarios (VaR and ES, the quantiles), two terms (single pass): float32
5e-5, TensorFloat-32 and float16 0.0078 (0.014), bfloat16 0.069 (0.17), FP8 e4m3 0.26 (1.7), e5m2
0.42 (1.8), the bias's significance 2.0 to 6.3 (55 to 172); P&L stored in the format (§15.4):
float16 0.096, bfloat16 0.18, FP8 1.95 and 5.4. P&L attribution against float64, worst over the
seeds: Spearman 1.000000 to 0.999633 (single pass to 0.995006) and Kolmogorov–Smirnov 0.00024 to
0.0045 (0.011), FP8 e5m2 the worst, for the evidence table to judge against MAR32's thresholds
(A-11).

**A third term, or a wider shared operand** (A-23's open option), on the FP8 cases where two
terms leave a significant bias, measured at 4,096 and 16,384 paths (a scratch run of the
campaign with `Format.shared_terms` at 3, and at 8, which is as good as the shared operand
unrounded, the limit of any wider one): every case's means within 0.77 to 1.33 of its two-term
value, both ways (pricing in e4m3 at 16,384 paths 30 to 23, the market in e4m3 18 to 23), its
quantiles within 4%, and eight terms equal to three within 3%. On the product alone (a per-path
operand near 1 ± 0.3, 65,536 paths) the split is plain: e4m3's per-path rounding alone biases the
mean 15.0 standard errors, with the shared operand in one term 258, two 19.2, three 15.2; e5m2
26.8, then 405, 23.5, 26.2. A third term removes the second term's own rounding, which shows only
from about 16,384 paths on, and not FP8's rest, so it is not adopted ([I-98](../known-issues.md#i-98)).

**Timing** (RTX 5060, jax 0.10.2, 65,536 paths, interleaved, warm): where the card's units are
native (float16, bfloat16, TensorFloat-32) two terms take 0.96 to 1.09 of the single pass's time
on the engine's product shapes (the legs `[S, 40] x [40, 3]`, the market `[6, S, 1] x [6, 1, 45]`,
the bridge `[24, 24] x [24, S]`, and `[S, 256] x [256, 64]`) and 0.97 to 1.08 for a whole job
(the `linear` trades, every stage in the format): no cost measurable. The engine's products are
narrow on their shared side (3 to 64 columns), so dispatch and memory traffic, not the doubled
arithmetic, decide their time; a compute-bound product, the rollback's, is where the factor
two shows: 1.37 to 1.62 times the single pass's time on the same card (2026-10-13, §15.7), and the
timing on Ironwood and H100 (F-07) measures both.

Reading it:

- **float32 is better than before the rewrite**: every stage in float32 0.0015, 0.0041 and 0.045
  of the statistical error at 1,024, 4,096 and 16,384 paths, against 0.003, 0.014 and 0.069 on
  the code before (§15.4's `throughout:float32`). The difference form keeps the legs'
  cancellation in the level, and the levels meet the paths in two terms (§8.2). A first build
  that added a level rounded once to float32 to every path was ten times worse than the code
  before on a bond (its mean moved 0.04 near 1e6, significance 287), which the two terms fixed.
- **bfloat16 passes are float32's equal** within a factor two to four; **TensorFloat-32 and
  float16 measure alike** (10-bit operands both).
- **The single pass is all bias; two terms are not** (A-23, I-96). In one pass the error of every
  case below float32 is bias (relative bias equal to relative error, significance 100 to 2,000):
  each product's path-independent operand (the bridge's matrix, the step moments, the market's
  loadings, each flow's present value at the level) is rounded once for every path, so its error
  moves every path alike, and grows against the statistical error with the paths. In two terms one
  stage's means at 16,384 paths are 0.15 to 0.20 statistical errors off in float16 and
  TensorFloat-32 and 0.95 to 1.2 in bfloat16 (single pass 4.4 to 56), the bias's significance
  2.9 to 7.7 (98 to 1,300), so they grow far more slowly with the paths. Compute is now closer than
  storage in the same format, where it was 5 to 50 times farther: one stage stored in bfloat16 is
  2.0 to 2.9 statistical errors off at 16,384 paths (§15.4), computed in it 0.95 to 1.2; so
  `throughout:bfloat16`'s 17 is mostly its storage.
- **FP8 keeps a bias of its own** ([I-98](../known-issues.md#i-98)): one stage's means 8 to 34
  times closer than the single pass's, yet 18 to 147 statistical errors off at 16,384 paths. It is
  each path's own rounding: to nearest on a grid whose spacing doubles at every power of two, a
  smooth spread of values has a mean error that is not zero (−2.7% of the rounding's spread for
  e4m3, −5% for e5m2, 0.02% for float16). More terms do not touch it (above); rounding the
  per-path operand stochastically does, on the product alone (e4m3 at 65,536 paths, significance
  12.3 to 4.7, the rest the second term's own), at 40% more root mean square error.
- **Quantiles**: in market risk two terms bring VaR and ES 1.7 to 6.4 times closer. In the
  portfolio's PFE the 10-bit formats' rows are noise either way (no significant bias; 0.06 to
  0.54 statistical errors to 4,096 paths), and FP8 e5m2's pricing quantiles are 1.8 times
  farther with two terms: a payer swap's 5-year PFE, −41 statistical errors in one pass and −73 in
  two at 4,096 paths, the same sign, so the single pass's shared bias had offset part of FP8's own
  (I-98).
- **No recompiles**: every case, in both campaigns, compiles nothing after its first seed (168
  rows).

### 15.6 The evidence table (2026-10-11)

All five campaigns were measured again for the table on one revision (commit `adc4359` with
the evidence table's changes, engine digest `805c1a7f8cad963b`; CPU, jax 0.10.2; 16 seeds, the
`options` campaign 8), 2026-10-11/12, about 57 minutes in all. Every one of their 23,736 rows
equals its previous record bit for bit, those of 2026-10-11 (`compute`, `compute_market_risk`)
and those of 2026-10-09/11 (`linear`, `options`, `market_risk`): the table's changes move no
number, and nothing since the kernel rewrite moved a storage campaign's figure. No case compiles
after its first seed. The table is
[compliance/precision-evidence.md](../../../compliance/precision-evidence.md); A-11's rules are
§10.4's.

**Measured again on 2026-10-12** after I-32 (ORE's default `ShiftHorizon`, an American's as-of
window on the paths): the two campaigns with Bermudans or Americans, `options` and
`market_risk` (engine digest `15de886690232ba1`, 46 minutes), the table regenerated. Every
verdict is as before but two rows, one gaining a count and one losing one: the options' netting
set EPE_B under float16 market storage with block offsets passes at 256 paths, and market risk's
ES_99 with the P&L stored in float16 by stochastic rounding passes to 2,048 scenarios, no longer
8,192 (reported, not redefined: its RMS error there went from 13.8 to 45.4 of a 1,873 statistical
error, one of the 16 seeds 173 where the record before had 26 at most, with no significant bias,
6.4 ± 11.6, as when the random rounding of the stored P&L reorders scenarios at the tail's edge;
the same case's ES_97.5 and the nearest rounding's ES_99 still pass to 8,192). Every market-risk case stays in MAR32.42's green zone at every seed. The other rows
moved within their statistics' noise.

**Measured again on 2026-10-12** after I-49 (each option's LGM calibrated once where ORE builds
the trade, no longer on every path): `options` (engine digest `eec51de554711f73`, 17 minutes;
`market_risk` revalues on the as-of date, which the change does not touch), the table
regenerated. Every verdict is as before but one, the options' netting set EPE_B under float16
market storage with block offsets, which no longer passes at 256 paths (it had gained the count
in the measurement before; its worst figure, the American's EPE_B, is 0.16 of its statistical
error, §15.4's options).

**Verdicts** (ceilings in paths, or scenarios for market risk; the largest measured counts are
16,384, 256 for the options and 8,192 for market risk):

| Combination | Ceiling |
|---|---|
| One stage stored in float32 | 16,384 for every scope, but the bond's cube mean under float32 cube storage, 4,096; market risk 8,192 |
| Every stage in float32 (`throughout:float32`) | 4,096 for every scope |
| One stage computed in float32 or `bfloat16_3x` | 4,096 to 16,384 (the simulation's 4,096; a European under `bfloat16_3x` in the market or throughout, 1,024) |
| Market risk: P&L computed in float16 or TensorFloat-32 (two terms) | VaR and ES 8,192; the portfolio's mean P&L 2,048; some trades' mean P&L at no count |
| Market risk: shifts or P&L stored in float16; P&L computed in bfloat16 | ES 8,192, but ES_99 with the P&L stored by stochastic rounding 2,048 since 2026-10-12 (VaR 2,048 to none); the trades' mean P&L at most 2,048 |
| Anything else below float32, in any stage | No ceiling for the cube's mean, EPE, PFE or a P&L mean. The first failing figure of a case, at the smallest count measured, with block offsets: 0.010 to 0.37 statistical errors for float16 and TensorFloat-32, 0.016 to 2.2 for bfloat16, 0.28 to 119 for FP8 (single passes the high ends; without offsets up to 74,000) |

What does pass below float32 in the portfolio campaigns is of two kinds. Figures no format
moves: a bond's ENE (zero on every path, whatever the rounding) and its EEE_B, EEPE_B and Basel
EEPE where they are held at their t=0 value pass by the fixed rule in every format, FP8
included, and the options' ENE in float16. And scattered figures in float16: pricing computed in
it passes a swap's Basel EPE and EEPE to 16,384 paths and a European's EPE and cube mean to 1,024.
In two terms and with block offsets the 10-bit formats fail by a factor of one to ten at 1,024
paths, and pass where a figure's statistical error is large. Every market-risk case is in
MAR32.42's green zone at every seed (the smallest Spearman 0.939 and the largest KS 0.030, both
for P&L stored in FP8 e5m2 with stochastic rounding at 2,048 scenarios): the test is a floor that
every format clears, as A-11 expected, and the statistical rule decides.

**Reading it.**

- **The 1% rule leaves little room below float32.** A figure passes when its precision error is a
  hundredth of its statistical error, so the format adds a ten-thousandth to the variance: an
  acceptance rule for results that must not move, not a measure of equal accuracy. Under it,
  float32 storage and compute, and the three bfloat16 passes, are as good as float64 to 4,096 or
  16,384 paths; float16 and TensorFloat-32 are within a factor of three of it at 1,024 paths and
  pass the market-risk quantiles; bfloat16 and FP8 do not pass a figure they move. The research
  question (3.18 in the roadmap: many low-precision paths against fewer float64 paths at equal
  accuracy) needs a different reading of the same measurements: a format whose root mean square
  precision error is r statistical errors at N paths is worth float64 at N / (1 + r²) paths, so
  float16 at 1,024 paths (r about 0.03) loses nothing measurable and FP8 (r about 3) is worth a
  tenth of its paths. The table keeps A-11's verdicts; the harness's relative errors, kept per
  point, are what the research result reads.
- **Path ceilings are coarse**: three path counts a factor of four apart, two for market risk, one
  for the options. A ceiling says the row passes there and fails at the next count, or that the
  next count was not measured (16,384 and 8,192 are the largest measured, not ceilings found).
- **The options' rows were thin** (256 paths, 8 seeds, the rollback before its rewrite): they were
  measured again after `ShiftHorizon` and after the recalibration fix (both 2026-10-12), and since
  the rollback's rewrite (2026-10-13) at the linear campaign's path counts and seeds, with their
  compute formats (§15.7).

**A robust standard error for the paired correction, by blocks of paths** (I-55's proposal),
measured on 2026-10-11 against a threshold fixed before: on the `linear` campaign's paired cases (a
paired eighth of the paths, every stage stored in each scaled format, nearest rounding, block
offsets; 16 seeds; EPE and ENE of the netting set and every trade, per date), the standardized
errors of the two-level estimate against the run's own float64 figure, with the correction's
standard error by batch means over the 32-path storage blocks against the one treating paths as
independent. Threshold: adopted if its standardized errors' root mean square is within 0.75 to
1.33 and its coverage at least 0.90 for every case at 1,024 and 4,096 paths.

| Stored | Paths | Independent paths: root mean square, coverage, largest | Blocks of 32: root mean square, coverage, largest |
|---|---|---|---|
| shocks, float16 | 1,024 / 4,096 | 0.94, 0.97, 3.2 / 1.83, 0.95, 34 | 1.58, 0.86, 16 / 1.94, 0.93, 37 |
| shocks, bfloat16 | 1,024 / 4,096 | 1.48, 0.94, 18 / 1.65, 0.91, 15 | 2.51, 0.85, 28 / 1.70, 0.89, 16 |
| shocks, FP8 (both) | 1,024 / 4,096 | 0.94 to 1.09, 0.94 to 0.97 | 1.01 to 2.01, 0.85 to 0.94 |
| curves, float16 | 1,024 / 4,096 | 3.30, 0.95, 48 / 3.22, 0.92, 74 | 3.09, 0.84, 37 / 3.24, 0.91, 72 |
| curves, bfloat16 | 1,024 / 4,096 | 1.55, 0.93, 19 / 1.57, 0.92, 15 | 2.67, 0.83, 42 / 1.69, 0.90, 19 |
| curves, FP8 (both) | 1,024 / 4,096 | 0.95 to 1.04, 0.94 to 0.96 | 1.05 to 1.84, 0.86 to 0.94 |
| cube, every format | 1,024 / 4,096 | 0.95 to 1.03, 0.94 to 0.96 | 1.03 to 1.70, 0.84 to 0.94 |

It fails: blocks of paths are no better anywhere, and at 1,024 paths (four blocks, three degrees
of freedom) much noisier. The misses are not correlation within a block but rare large paired
differences (single errors to 74 standard errors), with float16 and bfloat16 shocks or curves and
not with FP8, whose errors are many and small: a rounded curve flips a payoff on one path, and a
paired sample that misses those paths states too small an error. No standard error from the
paired paths alone sees what they miss. Independent replicates estimate variance but
also miss rare events; I-111 now declines nominal intervals without a validated tail bound. The evidence table does not use these standard errors, and
the result says so where they understate.

### 15.7 The rollback in difference form (2026-10-13)

The Bermudan/American induction as matrix products in difference form (§8.2, I-83): one rollback
operator per step and the cashflows' functions of the state shared by every path, each path's
deviation from the induction on the level curves, a cube's dates one program, its paths in blocks
within `INDUCTION_MEMORY_BUDGET`. All on CPU unless the GPU is named; `jax==0.10.2`.

**Equal at float64.** Each path of a cube is the induction run on that path's own curves and
fixings (the code before, per path) to 1e-12 (`tests/test_rollback_difference_form.py`); the
shared portfolio's option cubes against the code before 8.2e-15 of their scale at both grids;
ORE's engine today (1e-11) and ORE's cube on its paths (1.9e-11) as before; the golden snapshot
within the margin (§13.1). float32 is closer to float64 than before: the option cubes' largest
error 2 to 6 times smaller and their bias over the paths 5 to 20 times, the level being float64
where the code before ran the whole rollback in float32.

**Memory and time.** XLA's memory analysis of the compiled induction, per path at float64: 9.7
KB (a Bermudan) and 26.7 KB (the American) on the demo's grid of 193 nodes, 14.9 and 41.4 KB on
ORE's 301; the code before 6.1 MB on the demo's grid, a `[nodes, nodes]` operator per path and
step. The demo's job against path count is profiling §2.0's table: 262,144 paths in 2.98 s warm
and a 2.0 GB peak on the CPU and 2.37 s and 0.87 GB on an RTX 5060 at float64, where the code
before ran out of memory at 65,536 and 1,024 paths; 4,096 paths in 0.26 s against 13.2 s. Blocks
of paths cost nothing on the CPU (3.70 s and 9.5 GB in one block at 262,144 paths) and a third
of the warm time on the GPU where 65,536 paths take two (the budget is now chosen per device, §17).
Market risk's Bermudan revaluation 0.035 ms per scenario at 16 points per standard deviation and
0.18 ms at 64, against 5.6 and 75 ms; the `market_risk` campaign took 34 s against about 40
minutes.

**The campaigns** (engine digest `b0b17902865e9627`, CPU, 16 seeds, 2026-10-13, about 2.6 hours;
no case compiles after its first seed). `options` at 1,024, 4,096 and 16,384 paths (256 and 8
seeds before), `compute_options` new (the pricing stage, the induction, computed in each format,
and every stage in it), `compute_market_risk` with its Bermudan, `market_risk` measured again (its
Bermudan's and portfolio's rows moved by at most 8e-11 of their relative error, the float64
reference's rounding; 144 of 360 rows identical), and `linear` and `compute`, whose 13,680 and
8,640 figure rows equal their records bit for bit, gaining their Europeans' exercise rows. Under
A-11's 1% rule:

| Combination (options' figures) | Ceiling |
|---|---|
| Pricing computed in float32 | 16,384 (the American's figures 4,096) |
| Pricing computed in `bfloat16_3x`; every stage in float32 or `bfloat16_3x` | 4,096 (under `bfloat16_3x` the cash Bermudan's ENE at none, I-113) |
| One stage stored in float16, or computed in float16, TensorFloat-32, bfloat16 or FP8, or every stage in them | none: 0.05 to 0.33 statistical errors at 1,024 paths for float16 and TensorFloat-32, 0.6 to 1.6 for bfloat16, 3 to 380 for FP8 |
| Market risk: the Bermudan's P&L computed in float16, TensorFloat-32 or bfloat16 (two terms) | 0.009 to 0.18 statistical errors at 2,048 scenarios, bias significance below 2: as the other products' |

**Reading it.**

- **The rollback's rounding accumulates.** The induction rounds each path's columns at every
  step of its rollback, 6 for a Bermudan and about a hundred for the American, and round to
  nearest of a smooth function on a grid has a mean error that is not zero (I-98's mechanism, so
  far FP8's alone): with the pricing stage in float16 the American's mean figures are 7.0
  statistical errors off at 16,384 paths, significance 99, the Bermudans' 0.49. Two terms of the
  shared operand (A-23) cut the Bermudans' error twelvefold against the single pass (0.49 against
  6.1) and not the American's (7.0 against 2.7): its bias is its per-path operand's, which an
  unbiased rounding of the per-path operand would address (I-98). Market risk's
  revaluation today, a few steps, shows none of it.
- **float32 is as good as float64 for the options to 16,384 paths** (their worst figure 0.011 of
  its statistical error), its errors significant (up to 460 of their standard errors at 16,384
  paths: float32's rounding of each step is systematic too) at a scale the 1% rule never sees.
- **Exercise decisions flipped** (decision A-24, `ExerciseRow`; the evidence table's section): a
  stage stored or computed in a 10-bit format flips at most 0.03–0.05% of an option's paths on a
  date, bfloat16 compute 0.25%, FP8 storage of the states or curves 1.3 to 6% (the cash Bermudan
  1.3–1.5% under stored states, the physical one 3% and the American 6% under e5m2 curves without
  offsets), FP8 compute 2.8 to 11% (the American's 11% throughout e5m2). The cube's own storage
  flips none: its decisions are made before it is stored. The Europeans of the `linear` campaign
  flip at most 0.06% in the 16-bit formats and 0.6 to 6.9% in FP8.
- **The storage sanity bounds judged without the flips** (§13.5): bfloat16 simulation storage's
  bounds fell thirtyfold, to 2.3e-3 and 1.1e-5 of notional; FP8's largest errors are rounding.
- **The second term's cost.** Emulated (the `compute_options` campaign's warm seconds, its coarse
  grid of 49 nodes, CPU), two terms took 0.76 to 1.34 of the single pass's time, in no order
  across formats and path counts: too small a product for the doubled width to show through the
  CPU's emulation and dispatch. Native, on the RTX 5060's float16, bfloat16 and TensorFloat-32
  units (the options' path pricing at 65,536 paths, warm, the two alternated five times, medians):
  on ORE's grid of 301 nodes two terms take 1.37, 1.59 and 1.62 times the single pass's time
  (11.2 s against 8.1 s in float16), on the coarse grid 1.23, 1.36 and 1.32. Unlike the engine's
  narrow products (0.96 to 1.09, §15.5) the rollback is compute-bound, and the second term of its
  shared operator costs a third to three fifths of its time; it is where the single pass's bias
  (A-23) is traded against speed, which the timing on Ironwood and H100 (F-07) reads.

## 16. Decisions this document implements

Recorded in [compliance/decisions.md](../../../compliance/decisions.md), 2026-10-01.

| # | Decision |
|---|---|
| D-9 (revised) | Precision is a free choice for the adjustable stages, per run, product and trade; a run is never refused on precision; unvalidated combinations carry a warning |
| A-9 | Adjustable precision, the old mechanism removed once the replacement works (the structure stage, 2026-10-01 to 10-04) |
| A-10 | Only the heavy scenario-axis stages are adjustable; calibration, t=0, Greeks and reductions stay float64, built on the same mechanism so a stage can be opened later |
| A-11 | Acceptance standard: Basel III tests where they exist (P&L attribution, plus the Basel plan's P6.2 rule); the P6.2 statistical rule, labelled as engineering, for figures Basel does not cover |
| A-12 | The old precision configuration is refused, not translated |
| A-13 | Means are corrected by the two-level estimator; quantiles are measured |
| A-14 | Execution: API and one single-threaded engine worker process per host, through a durable SQLite queue |
| A-15 | Precision per stage, overridable per product and per trade (trade over product over stage) |
| A-16 | One kernel implementation for every precision; float64 moves at rounding level, within the margin, after ORE parity passes (the first families 2026-10-10, §8) |
| A-22 (2026-10-06) | A matrix product's precision is its compute format's, stated by the product (`engine.precision.matmul`, §6.7), never a process-wide default (the stated product precision (2026-10-06)) |
| A-23 (2026-10-11) | Below float32 a product carries the operand every path shares in two terms of the format; the single pass is a format of its own, `<name>_1x` (§6.1, §8.3, §15.5) |
| A-24 (5) (2026-10-12, implemented 2026-10-13) | Exercise decisions are measured, not hidden in bounds: the option wrapper's exercise state on every path and date is in the result, the harness measures the share each format flips, and the storage sanity bounds judge rounding on the cells not flipped (§13.5, §15.7) |

Engineering defaults, changeable without a decision: block size 32 along the scenario axis;
the block layout (`offset="block"`) and the rounding (`"nearest"`), both chosen by the storage
harness (2026-10-09, §15.4); `paired_fraction` 0 by default, 0.02 suggested for reduced-precision
runs; the evidence table's location (`compliance/precision-evidence.md`) and its matching rules
(§10.4); the fixed-entries rule's share, 1e-6 of a figure's scale (P6.2's rule for figures
without randomness, applied where float64 holds an entry fixed; 2026-10-11).

## 17. Exposure storage and peak memory (I-105, 2026-10-15)

Executed plan:

1. Prove the defect with a regression: a request omitting the output cube must never
   access `PortfolioValuation.cube`, while retaining every exposure and precision figure.
2. Reduce stored columns individually at float64, using the existing exposure definitions
   and exact sorted PFE. Keep a running float64 netting sum and a separate paired sum;
   keep the paired valuation as columns too. Complete each trade's reductions before loading
   the next, bounding live temporary arrays even under asynchronous device dispatch.
3. Add a Python output switch, defaulting to today's cube behavior, and wire HTTP
   `cube_output="none"` to it. Materialize requested cubes after the reductions, in one
   compiled assembly. Preserve inline and artifact results and scenario-risk metadata.
4. Size the rollback's path blocks by a stable device memory budget, with the existing
   CPU fallback. Avoid decisions based on fluctuating free memory and budget each device's
   share rather than the global path count.
5. Test mixed storage, paired estimates, cancellation, empty portfolios, compilation reuse
   and scenario sharding. Compare pre/post snapshots, measure fresh-process peak memory
   over increasing path counts at float64 and FP8, run ORE parity and the full suite, and
   record results and any remaining limitations here and in the issue ledger.

The arithmetic of pricing and storage stays unchanged. The netting reduction changes its
addition order only; accuracy is judged by the existing margins, not bit identity.

**Implementation.** `column_profiles` uses `exposure_profile` for both standalone trades
and the netting set. A module-level jitted decode/add kernel donates only the running sum;
stored input columns remain valid for a requested cube. Each profile is synchronized on
device before proceeding, including its paired estimates, so temporary decoded arrays
cannot accumulate behind asynchronous dispatch. Small reduced arrays remain in the result;
path arrays are not transferred to the host. The extra reduction working set is a bounded
number of `[paths, dates]` arrays, independent of trade count; the stored columns themselves,
the scenario market, option exercise states and the paired sample still take memory.

`PortfolioRequest.include_npv_cube=True` retains the Python default. False returns an empty
`(0, 0, 0)` cube while keeping exposure and precision figures. HTTP `outputs.npv_cube: "none"`
(`cube_output="none"` until 2026-10-18) sets it to False; inline and artifact output keep True. `PortfolioValuation.float64_cube()`
decodes and stacks requested output in one compiled program, after reductions and after
releasing the paired sample. Its specialization depends on storage and shapes, not trade
identities or values. The decode/add kernel similarly reuses programs across trades.

The rollback's budget is an eighth of the smallest local accelerator allocator's `bytes_limit`,
with the existing 1 GiB fallback on CPU or when the backend supplies no limit. It depends
on capacity, never currently free memory. Path blocks now budget each device's share; market
risk uses the same budget for its batches. The RTX 5060 reports 6,410,993,664 bytes of
allocator capacity and receives 801,374,208 bytes for the induction. This is a working-set
budget rather than a guarantee that an arbitrarily large stored market or requested cube fits.

**Accuracy and regression evidence.** The initial regression failed on the pre-change
pipeline at `load(valuation.cube, float64)` for a request omitting output, and the HTTP
conversion test failed because there was no engine output switch. The new tests cover
all storage formats mixed in one netting set, both roundings, partial storage blocks,
paired corrections and quantile estimates, offsetting and empty portfolios, malformed
column shapes, cube assembly, HTTP modes and device budgets. Growing from one to nine
trades and changing their values compiles nothing after warming the run and paired shapes.
The existing four-device scenario-sharding test passes with streamed exposure too.

The before/after snapshot (`tests.support.exposure_storage_check --snapshot PATH`) covers
675 numeric outputs, all identical: the shared sloped portfolio's swaps, European, Bermudan,
American and bond, Greeks included, at float64, float32 throughout and mixed FP8/float32
storage with a paired sample. Pricing and storage kernels are unchanged; their evidence rows
are not regenerated. A targeted rerun of the linear precision campaign checks FP8 pricing
storage at 1,024 and 4,096 paths, with both roundings, both layouts and paired cases.
At three seeds this rerun produced 720 figure rows, 48 coverage rows and 12 case/path
runs, with zero recompilations after warming each shape. Full-suite and platform results
are recorded in [known issues' verification status](../known-issues.md#verification-status).

**Peak memory.** Fresh processes of `tests.support.exposure_storage_check`, 24 alternating
payer/receiver swaps, 12 dates, today's sloped market, no Greeks or paired sample, output
cube omitted. Only pricing storage changes; simulation and market remain float64. The
baseline is the original engine at `c3d7c6a`. Whole-process Windows peak working set in MB
(decimal), including preparation, compilation, pricing and reductions:

| Paths | Float64 before | Float64 after | FP8 e4m3 before | FP8 e4m3 after |
|---|---:|---:|---:|---:|
| 4,096 | 605.8 | 599.3 | 607.7 | 586.4 |
| 16,384 | 701.6 | 661.2 | 678.9 | 644.2 |
| 65,536 | 1,092.7 | 1,053.3 | 1,052.0 | 938.4 |

RTX 5060 peak device allocation in MB, with the same fresh-process workload:

| Paths | Float64 before | Float64 after | FP8 e4m3 before | FP8 e4m3 after |
|---|---:|---:|---:|---:|
| 4,096 | 152.3 | 152.3 | 152.3 | 152.3 |
| 16,384 | 242.7 | 201.3 | 201.3 | 201.3 |
| 65,536 | 725.6 | 459.3 | 609.7 | 394.3 |

At 65,536 paths FP8's peak device allocation falls 35%. These are end-to-end peaks,
not a claim that the entire job shrinks by
the storage format's byte ratio. The scenario market, pricing and allocator high-water
marks limit the gain; requested cubes necessarily still allocate their full float64 output.
Reproduce a point in a fresh process with `python -m tests.support.exposure_storage_check
--paths 65536 --storage float8_e4m3fn`; `--paired 0.125` adds the paired sample and `--cube`
requests output. Measurements are local hardware evidence, not TPU/H100 speed results.

**Rollback budget and timing check.** The small profiling demo's five trades at 65,536
paths, Greeks off and the full output cube requested, run directly through the Python
entry on the GPU. Three synchronized warm observations: 0.97–1.10 s before, 1.02–1.12 s
after; zero warm compilations and identical cube, EPE and ENE. Peak device allocation is
660 MB before and 609 MB after. Initial calls took 36.65 s and 8.60 s respectively, with
compilation/cache loading included; the disk cache already held test programs, so those
are not a controlled cold-compilation comparison. A trial budget of half the allocator
limit raised the option-workspace peak to 2,219 MB while warm time remained about 1.04 s;
the conservative eighth was chosen from this measurement. The hardware timing campaign
must measure the budget/time/peak tradeoff on TPU and H100 rather than assuming a larger
block is faster.

**Other findings.** The benchmark extra already in the working tree declared matplotlib
without a constraint, failing the full suite's dependency check. The installed Linux
benchmark version, 3.11.2, and its previously unpinned dependencies are now constrained.
The precision harness accepts two seeds, sufficient for its figure errors, but paired
total-coverage checks use a leave-one-out sample variance and need at least three: with
two it emits NumPy degrees-of-freedom warnings and reports zero total checks. Use at least
three seeds for paired coverage. I-111 now reports unavailable diagnostics for two
outer seeds, without invalid sample-variance warnings.

**Remaining plan at this measurement (2026-10-15).** I-105 is closed. The following
section records the subsequent rounding, bounds and independent-replicate implementation;
I-111 subsequently adopts the conservative interval-availability rule. Supported
confidence intervals remain future capability F-10; FP4 remains deferred (F-07).
Equal-accuracy timing on TPU/H100 and multi-host execution remain planned; omitting the
cube is now an explicit output choice, so future timing comparisons must use equal output
settings on both sides.

## 18. Existing-format rounding and independent errors (2026-10-16)

`StagePrecision.compute_rounding` selects nearest (default) or stochastic rounding of
single-grid per-path product operands below float32. Shared operands retain two terms.
Keys and sample identities are traced data; arithmetic formats and rounding policy are
pytree metadata. Dates, products and rollback steps use independent derived streams;
blocking and sharding preserve sample identities. The control reaches Python and HTTP,
and a setting a product-free market-risk stage cannot read is refused by name.

A long option's continuation is projected onto zero before exercise/LongShort; physical
delivery can still produce a negative underlying. Market-risk changes are projected
before storage and after decoding, without subtracting a large reconstructed base.
The zero-bound rule is unchanged in the evidence table (I-113).

Nearest and stochastic compute rounding have distinct evidence keys, as do replicate
counts. `rounding`, `rounding_options`, `rounding_market_risk` and `replicated_storage`
measure their errors; the affected option/market-risk campaigns are refreshed. Stochastic
rounding is conditionally unbiased at an operand, but can increase nonlinear payoff bias
and variance. Failing FP8 rows remain failures. Nearest remains the default baseline.
At 4,096 paths the float16 rollback's worst bias significance falls from 41.4 to 2.66;
FP8 e4m3 market-stage European EPE from 5.32 to 1.80. That does not establish A-11 acceptance.

The full design, verification and conservative interval-availability rule are in
[precision-accuracy.md](precision-accuracy.md). FP4 and quantile correction remain separate
later research, with the current raw quantiles retained.

## 19. The research timing: wall time per figure at equal accuracy (F-07, 2026-10-19)

The first research result (F-07) times the existing formats against float64 at equal
accuracy. Its harness is `tests/support/precision_timing.py`; its method is fixed here before
anything is measured on the hardware of record (Ironwood, H100), and a format that does not
pay is a result.

**The deadline comparison (implemented 2026-10-20).** [`benchmarks.comparison`](../../../benchmarks/README.md)
composes this harness with `precision_harness.py` and `demos/benchmark_reporting.py`, reusing
campaign inputs and estimators. It joins CPU-reference errors, whole-job timings, memory
availability, dispatch and exact reproduction settings. Both bounded workloads smoke-checked
on CPU and the RTX 5060; target-device qualification and larger-count research remain open.
The [plan and outcome](comparison-report.md) record the implementation and its limits.
Use that report for the remaining research work below:

- Run one single trade and a representative portfolio of simple trades and options, using
  existing measured engine/grid settings. Compare CPU FP64/FP32 references and accelerator
  FP64/FP32, native FP8 and BF16 where supported. State the mixed-precision policy: A-10's
  fixed stages stay FP64. Greeks need no precision change and need not be requested.
- Reuse identical source shocks for matched-scenario precision comparisons, distinguishing
  generation differences from arithmetic differences. Keep named trade scenario PV and
  portfolio P&L/VaR/ES errors, including absolute errors near zero, with accuracy targets and
  the FP64 reference's own sampling uncertainty. Use multiple independent seeds and scenario
  counts for accuracy-versus-time; do not turn empirical diagnostics into certified bounds.
- Join cold, compile and synchronized warm whole-job time with peak memory and execution
  evidence. Count host preparation, calibration, conversions, transfers, any pairing,
  aggregation and required output. State whether scenario generation is inside the timed
  analytic or supplied beforehand; include generation when claiming simulation throughput.
  Hold output settings and timing boundaries equal, including the fixed FP64 work.
- Qualify native FP8 on the actual device and representative engine shapes using device
  checks and lowering/trace evidence, recording fallbacks, hardware, revision, libraries and
  process flags. The format-level native label alone does not establish that every product
  reached FP8 units. The verified cuBLAS route can supply the H100 result while upstream XLA
  work continues; TPU FP8 needs its own verification and device dispatch.
- Smoke-check at small sizes, then include counts from 65,536 upward as memory permits to
  determine when scenario work dominates on the target. Keep failures, unavailable memory
  metrics and targets not reached explicit. Publish a scoped first-device report while
  other hardware and cluster scaling proceed. Extend that report to measured device/host
  counts before claiming cluster speedup. Tuning and exhaustive sweeps follow the report.

INT8, new multi-pass methods/Qwix, configurable precision for today's PV and Greeks, FP4,
quantile correction and confidence-interval extensions are outside the deadline. The
roadmap assigns the existing-format comparison first and retains separately planned later
experiments; none is a prerequisite to obtaining an honest accuracy/speed result.

**What is timed.** Each case of a campaign of record (the precision harness's campaigns,
`tests/support/precision_campaigns.py`; by default its `throughout:` cases, every stage in
one format) and float64 at each path count, in a process of its own, the persistent
compilation cache off. Each seed's run is the whole job, its scenarios drawn and `campaign.run` (host preparation,
calibrations and every fixed float64 stage, the simulation, pricing, a paired sample if the
case asks for one, the reductions), the clock stopped once its figures are on the host (each
array transferred, so the device has finished). Both sides compute the same outputs (the
campaign's figures, the cube included where the campaign reads it). The first seed's run is
the cold run, its XLA compile time stated apart (`backend_compile_duration`); the median of
the other seeds' runs, which compile nothing (recorded), is the case's time. Each process
also records the device's peak allocation, the grid engine's induction budget and how each
compute format's products ran there (`product_execution`, native or emulated: the timing of
an emulated format measures the emulation, and says so).

**Accuracy, measured.** Per figure entry, the truth is float64's mean over the seeds at the
largest path count. A case's error at N paths is the root mean square over the seeds of its
figure less the truth: the statistical error, the precision error, their covariance and any
bias, as observed on the Sobol paths (RQMC), no convergence rate assumed. A figure's error is
the root mean square over its entries float64 does not hold fixed (`precision_harness.FIXED`),
relative to its scale. The truth's own error, float64's spread at the largest count over the
square root of the seeds, is why that count is never a target.

**Equal accuracy.** For each float64 path count but the largest, the target is float64's
error there. A case reaches it at the path count where its error falls to the target,
log-linear between the two measured counts that bracket it (the first measured count when it
is already there), and costs its warm time there, log-linear between measured counts alike;
a case whose error stays above the target at every measured count does not reach it. The
speedup is float64's time over the case's. A job's figures must all be equally accurate, so
its speedup is the worst over its figures; the record keeps every figure's.

**The induction budget** (§17). `--budget-divisors D ...` times every case at the largest
path count again with the grid engine's budget at `1/D` of the accelerator's allocator limit
(1/8 by default): the time and peak-memory tradeoff per device, read from the same record.

**Where it runs.** On any devices the process has: `python -m tests.support.precision_timing
compute --paths 4096 16384 65536 262144 --seeds 8`, then `compute_options` and
`compute_market_risk`, writes `compliance/precision-timing/<campaign>-<device>.json.gz` and its
summary. Multi-device: the scenario axis is split as any run's (`engine.market_simulation.
sharding`, I-61), so a host's devices time one job together; a pod slice's hosts run jobs as a
worker set (`jax-risk-worker --distributed`, §11.3), whose queue records each job's start and end.
A CPU or a consumer GPU (float64 at 1/64 of its float32 rate) gives local evidence, not the
research result.

**The first runs (2026-10-19, local evidence).** `compute` and `compute_market_risk` at 6 seeds on
the CPU and the RTX 5060 (its products through cuBLAS, FP8 native), every case compiling nothing
warm (`compliance/precision-timing/`). On the CPU, whose formats below float32 are emulated, no
format pays: each case's worst speedup at equal accuracy is 0.34 to 1.04. On the card a whole job
takes 0.41 to 0.48 s whatever its format and paths, 1,024 to 16,384 (market risk 0.14 to 0.40 s;
worst speedups 0.39 to 1.13, the times' noise): the
fixed work of a job, its host preparation, float64 stages and dispatch, is all of it at these
sizes, and its float64 runs at 1/64 of the card's float32 rate. FP8 reaches float64's accuracy
on few figures (it misses 50 to 55 of the portfolio's 55 at the targets, 4 to 8 of market risk's
10). So the research result needs path counts where the paths' own work dominates the job (from
65,536 up), and reads FP8 against the figures it can reach.

# Known Issues

Open defects and important shortcomings: what each does to a number or a caller, and what
closing it takes. The order of work is in [roadmap.md](roadmap.md); the rules for this file
(statuses, severities, how to add and close an entry) are in [README.md](README.md). Fixed
issues keep one line in the [closed ledger](#closed).

Read this before trusting a figure: every trade of the shared portfolio equals ORE's
exposure simulation path by path (swaps, Europeans and bonds since 2026-10-07, Bermudans and
Americans since 2026-10-12, [I-49](#i-49)); the bump Greeks equal ORE's sensitivity analytic and
a calibrated Bermudan's market-risk P&L ORE's recalibrating historical P&L (I-51, I-112,
2026-10-17); the AD Greeks are ORE's in the limit of a vanishing shift, but for a grid-priced
option, whose price is piecewise smooth, the result says how they differ (I-78). Beyond the
shared portfolio, what is compared with ORE and what is not is the
[parity matrix](../reference/ore-parity.md#the-parity-matrix) (I-116, 2026-10-18).

## Verification status

Last full run, 2026-10-18, on the run request (I-87, I-88, I-89, I-94, I-95, I-103, I-76, I-77,
I-91, I-106). **3,887 collected** on Windows: the previous 3,653 plus 234, reconciled by test id
against the code before: removed, the standalone bootstrap's four files (48 ids, I-94), the
retired routes' tests and the standalone cases of `tests/test_root_solvers.py`; added,
`tests/test_run_request.py` and `tests/test_input_checks.py`, the run-request versions of the
HTTP tests, the queue's migration, idempotency, cancel, retention, lock and supervisor tests, and
the completeness test's value check (195 parametrized cases, I-88). Every green run below printed
its summary line:

- **Windows full suite**, `.venv/Scripts/python.exe -m pytest tests/ -q -n 6 --durations=15`:
  **3,883 passed, 3 skipped, 1 failed**, 25m28s. The failure was a real race in the new queue
  code: processes opening a file together had one fail switching it to WAL (`database is
  locked`; SQLite does not wait for that lock). Fixed (`JobQueue._use_wal`), measured 38 failed
  opens in 720 before and 0 after, the test strengthened to 30 trials per kind (red 3 of 3 times
  on the code before the fix, green after); the eleven files it and the line-ending restore
  touched rerun: **502 passed**.
- **Fast tier under strict dtype promotion** (`JAX_NUMPY_DTYPE_PROMOTION=strict`, as CI's
  `fast-strict-dtypes` job runs it): **3,646 passed, 3 skipped, 0 failed**, 4m08s at `-n 6`, on
  the final code.
- **Linux** (rule 5: the worker's lock path, its `fcntl` lock and owner record, the supervisor),
  WSL2 Ubuntu, `JAX_PLATFORMS=cpu`, `-n 2`, the final code: the worker, API, run-request, input,
  demo, layering and completeness files, **479 passed**.
- **Lint**: pyflakes passes on every changed Python file.
- **The demo over HTTP**: `demos/demo_api.py` ran end to end against a real server and worker
  (one run of four analytics, the cube by reference with every hash checked, an idempotent
  market-risk run).
- **Not run**: the GPU (nothing device-specific changed); the golden snapshot (no kernel or
  number path changed: each analytic is its entry point's direct call, shown bit for bit equal
  by `tests/test_run_request.py`; the cube's new branch prices an expired European as the zeros it
  priced before; the rest is host-side checks and labels); the precision campaigns (no kernel
  changed).

Red first: on the code before, an unsorted curve priced the European payer at 50,679.68 for
51,133.41 and a late European expiry raised `StopIteration` (I-103); a half-migrated version-1 file
failed to open (`duplicate column name: kind`) and two queue files on one host each took a worker
lock (I-106); every submission was a new job (I-89); the value check fails on a schema that drops
a field (I-88).

The fast tier alone is not a full verification. Rules: [README.md](README.md#verification-rules).

## Summary

| ID | Issue | Sev. | Status | Category | Roadmap |
|---|---|---|---|---|---|
| [I-04](#i-04) | Seasoned TraderX swaps: the export has no past fixings | High | OPEN | Scope | External |
| [I-05](#i-05) | No faithful USD-SOFR / ACT-360 swap construction | High | OPEN | Scope | External |
| [I-07](#i-07) | No corporate bond, equity or listed-option pricer | Medium | OPEN | Scope | By demand |
| [I-08](#i-08) | A running TraderX attempt is lost on restart | Medium | PARTIAL | API | TraderX |
| [I-16](#i-16) | `rateSensitivity` is parallel-only | Medium | OPEN | Scope | External |
| [I-18](#i-18) | No equity spot or FX source; equity positions refused | Medium | OPEN | Scope | External |
| [I-23](#i-23) | `accrualBasis` strictness rests on an unconfirmed reading | Medium | ASSUMPTION | API | TraderX |
| [I-27](#i-27) | Long full-suite runs can hard-abort inside XLA or lose a worker process | Medium | OPEN | Tooling | Fix what exists |
| [I-104](#i-104) | Edge cases valued wrong: a zero-variance European, a bond awaiting a delayed payment, H at small a·t | Low | OPEN | Correctness | Fix what exists |
| [I-107](#i-107) | TraderX: `bundleId` is trusted, not recomputed, so a changed bundle can get the old result | Medium | OPEN | API | TraderX |
| [I-108](#i-108) | TraderX: NaN amounts priced as `ok`, terms contradicting the row priced, duplicate rows counted twice | Medium | OPEN | API | TraderX |
| [I-109](#i-109) | TraderX: after a failed publication a same-id retry never prices or publishes | Medium | OPEN | API | TraderX |
| [I-110](#i-110) | The documentation contradicts the code in places | Low | OPEN | Tooling | Fix what exists |
| [I-117](#i-117) | ORE's sensitivity analytic is inconsistent with itself, or fails a calibration the engine fits, in four configurations | Low | OPEN | Validation | Fix what exists |
| [I-115](#i-115) | The Greeks on an accelerator: bumps one at a time, a calibrated option's AD compiling for a minute, nothing measured on a GPU | Medium | OPEN | Performance | Focus |
| [I-118](#i-118) | A run of several analytics repeats the work they share | Low | OPEN | Performance | Fix what exists |
| [I-97](#i-97) | XLA computes some native GPU products wrong, so FP8 and bfloat16 passes run emulated on a GPU | Medium | OPEN | Performance | Focus |
| [I-53](#i-53) | On Windows, pricing runs slower after AD Greeks; Newton's bootstrap compiles 3–4 s longer from scratch | Low | PARTIAL | Performance | Fix what exists |
| [I-54](#i-54) | No swaption smile: options away from the money read the ATM vol | Medium | OPEN | Correctness | New |
| [I-58](#i-58) | TraderX: two concurrent submissions of one workload under different ids both execute | Medium | PARTIAL | API | TraderX |
| [I-59](#i-59) | TraderX: `calculations` and `reportingCurrency` accepted, keyed, then ignored | Medium | OPEN | API | TraderX |
| [I-60](#i-60) | TraderX result schema has no stated policy on added fields | Low | ASSUMPTION | API | TraderX |
| [I-90](#i-90) | The HTTP API has no authentication, rate limit or body-size limit | Low | OPEN | API | TraderX |
| [I-61](#i-61) | Nothing runs on more than one host; multi-device speed unmeasured | Medium | PARTIAL | Performance | Focus |
| [I-66](#i-66) | No linter or type checker | Low | OPEN | Tooling | Fix what exists |
| [I-67](#i-67) | Test modules import each other and repeat fixtures | Low | OPEN | Tooling | Fix what exists |
| [I-81](#i-81) | A cold job compiles about 180 one-operation programs; market risk vmaps a closure | Low | OPEN | Performance | Fix what exists |
| [I-93](#i-93) | TraderX's submissions are priced by their own pricers, not the engine's | Medium | OPEN | API | TraderX |

**One pipeline.** Since 2026-10-01 every run is `price_portfolio` on a `Market`
(`engine.run.pipeline`; since 2026-10-18 reached through the run request, `engine.run.run` and
HTTP `POST /runs`, I-87): ORE's cross-asset model with a
model per currency
(`CamConfig.ir`: the LGM by default, or Hull-White, decision A-1 in
[compliance/decisions.md](../../compliance/decisions.md)), and ORE's valuation of every trade on
every path. "Market path" in older entries and commits means this pipeline; "the Hull-White
model" before 2026-10-01 meant the separate pipeline removed that day.

---

## Correctness

<a id="i-104"></a>
### I-104 — Edge cases valued wrong: a zero-variance European, a bond awaiting a delayed payment, H at small a·t

**Severity:** Low · **Status:** OPEN · **Category:** Correctness · **Found:** 2026-10-10, the
project audit (AUD-08, AUD-11, AUD-12, AUD-14); confirmed in the code 2026-10-11

**What is wrong.** Three narrow cases, each a number off on an input the engine accepts (a
fourth, Theta's add-back of a coupon fixing inside a long roll at 0%, went on 2026-10-18 with the
add-back itself, which ORE's sensitivity analytic does not make, I-116):

- **A zero-variance European before expiry is 0 on the paths.** `european_cube`
  (`engine/pricing/european.py`) marks a date alive by `variance > 0`, but expiry is a date.
  Under `ConstantVariance` with a zero volatility the option's discounted intrinsic value is
  dropped. (`ForwardVariance`, the default, floors the variance, so it never meets this.)
- **A bond between its maturity and a delayed last payment is 0.** A coupon period may pay
  after its accrual end (`CouponPeriod.payment_date`), so the last coupon can pay after
  `maturity_date`, which must equal the last accrual end; `value_on`
  (`engine/pricing/cube.py`) values the bond at 0 from `maturity_date` on, and the redemption
  is paid on `maturity_date` though the coupon beside it is paid later. A weekend maturity
  with a Monday payment loses that coupon for two days: a Theta date or exposure date there.
- **The LGM's H(t) cancels at small a·t.** `engine.models.lgm.H` computes (1 − exp(−a t))/a;
  the host's twin (`engine.models.cam`) uses −expm1(−a t)/a. The relative error is about
  ε/(a t): 2e-3 in float32 at a = 0.01 and a one-day t, its effect on a path's log discount
  of float32's own rounding size. float64 is unaffected at the reversions used.

**Reach.** Configurations no test, demo or default uses: zero volatility under
`ConstantVariance`, explicit bond payment dates after maturity; and float32 runs, by about one
rounding.

**Current handling.** None.

**To close.** Liveness from the expiry date, the Bachelier formula taking zero variance (it
does); a bond valued until its last payment, the redemption's payment date stated and checked
against ORE's bond; H by `expm1`, host and JAX alike, with the a → 0 limit for its derivative.
A test per case; the golden snapshot within the margin.

<a id="i-54"></a>
### I-54 — No swaption smile: options away from the money read the ATM vol

**Severity:** Medium · **Status:** OPEN · *Difference from ORE* · **Found:** 2026-09-29

**What is wrong.** `engine.market_data.market.SwaptionVolSurface` is an ATM normal matrix (expiry × tenor).
ORE reads a vol cube or SABR smile at each option's strike. Every European, and every
Bermudan/American calibration helper (struck at the deal rate, `CoterminalDealStrike`), reads
the ATM vol.

**Reach.** None while markets are ATM-only, as every market given to the engine so far is.
With a smile: European NPV and Vega away from the money, and calibrated Bermudan/American σ.

**To close.** Decided (X-5): a strike axis in the market's volatilities, read at each
option's and helper's strike as ORE reads its cube. SABR is [F-02](features.md#f-02).

<a id="i-97"></a>
### I-97 — XLA computes some native GPU products wrong, so FP8 and bfloat16 passes run emulated on a GPU

**Severity:** Medium (a stated goal unmet: the speed of native units) · **Status:** OPEN ·
**Category:** Performance · **Found:** 2026-10-10, the device check (`tests/support/precision_devices.py`);
located 2026-10-11

**What is wrong.** Three matrix products XLA compiles for the GPU return wrong numbers without
an error (RTX 5060, compute capability 12.0, jax and jaxlib 0.10.2, driver 592.01; each against
the exact product of the same operands, relative to `|a|·|b|`, five shapes):

- **FP8** (e4m3 and e5m2 operands, float32 result): 0.39 to 0.72 off on every shape but the
  matrix-vector one, under any precision flag. XLA compiles these products with its own Triton
  GEMM emitter; with `--xla_gpu_enable_triton_gemm=false` it calls cuBLASLt instead
  (`__cublas$lt$matmul$f8` for e4m3) and every shape is exact to 3e-8. The card's FP8 units
  and NVIDIA's library are right; XLA's Triton code for FP8 on this architecture is not.
  e5m2 by e5m2 goes to cuBLASLt's non-FP8 path, consistent with FP8 units that do not take two
  e5m2 operands: a native e5m2 product would pair it with e4m3.
- **bfloat16 passes** (`BF16_BF16_F32_X3`): 2e-6 to 1.4e-5, as three passes should be, on every
  matrix shape, Triton or cuBLASLt; 9.9e-4, a single pass's error, on the matrix-vector shape
  (512 by 64 by 1), which XLA compiles as a fusion, not a GEMM, ignoring the requested
  algorithm. `--xla_allow_excess_precision=false` changes nothing.
- **float32 operands with a float64 result** (`preferred_element_type`): float32's error (2e-7)
  through Triton; cuBLASLt refuses it ("GEMM is not supported"), so only Triton's answer is
  silent. Fixed in the engine on 2026-10-11 (`engine.precision.products`: float32's values
  multiplied as float64).

**Reach.** No number the engine reports: `product_execution` marks FP8 and bfloat16 passes
emulated on every GPU, and the emulation is exact. What it costs is speed: on a GPU the FP8
compute formats run on float16 units, and bfloat16 passes as three float32 products, so their
timing on a GPU (F-07's research result) is not yet measurable. Unknown elsewhere: H100
(compute capability 9.0) is XLA's most exercised FP8 target and may compile FP8 another way;
the matrix-vector and float64-result faults look independent of the architecture.

**Current handling.** Emulated on a GPU; the precision report says so per format. The device
check covers every allowed arithmetic and a matrix-vector shape.

**To close.** Native FP8 and bfloat16 passes on any GPU that has the units, each shown right by
the device check: (1) route the engine's FP8 products to cuBLASLt (the engine worker's
`--xla_gpu_enable_triton_gemm=false`, timed against Triton on every other product, or a
narrower mechanism if XLA offers one per product); (2) a product format's matrix-vector product
run in a shape XLA honours the algorithm for, or emulated; (3) e5m2 paired with e4m3 where the
units need it; (4) the device check on H100 and on a newer jax, and minimal reproductions
reported to XLA. Red first: the device check with the formats marked native fails on the RTX 5060
today (FP8 0.7 off) and passes after.

---

## Validation

<a id="i-117"></a>
### I-117 — ORE's sensitivity analytic is inconsistent with itself, or fails a calibration the engine fits, in four configurations

**Severity:** Low · **Status:** OPEN · **Category:** Validation · **Found:** 2026-10-18, the
seeded ORE parity sweeps closing [I-116](#i-116)

**What is wrong.** In four configurations ORE's sensitivity analytic is no reference: its
numbers disagree with ORE's own other numbers, or it fails, so the engine's Greeks there are
unverified, though each value they are built from equals one of ORE's or fits ORE's basket.

- **An as-of date that is not a TARGET business day** (an Easter Monday). ORE's simulation
  market reads the swaption volatilities from a matrix whose reference date floats to the next
  business day; its sensitivity analytic's base NPV takes that reference, its bumped scenarios
  do not, so every Delta and Vega of an option carries the difference (5.73 on a European of 8.8
  million notional, at tenors and quotes the trade cannot see). The engine values the as-of date
  on the as-of date, equal to ORE's NPV analytic.
- **A deal strike ORE's fallback rule moves, on an ACT/ACT (ICMA) swap index.** A Bermudan or
  American struck beyond three ATM standard deviations (`IrModelBuilder`'s fallback rule 1)
  whose helpers accrue ACT/ACT (ICMA): ORE's base NPV on its sensitivity market differs from its
  own recalibrations under every scenario, which equal the engine's (0.0122 on an American of
  385,434 notional, 8e-5 of its value); its NPV analytic equals the engine's. An ACT/365 swap
  index, or a strike inside the band, agrees to 1e-10.
- **An American whose window holds a calibration grid date today but none on the Theta date.**
  ORE's builder fails on the Theta date ("no calibrating swaption provided") and books the trade
  as failed, worth 0, so its Theta is minus its value; the engine refuses the run, naming the
  trade, as it refuses the trade today (decision A-24).
- **A large basket recalibrated under a 5bp volatility shift.** An American on a 3M reference
  grid over an eight-year window (32 helpers, negative reversion): under a 5bp shift of one quote
  ORE's Levenberg-Marquardt bootstrap stops at a calibration error of 8.1e-4, above its 1e-4
  tolerance, and fails the whole analytic; the engine's bucket roots fit the same basket to 2e-15
  (decision A-21: both solve ORE's equations, the engine's solver more exactly). At 1bp ORE fits
  it too.

**Reach.** The Greeks of these configurations only: every other value, ORE's NPV analytic and
its recalibrated scenarios, agrees. The sweeps draw none of them
(tests/support/parity_sweep.py).

**Current handling.** Documented here and in
[ORE Parity](../reference/ore-parity.md#the-parity-matrix); the reproductions are the three cases
above (tests/test_ore_parity_sweep.py's docstring).

**To close.** Report the first two to ORE with minimal reproductions (an OREApp sensitivity run
of one trade), the fourth as a limit of ORE's optimiser; decide whether an as-of date that is not a business day is refused (the run
request's preflight, `engine.run.validate_run`); a test per case once ORE's numbers are consistent, or each
recorded as a difference by decision.

---

## Performance

<a id="i-115"></a>
### I-115 — The Greeks on an accelerator: bumps one at a time in a host loop, a calibrated option's AD compiling for a minute, nothing measured on a GPU

**Severity:** Medium · **Status:** OPEN · **Category:** Performance · **Found:** 2026-10-17, closing
I-51, I-78 and I-112

**What is wrong.**

- **The bump Greeks reprice one shift at a time.** `engine.risk.greeks.bump.portfolio_sensitivities`
  values each trade on about sixty shifted markets in a Python loop (`value_on` per shift, a
  calibrated option recalibrated in each), every value a host round trip: an accelerator idles
  between small programs. The shifts are independent scenarios on the sensitivity market, which
  market risk already revalues as one batch, its options recalibrated per scenario
  (`engine.risk.market.revaluation`, `engine.risk.greeks.price_functions`).
- **A calibrated option's AD Greeks compile for about a minute.** Measured old and new code
  alternately, the shared portfolio's AD Greeks on ORE's default grid with no disk cache: 159 s
  against 32 s before 2026-10-17, warm 4.7 s against 0.7 s (the bump method 8.7 s); its American,
  16 helpers, alone 87 s cold, 0.6 s warm, 4.2 GB peak. Most of it is the helpers' prices, each
  with its own y\* solve, traced again in every derivative context of the basket's residuals in
  one program; compiled apart, the calibration's second derivatives took 12 s and the grid's
  along a given volatility 0.6 s (profiling §3.9). Two other forms of the tangent solve were
  measured and are worse.
- **Nothing of 2026-10-17's is measured on an accelerator**: the AD program through a calibration
  (batched forward derivatives over 26 directions, each moving the grid's operator), and market
  risk's recalibration (each scenario its own operator, batched, where a fixed model shares one).
  On the CPU, the parity sweep's heaviest case (2026-10-18: an American of 32 helpers on a
  301-node grid, 52 exercise steps a year, recalibrated under 256 scenarios) took the engine
  183 s cold and 170 s warm, and ORE's own analytic 1,041 s: compute, not compilation.

**Reach.** Wall time only; no number. Bump Greeks on an accelerator; the first AD Greeks of a
calibrated option's shape in a process; calibrated options in market risk on a GPU or TPU.

**Current handling.** The disk cache keeps the compiled programs, so a repeated job does not pay
the compile; bump Greeks on a CPU are as before. Where the float64 stages (calibrations, Greeks)
run on a host with an accelerator is the owner's open decision A-25
([compliance/decisions.md](../../compliance/decisions.md)), which step 1's measurements inform.

**To close**, in this order, the numbers equal before and after within the margin (the ORE parity
suites, the golden snapshot), measured old against new alternately:

1. Measure on the RTX 5060 (WSL): the 2026-10-17 tests, the shared portfolio's bump and AD Greeks
   cold and warm, peak memory, and recalibrated market risk's time per scenario against a fixed
   model's, so the work below is aimed at what costs on the device.
2. The bump Greeks as one batched revaluation: the shifts as scenarios of the sensitivity market
   (the price functions on `TenorNodes`, a calibrated option recalibrated per shift as market
   risk recalibrates per scenario, Theta as it is), one program per trade shape; equal to ORE's
   sensitivity analytic as now (tests/test_sensitivity_ore_parity.py).
3. The AD program in two: the calibration's first and second derivatives along each direction
   (the implicit function theorem on the basket's residuals, as now), then the grid's along the
   combined direction; the same Greeks.

<a id="i-53"></a>
### I-53 — On Windows, pricing runs slower after AD Greeks; Newton's bootstrap compiles 3–4 s longer from scratch

**Severity:** Low · **Status:** PARTIAL · **Category:** Performance · **Found:** 2026-09-29 ·
re-measured 2026-10-02, 2026-10-05, 2026-10-06, 2026-10-07

**Closed parts.** The pipeline was slow, first by compiling the same work again per trade, path
date, bump and call (pricers jitted once per shape, 2026-10-02; [I-21](#i-21), [I-22](#i-22));
then a repeated job compiled 9 programs, the AD Greeks' derivatives living only in JAX's internal
caches (`tests/test_profiling_and_jit.py::TestCompileCounts::test_a_repeated_job_compiles_nothing`,
`::test_an_option_s_greeks_are_one_program`, both red on the code before); and the
recalibration of every Bermudan/American, on every path date and in its Greeks, ran bisections
of 60 steps per bucket around 160 per y\*, the dates one after another (decision
A-21: one root solver, Newton by default, the dates of one basket shape calibrated together;
`tests/test_root_solvers.py`). Measured untraced, old and new code alternately: the path
recalibration 35–55 times faster on the CPU (262,144 paths: 144–185 s to 2.8–3.4 s) and 15–38
times on the RTX 5060 (7.3 s to 0.19 s); the demo's repeated job on the GPU 5.4–5.6 s to
1.35–1.43 s, and its trace 32 s to 1.6 s
([profiling §2.0](../concepts/profiling.md#20-the-demo-measured-2026-10-05)).

**What is wrong.**

- **On Windows only, the AD Greeks slow later pricing.** In a process that has run any AD
  Greeks, every later job's pricing phase runs 10–25% slower: pricing-only jobs 1.57–1.83 s
  before one Greeks call and 1.98–2.25 s after, the code before Newton and after it alike; on Linux
  1.32 → 1.32–1.38 s (2026-10-06). No compile, trace or garbage collection is involved and one large
  allocation does not do it; the Windows heap is suspected, not shown. The deployments that
  matter run Linux.
- **A cold job is 3–4 s (8–11%) slower since the Newton solver.** Each bootstrap bucket's program under Newton
  traces the helper's price, with its nested y\* solve, at the bracket's ends, in the step (under
  `jax.jvp`) and for the model value: the demo's ten bucket programs compile in 5.4 s against
  3.3 s (7.1 s before the bracket's two ends were evaluated in one call). The programs stay in
  the worker and on disk, so a repeated or restarted job does not pay it.
- **The leg kernel in difference form is a third slower on its own on a CPU** (since the kernel
  rewrite, 2026-10-10, F-07). A swap's cube on 4,096 paths and 20 dates takes 13.4 ms against
  10.2 ms in float64, though the operations are the same work (11.6 ms against 11.7 ms summed
  over a trace): the old code's one reduction overlapped its two operand fusions, the new
  kernel's products (one per block of columns) run after theirs. End to end it is within the
  run-to-run spread: a five-trade linear portfolio 4% slower (medians 0.370 s and 0.352 s, three
  interleaved passes), market risk on its trades three times faster (0.075 s against 0.23 s).
  Concatenating the blocks into one product's operand cost a quarter more; two columns per coupon
  doubled it (both measured and discarded). Accepted by the owner (2026-10-11): the one kernel at
  every precision stays (A-16), rather than a float64 kernel of its own beside it; the fix is the
  batching below, done with the timing on accelerators if it shows there, else when fixing what
  exists.

**Reach.** Wall time only; no number. Windows CPU runs; a worker's first job; float64 path
pricing of linear trades on a CPU. (A calibrated option's AD compile is [I-115](#i-115)'s.)

**Current handling.** None needed for correctness.

**To close.** The Windows slowdown's cause, and the helper's price traced once per
bucket program, the snapshot within the margin; the leg kernel's products scheduled with their
operand fusions (or the trades of one shape priced as one product, their columns side by side,
which a matrix unit wants anyway), timed against the code before the kernel rewrite. Related: the cold job's one-operation
programs ([I-81](#i-81)). (The grid rollback's memory and time on the paths, the other half of the 4,096-path job's time,
went with its matrix form, [I-83](#i-83), 2026-10-13: the demo's job 0.27 s warm there, against 13.2 s.)

<a id="i-61"></a><a id="p-1"></a>
### I-61 — Nothing runs on more than one host; multi-device speed unmeasured

**Severity:** Medium · **Status:** PARTIAL · **Found:** 2026-09-24, audit P-1

**Closed part (2026-10-04, the one-host half).** A job's scenarios are split across
the devices of its host (`engine/market_simulation/sharding.py`): the simulation's Sobol normals and
market risk's shifts are placed on a one-axis mesh along the scenario axis, and XLA's sharding
propagation carries the split through path evolution, the scenario market, pricing and the
stored cube, gathering at the reductions (exposure, VaR/ES). Market risk's memory-bounded
batches take their scenarios from every device's share. The device count is the largest that
divides the scenario count, capped by `JAX_RISK_SCENARIO_DEVICES`; on one device nothing is
placed, so one-device runs are unchanged bit for bit. On four XLA host devices a portfolio run
(LGM, Bermudan, paired sample), an FP8 stochastic-rounding run and a market-risk run hold
their results on all four devices and equal the one-device run to about 6e-16 relative (the
batch shape's rounding; FP8 bit for bit). Tests: `tests/test_sharding.py`; red first: before
the change every result sat on one device.

**What is wrong.** The project's goal is to run across multiple TPUs and compare many
low-precision paths against fewer FP64 paths in equal wall time. Two parts are missing:

- **Several hosts.** On a pod slice every host must run the same job (SPMD); the worker
  claims jobs for its own host only and uses `jax.local_devices()`.
- **Speed.** The split is verified for its numbers on CPU host devices, which share the
  host's cores, so it says nothing about speed. Scaling on real devices is unmeasured.
- **Awkward path counts use fewer devices, silently.** The device count is the largest divisor
  of the path count (`scenario_device_count`): 1,001 paths on eight devices run on seven, a
  prime count on one, with no warning (found by the project audit, AUD-45).

**To close.** One worker per host (`jax.distributed.initialize`), process 0
claiming each job and handing it to the others
([details](details/precision.md#113-multi-device-and-multi-host)); then wall time against
device count on TPU (and H100), feeding the research result ([F-07](features.md#f-07)). A run on fewer devices than
it was given says so in its precision report (or pads its paths to a multiple of the devices
and drops the padding before any reduction); the storage blocks' alignment with each device's
share is checked on the real devices.

<a id="i-118"></a>
### I-118 — A run of several analytics repeats the work they share

**Severity:** Low · **Status:** OPEN · **Category:** Performance · **Found:** 2026-10-18, building
the run request ([I-87](#closed))

**What is wrong.** `engine.run.run` calls each analytic's entry point as it is
(`engine/run/run_request.py`), so work two analytics share is done once per analytic: a run of
`exposure` and `calibration` bootstraps every calibrated currency of the cross-asset model twice
(`calibrate_cam`, then `build_cross_asset_model`); `market_risk` beside a portfolio analytic
values each trade on today's curves again (its base values) and builds its ORE schedules again;
and the request is validated once by `validate_run` and again by each entry point
(`validate_request`, `validate_portfolio`), each building every trade's ORE swap.

**Reach.** Wall time of a run of several analytics, not measured (the bootstrap of a few helpers
and the host-side checks are small beside a simulation; a large portfolio's schedules are not).
No number: each analytic equals its direct call bit for bit, by design
(`tests/test_run_request.py`).

**Current handling.** Documented ([HTTP API](../reference/http-api.md#design-one-request-for-every-analytic)).

**To close.** Share what the analytics read in common, where it is the same computation: the
cross-asset model's calibration handed from `calibration` to `exposure`
(`build_cross_asset_model(..., sigmas=...)` takes it), the request validated once, the trades'
schedules built once per run. A test that the combined run still equals each analytic alone, bit
for bit, and that each shared stage runs once (a call count), timed against the code before.

---

## API

<a id="i-08"></a>
### I-08 — A running TraderX attempt is lost on restart

**Severity:** Medium · **Status:** PARTIAL

**Closed part (2026-10-04).** Portfolio jobs (the run requests since 2026-10-18): the durable
SQLite job queue (`engine/api/job_queue.py`, decision A-14) replaced the in-memory `_JOBS` dict.
A job survives restarts, every API process on the queue serves every run id, a job running when
its worker died reads `interrupted`, and a failure names its class (`bad-terms`,
`missing-market-data`, `unsupported-product`, `numerical-failure`, `infrastructure`).
Tests: `tests/test_engine_worker.py` (`test_jobs_survive_reopening_the_file`,
`test_a_worker_killed_mid_job_leaves_it_interrupted`,
`test_supervisors_of_one_queue_share_one_worker`, the failure classes); red first: on the code
before 2026-10-04 a job id from one app process was a `404` in the next.

**What is wrong.** The TraderX path is durable for finished work
(`engine/traderx/publication.py`: manifest as commit point, scan recovery, idempotent
`submissionId` across restarts), but a *running* attempt is memory only and reads as unknown
after a restart. The store is single-machine.

Three more gaps of the same store (found by the project audit, AUD-24 to AUD-26, confirmed
2026-10-11):

- **A failed workload is unknown after a restart.** `ResultStore.lookup` serves successes only,
  by design, and `AttemptStore.lookup` falls back to attempts in memory; after a restart a
  workload whose only attempt failed, published, reads as `UNKNOWN_WORKLOAD` (its attempt is
  still found by id).
- **Bindings and publication are one process's.** `AttemptStore` holds an in-process lock and
  `ResultStore` has no cross-process coordination, so two API processes on one directory can
  bind one id to two workloads and both publish.
- **Power loss.** Publication fsyncs each file and renames it atomically, but does not fsync
  the directory after the rename, so on Linux a published manifest may not survive a power cut
  that the code's comments call durable; a failed stage can leave its temporary file behind.

**To close.** On the TraderX path, the durable accepted-attempt record, a boot sweep
and an `interrupted` lookup state (TraderX acceptance case A-09); the portfolio queue's
`interrupt_running` is the pattern. With it, attempts, bindings and terminal states in the
queue's SQLite transactions (one owner across processes), a workload's latest terminal state
served when it has no success, and stale stages swept at boot. Until then one TraderX process
per store, as [docs/reference/traderx-path.md](../reference/traderx-path.md) ("What this is
still not") says of publication, and now of bindings too. Tests: a
fresh store finds a published failure by workload; two processes binding one id to two
workloads, one refused.

<a id="i-23"></a>
### I-23 — `accrualBasis` strictness rests on an unconfirmed reading

**Severity:** Medium · **Status:** ASSUMPTION · **Raised:** 2026-09-16

**The premise.** `engine/traderx/terms.py` accepts exactly `dateBasis = SESSION_DATE`,
`settlementAdjustment = NONE`, `rounding = HALF_EVEN`, schema
`traderx.accrual-basis.v1`, and refuses anything else. TraderX has not said whether new
values land in `v1` or force a `v2` (asked in responses v4, v6 and v7).

**Risk.** If they add values in place, bundles they consider valid are refused on the day a
real calendar is exported. That fails safe (a refusal, not a wrong number), but reads as a
defect on call. A `TermsJoinError` naming these fields may be this allowlist, not a bad
bundle.

**To close.** Their answer. "New version": close with no change. "In place": widen
`SUPPORTED_DATE_BASES` / `SUPPORTED_SETTLEMENT_ADJUSTMENTS` / `SUPPORTED_ACCRUAL_ROUNDING`
with a test per value. `tests/test_traderx_terms_v2.py::TestUnrecognizedValuesAreRefused`
pins today's rule.

<a id="i-58"></a>
### I-58 — TraderX: two concurrent submissions of one workload under different ids both execute

**Severity:** Medium · **Status:** PARTIAL · **Found:** 2026-09-17, TraderX case A-03 (FR-08)

**Closed part (2026-10-10, with [I-57](#i-57)).** A retry under the *same* `submissionId`
while its attempt is pricing is answered `200 {"state": "running", "reused": true}` and
not executed: `AttemptStore.submit` returns `reused` for an attempt it did not create, and
the route prices only a new one. Before, the retry priced the same attempt again and the
second to finish raised on the immutable attempt, a `500`. Test:
`tests/test_traderx_routes.py::TestARetryOfARunningSubmissionIsNotExecutedAgain`; TraderX's
A-03 starter retries under one id, so it now passes and does not exercise the open part.

**What is wrong.** Two submissions of one workload under *different* ids, or with none, while
nothing is completed (or with `reuseExistingResult=false`) each get a running attempt, and
both run `price_bundle`. The lock protects the binding, not the computation.

**To close.** In `AttemptStore.submit`, under its lock, a per-workload running-attempt index:
a submission of a workload with a running attempt joins it (`reused`, so the route does not
price it), and its id is bound to that attempt. Decide there whether a joined or retried
running attempt answers `200 running` (today's retry) or `202`, the same for both.
Regression test with a deterministic barrier, not timing. Cross-process exclusion is not
claimed.

<a id="i-59"></a>
### I-59 — TraderX: `calculations` and `reportingCurrency` accepted, keyed, then ignored

**Severity:** Medium · **Status:** OPEN · **Found:** 2026-09-17, TraderX cases A-04, A-05
(FR-06); still present

**What is wrong.** `EodSubmissionSchema` accepts both, `_key_for` hashes both into the
workload key, and `price_bundle(bundle, request.marketInputs)` receives neither. An `EUR`
request returns USD results with 200 OK, cached separately from the identical USD run; an
unknown calculation name is accepted.

**To close.** As decided (A-18, as ORE fails on an unknown analytic): reject
an unknown calculation (400 `UNKNOWN_CALCULATION`, allowlist
`engine.traderx.result.CALCULATIONS`); add `reportingCurrencies: ["USD"]` to the
capability document and reject others (400 `UNSUPPORTED_REPORTING_CURRENCY`). Tests assert
the consequence, not that the field parses. Reporting in other currencies is
[F-08](features.md#f-08), once an FX source exists.

<a id="i-107"></a>
### I-107 — TraderX: `bundleId` is trusted, not recomputed, so a changed bundle can get the old result

**Severity:** Medium · **Status:** OPEN · **Category:** API · **Found:** 2026-10-10, the project
audit (AUD-22); confirmed in the code 2026-10-11

**What is wrong.** The exchange contract
(`reference/traderX/specs/YU18-eod-risk-bundles/contracts/bundle-v2-and-terms.md`) defines
`bundleId` as the SHA-256 of the canonical manifest without it (sorted keys, two-space
indentation, ASCII escaping, finite numbers, UTF-8, one final LF). `engine.traderx.bundle`
checks each artifact against the hash the manifest gives, but takes `bundleId` as given
(`Bundle.bundle_id`), and the workload key is built from it (`_key_for`,
`engine/api/traderx_routes.py`), not from the manifest's content. A bundle re-exported with a
corrected artifact and that artifact's new hash, under the old `bundleId`, loads, gets the old
workload key, and with `reuseExistingResult` is answered with the old bundle's result.

**Reach.** Only a bundle that breaks the contract's identity rule (an exporter defect or a hand
edit); then a stale result for changed positions or terms, with `200`. A conforming exporter
changes the id with any artifact.

**Current handling.** Artifact hashes are checked against the manifest; the id is not.

**To close.** On load, the canonical manifest hash recomputed and a mismatch refused (`422`,
naming both ids); test fixtures that edit an artifact rebuild the id. Tests: a changed artifact
under the old id refused; the rebuilt bundle gets a new workload key.

<a id="i-108"></a>
### I-108 — TraderX: NaN amounts priced as `ok`, terms contradicting the row priced, duplicate rows counted twice

**Severity:** Medium · **Status:** OPEN · **Category:** API · **Found:** 2026-10-10, the project
audit (AUD-27 to AUD-31); confirmed in the code 2026-10-11

**What is wrong.** A bundle's rows are checked for shape and identity, not for every property
the pricers assume:

- **Non-finite numbers.** `normalize._parse_float` accepts `nan` and `inf` as Python's `float`
  does (`quantity`, `coupon`, `closingMark`, `accruedInterestFraction`), and so do the terms'
  numbers (`note._terms_float`, `bill.py`). A NaN quantity is priced to a NaN value with status
  `ok`; publication writes it as `json.dumps`' `NaN`, which is not JSON, and an HTTP answer
  that serializes it fails.
- **Terms that contradict the row.** `terms.join_terms` matches rows to entries by identity and
  refuses duplicate or unused entries, but compares no economics: the row's `coupon` is parsed
  and the terms' `couponRatePercent` priced, never checked against each other, though the
  contract forbids a supplement changing a booked rate, index, date, currency or direction.
- **Duplicate rows.** Duplicate terms entries are refused; a position row repeated (one account
  and security twice) or a contract row repeated is priced and aggregated twice.
- **Structure.** A manifest that is not a JSON object, or a field of the wrong type, fails
  outside `BundleIntegrityError` (a `500`, not a `422`); `CalculationOutcome.to_dict`
  (`engine/traderx/result.py`) lets a `payload` overwrite `status` and `value`.

**Reach.** Bundles with such rows, an exporter defect: then a NaN or double-counted figure
reported `ok`, or a `500`. Conforming fixtures are unaffected.

**Current handling.** Artifacts are hash-checked, terms entries de-duplicated, and accrued
interest reconciled against the exporter's figure, which can catch a contradicting coupon when
the accrual is not zero.

**To close.** Every parsed number finite (the row refused with a reason, not priced); each
joined row's booked fields reconciled with its terms (rate, currency, maturity, direction), a
contradiction refused with its own reason; repeated position and contract identities refused;
the manifest's structure validated before `Bundle` is built (`422` naming the field);
`CalculationOutcome` keeping its own keys; publication with `allow_nan=False`. A test per case,
from the CSV to the HTTP answer.

<a id="i-109"></a>
### I-109 — TraderX: after a failed publication a same-id retry never prices or publishes

**Severity:** Medium · **Status:** OPEN · **Category:** API · **Found:** 2026-10-10, the project
audit (AUD-23); confirmed in the code 2026-10-11

**What is wrong.** `Attempt.complete` (`engine/traderx/workload.py`) publishes before it changes
state, so a publication error leaves the attempt `running` and its priced result dropped. The
route answers `500 RESULT_NOT_PUBLISHED` and asks the coordinator to retry; the retry under the
same `submissionId` gets the bound attempt back as `reused` ([I-58](#i-58)'s closed part), and
the route returns `running` without pricing or publishing. Until the process restarts, that
submission cannot complete, even after the storage recovers. The staging file's open, write
and fsync also raise outside the `PublicationError` the route catches.

**Reach.** A TraderX submission whose publication fails transiently (a full or unavailable
disk). A new `submissionId` recovers it; the retry the route asks for does not.

**Current handling.** None.

**To close.** A priced-but-unpublished state that keeps the result, a same-id retry publishing
it (or the attempt failed retryably and the retry pricing again); every staging error a
`PublicationError`. Test: the first publication fails, storage is restored, a retry under the
same id yields one discoverable result; red first.

<a id="i-60"></a>
### I-60 — TraderX result schema has no stated policy on added fields

**Severity:** Low · **Status:** ASSUMPTION · **Raised:** 2026-09-17, response v7 §5.2

**The premise.** `jaxrisk.eod-result.v1` does not say whether consumers may pin its exact
shape. TraderX's validator does, so the next field added breaks them. Proposed to TraderX:
`additionalProperties: false` at the root and any new field is a schema version bump.

**To close.** Their answer, recorded as a decision; then state it in the published schema
with a test.

<a id="i-90"></a>
### I-90 — The HTTP API has no authentication, rate limit or body-size limit

**Severity:** Low · **Status:** OPEN · **Category:** API · **Found:** 2026-10-08, review of
the API of 2026-10-07

**What is wrong.** Every route, TraderX's included, serves anyone who can reach the
port. Nothing limits how many jobs a caller queues or how large a body it sends; one large
body is held in memory and stored in the queue whole.

**Reach.** Only a server reachable from outside a trusted network. uvicorn binds
`127.0.0.1` unless started with `--host`, and every use so far has been local.

**Current handling.** None; deploy behind a trusted network or a reverse proxy that
authenticates.

TraderX's `POST /eod/price` also reads a server-side `bundlePath` and joins each manifest
path to it without containment (`engine/traderx/bundle.py`): an absolute path or `..` reads
any file the server can, its hash supplied by the same manifest; files are read whole, unbounded
(project audit, AUD-34).

**To close.** Before the API is reachable from outside a trusted network:
authentication chosen with TraderX (a bearer token or mutual TLS), a body-size limit, and a
per-caller limit on queued jobs; bundle paths confined to configured roots, every artifact
path resolved inside its bundle (symlinks included), and a byte limit per artifact; a test per
refusal (`401`, `413`, `429`, a path outside the root).

<a id="i-93"></a>
### I-93 — TraderX's submissions are priced by their own pricers, not the engine's

**Severity:** Medium · **Status:** OPEN · **Category:** API · **Found:** 2026-10-08, owner
review of the roadmap

**What is wrong.** The engine has two ways in. Portfolio, market-risk and calibration
requests reach the engine's pipeline; TraderX's end-of-day bundles (`POST /eod/price`) are
priced by the TraderX path's own closed-form pricers (`engine/traderx/bill.py`,
`note.py`) on a flat assumed rate, while the engine prices the same bills and notes in
`engine/instruments/treasury.py` on any curve, today and on every path. Two pricers of one
product can disagree, every fix is made twice, and nothing the engine gains (curves, paths,
Greeks, precision, Basel figures) reaches TraderX.

**Reach.** Every TraderX result. No disagreement between the two is known: both discount the
same cashflows, and on the assumed profile's flat curve they should agree to rounding, which
no test checks today.

**Current handling.** None.

**To close.** The TraderX path builds a run request from its bundle, after its convention
checks and refusals (so a refused booking never reaches a pricer, I-05), and the engine
prices it; its own pricers go. Its routes, result document, attempt store and the
accrued-interest reconciliation against the exporter's figure stay. First, a test that the
engine's bond and the TraderX note agree on every bundle row of the tests, to rounding
([details/package-layout.md §3](details/package-layout.md#3-merges)). The adapter then calls
`run/`, which sits above it in the layers `tests/test_import_layering.py` enforces (`traderx`
shares a layer with `models` and `instruments`): its orchestration moves above `run/` (or into
`api/`), its terms and provenance types stay low, rather than an exception in the layer test
(project audit).

---

## Scope gaps

Refused with an identified reason, never approximated. Each is listed because a current
consumer (TraderX) sends the input.

<a id="i-04"></a><a id="m-2"></a>
### I-04 — Seasoned TraderX swaps: the export has no past fixings

**Severity:** High · **Status:** OPEN · blocked on TraderX · **Found:** TraderX EOD review;
widened by the 2026-09-24 audit (M-2)

**What is wrong.** A coupon fixed before the as-of date needs its historical fixing. Every
trade config takes `fixings` and refuses a missing one (`MissingFixingError`), as ORE does,
but TraderX exports no `pastFixings`, so a seasoned TraderX swap cannot be priced.

**Reach.** Seasoned swaps from the TraderX path only; the engine itself prices seasoned
swaps given their fixings, and on every path paid flows drop out and coupons fix by
`FixingManager`'s rule, under either model (the first half of this entry, the Hull-White
model keeping paid flows, closed on 2026-10-01: `tests/test_hull_white_model.py::
TestPaidFlowsAndMaturity`, `tests/test_pricing.py`).

**To close.** `pastFixings` from TraderX ([details](details/traderx-integration.md)).

<a id="i-05"></a>
### I-05 — No faithful USD-SOFR / ACT-360 swap construction

**Severity:** High · **Status:** OPEN · blocked on the D03/D04 convention agreement

**What is wrong.** `build_vanilla_swap` builds a generic term-IBOR swap (`SimIndex6M`,
ACT/365 legs, TARGET, no compounding). A USD-SOFR booking is overnight, ACT/360, US
calendars, compounded in arrears, with lookback/lockout/payment lag. ACT/360 vs ACT/365 alone
moves every accrual by 1.389% (about 46 × a 1bp DV01 on a 5Y fixed leg).

**Current handling.** The TraderX path refuses any booking outside an explicit allowlist
(`engine/traderx/conventions.py`, `CONVENTION_NOT_SUPPORTED` naming the fields; the
TraderX SOFR fixture's 13 missing terms). A direct Python caller cannot express SOFR in a
`SwapConfig`.

**To close.** The convention set from TraderX, then a new builder beside the existing one,
accepted against a same-terms ORE reference
([details/traderx-integration.md](details/traderx-integration.md)). Do not start on guessed
conventions.

<a id="i-07"></a>
### I-07 — No corporate bond, equity or listed-option pricer

**Severity:** Medium · **Status:** OPEN

**What is wrong.** Treasuries price today and on every path; TraderX's position export also carries
corporate bonds (refused: a Treasury-discounted corporate is not credit pricing, it needs a
credit model), cash equities ([I-18](#i-18): a market-data gap, not a pricer gap), listed
options, TIPS and FRNs. The simulation's equities (`CamConfig.equity_volatilities`) drive risk-factor paths; they
are not a position pricer. FX and equity trades are [F-04](features.md#f-04).

**To close.** Per instrument, on demand: a config, a pricer, ORE parity tests.

<a id="i-16"></a>
### I-16 — `rateSensitivity` is parallel-only

**Severity:** Medium · **Status:** OPEN · blocked on an observed curve (TraderX W2)

**What is wrong.** The note's `rateSensitivity` is a 1bp parallel shift, labelled
`shockedFactor: "zero-curve-parallel"`. The only market input the TraderX path accepts is an
assumed flat profile, so there is no pillar structure to shift. Parallel DV01 cannot tell a
2y position from a 10y one of equal duration.

**Do not close it** by bumping the flat profile per pillar: that yields an all-equal or
single-non-zero vector that passes a shape check.

**To close.** `marketInputs.mode: "package"` with bootstrapped pillars (W2); the bump loop
then shifts one pillar at a time and names it.

<a id="i-18"></a>
### I-18 — No equity spot or FX source; equity positions refused

**Severity:** Medium · **Status:** OPEN · blocked on a market-data decision

**What is wrong.** An equity position is `signedQuantity × multiplier × spot × fx`, and the
TraderX path has no spot or FX input (`engine/traderx/market_inputs.py` registers flat
rate profiles only). Positions are read, validated and refused (`SPOT_SOURCE_NOT_SUPPLIED`,
`FX_SOURCE_NOT_SUPPLIED`), echoing the quantities read; `capabilities()` reports equity NPV
as `blockedOnMarketInput`.

**Do not close it** with `closingMark`: `quantity × closingMark × multiplier` reproduces
TraderX's own `marketValue` exactly, an echo presented as a valuation.
`tests/test_traderx_equity.py::TestDoesNotEchoTheExportedMark` fails against it.

**To close.** A registered spot/FX surface in `marketInputs`, or observed spots in the
bundle. The pricer is four multiplications.

---

## Tooling

<a id="i-27"></a>
### I-27 — Long full-suite runs can hard-abort inside XLA or lose a worker process

**Severity:** Medium · **Status:** OPEN · located, not root-caused · **Found:** 2026-09-17

**What is wrong.** A long `pytest tests/` run has died with `Fatal Python error: Aborted` and
no summary line, the crashing thread inside `jax/_src/compiler.py`
(`backend_compile_and_load`) while `ProcessPoolExecutor` threads were alive (the worker pool
of the time; it was removed on 2026-10-04, so no test starts one now). Intermittent: the
same command later passed in full, and no abort occurred in the recorded runs since
2026-09-24.

**Hypothesis, unproven.** The worker pool lives for the interpreter; before 2026-10-01
`tests/test_api.py` created pools and never shut them down, so later in-process compiles ran
with worker children attached. Pairing modules does not reproduce it. Since 2026-10-02 the
suite does share JAX's on-disk compilation cache between its processes (xdist workers and
the engine workers they start, `tests/conftest.py`); JAX writes an entry without a lock and recompiles
when one cannot be read, so a torn entry costs a compile, not a crash, but an abort after
that date should rule the cache in or out first (rerun with `JAX_COMPILATION_CACHE_DIR`
pointing at an empty directory).

**Second hypothesis: memory.** On 2026-10-04 a Linux full run (Docker, 4 CPUs, `-n 4`, 25.2 GB)
lost one xdist process with no traceback ("node down: Not properly terminated") inside an FP8
storage test that passes alone at a 2.9 GB peak; the green rerun peaked at 24.2 GB (cgroup
peak, page cache included). GitHub's 4-vCPU runner, where CI's `full` job runs with `-n auto`,
has 16 GB.

**2026-10-04, three Linux full runs, none green** (see the verification status): one
beside another container in the same Docker VM, 6 failed; one alone on a loaded host, the
four-device sharding test (a fifth JAX process with four XLA host devices) timed out after
30 minutes at a cgroup peak of 24.4 GB of 25.2; one on a quiet host where the Docker VM went
down at 98%. Memory is the leading suspect: the suite now runs at the VM's limit.

**2026-10-05, five Linux full runs under WSL2, all with summary lines** (a 23 GB
VM, `-n 4`): four on the GPU (the first two with failures in the code since fixed, none a lost
process or an abort; the last two green, 20–22 min) and one on CPU, green in 9m24s. WSL2 is the steadier Linux host here;
the Docker attempts above remain unexplained.

**CI, 2026-10-05: memory, confirmed and fixed for the fast tier.** Every CI run since `-n auto`
(`f51227c`, run #14) died after 7–9 minutes with "The operation was canceled": the 4-vCPU,
16 GB runner ran out of memory and lost its runner agent. Reproduced in Docker with the
runner's limits (4 CPUs, 16 GB, no swap, no compile cache): OOM-killed, three of four xdist
processes lost. Two causes, two test-only fixes, the engine unchanged:
`tests/test_ore_bermudan_oracle.py::test_engine_is_grid_converged` alone peaked at 11.5 GB on
its 384-per-std, 10-std control grid, now 192 and 8 (within 1.6e-6 of it, 3.3 GB; the
rollback's memory was fixed properly by its matrix form on 2026-10-13, [I-83](#i-83), [details](details/precision.md#83-emulation-and-native-speed));
and each test process kept every XLA program it compiled, now dropped after each module
(`jax.clear_caches()` in `tests/conftest.py`). The same container then passed the fast tier,
2,461 passed, 2 skipped, at a 13.1 GB peak (page cache included), in 5m22s. The full suite
on that runner is not yet shown.

**2026-10-08, the Linux run of the 2026-10-07 code under WSL2 (24 GB VM, CPU, `-n 4`)** reached 78% with no
failure when the kernel's global OOM killer took a test process (5.9 GB resident); the suite
gained the ORE simulation tests (an OREApp run and the engine on the same portfolio per model)
since the green runs of 2026-10-05. Afterwards the distro instance stopped about 90 s after each
start, idle or not, until WSL was restarted. Memory remains the leading suspect.

**2026-10-11, memory confirmed for WSL2 at `-n 8` and `-n 4`.** The Linux runs of the kernel
rewrite (24 GB VM, CPU) took the whole VM down twice, at about 11% and with no summary line. A
log of each xdist process's resident memory and running test every 3 s showed the VM's use
rising past 17 GB at a rate of 1.4 GB per 5 s: four processes at 2.9 to 5.1 GB each plus the
demos' subprocesses. A process holds more than its test needs alone (the American's finer
grids at 5.1 GB, against a 1.3 GB peak alone): it keeps memory from the tests it ran before,
despite `jax.clear_caches()` after each module. The heaviest tests at that point need the same peak alone before and
after the rewrite (`/usr/bin/time`, HEAD against the working copy: Hull-White's pipeline 2,342
and 2,323 MB, the American's finer grids 1,273 and 1,272, the calibration portfolio 1,590 and
1,591, the component demos 2,803 and 2,808), so the rewrite added nothing; four processes no
longer fit in 24 GB. At `-n 2` the run finished in 18m11s, the VM at a 19.4 GB peak, two
processes at 7.8 GB (`tests/test_ore_xva_parity.py`'s simulation tests) and 8.8 GB
(`tests/test_profiling_and_jit.py`'s Greeks compile counts). A Linux full run here now uses
`-n 2`, and `-n auto` on a 16 GB CI runner would not fit.

**Current handling.** Since 2026-10-04 no pool, and no executor thread, exists: HTTP tests
start one engine worker, a separate process with no thread in the test process, and each
module stops it when it ends (`tests/conftest.py`); a worker also exits once its parent is
gone. If an abort recurs, the pool hypothesis is ruled out.

**To close.** Measure each xdist process's peak resident memory over a full run (Linux,
`-n 4`) and the heaviest tests; if memory explains the deaths, lower `-n` for the `full` CI job
or slim the heaviest fixtures. Then repeated clean full runs against a known-bad baseline,
with `JAX_COMPILATION_CACHE_DIR` on an empty directory to rule the cache in or out; one green
run proves nothing. Once the full tier fits CI's runner, run it on a schedule as well as by
hand: it holds the slow ORE simulation tests (L4) and the worker tests, which no automatic run
covers today (project audit, AUD-47).

<a id="i-66"></a><a id="q-2"></a>
### I-66 — No linter or type checker

**Severity:** Low · **Status:** OPEN · **Found:** 2026-09-24, audit Q-2 (pins and CI done)

**What is wrong.** No ruff/flake8, mypy/pyright or pre-commit, so nothing keeps the code
clean. `pyflakes engine` reports nothing since the layout change of 2026-10-08 (I-92), which
removed its findings with the modules they were in (re-export shims, unused imports, a string
forward reference, an f-string without placeholders); `pyflakes tests demos` still reports 19
unused imports and variables. `requirements.txt` does not mention the `profiling` extra.
`pyproject.toml` declares `pandas`, which nothing in `engine/`, `tests/`, `demos/` or
`compliance/` imports (project audit, AUD-49).

**To close.** As decided (A-20), in two steps. First, now: ruff in
`pyproject.toml` and CI with pyflakes' rules only, each finding fixed or marked (a re-export
listed in `__all__`), so `engine/` stays clean; `pandas` dropped from the dependencies. Then, among the fixes of what exists: a type
checker on `engine/`.

<a id="i-67"></a><a id="q-3"></a>
### I-67 — Test modules import each other and repeat fixtures

**Severity:** Low · **Status:** OPEN · **Found:** 2026-09-24, audit Q-3

**What is wrong.** Most cross-module imports moved to `tests/support/` on 2026-10-01
(`portfolio`, `lgm_engine`, `greeks`); test modules still import `bermudan_references` and
`date_helpers` from the tests directory itself.
`FIXTURES = ...traderx-eod` is defined in 15 files and `MARKET = {...flat-3pct-v1}` in 7.
Many tests reach into private functions, which freezes internal structure and makes
refactors (the configurable engine of 2026-10-04) break tests without behaviour changing.

**To close.** Shared helpers and constants in `tests/support/`; replace private-symbol tests
with public-entry tests where those refactors touched them. Collection must stay
identical except where a test is deliberately rewritten.

<a id="i-81"></a>
### I-81 — A cold job compiles about 180 one-operation programs; market risk vmaps a closure

**Severity:** Low · **Status:** OPEN · **Category:** Performance · **Found:** 2026-10-06,
making the AD Greeks one program

**What is wrong.** Code outside any jit runs JAX operations one by one, and each operation
compiles once per shape: on a cold demo job (`demos/demo_profile_small.py`, CPU) 180 of the
217 programs are a single `multiply`, `where`, `stack` and the like, about 2.5 s of
compilation plus their tracing. Most come from the scenario market's construction
(`lgm_numeraire`, `implied_log_discounts` in `engine/market_simulation/scenario_market.py`, about 60),
the CAM calibration (`cam.py`'s `zeta`, `calibrate_currency`), the options' known path
fixings (`engine/pricing/bermudan.py::_known_rates`) and the sensitivity market. Separately,
market risk vmaps a fresh closure over the jitted pricers per run (`revalue_trade`), the
pattern whose derivative programs were found (2026-10-06) to be kept by JAX only in internal caches of 2,048
entries (profiling §3.8): a market-risk job large enough to overflow them would compile its
batched programs again on the next run.

**Reach.** First-call time only: each such program is compiled once per process and shape and
then reused (a repeated demo job compiles nothing). No number is wrong.

**Current handling.** None.

**To close.** Jit the scenario market's construction and the other eager sites as
module-level functions with their data as arguments, and give market risk's batch a
module-level jitted function of `TradePriceFunction.pricer` and `.terms`, as the AD Greeks
(`engine.risk.greeks.ad._derivatives`). Jitting can move float64 at rounding level (XLA
fuses what ran op by op), so it shows the golden snapshot's change per array. Measured by the
compile probe of profiling §3.8 on the demo's job.

<a id="i-110"></a>
### I-110 — The documentation contradicts the code in places

**Severity:** Low · **Status:** OPEN · **Category:** Tooling · **Found:** 2026-10-10, the
project audit (AUD-15, AUD-46, AUD-47); checked 2026-10-11

**What is wrong.**

- [docs/getting-started/overview.md](../getting-started/overview.md) says engines like ORE run
  single-threaded on CPUs, where the README describes ORE's threaded and GPU paths.
- [docs/concepts/architecture.md](../concepts/architecture.md) calls the codebase three
  independent modules, and `demo_precision.py` a demo at three precisions; it runs ten.
- The README says the full pipeline compiles through XLA; ORE schedules and calendars, Sobol
  draws and other host work do not.
**Reach.** Readers; no number.

**To close.** Each statement corrected. (The table of what equals ORE, per product and
analytic, was this entry's until 2026-10-17, and is the parity matrix since 2026-10-18,
[I-116](#i-116); so is the rule on parity tests and the slow tier, now a fixed sample in the fast
tier and its statistical or swept extension in the full tier.)

---

<a id="closed"></a>
## Closed

Fixed issues. The full account is in the commit that closed each (`git log --grep "I-NN"`,
or the register's text at commit `8306073`). The test named guards the fix.

| ID | Fixed | Regression test |
|---|---|---|
| <a id="i-01"></a>I-01 | Swap Delta/Gamma/Theta were silently absent from portfolio results | `tests/test_portfolio_gap_fixes.py::TestSwapGreeksReachThePortfolioPath` |
| <a id="i-02"></a>I-02 | Bermudan Vega was never computed | `tests/test_portfolio_gap_fixes.py::TestBermudanVegaReachesThePortfolioPath` |
| <a id="i-03"></a>I-03 | No per-instrument NPV | `tests/test_portfolio_gap_fixes.py::TestPerTradeBaseNpv` |
| <a id="i-06"></a>I-06 | American exercise ignored ORE's broken-period `couponRatio` (up to 6× off) | `tests/test_ore_lgm_parity.py`, `tests/test_bermudan_swaption.py::TestMidPeriodBermudanExercise` |
| <a id="i-11"></a>I-11 | Risk measure unlabelled; no Monte Carlo error reported | `tests/test_risk_measure_label.py::TestPortfolioResultStatesItsMeasure` |
| <a id="i-09"></a>I-09 | The whole scenario cube was serialized into the JSON response (a request now asks for it inline, as a chunked, hashed artifact reference, or not at all: `cube_output`, decision A-17) | `tests/test_api_portfolio.py::test_arrays_by_reference_are_the_inline_arrays_and_none_leaves_them_out`, `tests/test_api_artifacts.py` |
| <a id="i-10"></a>I-10 (results) | Per-trade results were keyed by position beside the echoed ids (now one row per trade with its `trade_id`, the arrays' trade axes in that order) | `tests/test_api_portfolio.py::test_the_result_equals_the_in_process_run`, `tests/test_api_market_risk.py::test_the_result_equals_the_in_process_run` |
| <a id="i-10-configs"></a>I-10 (configs) | Trade configs had no identity (now a required `trade_id`, echoed as `PortfolioResult.trade_ids`) | `tests/test_trade_configs.py::TestEveryTradeNamesItselfAndItsDate`, `tests/test_pipeline.py::test_a_repeated_trade_id_is_refused` |
| <a id="i-12"></a>I-12 | `/version` named the API process's backend, the only device a result could be attributed to (every result now carries a `PrecisionReport` built where the job ran: its devices, backend and the formats read from its arrays) | `tests/test_api_portfolio.py::test_a_result_carries_the_workers_precision_report`, `tests/test_precision_report.py::TestReport` |
| <a id="i-13"></a>I-13 | A negative curve index priced against the wrong curve (trades now name a currency and index; a missing curve is refused before pricing, naming the trade) | `tests/test_portfolio_gap_fixes.py::TestCurveIndexValidatedBeforeAllPricing` |
| <a id="i-14"></a>I-14 | `generate_paths(precision=32)` leaked `jax_enable_x64=False` (`generate_paths` removed on 2026-10-01; nothing toggles the flag) | `tests/test_portfolio_entrypoint.py::TestPricePortfolioConcurrency` |
| <a id="i-15"></a>I-15 | The worker-pool concurrency test could not observe concurrency (and, until 2026-10-01, failed after the pricing tests by reusing one worker) | The pool and its test went on 2026-10-04; jobs now run one at a time (`tests/test_engine_worker.py::TestEngineWorkerPricing`) |
| <a id="i-17"></a>I-17 | A malformed note date failed the whole bundle | `tests/test_traderx_note.py::TestRefusalsAreNotePricingErrors` |
| <a id="i-19"></a>I-19 | The accrual tolerance rounded its own bound | `tests/test_traderx_note.py::TestToleranceIsDerivedNotConstant` |
| <a id="i-20"></a>I-20 | Impossible calendar dates aborted the whole bundle | `tests/test_traderx_note.py::TestImpossibleCalendarDates` |
| <a id="i-21"></a>I-21 | AD Greeks recompiled about 30 XLA programs per repeated call (the pricers are jitted with the trade as an argument; the Greeks no longer jit a fresh closure) | `tests/test_profiling_and_jit.py::TestCompileCounts::test_repeated_greeks_call_compiles_nothing`, `::test_a_different_trade_gets_its_own_greeks_from_warm_programs` |
| <a id="i-22"></a>I-22 | Each LGM calibration recompiled its bisection (now one program per helper shape, the basket an argument) | `tests/test_profiling_and_jit.py::TestCompileCounts::test_calibrations_of_one_basket_shape_share_one_program` |
| <a id="i-25"></a>I-25 | A scalar Greek crashed the HTTP result serializer | `tests/test_api_bond_schemas.py::TestBondGreeksSerializeOverHttp` |
| <a id="i-24"></a>I-24 | Bonds had no scenario NPV on the Hull-White model (refused; a broadcast t=0 value gave VaR 0, ES NaN) | `tests/test_hull_white_model.py::TestBondsOnEveryPath`, `tests/test_portfolio_bond_wire_through.py::TestBondsArePricedOnEveryPath` |
| <a id="i-26"></a>I-26 | Greeks for a bond maturing tomorrow crashed on the Theta reprice | `tests/test_portfolio_bond_wire_through.py::TestBondGreeksReachThePortfolioPath` |
| <a id="i-28"></a>I-28 | The `var_es` module demo crashed on a moved date | `tests/test_demos.py::TestComponentDemosRun` |
| <a id="i-29"></a>I-29 | A rounded exercise time silently dropped a coupon (exercise now given as dates) | `tests/test_ore_bermudan_oracle.py::TestExerciseDatesAreExact` |
| <a id="i-30"></a>I-30 | The `A(t,T)` variance term was nearly uncovered at t=0 (that formula went with the Hull-White pipeline; the model's bonds are QuantLib's on every state) | `tests/test_cam.py::test_hull_white_path_curves_equal_quantlibs_hull_white` |
| <a id="i-31"></a>I-31 | Bermudan/American floating coupons projected over the wrong period | `tests/test_ore_lgm_parity.py` |
| <a id="i-32"></a>I-32 | The Bermudan/American engine refused ORE's `ShiftHorizon` but 0, where ORE's builder defaults to 0.5 (1.6e-6 to 1.5e-4 from ORE's price there); on a path date it measured the shift horizon and opened an American's window from that date where ORE keeps the trade it built on the as-of date (up to 1.1e-3 of the American's largest value). Now any horizon, 0.5 by default, equal to ORE's engine today (1e-13 to 3e-12) and on ORE's simulated paths (1.7e-11, the LGM uncalibrated) | `tests/test_ore_lgm_parity.py` (`shift-0.5`), `tests/test_ore_xva_parity.py::test_an_uncalibrated_option_equals_ores_cube_on_ores_paths`, `tests/test_shift_horizon.py` |
| <a id="i-33"></a>I-33 | On Linux, worker-pool jobs hung once the parent had run JAX (fork) | The engine worker is a fresh interpreter (`subprocess`, never fork) since 2026-10-04: `tests/test_engine_worker.py::TestWorkerProcesses`, the full suite on Linux |
| <a id="i-34"></a>I-34 | The ORE oracle tilted a curve's first segment (ORE's rebuild reads the t=0 rate at t = 1e-4; the oracle now hands it the as-of quote that maps onto the engine's, 2.3e-6 → 1.5e-12 on a sloped first segment) | `tests/test_ore_lgm_parity.py::test_a_curve_sloped_in_its_first_segment_is_the_engines` |
| <a id="i-35"></a>I-35 | An American already in its window was exercisable on the evaluation date | `tests/test_trade_dates.py::test_seasoned_bermudan_and_american_equal_ore` |
| <a id="i-36"></a>I-36 | A non-ACT/365 floating leg projected the wrong forward | `tests/test_trade_dates.py::test_any_leg_day_count_equals_ore` |
| <a id="i-37"></a>I-37 | A European silently ignored `floating_spread` (refused by the Jamshidian engine, priced by Bachelier) | `tests/test_jamshidian.py::TestConfiguration::test_a_spread_is_refused_naming_the_trade`, `tests/test_pricing.py::test_european_today_equals_ores_default_engine` |
| <a id="i-38"></a>I-38 | Theta rolled a business day; ORE rolls a calendar day | `tests/test_trade_dates.py::test_swap_theta_equals_ore`, `tests/test_sensitivities.py::test_theta_rolls_one_calendar_day_from_a_friday` |
| <a id="i-39"></a>I-39 | Bond Theta had no add-back for a coupon paid in the period; reversed on 2026-10-18, when ORE's own sensitivity analytic showed it adds back no flow of the roll (I-116), the owner keeping ORE's definition (decision A-26; the roll's flows as their own figure, F-11) | `tests/test_ore_parity_settings.py::test_theta_onto_a_coupon_adds_nothing_back_as_ores` |
| <a id="i-40"></a>I-40 | The note's `rateSensitivity` ignored `fractionDecimals` | `tests/test_traderx_note.py::TestSensitivityUsesTheDeclaredFractionDecimals` |
| <a id="i-41"></a>I-41 | A European at zero mean reversion priced at intrinsic value (now refused) | `tests/test_jamshidian.py::TestConfiguration::test_non_positive_reversion_is_refused` |
| <a id="i-42"></a><a id="m-1"></a>I-42 | Hull-White simulated curves were not arbitrage-free against the input curve (deflated 10y bond +6.8% at 2y on a 3→5% curve) | `tests/test_hull_white_model.py::TestCurveFittedDrift`, `tests/test_cam.py` |
| <a id="i-43"></a><a id="m-3"></a>I-43 | Hull-White options were worth 0 after expiry instead of carrying the swap | `tests/test_pricing.py::test_an_exercised_physical_option_becomes_its_swap_and_a_cash_one_leaves` (both models) |
| <a id="i-44"></a><a id="a-2"></a>I-44 | Hull-White scenario pricing mixed Hull-White and LGM | `tests/test_pricing.py::test_the_path_model_is_the_calibration_ore_built_the_trade_with` (both models), `tests/test_hull_white_model.py::TestOneModelOnePipeline` |
| <a id="i-45"></a>I-45 | The Hull-White numeraire was a left-point bank account (E[1/N] +1.8% at 3y) | `tests/test_hull_white_model.py::TestExactNumeraire` |
| <a id="i-46"></a>I-46 | Hull-White Europeans were priced off the model volatility, without Vega | `tests/test_hull_white_model.py::TestEuropeansOnTheMarketVolatility` |
| <a id="i-47"></a>I-47 | Hull-White options calibrated to one caller basket, not their own | `tests/test_pricing.py::test_option_today_equals_ores_calibrated_grid_engine`, `tests/test_hull_white_model.py::TestOneModelOnePipeline` |
| <a id="i-48"></a>I-48 | Zero curves extrapolated a flat zero rate; ORE a flat forward | `tests/test_treasury_instrument.py::TestCurveInterpolation`, `tests/test_curves.py` |
| <a id="i-49"></a>I-49 | Bermudans and Americans were recalibrated on every path and date, 0.5–4% above ORE's exposure simulation (and one decision in 160 cells the other way); ORE never recalibrates there (its `XvaAnalytic` gives the valuation engine no model builders) and keeps the LGM it calibrated when it built the trade on the simulation market of the as-of date, which the engine now keeps: ORE's cube to 1.9e-11 under either model (calibrated on today's market instead, 1e-5). Per-path recalibration, which no ORE analytic does, is removed (decision A-24) | `tests/test_ore_xva_parity.py::test_every_trades_cube_equals_ores_on_ores_paths` (the six option cases, strict expected failures before), `tests/test_pricing.py::test_the_path_model_is_the_calibration_ore_built_the_trade_with` |
| <a id="i-50"></a>I-50 | Nothing compared the assembled simulation, cube or exposure with an ORE simulation (now an OREApp XVA oracle: its scenario dump rebuilds the engine's simulation market, swaps, Europeans and bonds equal ORE's cube path by path, the exposure definitions on ORE's cube are ORE's, and L4 holds; Bermudans/Americans wait on I-49) | `tests/test_ore_xva_parity.py` |
| <a id="i-52"></a>I-52 | Cash settlement was priced as physical | `tests/test_pricing.py::test_a_cash_settled_european_uses_the_par_yield_annuity` |
| <a id="i-56"></a>I-56 | Market risk and the CAM calibration had no route, and two routes were named like versions (now `POST /portfolio/market-risk` and `POST /calibration/cam`, one poll route for every job, `/v2/portfolio/price` and `schema_version` removed, and a test that fails on any setting without an API field; 2026-10-07, decision A-2) | `tests/test_api_market_risk.py`, `tests/test_api.py::TestTheCalibrationAnalytic`, `tests/test_api_completeness.py` |
| <a id="i-57"></a>I-57 | TraderX: a cached result was served before the submission id's binding was checked, so an id reused for a cached workload came back with that workload's result (TraderX case A-02; now one atomic `AttemptStore.submit` checks the binding, then the cache, binds the id to whatever it returns, and records the binding durably first; 2026-10-10) | `tests/test_traderx_routes.py::TestTheSubmissionIdBindsBeforeAnyCacheReturn`, `::TestSubmissionIdCannotCrossWorkloads::test_store_level_conflict_raises_even_when_the_target_is_cached`, `tests/test_traderx_publication.py::TestSubmissionBindingsAreDurable` |
| <a id="i-62"></a><a id="p-2"></a>I-62 | Hull-White Bermudan/American scenario pricing ran on the host (that pricer is removed; options are priced by the vectorized per-path engine) | `tests/test_pricing.py::test_the_path_model_is_the_calibration_ore_built_the_trade_with` |
| <a id="i-63"></a><a id="a-3"></a>I-63 | Hull-White trade configs carried copies of model parameters | `tests/test_trade_configs.py::TestEveryTradeNamesItselfAndItsDate::test_a_model_or_curve_on_the_trade_is_refused` |
| <a id="i-64"></a><a id="a-4"></a>I-64 | A trade's evaluation date defaulted to ORE's thread-local global | `tests/test_trade_configs.py::TestEveryTradeNamesItselfAndItsDate::test_without_an_evaluation_date_it_is_refused` |
| <a id="i-65"></a><a id="a-6"></a>I-65 | Demo data, the modules' `__main__` demos and the ORE test oracle shipped inside `engine/` (now `demos/`, `tests/support/`) | `tests/test_import_layering.py::test_engine_ships_no_demo_or_test_code` |
| <a id="i-68"></a>I-68 | The Hull-White model was chosen by the market's type, not by the configuration (now `HullWhiteConfig` in `CamConfig.ir`) | `tests/test_run_config.py::TestEveryModelRunsWithEveryEngineAndMethod` |
| <a id="i-69"></a>I-69 | The market path silently ignored `precision.pricing`/`risk`/`calibration`, `calibration_targets`, and a `base_currency` contradicting the simulation (now refused by name) | `tests/test_run_config.py::TestWhatThePipelineDoesNotImplementIsRefused`, `::TestConfigurationValues`, `tests/test_api_portfolio.py::test_an_unpriceable_request_is_a_400_and_no_run` |
| <a id="i-70"></a>I-70 | Bump Theta of a bond maturing the next day raised `BondPricingError` (now redemption − NPV, as ORE) | `tests/test_portfolio_bond_wire_through.py::TestBondGreeksReachThePortfolioPath::test_a_bond_maturing_tomorrow_does_not_crash_the_greeks` |
| <a id="i-71"></a>I-71 | A float32-tier worker turned x64 off, so its pricing and exposure ran in float32 where an in-process run used float64 | `tests/test_engine_worker.py::TestEngineWorkerPricing::test_runs_queued_together_give_the_bits_of_runs_one_after_another` (float64 and float32 jobs equal the direct call) |
| <a id="i-72"></a>I-72 | HTTP jobs ran in a process pool that pickled each request with its ORE dates frozen as text, compiled every job shape once per worker, broke for every later job when one worker died (`BrokenProcessPool`), and would have needed chip pinning on TPU (now one engine worker per host behind a durable queue) | `tests/test_engine_worker.py` (`test_a_second_identical_job_compiles_nothing`, `test_a_failing_job_fails_only_its_own_row`, `test_a_worker_killed_mid_job_leaves_it_interrupted`, `test_the_worker_prices_the_request_the_route_validated`) |
| <a id="i-73"></a>I-73 | A calibration that missed its basket was accepted unless a bucket hit the bracket's ceiling, and on a path not even then: a volatility dip after a spike priced at σ's floor (relative helper errors to 151), a zero volatility gave a NaN market value. Now ORE's acceptance on every bootstrap (the root mean square of the relative helper errors below `bootstrap_tolerance`, ORE's `Tolerance` and `BootstrapTolerance`, 1e-4 by default, one for the cross-asset model, decision A-24), a miss refused naming the trade or currency | `tests/test_calibration_acceptance.py` |
| <a id="i-75"></a>I-75 | Storage below 32 bits kept few bits of a concentrated column's spread: FP8 put its bits on the level of a bond's cube (1e6 ± 1e4) or a long log discount factor, and every path of a block rounded alike, a bias more paths did not remove (now a concentrated block, one sign and the farthest entry at most three times the nearest, is stored as its deviations from its midpoint, `Precision.offset="block"`, the default; the storage harness measures every format, 2026-10-09) | `tests/test_precision_harness.py::test_a_bonds_fp8_cube_column_is_unbiased_with_block_offsets`, `tests/test_precision.py::TestBlockOffsets` |
| <a id="i-74"></a>I-74 | A calibration basket read ORE's global evaluation date, so a later date (another caller's, or the wall clock past a helper's fixing) failed it with a missing fixing | `tests/test_ore_lgm_calibration.py::test_the_basket_does_not_depend_on_ores_global_evaluation_date` |
| <a id="i-79"></a>I-79 | Never run on a GPU (now the `gpu` extra under Linux or WSL2, verified on an RTX 5060: no preallocation, the API's JAX on the CPU, float32 products at float32 rather than TensorFloat-32, deterministic kernels; the demo and the full suite run on the GPU) | `tests/test_accelerator_defaults.py`, `tests/test_environment.py::test_a_gpu_plugin_is_jaxlibs_version`, the full suite on the GPU |
| <a id="i-80"></a>I-80 | Importing `engine` changed JAX and XLA settings for the whole process (now x64 only: each matrix product states its precision through `engine.precision.matmul`; deterministic GPU kernels set by the engine worker and the tests; GPU preallocation off in the demos and the tests; 2026-10-06, decision A-22) | `tests/test_accelerator_defaults.py` (`TestImportingTheEngine`; `TestEveryMatrixProductStatesItsPrecision::test_in_every_pipeline`, red on the bare products) |
| <a id="i-82"></a>I-82 | An American priced on a path date after the last reference-grid date in its window and before its last exercise (no calibration helper left) raised `ValueError: Need at least one array to stack` (now the engine's volatility, as a calibration today with no helper; found on 2026-10-07) | `tests/test_root_solvers.py::test_a_date_with_an_exercise_left_and_no_helper_prices_on_the_model_as_built` (since 2026-10-12 no date is recalibrated, I-49) |
| <a id="i-84"></a>I-84 | The simulation market held each curve's tenor points at the tenors' times from the simulation date; ORE's `ScenarioSimMarket` holds them at the times from the as-of date (swaps and Europeans up to 0.45% of their largest path value off ORE's cube; found by the ORE simulation oracle's L3, 2026-10-07) | `tests/test_ore_xva_parity.py::test_every_trades_cube_equals_ores_on_ores_paths`, `tests/test_pricing.py` (path curves at ORE's times) |
| <a id="i-83"></a>I-83 | A Bermudan/American on the paths held each step's `[nodes, nodes]` interpolation per path (0.3–0.7 MB per path and step; the demo's job asked for 277 GB at 65,536 paths) and computed in one format, float32 or float64 (now the induction as matrix products in difference form, one operator per step shared by every path and column, the cashflows a product of each path's coefficients with functions of the state, the coupons cached in progress in a few reused columns: a few copies of a path's columns on the grid, 10 to 41 KB per path at float64, a date's paths priced in blocks within a budget; the demo's job at 262,144 paths in 3.0 s and 2.0 GB on the CPU, 2.4 s on an RTX 5060, where 1,024 had run out of memory; every compute format, the Bermudan's market-risk P&L in difference form too, the exercise decisions on the paths exposed and measured, decision A-24; 2026-10-13, F-07) | `tests/test_rollback_difference_form.py::TestMemory` (red on the code before: 6.1 MB per path against a 0.085 MB bound; the demo's job failed there at 65,536 paths), `::TestDifferenceForm`, `::TestOneProgramPerTrade` |
| <a id="i-85"></a>I-85 | Exposure profiles started from each trade's value on today's market; ORE's start from its value on the simulation market of the as-of date, the cube's `T0` (up to 4.2% on the shared portfolio) | `tests/test_ore_xva_parity.py::test_the_pipelines_profiles_start_from_ores_t0` |
| <a id="i-86"></a>I-86 | A cash-settled option's exposure stopped at its underlying's maturity, so its time-weighted EPE kept accruing after it settled; ORE's `Trade::maturity()` is its last exercise date (2026-10-07) | `tests/test_ore_xva_parity.py::test_the_profiles_of_ores_cube_are_ores_reports` |
| <a id="i-92"></a>I-92 | `engine/`'s packages were named for layers and two modules sat at its root (now subpackages only, each named for what it holds and layered: `market_data`, `models`, `instruments`, `traderx`, `calibration`, `market_simulation`, `pricing`, `risk/{greeks,market,counterparty}`, `run`, `api`; [details/package-layout.md](details/package-layout.md)) | `tests/test_import_layering.py::test_no_module_but_init_at_the_root`, `::test_packages_import_only_the_layers_below`, `::test_risk_kinds_import_only_the_kinds_below` |
| <a id="i-96"></a>I-96 | Compute below float32 biased every figure: each product's path-independent operand was rounded once for every path (now carried in two terms of the format, the single pass a format of its own, `<name>_1x`; 2026-10-11, decision A-23) | `tests/test_compute_precision.py::TestPipeline::test_no_rounding_of_a_shared_operand_biases_the_paths` (red on the code before: 14 to 24 standard errors), `TestProducts::test_the_shared_operands_rounding_biases_a_single_pass_not_two_terms` |
| <a id="i-99"></a>I-99 | On a GPU a block-offset storage took a block's deviations from its unrounded midpoint while keeping the float32 one, so a stored entry missed its value by float32's rounding of it: XLA removed the float64 -> float32 -> float64 round trip of the offset's cast (found and fixed 2026-10-11, the offset now rounded on float32's grid) | `tests/test_precision.py::TestScaledStorage::test_any_scenario_count_and_axis_short_last_block` (red on an RTX 5060 at the code before, its one-entry blocks 5e-8 off) |
| <a id="i-100"></a>I-100 | Market risk added the P&Ls of trades in different currencies unconverted (now every amount is in the request's `base_currency`, USD by default as a portfolio run's without a simulation, each trade converted at the as-of FX spot in float64, a missing spot refused before any work naming the trade and the pair, the result naming its currency and warning that FX is not shocked; the portfolio run's preflight refuses a missing spot, and a trade in a currency its simulation does not model, before any work; 2026-10-11) | `tests/test_market_risk.py::TestReportingCurrency` (red on the code before: the figures did not move with the EURUSD spot; VaR_99 601,048 where the converted P&Ls give 633,967 at 1.10), `tests/test_run_config.py::TestWhatThePipelineDoesNotImplementIsRefused::test_a_trade_with_no_spot_into_the_reporting_currency_is_refused_naming_it`, `::test_a_trade_in_a_currency_the_simulation_does_not_model_is_refused` |
| <a id="i-101"></a>I-101 | A negative option notional booked a short European but a long Bermudan or American on the reversed swap (the Bermudan payer at -1e6 was +15,017.85 where ORE's short is -35,712.95), and a zero one priced a European to NaN (now every option has ORE's `LongShort`, `long_short` on the configs and the API, `Long` by default; a short is minus the long one in every engine, today, on the paths with the holder's exercise as ORE's `OptionWrapper`, in the AD and bump Greeks and in market risk; a zero or negative option notional refused naming the trade; 2026-10-11) | `tests/test_option_position.py`, `tests/test_shared_portfolio.py::test_a_short_option_today_is_ores_short_option` (against ORE's trade builder with `LongShort` Short), `tests/test_ore_xva_parity.py::test_a_short_option_on_ores_paths_is_ores_short_option` (each red on the code before) |
| <a id="i-102"></a>I-102 | VaR took index floor(N(1−p)), one above QuantLib's ceil(N(1−p)) − 1 where N(1−p) is a whole number (VaR 15 where ORE's is 16 on 16 scenarios at 0.9375), and quantile keys were rounded to six decimals, so close levels shared one (now QuantLib's rank, `var_rank`; each key the level's percentage exactly; a run's levels in (0, 1), each once, documented as broader than ORE's [0.9, 1); 2026-10-11) | `tests/test_var_es.py::TestExactRankBoundaryAgainstORE`, `::TestQuantileLabels`, `tests/test_market_risk.py::TestRunValidation::test_close_quantiles_keep_their_own_figures`, `tests/test_exposure.py::TestDefinitions::test_close_quantiles_keep_their_own_keys` (each red on the code before) |
| <a id="i-55"></a><a id="a-1"></a>I-55 | No result said whether its precision combination had been shown adequate for its figures, so a float32 exposure profile looked exactly like a validated one (now the evidence table, `compliance/precision-evidence.md`, generated from the precision harness's measurements against decision A-11's standard, `compliance/precision-acceptance.json`: the statistical rule, an absolute rule for figures float64 holds fixed, MAR32.42's P&L attribution test for market risk, each row's path ceiling, code revision, market, seeds and device; every portfolio and market-risk result warns of each figure without a passing row; the paired standard errors' limits are [I-111](#i-111); 2026-10-11) | `tests/test_precision_evidence.py` (the rules, the lookup, the files of record against the measurements, a reduced run's warnings end to end; red on the code before, where an FP8 run's `warnings` were `[]`, and the ceiling test red under a ceiling that skips a failing path count), `tests/test_precision_harness.py::TestMeasure::test_the_figures_measured_are_those_the_engine_warns_of` |
| <a id="i-105"></a>I-105 | Exposure reductions expanded every stored column into a float64 cube even when output was omitted; now streamed at float64 with an exact netting sum and PFE, paired columns retained separately, cube assembly only on request, rollback budget per device (2026-10-15; precision design §17) | `tests/test_exposure_storage.py`, `tests/test_sharding.py` (initial output-omission regression red on the pre-fix pipeline) |
| <a id="m-4"></a>Audit M-4 | Trades were defined relative to the evaluation date (now absolute dates) | `tests/test_trade_dates.py` |
| <a id="m-5"></a>Audit M-5 | Theta re-rolled the trade instead of ageing it | `tests/test_trade_dates.py` |
| <a id="r-1"></a>Audit R-1 | Cube quantiles were reported as VaR/ES (now exposure profiles; market-risk VaR/ES by t=0 revaluation) | `tests/test_exposure.py`, `tests/test_market_risk.py`, `tests/test_market_risk_ore_parity.py` |
| <a id="p-3"></a>Audit P-3 | The precision study bypassed the engine's own FP32 path | `demos/demo_precision.py` |
| <a id="a-5"></a>Audit A-5 | Instruments imported validators from the layer above | `tests/test_import_layering.py` |
| <a id="a-7"></a>Audit A-7 | The bond pricer was a separate plain-Python implementation | `tests/test_market_risk.py::TestRevaluation` |
| <a id="i-98"></a>I-98 | Per-path product operands had no unbiased rounding policy; keyed stochastic compute rounding removes their conditional rounding bias, nearest retained as the measured baseline and nonlinear FP8 failures left failing (2026-10-16) | `tests/test_precision_accuracy.py::test_stochastic_product_has_no_binade_rounding_bias`, `::test_stochastic_keys_are_data_and_sample_prefixes_reproduce`; `rounding` and `rounding_options` campaigns |
| <a id="i-114"></a>I-114 | The precision evidence lookup ignored pricing settings, so rows measured with Bachelier, on the coarse grid or on a fixed LGM validated a Jamshidian European, ORE's default grid and a Bermudan recalibrated under each market-risk scenario; every row now holds under its campaign's per-product kernel settings (`PricingConfig.kernel_settings`: the European engine and Jamshidian's solver, the grid's `nx`/`sx` and an American's steps, `model=fixed`/`recalibrated`), an unmeasured configuration is warned of naming what was measured, and the `market_risk_calibrated` campaign measures the recalibrated kernel (2026-10-18) | `tests/test_precision_evidence.py::TestSettingsEndToEnd` (each red on the code before: no warning), `::TestSettingsLookup`, `::TestKernelSettings`, `::TestRecord::test_each_record_holds_its_campaigns_pricing_settings` |
| <a id="i-116"></a>I-116 | ORE parity was shown on one market and the default settings, so a Theta 60–290% off had passed: the parity matrix (every product by analytic by setting, a test or a stated reason, generated from the tests' `ore_parity` markers into docs/reference/ore-parity.md) and seeded sweeps against ORE's NPV, sensitivity and historical-simulation VaR analytics, which found five defects, each fixed (a deal strike ORE writes to six decimals, a helper ending on the underlying's last payment date, an American without a calibrating helper refused, Theta without an add-back, the Theta market's volatilities from the next business day), and the settings runs on ORE's paths a sixth (an American whose window opened before the as-of date never exercised on a path) (2026-10-18) | `tests/test_ore_parity_matrix.py`, `tests/test_ore_parity_sweep.py`, `tests/test_ore_parity_settings.py`, `tests/test_ore_xva_parity.py::test_other_settings_equal_ores_simulation_on_ores_paths` (each red on the code before) |
| <a id="i-113"></a>I-113 | Long continuation values and stressed market-risk revaluations crossed zero; bound before LongShort/delivery and before/after P&L storage (2026-10-16) | `tests/test_precision_accuracy.py::test_long_option_bound_precedes_position_and_physical_delivery`, `::test_market_risk_cannot_revalue_a_long_option_below_zero` (both red before fix) |
| <a id="i-111"></a>I-111 | Raw variance diagnostics could be read as nominal confidence bounds despite rare-event undercoverage; the shared Python/HTTP availability contract now declines all nominal intervals without a validated tail bound, retains raw errors, audits each date including zero-width misses, and renames the conditional ES diagnostic (2026-10-16). Supported intervals are F-10 | `tests/test_uncertainty_availability.py`, `tests/test_precision_harness.py::TestMeasure::test_replicated_diagnostics_are_per_date_and_never_certify_empirical_constants`, `tests/test_precision_accuracy.py` (wire metadata and warm compile reuse) |
| <a id="i-51"></a>I-51 | The sensitivities were never run against ORE's sensitivity analytic, and their Theta market kept each curve's discount factors fixed in dates where ORE's holds its tenor points at their times from the as-of date as the reference date moves (60-290% of every trade's Theta off ORE's); now every Delta, Gamma, Vega and Theta of the shared portfolio equals ORE's own analytic, run in-process, within 2e-14 of the trade's NPV, calibrated options recalibrated, and `SensitivityConfig.swaption_vol_decay`, which a simulated volatility does not read, is gone (2026-10-17) | `tests/test_sensitivity_ore_parity.py` (red first: all eight trades on Theta) |
| <a id="i-78"></a>I-78 | The AD Greeks differentiated the market curves' pillars, not ORE's sensitivity market (a near-par swap's discount Delta 2% off the bump method's), and held a Bermudan's calibration where the bump recalibrates; now they are derivatives on ORE's sensitivity market with the calibration inside the price function (its implicit derivative on the whole basket), ORE's Greek as the shift vanishes for swaps, Europeans and bonds, and for a grid-priced option, whose price is piecewise smooth, labelled with the gap on the result (2026-10-17) | `tests/test_greeks.py::TestAgainstTheBumpMethod`, `tests/test_greeks_bermudan.py`, `tests/test_sensitivity_ore_parity.py::test_the_ad_greeks_are_ores_as_the_shift_vanishes` |
| <a id="i-112"></a>I-112 | Market risk held a calibrated Bermudan's/American's LGM under every scenario where ORE's historical P&L recalibrates it (up to 0.19% of the value, flagged); now recalibrated on every scenario's curves, in float64, a scenario that misses refused, each scenario priced on its own model in difference form, equal to ORE recalibrating to 7e-11 (2026-10-17) | `tests/test_market_risk_ore_parity.py::test_every_scenario_revalues_as_ore_does[bermudan-recalibrated]`, `[american-recalibrated]` (red first: 89.4 off), `tests/test_market_risk.py::TestRecalibration` |
| <a id="i-103"></a>I-103 | Malformed inputs were accepted: an unsorted curve priced silently wrong, a NaN rate or a negative pillar time priced, unequal times and rates failed inside the interpolation, a European expiry inside a leg's last accrual period raised a bare `StopIteration`, and four API models dropped unknown fields; now a curve's pillars are checked when built (`ZeroCurveConfig`), a live European entering no whole period is refused naming the trade and the dates (an expired one is worth nothing, as ORE builds no option; a Bermudan's empty exercise stays ORE's), every API model is strict, and PFE quantiles are checked (2026-10-18) | `tests/test_input_checks.py` (red first, measured: 50,679.68 for 51,133.41), `tests/test_run_request.py::TestTheCheckBeforeAnyWork::test_a_pfe_quantile_outside_the_unit_interval_or_repeated_is_refused` |
| <a id="i-87"></a>I-87 | The HTTP API was one route per engine function: now one run request, `POST /runs` with `market`, `portfolio`, an `analytics` list and a `config` whose sections follow ORE's files, one result that says what each figure is (the configuration as run, the dates, the reporting currency, trade ids on every axis, each Greek's factors, each analytic's pricing settings, the engine's revision and its jax, jaxlib and ORE versions), the exercise decisions on request, `GET /config/defaults`, `?wait=` long polls, and `engine.run.run` in Python; the four routes and `/jobs` removed; a run of one analytic is its entry point's direct call, a run of several each alone (2026-10-18) | `tests/test_run_request.py::TestOneAnalyticIsItsEntryPoint`, `::test_a_run_of_several_analytics_equals_each_alone_and_compiles_nothing_more`, `tests/test_api.py::TestRunsOverHttp` |
| <a id="i-88"></a>I-88 | The API completeness test checked field names only; now every field of every request schema set to a value other than its default must change its own setting of the engine's configuration and nothing else, and the run request's mapping onto `price_portfolio`, `run_market_risk` and the cross-asset model is checked field by field (2026-10-18) | `tests/test_api_completeness.py::test_every_fields_value_arrives_in_its_setting_and_only_there` (red: `::test_a_schema_that_drops_a_field_fails_the_value_check`) |
| <a id="i-89"></a>I-89 | A retried submission ran twice; now an optional `idempotency_key`: the same key and body (its canonical JSON, defaults filled in) return the first run's id, another body is a `409` checked before anything is returned, and two API processes on one queue get one job (2026-10-18) | `tests/test_engine_worker.py::TestIdempotency`, `::TestRoutesOverTheQueue::test_a_retried_submission_is_the_same_run_and_another_body_a_409` |
| <a id="i-95"></a>I-95 | A precision setting the run did not read was accepted, and echoed by its precision report; now refused, naming the field: any precision with no analytic that reads it (today's values, Greeks and calibration are float64, A-10), and by market risk its `market` stage and a `simulation.compute` other than its accumulate format; the precision demo's market-risk policy sets only the two stages it uses (2026-10-18) | `tests/test_run_request.py::TestTheCheckBeforeAnyWork::test_market_risk_refuses_the_precision_it_does_not_read`, `tests/test_api.py::TestPrecisionOnTheWire::test_a_precision_no_analytic_reads_is_a_400` |
| <a id="i-94"></a>I-94 | The LGM was calibrated by two implementations: the standalone bootstrap (`calibrate_lgm_sigma`, its basket and Bachelier helper prices, `POST /calibration/lgm`) is removed with its tests, the run request's `calibration` analytic returning each currency's sigma, each helper's market and model value and ORE's calibration error; ORE's bootstrap is `engine.calibration.lgm` (2026-10-18) | `tests/test_api.py::TestTheCalibrationAnalytic`, `tests/test_ore_lgm_calibration.py` |
| <a id="i-76"></a>I-76 | The job queue kept every job and result forever; now the worker purges, between jobs, the finished jobs older than `JAX_RISK_RETENTION_DAYS` (30 by default; `off` keeps every one) with their artifacts, never a run submitted with `keep` (2026-10-18) | `tests/test_engine_worker.py::TestCancelAndRetention`, `::TestWorkerLoop::test_the_worker_purges_finished_jobs_past_the_retention_period` |
| <a id="i-77"></a>I-77 | A worker that could not start left jobs `pending` with no signal, and a start that raised was an error for a job already queued; now the supervisor records each child's exit, a pending run's status and `/health` say the worker cannot start (two failed starts running, or one it could not launch) with the tail of its output, and the submission is answered with the run's id (2026-10-18) | `tests/test_engine_worker.py::TestSupervisor`, `::TestRoutesOverTheQueue::test_a_worker_that_cannot_start_leaves_the_run_pending_with_the_reason` |
| <a id="i-91"></a>I-91 | No job could be cancelled; now `POST /runs/{run_id}/cancel`: a pending run at once, a running one at its next phase boundary (each phase of the pipeline and of market risk, each trade's pricing and Greeks), the worker going on to the next run with its compiled programs (2026-10-18) | `tests/test_engine_worker.py::test_a_real_run_stops_at_the_phase_boundary_a_cancel_reaches`, `::TestWorkerLoop::test_a_job_cancelled_while_running_stops_at_its_next_phase_boundary`, `::TestRoutesOverTheQueue::test_cancelling_a_pending_run_and_a_finished_one` |
| <a id="i-106"></a>I-106 | The queue's migration ran in autocommit statements, so a crash after its `ALTER` stopped the file opening (`duplicate column name: kind`), and the worker's lock was per queue file, so two queues on one host had two workers; now the migration (to schema 3) is one `BEGIN IMMEDIATE` transaction, and the lock is per host (`JAX_RISK_WORKER_LOCK` for another device set), recording the queue its worker serves, a run on another queue of the host refused with a `503` (2026-10-18) | `tests/test_engine_worker.py::TestMigration` (red first on the code before), `::TestRoutesOverTheQueue::test_a_host_whose_worker_serves_another_queue_refuses_the_run`, `::TestWorkerLoop::test_a_second_worker_on_the_host_exits_at_once` |

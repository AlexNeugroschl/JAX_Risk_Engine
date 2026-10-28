# American-swaption VaR/ES accuracy and timing experiment

Recorded 2026-10-28 at the owner's request. Planned work under
[F-07](../features.md#f-07), [I-121](../known-issues.md#i-121) and
[I-122](../known-issues.md#i-122); this document records no new measurements.

## Question and scope

For a portfolio of at least 100 American swaptions, measure scenario valuation,
portfolio P&L, VaR and ES accuracy and runtime against ORE on CPU FP64. Compare this
engine on CPU FP64, a CPU FP32 control, an H100 and a TPU. Start with basic products;
then measure the benefit and cost of explicit multipass/correction options.

This is a bounded precision experiment, not a production risk-model validation.
The initial risk factors are curve pillar rates; swaption volatility quotes remain
fixed, although the LGM is recalibrated under each scenario when Bootstrap is selected.
American exercise is discretized on a grid. Matching ORE at the same grid establishes
implementation parity, not convergence to the continuous-exercise price.

Use the current [precision/execution contract](../../reference/explicit-precision-policy.md).
Accelerator cases use `profile="float32"` for internal kernels and explicit pricing
product overrides. FP64 on TPU remains refused. No implicit CPU placement, precision
downgrade or software emulation is part of this experiment. Host preprocessing and
result assembly remain included and reported. Older plans retaining fixed FP64
internals do not describe the TPU configuration for this experiment.

## First deliverable: a frozen synthetic dataset

Start from [the market-risk fixtures](../../../tests/market_risk_support.py),
not a live market-data feed. Their as-of date is 2026-07-30; the curves, normal
volatilities and shock covariance are synthetic. The named USD 6M simulation index
is a test convention, not a claim to represent a currently traded index.

Freeze the following together before any hardware timing:

| Input | Initial choice |
| --- | --- |
| Market date and currency | The fixture's common as-of date, USD |
| Discount curve | The fixture's sloped zero-rate curve and its pillar times |
| Forwarding curve | Its separate 6M-index zero-rate curve and pillar times |
| Volatilities | The fixture's ATM normal-volatility expiry/tenor matrix |
| Scenarios | Fixed, saved curve shocks with named factor order, horizon, generator and seeds |
| Conventions | Explicit index, calendar, day-count, payment, settlement and interpolation conventions shared with ORE |
| Pricing | Matching LGM model, calibration strategy/tolerance, state grid and exercise grid |

Normal-volatility units and zero-rate units must be explicit: rates are decimal
rates, not percentage or basis-point strings. This ATM-only surface assumes no
strike smile; varying trade strikes does not add observed smile information
([I-54](../known-issues.md#i-54)). Curve pillars and volatility coverage must reach
the trades and calibration helpers; identify or avoid extrapolation.

Construct a deterministic portfolio from ten valid schedule/exercise-window
combinations, five strike offsets around each combination's forward par swap rate,
and both payer/receiver directions: **10 x 5 x 2 = 100 different trades**.
Choose the strike offsets and notionals in the dataset manifest before collecting
results. Use positive notionals, unique IDs, physical settlement and the supported
USD index conventions. Prefer unseasoned forward-starting swaps initially so no
historical fixings are needed. Check the exercise windows against the underlying
cashflows. Reuse a manageable number of schedule shapes while recording their count;
do not disguise 100 copies of one trade as a diverse portfolio.

Begin with long positions in the main portfolio. Keep a separately named,
partially offsetting long/short portfolio as a later cancellation stress case;
avoid exact cancellation making relative VaR/ES errors meaningless.

Before expanding to all 100, take a small representative subset and verify on CPU
FP64: trade construction, base NPVs, calibration fit, a zero shock, and positive/
negative curve shocks against ORE. Invalid trades, calibration failures and material
oracle differences must be diagnosed, not removed silently or fixed by relaxing
tolerances. Passing this pilot is the first dataset acceptance check.

Store the market, portfolio, pricing settings, canonical scenario inputs and a
manifest together. The manifest records synthetic provenance, all generation rules,
units, seeds, trade/factor ordering, file hashes, source revision and library versions.
Keep generated outputs separate from these inputs. Shared inputs may be cast by the
explicit precision policy, but must never be regenerated independently per device.

Real market data is a later, separately named dataset. It needs a consistent as-of
snapshot of compatible discount/forward curves and swaption normal vols, supported
index conventions and documented provenance. A real historical scenario study also
needs consistent curve history; a spot snapshot alone supplies no empirical shock
distribution. The engine consumes prepared curves, so curve bootstrapping from raw
instrument quotes would be additional work. Do not label synthetic history as observed
history or synthetic valuations as evidence of real-market accuracy.

## Reuse and remaining preparation

Reuse [the API comparison](../../../benchmarks/README.md) and
[its ORE worker](../../../benchmarks/api_worker.py). The worker already uses native
portfolio historical-P&L batches of up to 4,096 scenarios, rather than launching an
ORE application for every individual trade-scenario price. It reuses the resulting
P&L in `ORE.RiskStatistics` to match the engine's VaR/ES conventions. ORE's historical
VaR report has a different tail-inclusion convention; do not compare its ES directly
without accounting for that difference.

Extend the workload registry/request preparation for the new 100-American portfolio,
the FP32 internal profile and the explicit product variants. The old single-swap and
five-trade comparison campaigns retain older precision/grid choices and are not the
new workload. Preserve the existing batching, common output mode, exact request
recording, resumable evidence where available, and ORE reference reuse.

The API benchmark encodes common shocks as synthetic history for ORE's historical
analytic. Check that reconstructed window moves match the canonical shocks within a
declared floating-point bound and that both implementations use the same windows.
Account for any actual submitted-input differences in numerical comparisons.

## Precision cases

Keep scenario inputs, stored P&L and risk reductions in FP32 initially on the
accelerators. Change pricing product precision independently to isolate product error;
study lower storage precision separately afterward. FP16/FP8 products retain FP32
accumulation and surrounding FP32 arithmetic, rather than making every engine
operation FP16/FP8.

| Case | CPU | H100 | TPU |
| --- | --- | --- | --- |
| FP64 | ORE oracle and engine reference | Optional bounded hardware control | Refused |
| FP32 profile | Control separating profile changes from device changes | First accelerator control | First accelerator control; full workload unverified |
| Basic FP16 products | No emulation campaign required | Preflight known subnormal-sensitive cases | Bounded diagnostic; earlier native attempts rejected or failed arithmetic |
| Basic BF16 products | No emulation campaign required | Test | Test |
| FP8 E4M3FN and E5M2 products | No emulation campaign required | Experimental; instruction use and arithmetic need verification | Experimental; matrix probes passed, vector widening remains relevant |
| BF16 three-pass | Optional diagnostic only | Explicit comparison | Explicit comparison |
| Selected residual/scaling options | Optional diagnostic only | Add after basic results | Add where the compiler route supports them |

Distinguish BF16 three-pass from shared-operand residual terms. Test those options
separately before combining them. Scaling, FP16 subnormal normalization, BF16 vector
padding and special FP8 algorithms remain explicit variants whose cost is counted.
Do not enable an entire collection of corrections silently. TPU FP32 can itself use
multiple BF16 passes; record the algorithm as well as the requested format.

## Execution order and evidence

1. **Dataset and CPU pilot.** Establish the small-subset checks above and measure
   representative ORE and engine FP64 runtime/memory. Use matching ORE Grid solver
   settings. Refine the state/exercise grids on representative trades separately to
   assess discretization error; do not change grids between precision comparisons.
2. **CPU portfolio reference.** Expand to 100 trades and save per-trade base/scenario
   PVs, P&L and portfolio statistics from ORE and the engine in FP64. Start at 2,048
   scenarios after the small pilot. Investigate discrepancies before using this
   engine's FP64 output as a reusable reference for accelerator comparisons.
3. **Accelerator preflight.** On each device, run representative real American
   workload shapes with about 32-64 scenarios. Check compilation, calibration,
   finite values, zero-shock P&L and repeatability. Retain actual quantized operands
   and product checks when diagnosing arithmetic faults. Inspect optimized lowering
   and, where available, instruction/profiler evidence for the kernels measured.
4. **Bounded full comparison.** Run the successful configurations on the same 100
   trades and saved scenarios. Consider 16,384 scenarios only after the pilot gives
   a practical runtime estimate. This is 1,638,400 trade-scenario valuations per
   configuration, with roughly 164 observations in a 1% tail. Choose VaR/ES levels
   explicitly (initially 95%, 97.5% and 99%) and report actual tail counts/ties.
5. **Explicit improvements and stability.** Compare selected multipass/correction
   variants with their basic counterpart. Add independent seeds, higher counts and
   the cancellation holdout as measured cost and tail uncertainty justify them.
   Keep tuning inputs/seeds separate from holdouts. Do not promise confidence
   intervals unsupported by the sampling design.

Report absolute and relative errors for per-trade prices/P&L, portfolio P&L and
VaR/ES, plus tail membership changes, non-finite values and calibration failures.
Use absolute or scale-normalized errors near zero. Declare comparison bounds before
the main runs; distinguish implementation/arithmetic failures from expected
quantization error and from poor financial accuracy of an otherwise correct product.
Matched scenarios isolate numerical differences; independent seeds/counts assess
sampling stability. An ORE match at finite scenario count does not establish the
true distribution's VaR/ES.

Measure first-job/compilation time, repeated completed-job wall time and memory
(or explicit counter unavailability). Include preparation, calibration, conversions,
transfers, reductions and requested output retrieval in API timings. Synchronize
accelerator completion. Save evidence writing outside the clock. Record scenario
batch sizes, distinct compiled shapes, hardware/compiler/library settings and the
source/input hashes. Keep direct-kernel timings separate from complete API timings.

Retain unsupported algorithms, compiler widening, arithmetic failures, inaccurate
results, timeouts, memory failures and no-speedup cases. A dispatch label or low-dtype
array alone does not prove physical low-precision instructions. Experimental runs
may proceed without prior qualification; limit any native-speed or accuracy claim
to the operations, devices and workload actually verified.

## Compute limits and completion

Prepare inputs, the runner and CPU pilot locally first. This planning task authorizes
no cloud provisioning. Before any paid cloud run, present the pilot-based resource/time
estimate, proposed spend cap, automatic deletion deadline and cleanup procedure for
authorization. Do not assume a previous trial's allowance covers a new campaign.
Use the smallest single-device trial that resolves the next question; save results
incrementally and clean up temporary cloud resources at completion or the deadline.

The deliverable is a reproducible dataset and report linking ORE/CPU references to
each accelerator/precision result, accuracy and timing together, with failed and
unverified cases visible. Updating the engine's shipped acceptance evidence or
claiming cluster scaling is separate work.

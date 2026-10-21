# Native FP8 and numerical reliability

Execution plan and implementation status, 2026-10-20; overnight source verification
completed 2026-10-21 UTC. Tracks [I-97](../known-issues.md#i-97),
[I-121](../known-issues.md#i-121), [I-122](../known-issues.md#i-122) and
[F-07](../features.md#f-07). The [roadmap](../roadmap.md) lists remaining work.
The [measured report](../../../compliance/precision-research/2026-10-20/README.md)
contains raw records, source snapshots, failures, traces and reproduction commands.

## Delivered implementation and evidence

| Work package | Implemented | Measured result / limit |
|---|---|---|
| Comparison matrix and reference | FP16, combined/compute/storage policies, explicit independent seed identities, independent FP64 references with empirical SE/convergence and an adequacy gate | Real FP16 reference study completed; two of 64 equal-accuracy rows unresolved and their speedups withheld; no confidence-coverage claim |
| Numerical boundaries | Stable `expm1`/series H(t) with zero-reversion derivatives; zero-volatility European liveness by expiry; bond redemption with its final coupon and liveness through the last payment | Eight tests fail on the old engine; all nine pass after fixes. 460 FP64/FP32 golden arrays remain within existing margins; I-104 closed |
| Native qualification | Seven shape families, awkward/batched/GEMV cases, wide ranges/zeros/cancellation, both shared directions and rounding modes, two-term/single-pass controls; optimized HLO and actual engine captures | RTX 5060/cuBLAS: 504 passing operation checks, 264 observed FP8 lowerings; twelve warm workload captures with finite figures and zero in-capture recompiles. H100/TPU unmeasured |
| Worker route | `jax-risk-worker --gpu-gemm cublas`; comparison GPU FP8 auto-selects cuBLAS before JAX starts | Process flags preserved in evidence; default FP64 worker environment remains available |
| Accuracy harness | Bounded full-array lifetime per seed; fresh-process, resumable campaigns; fixture/model/grid/correlated-FX variants; missing/failed counts cannot extend ceilings | All eleven original campaigns refreshed, original configurations preserved, four ceilings changed; no A-11 relaxation |
| Larger counts | Selected single-swap market risk 65K/262K/1M/4M ×32 seeds; exposure 16K/65K/262K ×16 seeds under both IR models | Mean/quantile/tail figures have different limits; FP8 E4M3 passes none of the large single-swap figures, FP16 passes some ES figures |
| Holdout/oracles | Disjoint stress seeds and ORE fixture families; default/refined option grids; independent component pricing | 68 today NPVs and 68 trade cubes agree with ORE. Both option grids agree with ORE individually, yet FP64 American cube moves up to $54.44 between grids. Joint multicurrency oracle remains open |
| Qualification reporting | Python/HTTP empirical scope distinguishes precision, sampling, discretization, intervals and intended dispatch from measured hardware use | Experimental domains are not silently installed as general acceptance evidence; FP64 remains the default |
| Full-suite reliability | Fresh batches of eight, exact collection/JUnit/log audit, Windows/POSIX child ownership, UTF-8 logs and ORE failure cleanup; scheduled/manual CI with retained artifacts | Complete Windows CPU source audit and repeated known-bad boundary controls retained. Repeatable 16 GB hosted-CI qualification remains open |
| TPU installation | Compatible optional extra and pinned libtpu candidate, environment checks and user-guide commands | Package selection only; no target installation or numerical result |

The original acceptance records cover exposure through 16,384 paths and market risk
through 8,192 scenarios, generally 16 seeds (32 outer seeds/eight inner scrambles for
replicated storage). They retain recorded product settings and calibrated/fixed-model
distinctions. Nine refreshed records are CPU and two GPU/cuBLAS. CPU compute results
check rounding behavior; they do not measure native performance. Larger studies are
separate experimental records with exact policies, markets, models and seeds.

## Why emulation remains available

Native FP8 means an eligible product executes on FP8 matrix units with FP32 accumulation.
The existing scaled difference-form kernels retain FP64 calibration, today's PV, Greeks
and final reductions (A-10); a full job is mixed precision.

The available NVIDIA device has a correct cuBLASLt FP8 route when
`--xla_gpu_enable_triton_gemm=false` is set before backend initialization. Its default
emitter produced wrong FP8 products in the recorded checks, so that route stays emulated.
E5M2 uses an E4M3 shared operand on the verified mixed-format cuBLAS route. GEMV fusions
can bypass matrix units even when a format has a native dispatch label. Emulation also
supports CPU, unqualified devices and matched numerical controls.

This follows [JAX's matrix precision controls](https://docs.jax.dev/en/latest/201/precision.html)
and [NVIDIA's FP8 cuBLASLt operations](https://docs.nvidia.com/cuda/cublas/index.html).
[Ironwood documents native FP8](https://docs.cloud.google.com/tpu/docs/ironwood-performance),
but neither that capability nor accepting a dtype qualifies this engine/runtime/shapes.
No H100 host was available. No local consumer-GPU optimization is required before H100
work; the correct cuBLAS route is already selectable and the target commands are ready.

## Remaining gates, in execution order

1. **Obtain H100 access and qualify execution.** Install the pinned Linux GPU environment;
   run the [device checks](../../../benchmarks/README.md#h100-native-qualification) under
   default and cuBLAS emitters in separate processes. Retain numerical comparisons,
   optimized lowering, actual engine HLO/traces and fallbacks for both FP8 formats and
   single/two-term controls. Run ORE parity, the full suite and the golden snapshot on
   that host. Do not equate a dispatch label with instruction use. Recheck after runtime,
   driver or accelerator-library upgrades. Existing XLA reproductions remain upstream
   follow-up; a correct cuBLAS route suffices without waiting for an emitter fix.
2. **Qualify the TPU candidate when hardware is available.** Validate installation and
   fixed-stage/effective FP64 behavior against CPU before native-format dispatch. Check
   FP32 algorithms and lowering as well as BF16/FP8. Preserve unmeasured status until
   numerical and device evidence exists; requesting a dtype proves no accuracy guarantee.
3. **Extend policy accuracy where a use case needs it.** Use the completed large/holdout
   studies to select per-figure mixed policies. Diagnose storage, product, cancellation,
   nonlinear/exercise and P&L-anchor errors on matched draws before increasing counts.
   Extend selected multi-trade exposure/market-risk domains, near-cancelling portfolios,
   scenario recalibration, maturities/volatility/moneyness and joint multicurrency ORE
   checks. Require disjoint holdouts; preserve failures and capped/OOM counts. If a lower
   format fails, retain FP32/FP64. Additional multi-pass methods or quantile corrections
   require evidence of benefit; universal FP8 acceptance is not a completion condition.
4. **Establish sampling and grid convergence separately.** Increase independent FP64
   reference counts/seeds until empirical uncertainty resolves each declared comparison.
   Test finer rollback grids and American time steps at FP64, independently of format.
   The current grid difference and 128-scenario empty-tail example show why precision
   acceptance alone cannot certify a reliable tail number. Supported intervals remain
   [F-10](../features.md#f-10): validate coverage on independent outer seeds, rare and
   zero-event samples against a converged reference, or keep interval status unavailable.
5. **Measure useful performance on idle qualified hardware.** Repeat whole-job cold,
   compile and warm timing, with zero warm recompiles, recording host preparation,
   calibration, conversion/transfer, pairing, reductions, outputs and memory availability.
   Compare at equal accuracy with resolved references, including FP64 baselines under
   both emitter settings. Current timings are diagnostic because work overlapped and the
   host was suspended. Retain no-speedup and failed-accuracy outcomes. Extend to multiple
   devices/hosts only after fixed-total scenario parity, sample identities, alignment and
   collective completion are verified ([I-61](../known-issues.md#i-61)).
6. **Prove repeatable CI and maintain evidence.** The bounded runner and weekly/manual
   workflow are implemented; demonstrate repeated full runs and known-bad controls on
   CI's 16 GB runner with peak-memory evidence. Investigate remaining XLA aborts with a
   fresh cache (I-27). Promote only reviewed original-domain records; archive source,
   manifests, seeds, measurements and qualification evidence together. Broader runtime
   evidence-domain keying requires measured false-transfer cases and regression tests.

Existing unsupported sensitivity configurations (I-117), GPU reverse-mode compilation
(I-119), smile/SOFR gaps (I-54/I-05), interval assumptions and cluster limits remain explicit.
Local source tests and native execution do not close them. Completion is a qualified
number and policy on a declared domain, supported by independent references and repeatable
evidence; some valid configurations require FP64.

# Google Cloud correctness testing: CPU, H100 and TPU

Prepared October 21, 2026; command compatibility reviewed October 25, 2026 against
engine revision `f9cc4c16fa843859729a9fb0956dd8cdfd2fa09b`.
Cloud commands below are reproduction instructions. A bounded H100 trial completed
on October 26, 2026; its FP64/FP32 checks passed and its VM/disks were deleted. See the
[local trial results](../../artifacts/h100-trial-2026-10-26/RESULTS.md) for the exact
coverage, evidence and cleanup checks.

A second October 26 H100 trial exercised all 13 compute formats and **failed full
native-precision qualification**. BF16 three-pass arithmetic failed, default FP8
programs emitted FP16 matrix instructions, and the explicit cuBLAS FP8 route failed
the engine's arithmetic tolerance. See the
[expanded local results](../../artifacts/accelerator-native-2026-10-26/RESULTS.md).
The subsequent v6e TPU trial also failed qualification: FP64 precision-scaling
operations hit compiler type errors, FP16 raw products missed tolerance, and explicit
TF32 was unsupported. All 13 modes were attempted in raw/arithmetic checks and a
separately labeled one-swap engine workload. Calibrated TPU portfolio attempts timed
out and remain unqualified. The H100 and created TPU VMs/disks were deleted, with
resource absence and temporary SSH-key removal verified. The earlier FP64/FP32
pass is not a pass for all formats. The optional broader project inventory comparison
was blocked by automatic approval review's account usage limit.

The subsequent [failure analysis and fixes](hardware-correctness-findings.md)
separates confirmed compiler errors, unsupported operations and unresolved pipeline
differences. An [actual-hardware rerun](hardware-correctness-rerun.md) confirmed the
H100 BF16 three-pass fix, but rejected the proposed TPU FP64-grid and FP16-emulation
fixes. Native FP8 and full TPU qualification remain failed; use the rerun report
for the latest evidence, coverage limits and cleanup record.

The October 25 comparison CLI defaults to API transport. This guide explicitly uses
`--transport direct` for its saved CPU/accelerator scenario comparisons and JSON checks.
The local access audit and CPU baseline are recorded separately in
`artifacts/cloud-preflight-2026-10-25/`. The completed trial covers bounded workloads;
it does not establish full-suite or other precision-profile qualification.

The objective is numerical correctness and repeatability. Use the existing research
tools for their saved inputs, outputs and comparisons. Their timing and speedup columns
are incidental to this exercise. You do not need a scaling study, GKE, TraderX, a new
Cloud Build trigger or a GPU container to start.

## 1. What the supplied handoffs establish

The supplied `alex-gcloud-cicd-guide.md` and `alex-agent-gcloud-handoff.md` describe the
cloud setup as of October 14. Treat their setup instructions as context for this guide,
rather than as authorization to change shared resources.

| Item | Starting point |
| --- | --- |
| Project | `project-compsci-484414` |
| Your Google identity | `aneugro1@mail.yu.edu` |
| Shared-project administrator | Yaakov |
| Experiment storage | `gs://project-compsci-484414-risk-engine-alex-dev` |
| Possible attached VM identity | `risk-engine-alex-dev@project-compsci-484414.iam.gserviceaccount.com` |
| Proven cloud execution | One bounded synthetic CPU Batch pilot |
| GPU/TPU qualification | Neither established by those handoffs |

The earlier CPU container accepts one exact request. It is unsuitable as a general
accelerator test runner. Its four Treasury positions offset to zero and scenario risk
is disabled, so reproducing its net NPV would provide very little accelerator coverage.
Automatic candidate-image builds were also unconfirmed in those documents.

The current checkout adds a pinned `tpu` extra and useful qualification tools. Use this
checkout's commands below rather than assuming the older container contains them.

## 2. Verify your access and choose two small experiments

Start with the [step-by-step access checklist](gcloud-access-check.md). If those checks
pass, provision and run the experiment VMs yourself using the commands below. You do
not need to ask Yaakov to provision them merely because he maintains the shared project.

If a check identifies a gap, send Yaakov the failed command and exact error. The
following is an optional setup request if you need him to provision the resources;
this guide does not send it:

> Please provision isolated numerical-validation resources in project
> project-compsci-484414: one a3-highgpu-1g H100 VM, and, after the H100 checks,
> one single-host ct6e-standard-1t TPU v6e VM if available. Please confirm the billing
> account receives my credits and that the credit terms cover both resources; confirm
> regional accelerator quotas, capacity and a spending allowance. I need SSH and sudo
> for these experiment VMs, outbound access to GitHub/Python packages, and either an
> approved network with SSH access or IAP access. Please supply the names, zones,
> OS/image IDs, and attached service accounts. The H100 needs NVIDIA driver >=580.
> Keep TraderX services and IAM unchanged. Please agree who deletes both VMs and their
> experiment disks when the evidence is saved.

Credit balance, quota and available hardware are separate checks. An account still in
the introductory Free Trial has accelerator restrictions; promotional/research credits
on a paid billing account have their own terms. Check Billing in the console and
**IAM & Admin -> Quotas & System Limits** before allocating hardware.
[Google Free Program](https://docs.cloud.google.com/free/docs/free-cloud-features),
[Compute quotas](https://docs.cloud.google.com/compute/resource-usage).

Use these sizes for a first pass:

| Test | Resource | Reason |
| --- | --- | --- |
| CPU reference + NVIDIA | `a3-highgpu-1g` | One H100; CPU reference can run on the same VM |
| TPU compatibility | `ct6e-standard-1t` | One v6e chip, single host; specifically intended for testing |

The one-H100 A3 requires **Spot or Flex-start** provisioning. Spot can be interrupted;
interruption is an incomplete experiment, not a numerical failure. Start with Spot if
available; use Flex-start if allocation is unreliable. Do not substitute an
eight-GPU machine merely to obtain standard provisioning.
[A3 creation](https://docs.cloud.google.com/compute/docs/gpus/create-gpu-vm-accelerator-optimized).

Google lists v6e in `us-east1-d`, among other zones. Availability tables do not guarantee
free capacity or your quota. If a one-chip slice is unavailable, ask for a supported
single-host configuration and record the change.
[TPU sizes](https://docs.cloud.google.com/compute/docs/tpus/tpu-machines),
[TPU locations](https://docs.cloud.google.com/compute/docs/regions-zones/tpu-regions-zones).

Use the console's price estimate for the selected resources and agree a deletion time.
An alerts-only budget sends notifications and does not cap usage.
[Budget behavior](https://docs.cloud.google.com/billing/docs/how-to/budgets).

## 3. Use Cloud Shell as the control terminal

Open the Google Cloud console, sign in with the account above, select the project, and
click **Activate Cloud Shell** near the top right. Cloud Shell includes `gcloud`, but
does not supply an H100 or TPU. All commands in this guide are **Bash**, entered either
in Cloud Shell or after SSH into a Linux VM. Do not paste them into Windows PowerShell.

In Cloud Shell:

```bash
gcloud auth list
gcloud config set account aneugro1@mail.yu.edu
gcloud config set project project-compsci-484414
gcloud config list

export PROJECT_ID=project-compsci-484414
export ENGINE_REV=f9cc4c16fa843859729a9fb0956dd8cdfd2fa09b
export H100_VM=alex-h100-correctness
export TPU_VM=alex-tpu-correctness
# Replace these with the actual provisioned locations.
export H100_ZONE=REPLACE_WITH_H100_ZONE
export TPU_ZONE=us-east1-d
```

If Yaakov supplies different names, use those. Shell variables must be set again in a
new terminal. The H100 and TPU zones can differ from each other and from the CPU pilot.

If the resources already exist, skip creation and connect. If your access checks pass,
these are example creation commands you can run yourself. They use the development
service account from the handoff. Check that you can attach it and select the project's
approved network/subnet; consult Yaakov only if those details are unavailable.

```bash
# H100. Add the project's approved network/subnet settings.
gcloud compute instances create "$H100_VM" \
  --project="$PROJECT_ID" --zone="$H100_ZONE" \
  --machine-type=a3-highgpu-1g --provisioning-model=SPOT \
  --maintenance-policy=TERMINATE \
  --image-project=debian-cloud --image-family=debian-12 \
  --boot-disk-size=200GB --boot-disk-type=pd-balanced \
  --service-account="risk-engine-alex-dev@${PROJECT_ID}.iam.gserviceaccount.com" \
  --scopes=https://www.googleapis.com/auth/cloud-platform

# TPU through the current Compute Engine API.
gcloud compute instances create "$TPU_VM" \
  --project="$PROJECT_ID" --zone="$TPU_ZONE" \
  --machine-type=ct6e-standard-1t --maintenance-policy=TERMINATE \
  --image-project=ubuntu-os-accelerator-images \
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
  --boot-disk-size=200GB --boot-disk-type=hyperdisk-balanced \
  --service-account="risk-engine-alex-dev@${PROJECT_ID}.iam.gserviceaccount.com" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

These examples use a Debian GPU image and a TPU-specific Ubuntu image. Record the
resolved image ID, since image families change. Verify required APIs and provisioning/IAM
grants using the access checklist. The `cloud-platform` OAuth scope does not
itself grant bucket access.

Google now supports TPU creation through Compute Engine, using machine types and an
image family. That is the TPU lifecycle used throughout this guide. A TPU created with
the older `gcloud compute tpus tpu-vm` commands uses a different management interface;
have Yaakov supply matching connect/delete commands if he provisions that way instead.
[TPU creation](https://docs.cloud.google.com/compute/docs/tpus/create-tpu-vm-instance),
[OS image families](https://docs.cloud.google.com/compute/docs/images/os-details).

Connect from Cloud Shell:

```bash
gcloud compute ssh "$H100_VM" --project="$PROJECT_ID" --zone="$H100_ZONE"
```

Later, use the same command with `"$TPU_VM"` and `"$TPU_ZONE"`. If the approved access
path is IAP, add `--tunnel-through-iap`. A permission/firewall error needs Yaakov's access
setup; it does not indicate an engine defect. Your user identity and the VM's attached
service account are separate identities.

## 4. Prepare the H100 driver, then the same source on both VMs

### H100 driver, inside the H100 VM

```bash
nvidia-smi
```

Require an H100 and driver **580 or newer** for this repository's CUDA 13 extra. If the
driver is already suitable, leave it installed. If it is missing and you have sudo, install it
using Google's supported installer:

```bash
curl -fL https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz \
  -o "$HOME/cuda_installer.pyz"
sudo python3 "$HOME/cuda_installer.pyz" install_driver \
  --installation-mode=repo --installation-branch=prod
```

The installer can reboot; reconnect and rerun the installer if required to complete it.
Then recheck `nvidia-smi`. If the chosen branch is below 580, the provisioner must select
a compatible driver. The JAX GPU extra supplies CUDA user libraries through pip; a
separate system CUDA toolkit is not required for this installation path. The CUDA
version printed by `nvidia-smi` is not proof that JAX imported the intended libraries.
[Google driver installation](https://docs.cloud.google.com/compute/docs/gpus/install-drivers-gpu),
[JAX CUDA requirements](https://docs.jax.dev/en/latest/installation.html).

### Source and Python, inside each VM

Use a fresh checkout at the same commit on both machines. A private GitHub repository
needs your approved GitHub authentication independently of Google Cloud login.

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv ca-certificates
git clone https://github.com/AlexNeugroschl/JAX_Risk_Engine.git
cd JAX_Risk_Engine
git checkout --detach f9cc4c16fa843859729a9fb0956dd8cdfd2fa09b
git rev-parse HEAD
git status --short
```

Do not initialize the large reference submodules just for these checks; the Python ORE
dependency is installed below. If using a later engine revision, select that exact SHA
on both machines and verify its CLI/options still match this guide.

If Python 3.11 with venv is available:

```bash
python3.11 -m venv .venv
```

If the image supplies only Python 3.10, use an isolated uv bootstrap to obtain 3.11:

```bash
python3 -m venv "$HOME/risk-python-bootstrap"
"$HOME/risk-python-bootstrap/bin/python" -m pip install uv
"$HOME/risk-python-bootstrap/bin/uv" python install 3.11
"$HOME/risk-python-bootstrap/bin/uv" venv --python 3.11 --seed .venv
```

Use the same Python 3.11 patch version on both VMs where practical; record `python -VV`.
uv is only an environment setup tool here.
[uv Python installation](https://docs.astral.sh/uv/guides/install-python/),
[uv environments](https://docs.astral.sh/uv/pip/environments/).

Install **one** appropriate extra in that VM's fresh environment:

```bash
# H100 VM only:
.venv/bin/python -m pip install -c constraints.txt -e '.[api,dev,gpu]'

# TPU VM only, when you reach the TPU stage:
.venv/bin/python -m pip install -c constraints.txt -e '.[api,dev,tpu]'
```

Do not replace these commands with an unconstrained `pip install -U jax`. The checkout
pins JAX/jaxlib 0.10.2 and libtpu 0.0.42; these are the experiment's versions. An install
failure is an environment compatibility issue to resolve before numerical testing.

In each VM, save a record outside the source checkout:

```bash
export EVIDENCE="$HOME/risk-correctness-evidence"
mkdir -p "$EVIDENCE"
git rev-parse HEAD > "$EVIDENCE/commit.txt"
git status --short > "$EVIDENCE/source-status.txt"
cp pyproject.toml constraints.txt "$EVIDENCE/"
.venv/bin/python -VV > "$EVIDENCE/python.txt"
.venv/bin/python -m pip freeze > "$EVIDENCE/packages.txt"
uname -a > "$EVIDENCE/os.txt"
printenv | sort | sed -n '/^JAX_/p; /^XLA_/p; /^CUDA_/p; /^LD_LIBRARY_PATH=/p; /^LIBTPU_/p' \
  > "$EVIDENCE/inherited-runtime-settings.txt"
```

Use a fresh shell/environment with no inherited JAX/XLA overrides for the first run.
In particular, do not copy the RTX 5060 workarounds into the H100 environment. If a
provisioned image sets library paths or flags, inspect them with Yaakov rather than
silently accepting a different runtime.

## 5. H100: validate the environment and existing calculations

Run from the engine checkout in the H100 VM. Keep logs; `pipefail` prevents `tee` from
hiding a failing command. Inspect each command's result before proceeding.

```bash
set -o pipefail
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
nvidia-smi > "$EVIDENCE/nvidia-smi.txt"
.venv/bin/python - <<'PY' 2>&1 | tee "$EVIDENCE/h100-device.txt"
import engine
import jax
assert not jax.config.jax_disable_jit
devices = jax.devices('gpu')
assert devices and all('H100' in d.device_kind for d in devices), devices
assert jax.default_backend() == 'gpu'
print('JAX:', jax.__version__, 'x64:', jax.config.jax_enable_x64)
print('Devices:', devices)
PY

.venv/bin/python -m pytest tests/test_environment.py -q \
  --junitxml="$EVIDENCE/h100-environment.xml" \
  2>&1 | tee "$EVIDENCE/h100-environment.log"

# An independently checked CPU baseline at the exact same source/dependency versions.
JAX_PLATFORMS=cpu .venv/bin/python -m pytest -m ore_parity -q \
  --junitxml="$EVIDENCE/cpu-ore.xml" 2>&1 | tee "$EVIDENCE/cpu-ore.log"

# Repeat those ORE checks with GPU as the default backend.
.venv/bin/python -m pytest -m ore_parity -q \
  --junitxml="$EVIDENCE/h100-ore.xml" 2>&1 | tee "$EVIDENCE/h100-ore.log"
```

Every command above should exit zero, with no unexplained failures or unexpected skips.
The marked ORE tests run independent library comparisons at their existing tolerances;
do not loosen those tolerances to obtain a pass. Some tests explicitly select CPU for
their own purpose. A green pytest run alone does not prove all calculations used H100;
the next comparison checks the execution devices of its actual outputs.

## 6. Compare H100 with CPU on identical scenarios

The tool `benchmarks.comparison` creates CPU-generated scenarios and refuses a matched
comparison if stored inputs differ. It checks a single swap and a five-trade portfolio
with swaps, European/Bermudan options and a bond. It retains per-scenario values and
named risk results, avoiding cancellation hidden in a net total.

Start with default FP64. These tolerances are an explicit **initial engineering gate**,
not a financial acceptance certificate: absolute floor `1e-8`, relative limit `1e-10`
of each named output's CPU maximum absolute value. ORE tests retain their own rules.

```bash
.venv/bin/python -m benchmarks.comparison --transport direct \
  --out "$EVIDENCE/h100-fp64" --backends cpu gpu --expected-gpu H100 \
  --formats float64 --paths 512 4096 --seeds 1 2 3 4 5 6 7 8 \
  --atol 1e-8 --rtol 1e-10 \
  2>&1 | tee "$EVIDENCE/h100-fp64.log"
```

Use the JSON gate in section 9 on `h100-fp64`. If this fails, investigate before adding
lower precision. CPU FP64 is a reproducible reference; agreement with it does not prove
all financial conventions correct. That is why the independent ORE checks also matter.

If you intend to use the existing float32 pricing profile, test it separately:

```bash
.venv/bin/python -m benchmarks.comparison --transport direct \
  --out "$EVIDENCE/h100-fp32" --backends cpu gpu --expected-gpu H100 \
  --formats float64 float32 --paths 1024 4096 --seeds 1 2 3 4 5 6 7 8 \
  --atol 1e-8 --rtol 1e-4 \
  2>&1 | tee "$EVIDENCE/h100-fp32.log"
```

This changes pricing compute/storage according to the named profile. It does not turn
every engine stage into float32. Today's PV, calibration, Greeks and final reductions
keep the engine's policies. A float32 pass under `rtol=1e-4` is a separate conclusion
from the FP64 gate. Absolute errors must also make sense in each output's units.

For repeatability, rerun the FP64 command into a new directory, keeping source, flags,
seeds and counts identical:

```bash
.venv/bin/python -m benchmarks.comparison --transport direct \
  --out "$EVIDENCE/h100-fp64-repeat" --backends cpu gpu --expected-gpu H100 \
  --formats float64 --paths 512 4096 --seeds 1 2 3 4 5 6 7 8 \
  --atol 1e-8 --rtol 1e-10 \
  2>&1 | tee "$EVIDENCE/h100-fp64-repeat.log"
```

Apply section 9's numerical gate to both collections, then compare corresponding
snapshots directly with the repository's existing comparison function:

```bash
.venv/bin/python - "$EVIDENCE/h100-fp64" "$EVIDENCE/h100-fp64-repeat" <<'PY' | tee "$EVIDENCE/repeat-check.log"
import json, pathlib, sys
from demos.benchmark_reporting import named_errors
def load(directory):
    root = pathlib.Path(directory)
    requests = json.loads((root / 'manifest.json').read_text())['requests']
    return {tuple(r[k] for k in ('campaign', 'backend', 'label', 'paths')):
            (root / r['directory'], r) for r in requests}
first, repeat = map(load, sys.argv[1:])
assert first.keys() == repeat.keys() and first
failures = 0
for key, (folder, request) in first.items():
    other, repeated_request = repeat[key]
    assert request['seeds'] == repeated_request['seeds']
    for seed in request['seeds']:
        rows = named_errors(folder / f'seed-{seed}.npz',
                            other / f'seed-{seed}.npz', 1e-10, 1e-8)
        assert rows
        failures += sum(r['status'] != 'PASS' for r in rows.values())
        worst = max((r.get('max_absolute_error', 0) for r in rows.values()), default=0)
        print(key, 'seed', seed, 'worst absolute difference', worst, flush=True)
if failures:
    raise SystemExit(f'Repeatability failed: {failures} output comparisons')
print('PASS: all repeated snapshots match within the declared tolerances')
PY
```

Scenario inputs must match exactly; outputs must reproduce within the declared
tolerance. Inspect any nonzero repeat differences, with deterministic bits expected
where the runtime promises them. Do not use `--merge` for repeated points: the tool
deliberately rejects duplicates. For TPU repeats, substitute the TPU collection paths
after its CPU comparisons have passed.

After the bounded checks pass, collect the complete regression suite once:

```bash
.venv/bin/python -m benchmarks.test_suite \
  --out "$EVIDENCE/h100-full-suite" --batch-size 8 \
  2>&1 | tee "$EVIDENCE/h100-full-suite.log"
```

Require `suite.json` to contain `complete: true`. Inspect skips and each batch's JUnit
and logs. This includes simulation, pricing, Greeks, risk, API/worker and validation
regressions beyond the bounded comparison. Run one process at a time initially; TPU
runtime ownership and accelerator memory make `pytest -n auto` a poor starting choice.

## 7. TPU: test compatibility before launching the suite

Save and delete the H100 experiment when finished, then connect to the TPU VM from
Cloud Shell and perform section 4's source/environment setup with the `tpu` extra.

**The main compatibility question is FP64.** JAX states that TPUs have no true float64
hardware. Setting `jax_enable_x64=True`, allocating a float64-looking array, or finding
`jax.devices('tpu')` does not establish usable FP64 arithmetic. The engine requires
FP64 fixed stages even in its lower-precision profiles. An unsupported compilation,
truncation or precision failure is a legitimate incompatibility result. A CPU fallback
does not count as TPU correctness.
[JAX numerical accuracy](https://docs.jax.dev/en/latest/accuracy.html).

On the TPU VM:

```bash
set -o pipefail
export JAX_PLATFORMS=tpu,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python - <<'PY' 2>&1 | tee "$EVIDENCE/tpu-device.txt"
import engine
import jax
import jax.numpy as jnp
import numpy as np
assert not jax.config.jax_disable_jit
assert jax.default_backend() == 'tpu'
device = jax.devices('tpu')[0]
print('JAX:', jax.__version__, 'devices:', jax.devices(), flush=True)
with jax.default_device(device):
    # Compilation and arithmetic, rather than array allocation alone.
    result = jax.jit(lambda x: (x + jnp.float64(1)) - x)(
        jnp.asarray([2**30], dtype=jnp.float64))
    result.block_until_ready()
assert result.dtype == jnp.float64, result.dtype
assert all(d.platform == 'tpu' for d in result.devices()), result.devices()
np.testing.assert_array_equal(np.asarray(result), np.asarray([1.0]))
print('Minimal FP64 probe passed; engine qualification still required.')
PY
```

The addition probe detects a basic float32 substitution because float32 loses the unit
increment at `2**30`. Passing it is only permission to try real engine operations; it
does not establish FP64 support for matrix products, transcendental functions, AD or
reductions. If it fails, save its output and exception and stop the expensive test
sequence. Record **"current engine FP64 policy incompatible with this TPU runtime"**,
with the exact runtime/device details. Do not disable x64 or relax fixed-stage policies
to turn this into a pass; that would be a separate engine design project.

If the probe passes:

```bash
.venv/bin/python -m pytest tests/test_environment.py -q \
  --junitxml="$EVIDENCE/tpu-environment.xml" \
  2>&1 | tee "$EVIDENCE/tpu-environment.log"

.venv/bin/python -m tests.support.precision_devices --platform tpu \
  --formats float64 float32 --out "$EVIDENCE/tpu-arithmetic" \
  2>&1 | tee "$EVIDENCE/tpu-arithmetic.log"

.venv/bin/python -m pytest -m ore_parity -q \
  --junitxml="$EVIDENCE/tpu-ore.xml" 2>&1 | tee "$EVIDENCE/tpu-ore.log"
```

The arithmetic check exercises adversarial matrix/vector shapes and accumulation
formats against CPU emulation. Require `qualification.json` to have both
`complete: true` and `passes: true`, and inspect the per-operation rows. If fixed-stage
operations fail, stop and retain the failure. Do not assume the rest of the engine can
be qualified just because a float32 matrix multiply works.

Only if these gates pass, collect a bounded TPU comparison:

```bash
.venv/bin/python -m benchmarks.comparison --transport direct \
  --out "$EVIDENCE/tpu-fp64" --backends tpu \
  --formats float64 --paths 512 4096 --seeds 1 2 3 4 5 6 7 8 \
  --atol 1e-8 --rtol 1e-10 \
  2>&1 | tee "$EVIDENCE/tpu-fp64.log"
```

This collection intentionally has no local CPU rows. It will exit 1 with
`CPU FP64 reference unavailable` until merged with the matching H100/CPU collection.
That specific partial-report condition is expected; worker compilation failures or
wrong devices are not. If desired, run `--backends cpu tpu` instead for a self-contained
local comparison, but do not merge its duplicate CPU points with `h100-fp64`.

If the merged FP64 comparison passes, repeat with identical settings in a new directory
and compare saved snapshots as for H100. Then run the complete suite with
`JAX_PLATFORMS=tpu,cpu` and a fresh evidence directory. Inspect test device selections;
a CPU-only test in the suite does not establish that operation's TPU support.

## 8. Additional coverage after the default policies work

For FP16, BF16 or FP8, qualify each profile separately. Device correctness and financial
precision are different questions: faithfully executing FP8 can still miss your error
budget. Full-precision tests do not qualify these formats.

For example, on H100, with the additional profiler dependencies installed through
`pip install -c constraints.txt -e '.[api,dev,gpu,profiling]'`:

```bash
.venv/bin/python -m tests.support.precision_devices --platform gpu \
  --expected-device H100 \
  --formats float16 float8_e4m3fn float8_e5m2 --require-native-fp8 \
  --out "$EVIDENCE/h100-low-precision-arithmetic" \
  --engine-workloads comparison_single comparison_portfolio
```

This retains optimized HLO and traces as well as numerical checks. If a compiler route
fails, retain that result. A separate `--gemm cublas` run into a new directory can
diagnose it, but any pass then qualifies that explicit configuration. Actual native
instruction claims require trace inspection and are unnecessary for the initial
FP64/FP32 correctness exercise.

The October 26 expanded trial exposed a false positive in the old `--require-native-fp8`
check: typed FP8 HLO could emit FP16 instructions. The corrected check retains GPU PTX
and requires matching FP8 matrix instructions for every FP8 operation, alongside
native dispatch, typed lowering and numerical checks. Missing compiler output, including
cached executables, fails this strict gate. An opaque cuBLAS kernel remains unverified
when its instructions cannot be extracted; that does not establish emulation. Retain
actual GPU traces too and keep compiler routes in separate result directories.

To add exposure simulation and other fixtures beyond the bounded market-risk workload:

```bash
# H100; add --multi-currency for a separate USD/EUR/FX exposure study.
.venv/bin/python -m benchmarks.qualification \
  --out "$EVIDENCE/h100-exposure-holdout" --backend gpu --expected-gpu H100 \
  --campaigns compute --paths 1024 4096 --seed-ids 101 102 103 104 \
  --fixture-seeds 40 41 --models LGM HullWhite --grids recorded \
  --cases pricing:compute:float32
```

This harness compares precision policies against FP64 **on the selected device**. It
adds statistical and fixture coverage, not a replacement for matched CPU/accelerator
comparisons. It currently explicitly selects cuBLAS on GPU; record that configuration
separately from native-default runs. Use its per-figure evidence and failure records;
the A-11 statistical acceptance rules are separate from the comparison tool's
engineering tolerance. Do not overwrite shipped `compliance/precision-measurements`
or regenerate published evidence merely to store an experiment.

For your intended production workload, add representative portfolios after these
checks: both interest-rate models you use, European/Bermudan/American settings,
recalibration modes, Greeks, multicurrency exposures and stressed curves as applicable.
The bounded comparison uses a fixed-LGM coarse-grid Bermudan, does not request Greeks,
and does not qualify calibrated-scenario options or multi-host/multi-device scaling.
512/4096 paths are correctness checks on a finite sample, not evidence that 99% tail
risk has converged. Increase path counts/seeds if tail estimation itself is in scope.

## 9. Read the results and enforce an actual numerical pass

For each comparison collection:

| Artifact | Inspect |
| --- | --- |
| `report.md` | Status and matched-scenario error tables; every trade/observable |
| `comparison.json` | Per-seed max absolute/RMS errors, bias, reference scale and warnings |
| `manifest.json` | Requested workloads, formats, paths and seed identities |
| Point `result.json` | Actual execution devices, precision, versions and flags |
| Point `worker.log` | Exceptions, unsupported lowerings, OOM or timeouts |
| Point `seed-*.npz` | Exact scenario shifts and named output arrays |

**The comparison command can return exit 0 while numerical rows say FAIL or hardware
is unavailable.** Check its rows explicitly. For example, inside the matching checkout
and environment, run this against the completed H100 collection:

```bash
.venv/bin/python - "$EVIDENCE/h100-fp64" <<'PY'
import json, pathlib, sys
directory = pathlib.Path(sys.argv[1])
manifest = json.loads((directory / 'manifest.json').read_text())
report = json.loads((directory / 'comparison.json').read_text())
points, rows = report['points'], report['comparisons']
assert len(points) == len(rows) == len(manifest['requests']) > 0
bad = [r for p, r in zip(points, rows)
       if p['status'] != 'ok' or r.get('accuracy') != 'PASS']
for row in bad:
    print(json.dumps(row, indent=2))
if bad:
    raise SystemExit(f'NOT QUALIFIED: {len(bad)} missing/failed numerical points')
print(f'PASS: all {len(rows)} requested points passed their recorded tolerances')
PY
```

For a merged report, there is no new collection manifest. Replace the manifest line
and length assertion with `assert len(points) == len(rows) > 0`, and verify its point
count against the sum of the input manifests. The prescribed FP64 H100 collection has
8 points (2 workloads x 2 backends x 2 path counts); TPU-only adds 4, giving 12 merged
points. Apply the gate to the merged report only after adding the CPU reference.

Use these interpretations:

| Observation | Conclusion / next action |
| --- | --- |
| All requested numerical rows pass, devices verified, regressions complete | Qualified for the recorded workloads, profiles and runtime |
| Numerical `FAIL` | Inspect worst per-trade/per-scenario output, units, absolute error and bias; preserve the failure |
| Different shifts, source digest or numerical libraries | Invalid comparison; align inputs/environment and recollect |
| `unmeasured`, missing row, wrong device, timeout or OOM | Incomplete coverage; cannot call that point a pass |
| Non-finite result | Numerical failure, unless an explicitly designed negative test |
| GPU/TPU compiler error | Runtime/operation incompatibility; retain full logs |
| Small net portfolio error but large opposing trade errors | Failure hidden by cancellation; use named outputs |
| FP32 works but required FP64 stages fail | Current engine policy remains incompatible |

The tool uses `max_abs_error <= atol + rtol * max_abs(CPU output)` for each named
array. Near zero, absolute error matters. Array-wide scaling can hide a small component's
relative error, so also inspect economically significant entries and use their own
domain tolerances. Do not reinterpret `rtol` as a universal currency or Greek limit.
Differences between identical scenarios measure arithmetic effects; changes with
independent seeds/path counts measure sampling effects. Bitwise identity across
different hardware is not the general acceptance requirement.

The API is unnecessary for the first numerical checks. For a later end-to-end smoke,
use the supplied guide's regular `POST /portfolio/price` / `GET /jobs/{job_id}` flow
on the **same qualified source/runtime**, with a fresh queue. Require a completed job,
expected trade coverage and per-trade prices. Inspect the completed worker's precision
and device report: `/version` can say CPU because the API control process stays on CPU.
An HTTP 202 only means submission was accepted.

## 10. Save evidence, join collections and clean up

Inside each VM, archive the evidence outside the checkout:

```bash
tar -czf "$HOME/risk-correctness-evidence.tar.gz" -C "$HOME" risk-correctness-evidence
sha256sum "$HOME/risk-correctness-evidence.tar.gz"
```

From Cloud Shell, retrieve each archive before deleting the machine:

```bash
mkdir -p "$HOME/risk-hardware-results"
gcloud compute scp "$H100_VM:~/risk-correctness-evidence.tar.gz" \
  "$HOME/risk-hardware-results/h100.tar.gz" \
  --project="$PROJECT_ID" --zone="$H100_ZONE"
gcloud compute scp "$TPU_VM:~/risk-correctness-evidence.tar.gz" \
  "$HOME/risk-hardware-results/tpu.tar.gz" \
  --project="$PROJECT_ID" --zone="$TPU_ZONE"
```

Use `--tunnel-through-iap` here too if that is your SSH path. A VM's service account
does not need bucket access for this transfer. Upload from Cloud Shell under your user
identity with unique, create-only object names, matching the handoff's bucket grants:

```bash
export EXPERIMENT_ID="hardware-$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
gcloud storage cp "$HOME/risk-hardware-results/h100.tar.gz" \
  "gs://project-compsci-484414-risk-engine-alex-dev/experiments/$EXPERIMENT_ID/h100.tar.gz" \
  --project="$PROJECT_ID" --account=aneugro1@mail.yu.edu --if-generation-match=0
gcloud storage cp "$HOME/risk-hardware-results/tpu.tar.gz" \
  "gs://project-compsci-484414-risk-engine-alex-dev/experiments/$EXPERIMENT_ID/tpu.tar.gz" \
  --project="$PROJECT_ID" --account=aneugro1@mail.yu.edu --if-generation-match=0
```

Read back the uploaded archives and verify their SHA256 against the downloaded copies.
Keep collection directories intact, including all `.npz` files and logs. On your local
workstation or a suitable analysis machine with the matching checkout and dependencies,
extract them into separate directories:

```bash
mkdir -p results/h100 results/tpu
tar -xzf h100.tar.gz -C results/h100
tar -xzf tpu.tar.gz -C results/tpu
.venv/bin/python -m benchmarks.comparison --transport direct --merge \
  results/h100/risk-correctness-evidence/h100-fp64 \
  results/tpu/risk-correctness-evidence/tpu-fp64 \
  --out results/joined-fp64 --atol 1e-8 --rtol 1e-10
```

This reuses the CPU FP64 reference collected on the H100 host. Source/tool digests,
workloads, seeds and numerical library versions must match. Accelerator-specific
libraries can differ and are recorded. The merge verifies exact scenario shifts.
If TPU compatibility failed earlier, there is no successful TPU collection to merge;
retain the incompatibility report alongside the H100 result instead.

Delete only your experiment VMs after confirming the archives are retrievable. Have
Yaakov do this if deletion is part of his provisioning workflow. For the resources
created with the Compute Engine commands above:

```bash
# Cloud Shell: inspect identities and disks before deletion.
gcloud compute instances describe "$H100_VM" --project="$PROJECT_ID" --zone="$H100_ZONE" \
  --format='yaml(name,zone,status,disks)'
gcloud compute instances describe "$TPU_VM" --project="$PROJECT_ID" --zone="$TPU_ZONE" \
  --format='yaml(name,zone,status,disks)'

# Delete the named experiment instances; these commands retain confirmation prompts.
gcloud compute instances delete "$H100_VM" --project="$PROJECT_ID" --zone="$H100_ZONE"
gcloud compute instances delete "$TPU_VM" --project="$PROJECT_ID" --zone="$TPU_ZONE"
```

Check Compute Engine -> VM instances and Disks, plus any provisioned reservations or
static IPs, for retained experiment resources. Delete experiment-only retained resources
with Yaakov. Merely closing SSH leaves compute running. Compute Engine TPU instances
cannot be stopped/suspended to end usage; delete them. A stopped GPU VM can still retain
billable disks. Keep the shared buckets and TraderX resources.
[TPU lifecycle limitations](https://docs.cloud.google.com/compute/docs/tpus/create-tpu-vm-instance).

Write one short decision record per device: commit; OS/Python/JAX/driver/libtpu versions;
device model; flags; test counts and skips; profiles/workloads/path counts/seeds;
declared tolerances and worst observed errors; repeated-run differences; evidence URI;
and **qualified**, **numerical failure**, **runtime incompatible**, or **incomplete**.
A defensible first outcome can be H100 passing and TPU incompatible with the current
FP64 policy. Report those separately rather than treating successful device discovery
as engine qualification.

# Google Cloud access for CPU, GPU and TPU precision research

Prepared for Alex and the cloud administrator. Requirements reviewed on **2026-10-15**.

## Purpose and requested outcome

Alex needs to independently provision, connect to, run and clean up **CPU, GPU and
TPU experiments** for the JAX Risk Engine. The research compares numerical accuracy,
memory use and execution time across precision policies and hardware, using a
float64 reference and reproducible workloads.

The required access covers interactive development, long precision campaigns,
CPU/GPU Batch jobs, profiling, and storing inputs and results. It should permit
routine experiments without a new administrator action for every machine or run.

The research scope comes from the [precision plan](planning/details/precision.md)
and [roadmap](planning/roadmap.md). The supplied `alex-gcloud-cicd-guide.md` and
`alex-agent-gcloud-handoff.md` describe the existing shared-project arrangement,
where Yaakov handles provisioning. Their resource names below are context from
those guides; this document proposes the access needed for independent research.
Live IAM bindings, quotas and resource availability have not been inspected.

## Existing project and identities

| Item | Existing value from the setup guides |
|---|---|
| Project | `project-compsci-484414` |
| Alex's Google account | `aneugro1@mail.yu.edu` |
| Experiment runtime service account | `risk-engine-alex-dev@project-compsci-484414.iam.gserviceaccount.com` |
| Development bucket | `gs://project-compsci-484414-risk-engine-alex-dev` |
| Candidate image repository | `us-east1-docker.pkg.dev/project-compsci-484414/risk-engine-candidates` |
| Candidate builder service account | `risk-candidate-builder@project-compsci-484414.iam.gserviceaccount.com` |
| Existing CPU build/Batch region | `us-east1` |

A **dedicated research project** is recommended for independent provisioning:
project-level Compute Instance Admin controls other VMs in that project too.
If a separate project is used, provision equivalent identities, storage and image
access there. Accelerator locations should follow hardware availability rather
than assuming the existing CPU region supports every target.

## Permissions for Alex's account

The existing ML Engineer role does not provide the complete Compute Engine and
Batch workflow. The administrator should compare effective permissions with this
table and grant the missing capabilities. Existing equivalent grants can remain.
See Google's [ML Engineer role definition](https://docs.cloud.google.com/iam/docs/roles-permissions/jobfunctions#iam.mlEngineer).

| Task | Role | Grant scope / condition |
|---|---|---|
| Create, configure, start, stop and delete CPU/GPU VMs and Compute Engine TPU instances | Compute Instance Admin (v1), `roles/compute.instanceAdmin.v1` | Research project; project scope is needed to create instances |
| SSH and install drivers/system packages with sudo | Compute OS Admin Login, `roles/compute.osAdminLogin` | Research VMs or project, where OS Login is enabled; ordinary `roles/compute.osLogin` suffices if the administrator maintains the system software |
| Connect through IAP | IAP Tunnel Resource Accessor, `roles/iap.tunnelResourceAccessor` | Research VMs/project, only if IAP is the configured connection route |
| Attach the runtime identity to VMs and jobs | Service Account User, `roles/iam.serviceAccountUser` | On the experiment runtime service account specifically |
| Submit, inspect and delete CPU/GPU Batch jobs | Batch Job Editor, `roles/batch.jobsEditor` | Research project |
| Inspect execution logs | Logs Viewer, `roles/logging.viewer` | Research project; check whether current grants already cover this |
| Inspect CPU/GPU/TPU metrics | Monitoring Viewer, `roles/monitoring.viewer` | Research project, if not already covered |
| Read development inputs and download results | Storage Object Viewer, `roles/storage.objectViewer` | Development bucket and separately authorized input locations |
| Upload new inputs and results | Storage Object Creator, `roles/storage.objectCreator` | Development bucket; unique object paths avoid overwrite permissions |
| Overwrite/delete development objects, if needed | Storage Object User, `roles/storage.objectUser` | Development bucket; alternative to Viewer + Creator |
| Pull private container images | Artifact Registry Reader, `roles/artifactregistry.reader` | Repositories used by experiments |

If Alex's identity is external to the project's organization and OS Login is used,
an organization administrator may also need to grant
`roles/compute.osLoginExternalUser` at organization scope.

Sources: [Compute IAM](https://docs.cloud.google.com/compute/docs/access/iam),
[OS Login and IAP access](https://docs.cloud.google.com/compute/docs/oslogin/set-up-oslogin),
[Batch user access](https://docs.cloud.google.com/batch/docs/get-started),
[Monitoring roles](https://docs.cloud.google.com/monitoring/access-control),
[Storage roles](https://docs.cloud.google.com/storage/docs/access-control/iam-roles),
[Artifact Registry access](https://docs.cloud.google.com/artifact-registry/docs/access-control).

### TPU provisioning: select the interface

| Provisioning route | Required access |
|---|---|
| Compute Engine TPU instances, using `gcloud compute instances ...` | The Compute Instance Admin and Service Account User access above; administrator enables the Compute Engine API |
| Cloud TPU API, using `gcloud compute tpus tpu-vm ...` | TPU Admin, `roles/tpu.admin`, on the research project, plus Service Account User on the runtime account and Compute Viewer, `roles/compute.viewer`, for supporting visibility |

Provision the route supported by the selected TPU generation and topology. A TPU
role for the Cloud TPU API does not replace Compute Engine permissions for the
Compute Engine route. SSH access must also match that route's configuration.

Sources: [Compute Engine TPU quickstart](https://docs.cloud.google.com/compute/docs/tpus/quickstart-create-tpu-instance),
[Cloud TPU VM access requirements](https://docs.cloud.google.com/tpu/docs/run-calculation-pytorch),
[Cloud TPU roles](https://docs.cloud.google.com/iam/docs/roles-permissions/tpu).

## Permissions for running jobs

Alex's user identity, the runtime service account and the image builder are
separate identities. Granting Alex access to a bucket does not grant a VM/job
access to it.

| Identity | Required access |
|---|---|
| Experiment runtime service account | Artifact Registry Reader on required image repositories; Storage Object Viewer on authorized inputs; Storage Object Creator on the development output bucket; Logs Writer (`roles/logging.logWriter`) on the project |
| Same runtime account, when used for Batch | Batch Agent Reporter (`roles/batch.agentReporter`) on the project |
| Candidate builder account | Artifact Registry Writer on its destination repository and Logs Writer; source/log storage access as required by its build configuration |

Use Storage Object User on the development bucket only if runtime jobs need
overwrite/delete. Exact-object pilot access and access to private market datasets
are separate grants; synthetic research can proceed with repository fixtures.
If a monitoring agent or the workload publishes metrics, give its runtime identity
Monitoring Metric Writer (`roles/monitoring.metricWriter`) on the project.

The Google-managed Batch service agent also needs its normal service-agent role.
For Shared VPC, the administrator must configure its subnet access in the host
project. Sources: [Batch runtime setup](https://docs.cloud.google.com/batch/docs/get-started)
and [Batch networking](https://docs.cloud.google.com/batch/docs/specify-job-network).

## Hardware and quota requirements

These are proposed research stages. The administrator should record the selected
machine type, zone, device count and approved concurrent capacity for each stage.

| Environment | Research purpose | Capacity to arrange |
|---|---|---|
| CPU Linux VM | ORE parity, float64 reference, precision campaigns and host-side comparisons | Capacity for one reference VM; enough host RAM and scratch disk for the selected path/trade grids |
| GPU Linux VM | GPU correctness, precision campaigns, scaling and kernel/Python profiling | Initially one GPU VM; request quota for its exact GPU model and region, global GPU quota, and the associated CPU/disk resources |
| H100 research target | The GPU target named in the precision plan | An available H100 machine configuration and quota for its actual device count; the supported allocation may contain multiple GPUs |
| TPU development environment | JAX/runtime compatibility, arithmetic checks and precision campaigns | A supported TPU configuration whose devices are reachable from one host for the initial worker model |
| Ironwood research target | The TPU target named in the precision plan | Confirm generation-specific access, supported topology, quota and available capacity; agree a separate implementation scope if multiple hosts are required |

A smaller GPU can qualify the experiment workflow before an H100 allocation is
used. Its measured performance should be reported for that hardware. Configure
host/device memory from the intended workloads rather than assuming a device
count guarantees a particular path or trade count.

GPU quotas depend on model, region and provisioning model (for example, standard
versus Spot). Check the global GPU quota as well. TPU quota needs to match the
selected accelerator, location and provisioning API: Compute Engine TPU quota
and Cloud TPU API quota are separate. Quota approval does not guarantee available
capacity at execution time.

Sources: [GPU machine configurations](https://docs.cloud.google.com/compute/docs/gpus),
[Compute/GPU quotas](https://docs.cloud.google.com/compute/resource-usage),
[Cloud TPU API quotas](https://docs.cloud.google.com/tpu/docs/quota),
[resource availability errors](https://docs.cloud.google.com/compute/docs/troubleshooting/troubleshooting-resource-availability).

## One-time administrator setup

1. Select the project, enable billing, and agree the initial hardware allocations
   and experiment budget. Define how temporary VMs, disks and outputs are cleaned up.
2. Enable Compute Engine, Batch, Cloud Logging, Cloud Monitoring, Cloud Storage and
   Artifact Registry APIs as used. Enable Cloud TPU API if that provisioning route
   is selected, and Cloud Build if automatic images are used.
3. Configure the runtime account and Alex's grants, including any cross-project
   image/input access. Confirm an approved network/subnet and SSH route.
4. Provide outbound access, or equivalent package mirrors, for the repository,
   Python packages and container images. Configure IAP/firewall rules and private
   API access or NAT as required by the selected network.
5. Obtain CPU/GPU/TPU quotas and confirm a usable provisioning option for each
   accelerator target. Check organization restrictions affecting locations,
   external IPs, identities or permitted machine configurations.
6. Supply Linux images and drivers compatible with the repository's dependencies.
   The current GPU extra uses JAX CUDA 13, requiring NVIDIA driver **580 or newer**.
   Qualify the TPU OS/JAX/libtpu combination separately against the repository pins.

The administrator can retain API activation and quota management. If Alex will
perform those tasks independently, add Service Usage Admin
(`roles/serviceusage.serviceUsageAdmin`) for API activation and Quota Administrator
(`roles/servicemanagement.quotaAdmin`) for quota adjustment requests on the project.

Sources: [API activation for Batch](https://docs.cloud.google.com/batch/docs/get-started),
[quota management permissions](https://docs.cloud.google.com/docs/quotas/view-manage),
[JAX installation and driver requirements](https://docs.jax.dev/en/latest/installation.html).
Repository dependencies are defined in [pyproject.toml](../pyproject.toml) and
[constraints.txt](../constraints.txt).

## Builds and images

Interactive source-based research can begin before container automation is ready.
For the image workflow, give Alex Cloud Build Viewer
(`roles/cloudbuild.builds.viewer`) and access to the configured log destination.
Add Cloud Build Editor (`roles/cloudbuild.builds.editor`) only if Alex will submit
manual builds or manage triggers, with Service Account User on the selected
builder account where required. Give Alex Artifact Registry Writer on a research
repository if Alex will publish images directly.

The supplied guides describe a prepared CPU build interface and a bounded
synthetic CPU pilot. A general research runner and accelerator images need their
own qualification before they are used for unattended precision campaigns.

Sources: [Cloud Build roles](https://docs.cloud.google.com/build/docs/iam-roles-permissions)
and [custom build service accounts](https://docs.cloud.google.com/build/docs/securing-builds/configure-user-specified-service-accounts).

## Research qualification and evidence

Access is ready when Alex can create a machine, connect, verify the requested
backend, execute a bounded calculation, save artifacts and remove the temporary
resources. Verify runtime storage access from inside the VM/job as well as from
Alex's terminal. For Batch, also verify job status and logs.

The numerical and performance qualification then follows the repository's
[precision plan](planning/details/precision.md),
[acceptance standard](../compliance/precision-acceptance.json) and
[evidence documentation](../compliance/precision-evidence.md):

- Run [precision device checks](../tests/support/precision_devices.py) on each
  target before qualifying its arithmetic, and after JAX/driver changes.
- Run the [precision campaigns](../tests/support/precision_campaigns.py): `linear`,
  `options`, `market_risk`, `compute`, `compute_options` and `compute_market_risk`.
  Keep experiment outputs in separate directories per hardware/environment.
- Establish the float64 reference and applicable ORE parity. TPU float64 behavior
  must be qualified for the engine's fixed float64 stages.
- Record source revision and changes, dependency versions, OS, driver/runtime,
  device model/count, seeds, path/trade/scenario counts and precision settings.
- Separate compilation from warm execution and synchronize device work for timing.
  Use the [CPU/GPU scaling study](concepts/gpu-scaling.md) and
  [Python/GPU profiling guide](concepts/profiling-python.md) for those measurements.
- Preserve numerical errors, native/emulated product execution, timing summaries,
  logs and selected traces under unique experiment paths.

At this revision, [products.py](../engine/precision/products.py) emulates FP8
products on all backends and bfloat16 passes on GPU pending numerical qualification.
Hardware access therefore needs follow-up device validation and engine work before
native FP8 speed can be claimed. The current worker model uses devices within one
host; multi-host TPU execution remains a separate planned capability.

## Request to send to the cloud administrator

> Please configure independent CPU, GPU and TPU research access for
> `aneugro1@mail.yu.edu`, preferably in a dedicated research project. I need to
> create, connect to, configure and clean up experiment machines; submit CPU/GPU
> Batch jobs; inspect logs and metrics; and retain inputs, outputs and profiles.
>
> Please grant the applicable Compute Instance Admin, OS Login/IAP, Batch Job
> Editor, storage, registry and observability access listed above, plus Service
> Account User on the experiment runtime account. If we use the Cloud TPU API,
> please add the corresponding TPU access. Please configure the runtime account's
> own permissions separately.
>
> Please confirm enabled APIs, networking, usable CPU/GPU/TPU quotas and available
> hardware configurations, including the H100 and Ironwood research targets.
> Record the approved device counts, locations, budget and cleanup arrangements.
> The existing ML Engineer grant needs to be supplemented for this workflow.

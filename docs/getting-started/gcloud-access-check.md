# Check your Google Cloud access before H100/TPU testing

Run these checks yourself first. Existing access may already be sufficient; involve
Yaakov only for a specific missing permission, billing confirmation or network detail.
These checks do not allocate compute. Step 6 creates one tiny test object in your
development bucket; step 7 connects only to an existing experiment VM.

Live account access was not verified while writing this checklist: `gcloud` was not
available in the local IDE terminal. Vendor references were checked October 21, 2026.

## 1. Open Cloud Shell and select the correct account/project

Open [Google Cloud Console](https://console.cloud.google.com/), sign in as
`aneugro1@mail.yu.edu`, and select `project-compsci-484414` from the project selector.
Click **Activate Cloud Shell** near the top right. Run all commands below in that
**Bash** terminal, not Windows PowerShell.

```bash
export PROJECT_ID=project-compsci-484414
export ACCOUNT=aneugro1@mail.yu.edu
export RUNTIME_SA="risk-engine-alex-dev@${PROJECT_ID}.iam.gserviceaccount.com"
gcloud auth list
gcloud config set account "$ACCOUNT"
gcloud config set project "$PROJECT_ID"
gcloud projects describe "$PROJECT_ID" --format='yaml(projectId,projectNumber,lifecycleState)'
```

**Pass:** the intended account is active and the project reports `lifecycleState: ACTIVE`.
Cloud Shell normally uses the console login. If the account is absent, sign into the
console with the intended account and reopen Cloud Shell. A project-access error is
different from the project being absent from a filtered selector. Keep its exact text.

You do not need `gcloud auth application-default login` or a service-account key for
these user-authenticated CLI checks.

## 2. Check billing and whether your credits apply

```bash
gcloud billing projects describe "$PROJECT_ID" \
  --format='yaml(billingAccountName,billingEnabled)'
```

**Pass:** a billing account is listed and `billingEnabled: true`. In the console, open
**Billing**, select the project's linked billing account, and open **Credits**. Check
the credit balance, expiration and covered services against H100/TPU usage. Having
credits on a different billing account does not fund this project automatically.
[Billing-status instructions](https://docs.cloud.google.com/billing/docs/how-to/verify-billing-enabled).

If you cannot view billing/credits, that alone does **not** mean you cannot create or
use a VM. Ask the billing administrator to confirm coverage for this project. Do not
change the shared project's billing account just to perform this check.

## 3. Check that the required service is enabled

```bash
gcloud services list --enabled --project="$PROJECT_ID" \
  --format='value(config.name)'
```

Look for `compute.googleapis.com` and `storage.googleapis.com`. The main correctness
guide creates **both H100 and TPU VMs through Compute Engine**. The older Cloud TPU API
and its separate quota pool are not the lifecycle used by that guide.

If Compute Engine is disabled, enable it if your role allows it:

```bash
gcloud services enable compute.googleapis.com --project="$PROJECT_ID"
```

Enabling the service does not allocate a VM. If this returns a permissions error, send
the error to Yaakov. If listing services itself is denied, record that as a visibility
gap; it does not independently prove that Compute Engine usage is denied.

## 4. Check core VM permissions and permission to attach the runtime identity

Being able to list VMs does not prove you can create one. Run this read-only permission
probe. It uses your current account's short-lived token internally and does not print it.

```bash
python3 - <<'PY'
import json, os, subprocess, urllib.error, urllib.parse, urllib.request
project, account = os.environ['PROJECT_ID'], os.environ['ACCOUNT']
service_account = os.environ['RUNTIME_SA']
token = subprocess.check_output(
    ['gcloud', 'auth', 'print-access-token', '--account=' + account], text=True).strip()
def check(title, endpoint, permissions):
    request = urllib.request.Request(endpoint,
        data=json.dumps({'permissions': permissions}).encode(),
        headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            granted = set(json.load(response).get('permissions', []))
    except urllib.error.HTTPError as error:
        print(title, 'CHECK FAILED:', error.code, error.read().decode())
        return
    print('\n' + title)
    for permission in permissions:
        print('GRANTED      ' if permission in granted else 'NOT CONFIRMED', permission)
check('Project: basic VM creation and cleanup',
      f'https://cloudresourcemanager.googleapis.com/v1/projects/{project}:testIamPermissions',
      ['compute.instances.create', 'compute.disks.create',
       'compute.instances.setServiceAccount', 'compute.instances.setMetadata',
       'compute.instances.get', 'compute.instances.list',
       'compute.instances.delete', 'compute.disks.delete',
       'compute.zoneOperations.get'])
encoded = urllib.parse.quote(service_account, safe='')
check('Selected runtime service account: attachment',
      f'https://iam.googleapis.com/v1/projects/-/serviceAccounts/{encoded}:testIamPermissions',
      ['iam.serviceAccounts.actAs'])
PY
```

**Expected:** `GRANTED` for the permissions needed by your selected configuration.
Deletion permissions let you clean up your own experiment resources. A role scoped to
particular existing resources or an IAM condition can make a project-level probe
inconclusive; treat `NOT CONFIRMED` as something to investigate, not definitive proof
that no valid configuration is possible.

The service-account check concerns **your permission to attach that account**. It does
not test the account's own bucket permissions. You do not need Service Account Token
Creator just to attach the account to a VM.

This is a core-permission check, not a full simulation of VM creation: the selected
subnet, images, IAM conditions and organization policies can add requirements.
If creating the VM reports a particular missing permission, use that resource and
permission in **IAM & Admin -> Policy Troubleshooter**, if you can access it, or send
the exact error to Yaakov. Avoid requesting Owner just to bypass a narrow gap.
[Project permission probe](https://docs.cloud.google.com/resource-manager/reference/rest/v1/projects/testIamPermissions),
[service-account permission probe](https://docs.cloud.google.com/iam/docs/reference/rest/v1/projects.serviceAccounts/testIamPermissions),
[VM creation requirements](https://docs.cloud.google.com/compute/docs/instances/create-start-instance).

## 5. Confirm quota for one H100 and one TPU

**This step only checks your project's accelerator allowance.** It does not create
machines or change cloud settings. If you have already found the matching quota rows,
check the amounts below, note your chosen regions, and move on to step 6.

1. Open [Quotas & System Limits for this project](https://console.cloud.google.com/iam-admin/quotas?project=project-compsci-484414)
   and filter **Service** to **Compute Engine API**.

2. Find the following allowances. In each case, **remaining = limit minus current usage**.
   Merely finding a quota's name is not enough; you need at least one remaining.

   | Planned machine | Quota to check | Amount needed |
   | --- | --- | --- |
   | One H100 Spot GPU | **Preemptible NVIDIA H100 GPUs** (`PREEMPTIBLE_NVIDIA_H100_GPUS`) | At least **1 remaining** in the region you will use for H100 |
   | One on-demand TPU v6e chip | **TPUs per TPU family**, dimension **`tpu_family:CT6E`** | At least **1 remaining** in the region you will use for TPU |

   For H100, also check **GPUs (all regions)** (`GPUS_ALL_REGIONS`) if it applies to
   your project: it must have at least **1 remaining**. This is a global allowance,
   so clear the region filter if necessary to find it. Ignore H100 **Mega** and
   **Committed** rows for this planned machine. If only standard H100 quota is shown,
   Google's Spot fallback rules can apply; do not infer a zero allowance solely from
   an absent preemptible row.
   [Quota names and allocation rules](https://docs.cloud.google.com/compute/resource-usage).

3. Write down the two regions you selected, for example in your own notes:

   ```text
   H100 region: <region with at least 1 H100 available under quota>
   TPU region:  <region with at least 1 CT6E chip available under quota>
   ```

   **The regions can differ.** If TPU quota is zero in your current region but another
   region has at least one remaining, choose that other region. You do not need a quota
   increase in the original region for this correctness experiment.

4. **You are done with step 5 when those allowances are confirmed and the regions are
   noted. Continue to step 6.** No machine-type lookup or network commands are required
   in this step. Zone selection, boot disks and the existing network/subnet will be
   checked when creating the experiment VM in the
   [hardware correctness guide](gcloud-hardware-correctness.md#3-use-cloud-shell-as-the-control-terminal).

Quota confirms what your project is allowed to allocate. Actual hardware availability
is established when VM creation succeeds. If no suitable region has enough quota,
request an increase or ask Yaakov to help with that specific quota.

## 6. Verify development-bucket upload and download

Use your own development bucket, rather than testing against TraderX data or relying
on whole-bucket listing permissions. This creates one small uniquely named object and
reads it back. It does not require overwrite or delete access.

```bash
export PROBE_ID="access-check-$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
export PROBE_DIR="$(mktemp -d)"
export PROBE_URI="gs://project-compsci-484414-risk-engine-alex-dev/experiments/$PROBE_ID/probe.txt"
printf 'Risk engine storage access check\n' > "$PROBE_DIR/probe.txt"
gcloud storage cp "$PROBE_DIR/probe.txt" "$PROBE_URI" \
  --project="$PROJECT_ID" --account="$ACCOUNT" --if-generation-match=0
gcloud storage cp "$PROBE_URI" "$PROBE_DIR/readback.txt" \
  --project="$PROJECT_ID" --account="$ACCOUNT"
cmp "$PROBE_DIR/probe.txt" "$PROBE_DIR/readback.txt" && echo 'PASS: upload and readback match'
```

**Pass:** both copies succeed and `cmp` reports a match. The tiny object remains as
evidence. `storage.objects.create` or `storage.objects.get` errors identify the missing
operation. A failed bucket listing is not equivalent to a failed exact-object read.

This tests **your user account**, which is enough for the main guide's workflow:
retrieve VM results with SCP, then upload from Cloud Shell. Direct VM-to-bucket access
would additionally require the attached VM service account to have the relevant
storage grants and suitable OAuth scopes.

## 7. Verify SSH, sudo and dependency access once an experiment VM exists

If no experiment VM exists, defer this step until you create the H100 using the main
guide. Do not use a TraderX production VM as an access probe.

```bash
export VM_NAME=REPLACE_WITH_EXPERIMENT_VM_NAME
export VM_ZONE=REPLACE_WITH_ITS_ZONE
gcloud compute instances describe "$VM_NAME" --project="$PROJECT_ID" --zone="$VM_ZONE" \
  --format='yaml(name,status,serviceAccounts,networkInterfaces)'
gcloud compute ssh "$VM_NAME" --project="$PROJECT_ID" --zone="$VM_ZONE"
```

If the VM uses IAP, add `--tunnel-through-iap`. After connecting, run inside the VM:

```bash
id
sudo -n true && echo 'PASS: sudo available'
git ls-remote https://github.com/AlexNeugroschl/JAX_Risk_Engine.git HEAD
python3 - <<'PY'
import urllib.request
with urllib.request.urlopen('https://pypi.org/simple/jax/', timeout=30) as response:
    print('Python package index reachable:', response.status)
PY
```

A private repository requires separate GitHub authentication; a GitHub denial is not
a Google IAM failure. Actual pinned package installation is the final dependency check.
If passwordless sudo is denied, try `sudo -v` to distinguish a password prompt from
missing sudo rights. You need sudo for the guide's OS/driver setup, or those components
must be supplied by the provisioner.

For an SSH failure, from Cloud Shell run:

```bash
gcloud compute ssh "$VM_NAME" --project="$PROJECT_ID" --zone="$VM_ZONE" --troubleshoot
# For IAP access, add --tunnel-through-iap to this command too.
```

The diagnostics can require additional visibility permissions; preserve any error.
With OS Login enabled, check OS Login access, **OS Admin Login** if you need sudo,
and Service Account User on the attached account. IAP also requires permission to
open the tunnel and a valid firewall route. External organization identities can need
an additional OS Login external-user grant. SSH timeout, public-key denial and missing
sudo are different failures.
[OS Login setup](https://docs.cloud.google.com/compute/docs/oslogin/set-up-oslogin),
[SSH troubleshooting](https://docs.cloud.google.com/compute/docs/troubleshooting/troubleshooting-ssh-errors).

## 8. Decide whether you can proceed

Proceed to [the hardware correctness guide](gcloud-hardware-correctness.md) when your
identity/project, billing coverage, Compute Engine service, applicable create/attach/
delete permissions, quota and development storage are established. Validate SSH and
runtime access immediately after provisioning your experiment VM.

| Error or result | Meaning |
| --- | --- |
| `PERMISSION_DENIED` / named IAM permission | Access gap; retain account, resource and exact permission |
| API disabled / `SERVICE_DISABLED` | Enable that service if allowed, or request enablement |
| `QUOTA_EXCEEDED` | Insufficient quota; increasing IAM privileges does not fix it |
| `ZONE_RESOURCE_POOL_EXHAUSTED` / insufficient capacity | Capacity issue; try another supported zone or supported provisioning option |
| Organization-policy violation | Configuration is restricted; use a compliant configuration or request a targeted exception |
| Cannot view IAM/billing diagnostics | Visibility gap; not by itself proof that workload execution is denied |

If blocked, send Yaakov just: **account, project, failed command, full error, relevant
resource/region, and the intended action**. There is no need to request broad additional
access if these checks already pass.

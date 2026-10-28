# Hardware testing

How the engine is tested on Google Cloud H100 GPUs and TPUs, and what those trials found.
The reports are dated records of what ran at a given engine revision; newer reports
supersede older ones where they overlap.

## Running a trial

- [Check your Google Cloud access](gcloud-access-check.md): billing, permissions, quota,
  storage and SSH, checked before any compute is allocated.
- [Correctness testing on CPU, H100 and TPU](gcloud-hardware-correctness.md): provisioning,
  setup, the comparison runs, reading the results and cleaning up.

## Findings, newest first

- [Native TPU probes](tpu-native-precision.md) (2026-10-27): FP8 matrix candidates pass;
  FP16 remains unsuccessful.
- [TPU precision survey](tpu-precision-survey.md) (2026-10-27): all tested non-FP64
  arithmetic passes through the engine's emulation paths.
- [Cloud failures: boundary analysis](cloud-root-cause-analysis.md) (2026-10-27): local
  reproductions of the October 26 failures and the corrections made.
- [Hardware correctness rerun](hardware-correctness-rerun.md) (2026-10-26): the H100 fix
  passes; the first TPU fixes fail.
- [H100 and TPU failures](hardware-correctness-findings.md) (2026-10-26): causes and fixes
  of the first full-precision trial's failures.

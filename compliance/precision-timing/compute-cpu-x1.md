# Timing: compute

Devices cpu (cpu), jax 0.10.2, commit 0b4ec6768e with changes, engine 5483f49b9633b217, 2026-10-20T02:07:58+00:00.

| Case | Paths | Budget | Cold s | Compile s | Warm s | Peak MB | Products |
|---|---:|---:|---:|---:|---:|---:|---|
| float64 | 1024 | default | 5.79 | 3.98 | 0.051 |  | float64 native |
| float64 | 4096 | default | 5.63 | 3.93 | 0.071 |  | float64 native |
| float64 | 16384 | default | 5.72 | 3.98 | 0.153 |  | float64 native |
| throughout:bfloat16 | 1024 | default | 7.44 | 4.82 | 0.052 |  | bfloat16 emulated |
| throughout:bfloat16 | 4096 | default | 7.36 | 4.82 | 0.073 |  | bfloat16 emulated |
| throughout:bfloat16 | 16384 | default | 7.63 | 5.04 | 0.141 |  | bfloat16 emulated |
| throughout:bfloat16_1x | 1024 | default | 7.30 | 4.78 | 0.051 |  | bfloat16_1x emulated |
| throughout:bfloat16_1x | 4096 | default | 7.17 | 4.71 | 0.071 |  | bfloat16_1x emulated |
| throughout:bfloat16_1x | 16384 | default | 7.47 | 4.92 | 0.137 |  | bfloat16_1x emulated |
| throughout:bfloat16_3x | 1024 | default | 6.98 | 4.55 | 0.061 |  | bfloat16_3x emulated |
| throughout:bfloat16_3x | 4096 | default | 6.84 | 4.46 | 0.070 |  | bfloat16_3x emulated |
| throughout:bfloat16_3x | 16384 | default | 6.95 | 4.55 | 0.136 |  | bfloat16_3x emulated |
| throughout:float16 | 1024 | default | 8.00 | 5.11 | 0.062 |  | float16 emulated |
| throughout:float16 | 4096 | default | 7.83 | 5.02 | 0.074 |  | float16 emulated |
| throughout:float16 | 16384 | default | 8.12 | 5.25 | 0.151 |  | float16 emulated |
| throughout:float16_1x | 1024 | default | 7.93 | 5.09 | 0.053 |  | float16_1x emulated |
| throughout:float16_1x | 4096 | default | 7.73 | 4.97 | 0.076 |  | float16_1x emulated |
| throughout:float16_1x | 16384 | default | 7.97 | 5.16 | 0.148 |  | float16_1x emulated |
| throughout:float32 | 1024 | default | 6.49 | 4.44 | 0.059 |  | float32 native |
| throughout:float32 | 4096 | default | 6.12 | 4.19 | 0.068 |  | float32 native |
| throughout:float32 | 16384 | default | 6.34 | 4.33 | 0.129 |  | float32 native |
| throughout:float8_e4m3fn | 1024 | default | 8.18 | 5.34 | 0.054 |  | float8_e4m3fn emulated |
| throughout:float8_e4m3fn | 4096 | default | 8.15 | 5.28 | 0.071 |  | float8_e4m3fn emulated |
| throughout:float8_e4m3fn | 16384 | default | 8.53 | 5.56 | 0.150 |  | float8_e4m3fn emulated |
| throughout:float8_e4m3fn_1x | 1024 | default | 8.08 | 5.24 | 0.055 |  | float8_e4m3fn_1x emulated |
| throughout:float8_e4m3fn_1x | 4096 | default | 7.95 | 5.12 | 0.074 |  | float8_e4m3fn_1x emulated |
| throughout:float8_e4m3fn_1x | 16384 | default | 8.25 | 5.42 | 0.150 |  | float8_e4m3fn_1x emulated |
| throughout:float8_e5m2 | 1024 | default | 8.78 | 5.63 | 0.056 |  | float8_e5m2 emulated |
| throughout:float8_e5m2 | 4096 | default | 8.06 | 5.25 | 0.076 |  | float8_e5m2 emulated |
| throughout:float8_e5m2 | 16384 | default | 8.36 | 5.48 | 0.145 |  | float8_e5m2 emulated |
| throughout:float8_e5m2_1x | 1024 | default | 8.12 | 5.24 | 0.053 |  | float8_e5m2_1x emulated |
| throughout:float8_e5m2_1x | 4096 | default | 7.86 | 5.10 | 0.073 |  | float8_e5m2_1x emulated |
| throughout:float8_e5m2_1x | 16384 | default | 8.15 | 5.35 | 0.144 |  | float8_e5m2_1x emulated |
| throughout:tensorfloat32 | 1024 | default | 7.10 | 4.50 | 0.052 |  | tensorfloat32 emulated |
| throughout:tensorfloat32 | 4096 | default | 6.83 | 4.43 | 0.071 |  | tensorfloat32 emulated |
| throughout:tensorfloat32 | 16384 | default | 7.02 | 4.53 | 0.134 |  | tensorfloat32 emulated |
| throughout:tensorfloat32_1x | 1024 | default | 6.91 | 4.48 | 0.051 |  | tensorfloat32_1x emulated |
| throughout:tensorfloat32_1x | 4096 | default | 6.67 | 4.33 | 0.070 |  | tensorfloat32_1x emulated |
| throughout:tensorfloat32_1x | 16384 | default | 6.88 | 4.45 | 0.130 |  | tensorfloat32_1x emulated |

Speedup at float64's accuracy (float64's warm time over the case's, at the path count where the case's error reaches float64's): the worst figure, and the figures not reached.

| Case | float64 paths | Worst speedup | Worst figure | Not reached |
|---|---:|---:|---|---:|
| throughout:bfloat16 | 1024 | 0.93 | netting_set/EEE_B | 0 of 55 |
| throughout:bfloat16 | 4096 | 0.53 | netting_set/EEPE_B | 0 of 55 |
| throughout:bfloat16_1x | 1024 | 0.88 | netting_set/EEPE_B | 5 of 55 |
| throughout:bfloat16_1x | 4096 | 0.52 | trades/bond/NPV | 35 of 55 |
| throughout:bfloat16_3x | 1024 | 0.84 | trades/european-receiver-otm-cash/Basel_EEPE | 0 of 55 |
| throughout:bfloat16_3x | 4096 | 1.01 | trades/european-payer/EEE_B | 0 of 55 |
| throughout:float16 | 1024 | 0.82 | trades/bond/Basel_EPE | 0 of 55 |
| throughout:float16 | 4096 | 0.79 | netting_set/EEPE_B | 0 of 55 |
| throughout:float16_1x | 1024 | 0.89 | trades/swap-receiver-seasoned-icma/Basel_EPE | 0 of 55 |
| throughout:float16_1x | 4096 | 0.50 | trades/european-receiver-otm-cash/EEE_B | 5 of 55 |
| throughout:float32 | 1024 | 0.86 | trades/european-payer/Basel_EEPE | 0 of 55 |
| throughout:float32 | 4096 | 1.04 | netting_set/Basel_EEPE | 0 of 55 |
| throughout:float8_e4m3fn | 1024 | 0.54 | trades/swap-payer/Basel_EEPE | 50 of 55 |
| throughout:float8_e4m3fn | 4096 | 0.51 | trades/swap-receiver-seasoned-icma/Basel_EPE | 54 of 55 |
| throughout:float8_e4m3fn_1x | 1024 | 0.84 | trades/european-payer/ENE | 54 of 55 |
| throughout:float8_e4m3fn_1x | 4096 |  |  | 55 of 55 |
| throughout:float8_e5m2 | 1024 | 0.45 | trades/swap-payer/Basel_EEPE | 52 of 55 |
| throughout:float8_e5m2 | 4096 |  |  | 55 of 55 |
| throughout:float8_e5m2_1x | 1024 | 0.74 | trades/european-payer/ENE | 53 of 55 |
| throughout:float8_e5m2_1x | 4096 |  |  | 55 of 55 |
| throughout:tensorfloat32 | 1024 | 0.98 | netting_set/EEE_B | 0 of 55 |
| throughout:tensorfloat32 | 4096 | 0.96 | netting_set/Basel_EEPE | 0 of 55 |
| throughout:tensorfloat32_1x | 1024 | 0.93 | trades/swap-receiver-seasoned-icma/Basel_EPE | 0 of 55 |
| throughout:tensorfloat32_1x | 4096 | 0.56 | trades/european-receiver-otm-cash/EEE_B | 5 of 55 |

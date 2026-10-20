# Timing: compute_market_risk

Devices cpu (cpu), jax 0.10.2, commit 0b4ec6768e with changes, engine 5483f49b9633b217, 2026-10-20T02:01:39+00:00.

| Case | Paths | Budget | Cold s | Compile s | Warm s | Peak MB | Products |
|---|---:|---:|---:|---:|---:|---:|---|
| float64 | 2048 | default | 2.66 | 1.73 | 0.022 |  | float64 native |
| float64 | 8192 | default | 2.47 | 1.62 | 0.060 |  | float64 native |
| pricing:compute:bfloat16 | 2048 | default | 3.28 | 2.03 | 0.026 |  | bfloat16 emulated, float64 native |
| pricing:compute:bfloat16 | 8192 | default | 3.30 | 2.00 | 0.077 |  | bfloat16 emulated, float64 native |
| pricing:compute:bfloat16_1x | 2048 | default | 2.94 | 1.84 | 0.023 |  | bfloat16_1x emulated, float64 native |
| pricing:compute:bfloat16_1x | 8192 | default | 3.01 | 1.84 | 0.067 |  | bfloat16_1x emulated, float64 native |
| pricing:compute:bfloat16_3x | 2048 | default | 3.24 | 2.00 | 0.030 |  | bfloat16_3x emulated, float64 native |
| pricing:compute:bfloat16_3x | 8192 | default | 3.21 | 1.97 | 0.090 |  | bfloat16_3x emulated, float64 native |
| pricing:compute:float16 | 2048 | default | 3.37 | 2.13 | 0.027 |  | float16 emulated, float64 native |
| pricing:compute:float16 | 8192 | default | 3.40 | 2.11 | 0.079 |  | float16 emulated, float64 native |
| pricing:compute:float16_1x | 2048 | default | 3.31 | 2.03 | 0.024 |  | float16_1x emulated, float64 native |
| pricing:compute:float16_1x | 8192 | default | 3.23 | 1.96 | 0.068 |  | float16_1x emulated, float64 native |
| pricing:compute:float32 | 2048 | default | 2.67 | 1.80 | 0.022 |  | float32 native, float64 native |
| pricing:compute:float32 | 8192 | default | 2.68 | 1.77 | 0.052 |  | float32 native, float64 native |
| pricing:compute:float8_e4m3fn | 2048 | default | 3.75 | 2.33 | 0.034 |  | float64 native, float8_e4m3fn emulated |
| pricing:compute:float8_e4m3fn | 8192 | default | 3.69 | 2.25 | 0.078 |  | float64 native, float8_e4m3fn emulated |
| pricing:compute:float8_e4m3fn_1x | 2048 | default | 3.43 | 2.07 | 0.028 |  | float64 native, float8_e4m3fn_1x emulated |
| pricing:compute:float8_e4m3fn_1x | 8192 | default | 3.29 | 2.03 | 0.071 |  | float64 native, float8_e4m3fn_1x emulated |
| pricing:compute:float8_e5m2 | 2048 | default | 3.42 | 2.18 | 0.027 |  | float64 native, float8_e5m2 emulated |
| pricing:compute:float8_e5m2 | 8192 | default | 3.50 | 2.20 | 0.081 |  | float64 native, float8_e5m2 emulated |
| pricing:compute:float8_e5m2_1x | 2048 | default | 3.22 | 2.01 | 0.027 |  | float64 native, float8_e5m2_1x emulated |
| pricing:compute:float8_e5m2_1x | 8192 | default | 3.29 | 2.02 | 0.072 |  | float64 native, float8_e5m2_1x emulated |
| pricing:compute:tensorfloat32 | 2048 | default | 3.05 | 1.90 | 0.023 |  | float64 native, tensorfloat32 emulated |
| pricing:compute:tensorfloat32 | 8192 | default | 3.12 | 1.92 | 0.073 |  | float64 native, tensorfloat32 emulated |
| pricing:compute:tensorfloat32_1x | 2048 | default | 2.97 | 1.84 | 0.021 |  | float64 native, tensorfloat32_1x emulated |
| pricing:compute:tensorfloat32_1x | 8192 | default | 3.34 | 1.99 | 0.065 |  | float64 native, tensorfloat32_1x emulated |

Speedup at float64's accuracy (float64's warm time over the case's, at the path count where the case's error reaches float64's): the worst figure, and the figures not reached.

| Case | float64 paths | Worst speedup | Worst figure | Not reached |
|---|---:|---:|---|---:|
| pricing:compute:bfloat16 | 2048 | 0.71 | trades/bermudan/PnL | 0 of 10 |
| pricing:compute:bfloat16_1x | 2048 | 0.78 | portfolio/VaR_99 | 0 of 10 |
| pricing:compute:bfloat16_3x | 2048 | 0.74 | trades/bermudan/PnL | 0 of 10 |
| pricing:compute:float16 | 2048 | 0.82 | portfolio/VaR_99 | 0 of 10 |
| pricing:compute:float16_1x | 2048 | 0.91 | trades/bermudan/PnL | 0 of 10 |
| pricing:compute:float32 | 2048 | 1.02 | trades/bermudan/PnL | 0 of 10 |
| pricing:compute:float8_e4m3fn | 2048 | 0.46 | portfolio/VaR_99 | 4 of 10 |
| pricing:compute:float8_e4m3fn_1x | 2048 | 0.79 | portfolio/ES_99 | 8 of 10 |
| pricing:compute:float8_e5m2 | 2048 | 0.60 | trades/european/PnL | 5 of 10 |
| pricing:compute:float8_e5m2_1x | 2048 | 0.34 | trades/swap/PnL | 5 of 10 |
| pricing:compute:tensorfloat32 | 2048 | 0.96 | portfolio/VaR_99 | 0 of 10 |
| pricing:compute:tensorfloat32_1x | 2048 | 1.03 | trades/bermudan/PnL | 0 of 10 |

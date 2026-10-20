# Timing: compute_market_risk

Devices NVIDIA GeForce RTX 5060 Laptop GPU (gpu), jax 0.10.2, commit 0b4ec6768e with changes, engine 5483f49b9633b217, 2026-10-20T02:14:00+00:00.

| Case | Paths | Budget | Cold s | Compile s | Warm s | Peak MB | Products |
|---|---:|---:|---:|---:|---:|---:|---|
| float64 | 2048 | default | 6.63 | 5.41 | 0.139 | 68 | float64 native |
| float64 | 8192 | default | 6.56 | 5.27 | 0.397 | 72 | float64 native |
| pricing:compute:bfloat16 | 2048 | default | 6.56 | 5.14 | 0.129 | 68 | bfloat16 native, float64 native |
| pricing:compute:bfloat16 | 8192 | default | 7.04 | 5.40 | 0.395 | 72 | bfloat16 native, float64 native |
| pricing:compute:bfloat16_1x | 2048 | default | 6.40 | 5.01 | 0.127 | 68 | bfloat16_1x native, float64 native |
| pricing:compute:bfloat16_1x | 8192 | default | 6.89 | 5.27 | 0.393 | 72 | bfloat16_1x native, float64 native |
| pricing:compute:bfloat16_3x | 2048 | default | 6.56 | 5.21 | 0.131 | 68 | bfloat16_3x native, float64 native |
| pricing:compute:bfloat16_3x | 8192 | default | 6.86 | 5.32 | 0.401 | 72 | bfloat16_3x native, float64 native |
| pricing:compute:float16 | 2048 | default | 7.12 | 5.60 | 0.130 | 68 | float16 native, float64 native |
| pricing:compute:float16 | 8192 | default | 7.69 | 5.89 | 0.397 | 72 | float16 native, float64 native |
| pricing:compute:float16_1x | 2048 | default | 6.82 | 5.33 | 0.137 | 68 | float16_1x native, float64 native |
| pricing:compute:float16_1x | 8192 | default | 7.28 | 5.55 | 0.394 | 72 | float16_1x native, float64 native |
| pricing:compute:float32 | 2048 | default | 6.26 | 5.04 | 0.156 | 68 | float32 native, float64 native |
| pricing:compute:float32 | 8192 | default | 6.35 | 5.00 | 0.391 | 72 | float32 native, float64 native |
| pricing:compute:float8_e4m3fn | 2048 | default | 7.14 | 5.59 | 0.137 | 68 | float64 native, float8_e4m3fn native |
| pricing:compute:float8_e4m3fn | 8192 | default | 7.67 | 5.89 | 0.405 | 72 | float64 native, float8_e4m3fn native |
| pricing:compute:float8_e4m3fn_1x | 2048 | default | 6.74 | 5.24 | 0.128 | 68 | float64 native, float8_e4m3fn_1x native |
| pricing:compute:float8_e4m3fn_1x | 8192 | default | 7.25 | 5.51 | 0.391 | 72 | float64 native, float8_e4m3fn_1x native |
| pricing:compute:float8_e5m2 | 2048 | default | 7.08 | 5.53 | 0.126 | 68 | float64 native, float8_e5m2 native |
| pricing:compute:float8_e5m2 | 8192 | default | 7.64 | 5.87 | 0.398 | 72 | float64 native, float8_e5m2 native |
| pricing:compute:float8_e5m2_1x | 2048 | default | 6.77 | 5.25 | 0.130 | 68 | float64 native, float8_e5m2_1x native |
| pricing:compute:float8_e5m2_1x | 8192 | default | 7.25 | 5.53 | 0.393 | 72 | float64 native, float8_e5m2_1x native |
| pricing:compute:tensorfloat32 | 2048 | default | 6.64 | 5.87 | 0.132 | 68 | float64 native, tensorfloat32 native |
| pricing:compute:tensorfloat32 | 8192 | default | 7.03 | 5.37 | 0.394 | 72 | float64 native, tensorfloat32 native |
| pricing:compute:tensorfloat32_1x | 2048 | default | 6.39 | 5.00 | 0.130 | 68 | float64 native, tensorfloat32_1x native |
| pricing:compute:tensorfloat32_1x | 8192 | default | 6.91 | 5.83 | 0.390 | 72 | float64 native, tensorfloat32_1x native |

Speedup at float64's accuracy (float64's warm time over the case's, at the path count where the case's error reaches float64's): the worst figure, and the figures not reached.

| Case | float64 paths | Worst speedup | Worst figure | Not reached |
|---|---:|---:|---|---:|
| pricing:compute:bfloat16 | 2048 | 0.87 | trades/bermudan/PnL | 0 of 10 |
| pricing:compute:bfloat16_1x | 2048 | 0.87 | portfolio/VaR_99 | 0 of 10 |
| pricing:compute:bfloat16_3x | 2048 | 1.06 | trades/bond/PnL | 0 of 10 |
| pricing:compute:float16 | 2048 | 1.06 | portfolio/VaR_99 | 0 of 10 |
| pricing:compute:float16_1x | 2048 | 1.00 | trades/bermudan/PnL | 0 of 10 |
| pricing:compute:float32 | 2048 | 0.89 | trades/bermudan/PnL | 0 of 10 |
| pricing:compute:float8_e4m3fn | 2048 | 0.64 | portfolio/VaR_99 | 4 of 10 |
| pricing:compute:float8_e4m3fn_1x | 2048 | 1.08 | portfolio/ES_99 | 8 of 10 |
| pricing:compute:float8_e5m2 | 2048 | 0.79 | trades/european/PnL | 5 of 10 |
| pricing:compute:float8_e5m2_1x | 2048 | 0.39 | trades/swap/PnL | 5 of 10 |
| pricing:compute:tensorfloat32 | 2048 | 1.05 | portfolio/VaR_99 | 0 of 10 |
| pricing:compute:tensorfloat32_1x | 2048 | 1.06 | trades/bermudan/PnL | 0 of 10 |

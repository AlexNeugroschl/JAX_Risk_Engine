# Timing: compute

Devices NVIDIA GeForce RTX 5060 Laptop GPU (gpu), jax 0.10.2, commit 0b4ec6768e with changes, engine 5483f49b9633b217, 2026-10-20T02:24:46+00:00.

| Case | Paths | Budget | Cold s | Compile s | Warm s | Peak MB | Products |
|---|---:|---:|---:|---:|---:|---:|---|
| float64 | 1024 | default | 9.63 | 7.26 | 0.431 | 69 | float64 native |
| float64 | 4096 | default | 9.80 | 7.41 | 0.484 | 69 | float64 native |
| float64 | 16384 | default | 9.95 | 7.49 | 0.419 | 175 | float64 native |
| throughout:bfloat16 | 1024 | default | 12.24 | 8.80 | 0.446 | 69 | bfloat16 native |
| throughout:bfloat16 | 4096 | default | 12.36 | 8.96 | 0.434 | 69 | bfloat16 native |
| throughout:bfloat16 | 16384 | default | 12.52 | 9.07 | 0.441 | 100 | bfloat16 native |
| throughout:bfloat16_1x | 1024 | default | 12.13 | 9.33 | 0.451 | 69 | bfloat16_1x native |
| throughout:bfloat16_1x | 4096 | default | 12.21 | 8.84 | 0.427 | 69 | bfloat16_1x native |
| throughout:bfloat16_1x | 16384 | default | 12.37 | 9.91 | 0.400 | 99 | bfloat16_1x native |
| throughout:bfloat16_3x | 1024 | default | 11.28 | 8.86 | 0.446 | 69 | bfloat16_3x native |
| throughout:bfloat16_3x | 4096 | default | 11.31 | 8.29 | 0.448 | 69 | bfloat16_3x native |
| throughout:bfloat16_3x | 16384 | default | 11.49 | 9.45 | 0.416 | 89 | bfloat16_3x native |
| throughout:float16 | 1024 | default | 13.25 | 9.53 | 0.446 | 69 | float16 native |
| throughout:float16 | 4096 | default | 13.28 | 9.57 | 0.430 | 69 | float16 native |
| throughout:float16 | 16384 | default | 13.30 | 9.63 | 0.440 | 100 | float16 native |
| throughout:float16_1x | 1024 | default | 12.69 | 9.13 | 0.462 | 69 | float16_1x native |
| throughout:float16_1x | 4096 | default | 12.58 | 9.12 | 0.438 | 69 | float16_1x native |
| throughout:float16_1x | 16384 | default | 13.08 | 9.53 | 0.406 | 101 | float16_1x native |
| throughout:float32 | 1024 | default | 10.62 | 7.89 | 0.439 | 69 | float32 native |
| throughout:float32 | 4096 | default | 10.80 | 8.93 | 0.449 | 69 | float32 native |
| throughout:float32 | 16384 | default | 10.73 | 8.04 | 0.412 | 89 | float32 native |
| throughout:float8_e4m3fn | 1024 | default | 13.26 | 9.57 | 0.428 | 69 | float8_e4m3fn native |
| throughout:float8_e4m3fn | 4096 | default | 13.37 | 10.54 | 0.434 | 69 | float8_e4m3fn native |
| throughout:float8_e4m3fn | 16384 | default | 13.40 | 9.74 | 0.429 | 94 | float8_e4m3fn native |
| throughout:float8_e4m3fn_1x | 1024 | default | 12.60 | 9.67 | 0.435 | 69 | float8_e4m3fn_1x native |
| throughout:float8_e4m3fn_1x | 4096 | default | 12.73 | 9.20 | 0.445 | 69 | float8_e4m3fn_1x native |
| throughout:float8_e4m3fn_1x | 16384 | default | 12.89 | 10.31 | 0.425 | 93 | float8_e4m3fn_1x native |
| throughout:float8_e5m2 | 1024 | default | 13.28 | 9.55 | 0.433 | 69 | float8_e5m2 native |
| throughout:float8_e5m2 | 4096 | default | 13.36 | 10.31 | 0.430 | 69 | float8_e5m2 native |
| throughout:float8_e5m2 | 16384 | default | 13.37 | 9.69 | 0.426 | 94 | float8_e5m2 native |
| throughout:float8_e5m2_1x | 1024 | default | 12.46 | 9.55 | 0.402 | 69 | float8_e5m2_1x native |
| throughout:float8_e5m2_1x | 4096 | default | 12.51 | 9.03 | 0.429 | 69 | float8_e5m2_1x native |
| throughout:float8_e5m2_1x | 16384 | default | 13.46 | 9.74 | 0.434 | 95 | float8_e5m2_1x native |
| throughout:tensorfloat32 | 1024 | default | 11.63 | 8.40 | 0.445 | 69 | tensorfloat32 native |
| throughout:tensorfloat32 | 4096 | default | 11.83 | 9.17 | 0.461 | 69 | tensorfloat32 native |
| throughout:tensorfloat32 | 16384 | default | 11.67 | 8.42 | 0.424 | 88 | tensorfloat32 native |
| throughout:tensorfloat32_1x | 1024 | default | 11.42 | 8.85 | 0.431 | 69 | tensorfloat32_1x native |
| throughout:tensorfloat32_1x | 4096 | default | 11.59 | 8.36 | 0.456 | 69 | tensorfloat32_1x native |
| throughout:tensorfloat32_1x | 16384 | default | 11.77 | 8.56 | 0.448 | 88 | tensorfloat32_1x native |

Speedup at float64's accuracy (float64's warm time over the case's, at the path count where the case's error reaches float64's): the worst figure, and the figures not reached.

| Case | float64 paths | Worst speedup | Worst figure | Not reached |
|---|---:|---:|---|---:|
| throughout:bfloat16 | 1024 | 0.97 | netting_set/Basel_EPE | 0 of 55 |
| throughout:bfloat16 | 4096 | 1.10 | netting_set/EEE_B | 0 of 55 |
| throughout:bfloat16_1x | 1024 | 0.96 | trades/european-payer/NPV | 5 of 55 |
| throughout:bfloat16_1x | 4096 | 1.13 | netting_set/PFE_95 | 35 of 55 |
| throughout:bfloat16_3x | 1024 | 0.97 | trades/european-payer/Basel_EEPE | 0 of 55 |
| throughout:bfloat16_3x | 4096 | 1.08 | trades/swap-receiver-seasoned-icma/PFE_95 | 0 of 55 |
| throughout:float16 | 1024 | 0.97 | netting_set/Basel_EPE | 0 of 55 |
| throughout:float16 | 4096 | 1.12 | netting_set/EEE_B | 0 of 55 |
| throughout:float16_1x | 1024 | 0.93 | netting_set/Basel_EEPE | 0 of 55 |
| throughout:float16_1x | 4096 | 1.10 | trades/bond/Basel_EPE | 5 of 55 |
| throughout:float32 | 1024 | 0.98 | trades/european-payer/Basel_EEPE | 0 of 55 |
| throughout:float32 | 4096 | 1.08 | trades/european-payer/ENE | 0 of 55 |
| throughout:float8_e4m3fn | 1024 | 0.99 | trades/swap-receiver-seasoned-icma/Basel_EPE | 50 of 55 |
| throughout:float8_e4m3fn | 4096 | 1.13 | trades/swap-receiver-seasoned-icma/Basel_EPE | 54 of 55 |
| throughout:float8_e4m3fn_1x | 1024 | 0.98 | trades/european-payer/ENE | 54 of 55 |
| throughout:float8_e4m3fn_1x | 4096 |  |  | 55 of 55 |
| throughout:float8_e5m2 | 1024 | 1.00 | trades/european-receiver-otm-cash/PFE_95 | 52 of 55 |
| throughout:float8_e5m2 | 4096 |  |  | 55 of 55 |
| throughout:float8_e5m2_1x | 1024 | 1.02 | trades/european-payer/ENE | 53 of 55 |
| throughout:float8_e5m2_1x | 4096 |  |  | 55 of 55 |
| throughout:tensorfloat32 | 1024 | 0.97 | netting_set/Basel_EEPE | 0 of 55 |
| throughout:tensorfloat32 | 4096 | 1.05 | trades/swap-receiver-seasoned-icma/EPE | 0 of 55 |
| throughout:tensorfloat32_1x | 1024 | 0.99 | trades/swap-receiver-seasoned-icma/Basel_EPE | 0 of 55 |
| throughout:tensorfloat32_1x | 4096 | 1.06 | netting_set/PFE_95 | 5 of 55 |

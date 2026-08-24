# JAX Risk Engine

A [JAX](https://github.com/google/jax)-based market simulation and trade-pricing engine
built to run pricing/simulation across multiple TPUs — the intended deployment target is
a Google Cloud TPU VM, which is how TPU access actually happens for this project — and to
study precision (float32 vs. float64) tradeoffs on that hardware. It's designed to
mathematically mirror [ORE (Open Source Risk Engine)](https://www.opensourcerisk.org/) —
a mature, real-world risk engine used by actual financial institutions — for correctness,
while exploiting JAX's vectorization and multi-device execution instead of ORE's
single-threaded CPU-based C++. The architecture is backend-agnostic by construction (it
runs correctly on CPU and GPU too, and the current dev/test suite runs entirely on CPU)
but is optimized specifically for TPU.

Every pricing and risk formula in this codebase has been checked, line-by-line where
possible, against ORE's own installed software and C++ source, not against a textbook
description. See [ORE Parity](docs/reference/ore-parity.md) for the full
algorithm-by-algorithm mapping.

## What's implemented

| Component | Status |
|---|---|
| Cross-asset market simulation (rates, equities, FX) | ✅ |
| Interest rate swaps | ✅ |
| European swaptions (Jamshidian's decomposition) | ✅ |
| Bermudan & American swaptions (numeric LGM backward induction) | ✅ |
| Value at Risk / Expected Shortfall | ✅ |
| XVA (CVA/DVA) | 🔜 Planned |
| Live API (TraderX integration) | 🔜 Planned |

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q
```

See the [User Guide](docs/getting-started/user-guide.md) for a full setup walkthrough and
runnable pricing examples.

## Documentation

Start at **[docs/README.md](docs/README.md)** for the full documentation index —
architecture, per-instrument deep dives, API reference, ORE parity mapping, glossary, and
the development roadmap.

If you're new to the project, [docs/getting-started/overview.md](docs/getting-started/overview.md)
explains what this does and why, with no finance or math background assumed.

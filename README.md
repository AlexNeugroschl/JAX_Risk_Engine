# JAX Risk Engine

A GPU-accelerated market simulation and trade-pricing engine built in
[JAX](https://github.com/google/jax), designed to mathematically mirror
[ORE (Open Source Risk Engine)](https://www.opensourcerisk.org/) — a mature, real-world
risk engine used by actual financial institutions — while running orders of magnitude
faster by exploiting GPU vectorization instead of ORE's CPU-based C++.

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

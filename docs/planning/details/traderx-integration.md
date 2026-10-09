# TraderX integration

The end-of-day (EOD) integration with [TraderX](https://github.com/finos/traderX): what is
agreed, what is open with TraderX, and the remaining work. How the boundary works today is in
[EOD Integration](../../reference/eod-integration.md). The negotiation itself (the proposal
and responses v2 to v7, the readiness and bond plans) is in git history, last present at
commit `8306073` under `docs/planning/traderX_integration/`.

## State

TraderX exports an immutable, hash-pinned EOD bundle (positions, OTC contracts, an
`instrument-terms` artifact). Their coordinator owns the durable logical job and calls
`/eod`; this engine verifies hashes, joins terms, refuses what it cannot price faithfully,
prices what it can against an explicitly requested curve, and publishes immutable,
identified results. Tasks W0 (contract and refusal machinery) and W1 (bill, note, equity
refusal, contract interface, portfolio wire-through) are done. TraderX independently
reproduced every priced figure (bill ±98,507.15, note ±103,308.33, accrued ±1,857.10, +1bp
sensitivity ∓15.28).

Remaining engine work: the three bugs in the submission path that TraderX's acceptance kit found
([I-57](../known-issues.md#i-57), [I-58](../known-issues.md#i-58),
[I-59](../known-issues.md#i-59)), running-attempt durability ([I-08](../known-issues.md#i-08)),
and W2 below once its inputs arrive.

## Settled — do not reopen

| Topic | Agreement |
|---|---|
| Bundle transport | Immutable bundle plus versioned manifest; pull-discoverable, notifications only accelerate |
| Job ownership | TraderX owns the durable logical job and retry decisions; the engine owns computation attempts and result publication |
| Refusal policy | Unmapped conventions → `unsupported` with `CONVENTION_NOT_SUPPORTED`, even if that covers every OTC row |
| Market data | TraderX supplies dated observations; the engine builds curves and returns repricing diagnostics |
| Assumed curves | Requested explicitly by profile id; missing market data fails the job, never falls back |
| Units | Coupon in annual percent; marks and accrued as fraction of par; quantity as signed face; normalization in the engine's adapter, `mappingVersion` echoed |
| Accrued sign | `fraction × signed face` in one step (short → negative); no separate `sign()` factor |
| Accrued source | Labelled: `exported-fraction` (the value used) or `recomputed-schedule`; `structural-zero` distinct |
| Accrual tolerance | `round(0.5 × 10^(−fractionDecimals) × abs(face), 2) + 0.01`, derived from `accrualBasis`; the two monetary paths are never compared as exact equals |
| Sensitivities | `rateSensitivity` with `method` (`ad-first-order` or `bumped-revaluation`), numeric bump, factor, units |
| Gamma | Scaled second derivative, no ½, diagonal only |
| Coverage | Per calculation per item, five statuses (`ok`, `unsupported`, `unavailable`, `failed`, `not-applicable`); `allOutcomesAccountedFor` vs `allApplicableComputed` |
| Identity | Opaque `itemId` plus source identity (kind, account, security or contract) and `clusterEpoch`, on refused rows too |
| Cube output | An artifact reference with shape, dtype, ordering and hash plus a hashed item-order file; never inline JSON |
| Attempts | Lookup returns the most recent successful attempt; attempts are immutable; the result manifest is the commit point and lookup falls back to a scan when the pointer is behind |
| Lookup states | `UNKNOWN_WORKLOAD`, `running`, `failed`, `completed`: never a bare 404 for accepted work |
| Submission identity | Caller-generated idempotent `submissionId`; a retry recovers the same attempt. `reuseExistingResult: false` means "do not serve the cache", not "always start a new attempt" |
| `synthetic` vs `assumed` | Distinct: their `provenance.origin` is `synthetic` or `supplied`; the engine's curve `inputOrigin` is separate |
| `accrualBasis` shape | `traderx.accrual-basis.v1` field shape frozen; its value vocabulary is not ([I-23](../known-issues.md#i-23)) |
| Schemas | Engine-owned and served at `/eod/schemas/*`; a joint spec pack references them by URL and version rather than copying them |

<a id="open-with-traderx"></a>
## Open with TraderX

| Ask | First asked | Blocks |
|---|---|---|
| Do new `accrualBasis` values land in `v1` or force `v2`? | v4 §1.3; again v6, v7 | [I-23](../known-issues.md#i-23) |
| Additive-field policy for the result schema (proposed: exact shape, any new field is a version bump) | v7 §5.2 | [I-60](../known-issues.md#i-60) |
| The D03/D04 USD-SOFR convention set (the fixture's 13 missing terms) | Proposal §2.2 | [I-05](../known-issues.md#i-05), W2 |
| `pastFixings` for seasoned swaps | Proposal §2.2 | [I-04](../known-issues.md#i-04) (data half) |
| An equity spot and FX source, or observed spots in the bundle | W1.4 | [I-18](../known-issues.md#i-18) |
| Their compatibility work pushed with a commit SHA (`reference/traderX` is pinned at `a102e498`, which lacks their `.gitattributes` and checkout proof), so their tests can be re-run here | v4 | Independent re-verification |
| Their side of v7's sequence: `coordinator.py` calls `/eod`; retire their `fake_worker.py`; move `pricing_result.py` / `w0_result.py` under their tests as an independent reference; a consumer-side validator pinned to `jaxrisk.eod-result.v1` | v7 §6 | Nothing here; their duplicate pricer drifts until done |

## W2 — observed market data and USD-SOFR

Blocked on D03/D04 and a market-data package. Closes [I-05](../known-issues.md#i-05) and
[I-16](../known-issues.md#i-16).

- **Do not start on assumed conventions.** Guessing produces confident wrong numbers.
- **Required first:** both legs' schedules and dates, fixed/float day counts, calendars,
  business-day adjustment, payment lags, overnight compounding, lookback, lockout,
  observation shift, fixing calendar and fixing history.
- **A separate builder.** Do not change `build_vanilla_swap`: the calibration baskets and
  every swap-bearing trade's ORE parity depend on its conventions.
- **`marketInputs.mode: "package"`**: observed data with bootstrapped pillars, which is what
  lets `rateSensitivity` shift one pillar at a time and name it.
- **Acceptance** against a same-terms ORE reference, not this engine's own tests.

Out of scope until specified: corporate bonds (need a credit model;
[I-07](../known-issues.md#i-07)), listed options, TIPS and FRNs.

## Working rules

What the exchange established; they apply to all boundary work.

1. Never silently approximate: an explicit `unsupported` is recoverable, a plausible wrong
   number is not.
2. Key decisions on terms, not on blanks: a missing field and a structural zero look
   identical in CSV and mean opposite things.
3. A regression test must fail against the pre-fix code, and against the plausible wrong fix
   (a defaulted curve, a defaulted day count).
4. Distinguish "ORE can represent it" from "this engine prices it".
5. Fixed is not flagged: a warning improves honesty, not accuracy.
6. Read hashed bytes in binary; line-ending translation breaks every hash.
7. Identity never rides on array position.
8. Test the caller, not just the guard: every guard needs a test through the public entry
   point ([I-13](../known-issues.md#i-13) hid behind a passing unit test of its validator).
9. Test consequences, not parsing: [I-59](../known-issues.md#i-59)'s fields were tested to
   parse and never to take effect.

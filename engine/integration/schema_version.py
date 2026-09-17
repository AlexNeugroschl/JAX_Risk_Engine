"""
W1.6.2 -- the published document schema versions, as a dependency-free leaf.

**Why this is its own module.** `engine.integration.schema` *derives* the
JSON Schemas from `engine.integration.result`'s frozen vocabulary, so it
imports `result`. But `result` has to stamp its own version onto every
document it emits, so it needs the version constant -- which would import
`schema` right back. Putting the bare strings in a leaf that imports nothing
breaks that cycle at its narrowest point, rather than by deferring an import
inside a function (which hides the dependency) or by duplicating the
constants (which lets the published version drift from the schema that
describes it).

The same shape as `engine/day_count.py`, created for the same reason during
W1.3: a vocabulary that several layers need belongs below all of them.

**Nothing but constants belongs here.** The moment this module needs an
import, the cycle it exists to prevent is back.
"""

#: The published result document's schema version (W1.6.2).
#:
#: **Bump whenever a consumer's validator would need to change**: a removed
#: field, a renamed one, a narrowed type, or a new *required* field. Adding
#: an optional field does not require a bump -- a pinned consumer ignores it
#: -- but removing or repurposing one always does.
RESULT_SCHEMA_VERSION = "jaxrisk.eod-result.v1"

#: The capability document's schema version. Separately versioned from the
#: result schema because the two change for different reasons: a new pricer
#: changes what `capabilities()` advertises without changing the result
#: document's shape at all.
CAPABILITY_SCHEMA_VERSION = "jaxrisk.eod-capabilities.v1"

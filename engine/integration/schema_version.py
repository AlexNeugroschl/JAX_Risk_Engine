"""
Published document schema versions. A leaf with no imports: `schema` derives from `result`,
and `result` stamps these versions on its documents, so the constants live below both.
"""

#: Result document schema version. Bump when a pinned consumer's validator would need to
#: change (a field removed, renamed or narrowed, or a new required field); adding an
#: optional field does not require it.
RESULT_SCHEMA_VERSION = "jaxrisk.eod-result.v1"

#: Capability document schema version, versioned separately (a new pricer changes what is
#: advertised, not the result document's shape).
CAPABILITY_SCHEMA_VERSION = "jaxrisk.eod-capabilities.v1"

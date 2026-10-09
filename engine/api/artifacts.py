"""
Arrays returned by reference instead of inline (decision A-17, I-09): a result's
NPV cube or P&L matrix, split into chunks of bytes that the job queue stores beside the result
(`engine.api.job_queue`, `artifacts`) and `GET /jobs/{job_id}/artifacts/{name}/{chunk}` serves.

A reference says everything needed to rebuild and verify the array without reading the result
again (the contract `docs/planning/details/traderx-integration.md` settled for the EOD cube):

    {"name": "npv_cube", "dtype": "float64", "byte_order": "little",
     "shape": [S, D, N], "axes": ["scenario", "date", "trade"],
     "chunks": [{"url": "/jobs/<id>/artifacts/npv_cube/0", "rows": [0, 4096], "bytes": ..., "sha256": ...}, ...],
     "sha256": <of every chunk's bytes in order>,
     "items": {"axis": "trade", "ids": [...], "sha256": <of the canonical JSON of the ids>}}

The array is C-ordered (the last axis fastest) and split along its first axis, whole rows per
chunk, each chunk at most `CHUNK_BYTES`. The item order travels as its own hashed record, so a
consumer can check that the trade order it read is the order published, not inferred from array
position (as the EOD result's `itemOrder`, `engine.integration.identity.item_order_artifact`).

No JAX: the API process and the tests read references with `read_array` as any client would.
"""
import hashlib
import json
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

#: The largest chunk in bytes (whole rows of the first axis, so a row larger than this is one chunk).
CHUNK_BYTES = 8 * 1024 * 1024

#: How a request asks for a large array: in the result (`inline`, the default), by reference
#: (`artifact`), or not at all (`none`), as ORE writes its cube only when asked (A-17).
ARRAY_OUTPUTS = ("inline", "artifact", "none")


class ArtifactError(ValueError):
    """An artifact whose bytes do not match its reference."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def items_record(axis: str, ids: Sequence[str]) -> Dict:
    """The hashed order of an axis's items (`ids` in array order)."""
    ids = [str(i) for i in ids]
    canonical = json.dumps(ids, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return {"axis": axis, "ids": ids, "sha256": _sha256(canonical)}


def array_artifact(array, name: str, url: str, axes: Sequence[str], items: Dict,
                   chunk_bytes: int = CHUNK_BYTES) -> Tuple[Dict, List[bytes]]:
    """The reference to `array` and its chunks. `url` is where the chunks are served, each at
    `<url>/<index>`; `axes` names every axis; `items` is `items_record` of one axis."""
    values = np.asarray(array)
    values = np.ascontiguousarray(values, dtype=values.dtype.newbyteorder("<"))
    if values.ndim < 1 or len(axes) != values.ndim:
        raise ValueError(f"{name}: {len(axes)} axis names for an array of shape {values.shape}")
    if items["axis"] not in axes or len(items["ids"]) != values.shape[list(axes).index(items["axis"])]:
        raise ValueError(f"{name}: {len(items['ids'])} {items['axis']} ids for shape {values.shape}")
    row_bytes = values[:1].nbytes if values.shape[0] else 1
    rows = max(1, chunk_bytes // max(row_bytes, 1))
    chunks, described, whole = [], [], hashlib.sha256()
    for start in range(0, values.shape[0], rows):
        data = values[start:start + rows].tobytes()
        whole.update(data)
        described.append({"url": f"{url}/{len(chunks)}", "rows": [start, min(start + rows, values.shape[0])],
                          "bytes": len(data), "sha256": _sha256(data)})
        chunks.append(data)
    reference = {"name": name, "dtype": values.dtype.name, "byte_order": "little", "shape": list(values.shape),
                 "axes": list(axes), "chunks": described, "sha256": whole.hexdigest(), "items": items}
    return reference, chunks


def read_array(reference: Dict, fetch: Callable[[str], bytes]) -> np.ndarray:
    """The array a reference describes, its chunks read by `fetch(url)` and every hash checked
    (each chunk's, the whole array's, the item order's); `ArtifactError` on any mismatch."""
    whole, parts = hashlib.sha256(), []
    for chunk in reference["chunks"]:
        data = fetch(chunk["url"])
        if len(data) != chunk["bytes"] or _sha256(data) != chunk["sha256"]:
            raise ArtifactError(f"{reference['name']}: chunk {chunk['url']} does not match its sha256")
        whole.update(data)
        parts.append(data)
    if whole.hexdigest() != reference["sha256"]:
        raise ArtifactError(f"{reference['name']}: the chunks do not match the array's sha256")
    items = reference["items"]
    if items_record(items["axis"], items["ids"]) != items:
        raise ArtifactError(f"{reference['name']}: the {items['axis']} order does not match its sha256")
    dtype = np.dtype(reference["dtype"]).newbyteorder("<" if reference["byte_order"] == "little" else ">")
    return np.frombuffer(b"".join(parts), dtype=dtype).reshape(reference["shape"])

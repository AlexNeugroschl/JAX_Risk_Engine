"""
The array artifact format (`engine.api.artifacts`, decision A-17, I-09): an array split into
chunks of whole rows, each hashed, the whole array hashed, the trade order hashed beside it;
`read_array` rebuilds the array bit for bit and refuses any chunk, order or array that does not
match its reference.
"""
import json

import numpy as np
import pytest

from engine.api.artifacts import ArtifactError, array_artifact, items_record, read_array
from engine.api.schemas import ArrayArtifactSchema

IDS = ["swap", "bond", "bermudan"]


def _cube(scenarios=7, dates=4):
    return np.random.default_rng(1).normal(size=(scenarios, dates, len(IDS)))


def _artifact(array, chunk_bytes):
    return array_artifact(array, "npv_cube", "/jobs/j/artifacts/npv_cube", ("scenario", "date", "trade"),
                          items_record("trade", IDS), chunk_bytes=chunk_bytes)


@pytest.mark.parametrize("chunk_bytes", [1, 96, 200, 10 ** 9], ids=["row-per-chunk", "one-row", "two-rows", "one"])
def test_the_chunks_rebuild_the_array_bit_for_bit(chunk_bytes):
    cube = _cube()
    reference, chunks = _artifact(cube, chunk_bytes)
    by_url = {c["url"]: data for c, data in zip(reference["chunks"], chunks)}
    rebuilt = read_array(reference, by_url.__getitem__)
    assert rebuilt.dtype == np.float64 and rebuilt.shape == cube.shape
    np.testing.assert_array_equal(rebuilt, cube)
    rows = [c["rows"] for c in reference["chunks"]]
    assert rows[0][0] == 0 and rows[-1][1] == cube.shape[0]
    assert all(a[1] == b[0] for a, b in zip(rows, rows[1:]))
    assert all(len(data) <= max(chunk_bytes, cube[:1].nbytes) for data in chunks)
    assert [c["url"] for c in reference["chunks"]] == [f"/jobs/j/artifacts/npv_cube/{i}" for i in range(len(chunks))]
    ArrayArtifactSchema.model_validate(reference)


def test_a_float32_array_keeps_its_dtype_and_an_empty_trade_axis_is_one_empty_chunk():
    cube = _cube().astype(np.float32)
    reference, chunks = _artifact(cube, 64)
    assert reference["dtype"] == "float32"
    np.testing.assert_array_equal(read_array(reference, dict(zip([c["url"] for c in reference["chunks"]], chunks)).get),
                                  cube)
    empty, chunks = array_artifact(np.zeros((5, 2, 0)), "npv_cube", "/u", ("scenario", "date", "trade"),
                                   items_record("trade", []))
    assert chunks == [b""] and read_array(empty, lambda url: b"").shape == (5, 2, 0)


def test_the_item_order_is_hashed_on_its_canonical_json():
    record = items_record("trade", IDS)
    canonical = json.dumps(IDS, separators=(",", ":")).encode()
    import hashlib

    assert record == {"axis": "trade", "ids": IDS, "sha256": hashlib.sha256(canonical).hexdigest()}


def test_a_tampered_chunk_order_or_array_is_refused():
    reference, chunks = _artifact(_cube(), 200)
    by_url = {c["url"]: data for c, data in zip(reference["chunks"], chunks)}
    flipped = dict(by_url)
    first = reference["chunks"][0]["url"]
    flipped[first] = bytes([by_url[first][0] ^ 1]) + by_url[first][1:]
    with pytest.raises(ArtifactError, match="chunk .* does not match its sha256"):
        read_array(reference, flipped.__getitem__)
    with pytest.raises(ArtifactError, match="trade order"):
        read_array({**reference, "items": {**reference["items"], "ids": IDS[::-1]}}, by_url.__getitem__)
    swapped = {**reference, "chunks": reference["chunks"][::-1]}
    with pytest.raises(ArtifactError, match="the array's sha256"):
        read_array(swapped, by_url.__getitem__)


def test_axis_names_and_item_ids_must_fit_the_array():
    with pytest.raises(ValueError, match="axis names"):
        array_artifact(_cube(), "x", "/u", ("scenario", "trade"), items_record("trade", IDS))
    with pytest.raises(ValueError, match="trade ids"):
        array_artifact(_cube(), "x", "/u", ("scenario", "date", "trade"), items_record("trade", IDS[:2]))

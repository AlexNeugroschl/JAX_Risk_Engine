"""
The ORE parity matrix (I-116, tests/support/ore_parity_matrix.py): every public setting and
analytic has a cell compared with ORE by a test, or a stated reason; and the matrix published in
docs/reference/ore-parity.md is the one the tests declare.

The declarations are collected from the whole suite in a fresh process (pytest's collection with
the matrix's `Collector` plugin), so a parity test added, removed or changed changes the matrix,
and `python -m tests.support.ore_parity_matrix --write` publishes it again.
"""
import pytest

from tests.support import ore_parity_matrix as matrix


@pytest.fixture(scope="module")
def declared():
    return matrix.collect()


def test_writing_the_matrix_keeps_the_documents_line_endings(tmp_path, monkeypatch):
    """The published document is CRLF: a rewrite on Linux or macOS must not flatten all of it."""
    begin, end = matrix.BEGIN.encode(), matrix.END.encode()
    doc = tmp_path / "ore-parity.md"
    doc.write_bytes(b"\xef\xbb\xbf# Title\r\n\r\n" + begin + b"\r\nold\r\n" + end + b"\r\nafter\r\n")
    monkeypatch.setattr(matrix, "DOC", doc)
    monkeypatch.setattr(matrix, "render", lambda declared: f"{matrix.BEGIN}\nnew\n{matrix.END}")
    matrix.write([])
    assert doc.read_bytes() == b"\xef\xbb\xbf# Title\r\n\r\n" + begin + b"\r\nnew\r\n" + end + b"\r\nafter\r\n"


def test_every_field_of_the_public_requests_is_a_setting_an_input_or_has_a_reason():
    """A field added to a request, a trade or a configuration fails here until it is a setting
    of the matrix (`SETTINGS`), an input (`INPUTS`) or in a class with one reason."""
    assert matrix.unclassified_fields() == []


def test_every_http_route_names_its_analytics():
    from engine.api.app import app
    assert sorted(app.openapi()["paths"]) == sorted(matrix.ROUTES)
    for route, analytics in matrix.ROUTES.items():
        assert set(analytics.split()) <= set(matrix.ANALYTICS), route


def test_every_declared_cell_is_a_row_of_the_matrix(declared):
    assert declared, "no test declares an ORE parity cell"
    assert matrix.undeclarable(declared) == []


def test_every_cell_has_a_test_or_a_reason(declared):
    """Each product of each analytic, and each value of each setting for every analytic and
    product it applies to (`required_cells`)."""
    missing = matrix.uncovered(declared)
    assert missing == [], "\n".join(map(str, missing))


def test_every_reason_stands_for_a_cell_no_test_compares(declared):
    """A reason a test has overtaken is deleted, so the matrix never gives one beside a test."""
    covered = {cell for d in declared for cell in d.cells}
    used = {key for cell in matrix.required_cells() if cell not in covered
            for key in [matrix.reason_key(cell)] if key is not None}
    assert set(matrix.REASONS) - used == set()


def test_the_published_matrix_is_the_tests(declared):
    assert matrix.published() == matrix.render(declared), \
        "docs/reference/ore-parity.md is stale: run python -m tests.support.ore_parity_matrix --write"

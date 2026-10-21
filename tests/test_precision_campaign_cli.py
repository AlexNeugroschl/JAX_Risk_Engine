"""Campaign inputs must describe the independent experiments actually executed."""
import pytest

from tests.support import precision_campaigns as campaigns


@pytest.mark.parametrize("args", [["--seeds", "1"], ["--seeds", "0"], ["--seed-ids", "1", "1"],
    ["--seed-ids", "-1", "2"], ["--paths", "31"], ["--paths", "64", "64"], ["--cases", "nonexistent"]])
def test_invalid_experiments_fail_before_output(tmp_path, args):
    with pytest.raises(SystemExit) as exc:
        campaigns.main(["linear", "--out", str(tmp_path), *args])
    assert exc.value.code == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("arguments,expected", [(["--seeds", "64"], tuple(range(1, 65))),
                                               (["--seed-ids", "101", "203", "409"], (101, 203, 409))])
def test_requested_seed_identities_are_never_truncated(tmp_path, monkeypatch, arguments, expected):
    found = []
    monkeypatch.setattr(campaigns, "measure", lambda c, paths, **kw: found.append(c.seeds))
    monkeypatch.setattr(campaigns, "write", lambda *a: None)
    campaigns.main(["linear", "--paths", "64", "--out", str(tmp_path), *arguments])
    assert found == [expected]

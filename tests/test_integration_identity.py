"""
Identity (`engine.integration.identity`, W0.7; closes I-10 at the integration boundary):
one security in two accounts gives distinct rows with correct signs; an unsupported row has
identity; the item-order artifact hash matches the result ordering.
"""
import json
from pathlib import Path

import pytest

from engine.integration import price_bundle
from engine.integration.identity import (
    ITEM_ID_SCHEME,
    ItemIdentity,
    contract_identity,
    identity_for,
    item_id,
    item_order_artifact,
    position_identity,
)

FIXTURES = Path(__file__).parent / "fixtures" / "traderx-eod"


class TestSameSecurityInTwoAccounts:
    """Keyed by array position (I-10), the long and short rows were indistinguishable."""

    def test_distinct_rows_with_distinct_ids(self):
        result = price_bundle(FIXTURES / "note" / "v2")
        assert len(result.items) == 2

        ids = {item.item_id for item in result.items}
        assert len(ids) == 2, "same security, different accounts -> different items"

    def test_same_security_different_accounts(self):
        result = price_bundle(FIXTURES / "note" / "v2")
        identities = {item.identity.account_id: item.identity for item in result.items}

        assert set(identities) == {"22214", "42422"}
        assert {i.security for i in identities.values()} == {"UST-NOTE-20261215"}

    def test_correct_signs_per_account(self):
        """Identity and sign travel together."""
        result = price_bundle(FIXTURES / "note" / "v2")
        accrued = {
            item.identity.account_id: item.calculations["accruedInterest"].value
            for item in result.items
        }
        assert accrued["22214"] > 0
        assert accrued["42422"] < 0
        assert accrued["22214"] == pytest.approx(-accrued["42422"])


class TestItemIdProperties:
    def test_stable_across_calls(self):
        identity = ItemIdentity(kind="position", account_id="1", cluster_epoch="e", security="S")
        assert item_id(identity) == item_id(identity)

    def test_stable_across_processes(self):
        """A SHA-256 of the canonical identity, not Python's `hash()` (salted per process by
        PYTHONHASHSEED)."""
        import subprocess, sys
        code = (
            "from engine.integration.identity import ItemIdentity, item_id;"
            "print(item_id(ItemIdentity(kind='position', account_id='1',"
            " cluster_epoch='e', security='S')))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=str(Path(__file__).parents[1]),
        )
        assert out.returncode == 0, out.stderr
        local = item_id(ItemIdentity(kind="position", account_id="1", cluster_epoch="e", security="S"))
        assert out.stdout.strip() == local

    def test_differs_by_account(self):
        a = ItemIdentity(kind="position", account_id="1", cluster_epoch="e", security="S")
        b = ItemIdentity(kind="position", account_id="2", cluster_epoch="e", security="S")
        assert item_id(a) != item_id(b)

    def test_differs_by_epoch(self):
        """A `contractId` is unique only within its epoch, so the epoch is in the id
        preimage (else SW-3 in two epochs collides)."""
        a = ItemIdentity(kind="contract", account_id="1", cluster_epoch="e1", contract_id="SW-3")
        b = ItemIdentity(kind="contract", account_id="1", cluster_epoch="e2", contract_id="SW-3")
        assert item_id(a) != item_id(b)

    def test_is_opaque_and_fixed_width(self):
        identity = ItemIdentity(kind="position", account_id="1", cluster_epoch="e", security="S")
        value = item_id(identity)
        assert len(value) == 32
        assert all(c in "0123456789abcdef" for c in value)

    def test_exactly_one_of_security_or_contract_id(self):
        with pytest.raises(ValueError, match="exactly one"):
            ItemIdentity(kind="position", account_id="1", cluster_epoch="e")
        with pytest.raises(ValueError, match="exactly one"):
            ItemIdentity(kind="position", account_id="1", cluster_epoch="e",
                         security="S", contract_id="C")


class TestSourceIdentityBlock:
    def test_position_block_shape(self):
        block = position_identity(
            {"accountId": "22214", "security": "UST-NOTE-20261215"}, "epoch-1"
        ).to_dict()
        assert block == {
            "kind": "position", "accountId": "22214",
            "clusterEpoch": "epoch-1", "security": "UST-NOTE-20261215",
        }

    def test_contract_block_shape(self):
        block = contract_identity({"accountId": "22214", "contractId": "SW-3"}, "epoch-1").to_dict()
        assert block == {
            "kind": "contract", "accountId": "22214",
            "clusterEpoch": "epoch-1", "contractId": "SW-3",
        }

    def test_dispatch_by_source(self):
        assert identity_for("positions", {"accountId": "1", "security": "S"}, "e").kind == "position"
        assert identity_for("contracts", {"accountId": "1", "contractId": "C"}, "e").kind == "contract"
        with pytest.raises(ValueError, match="unknown row source"):
            identity_for("elsewhere", {}, "e")


class TestUnsupportedRowsCarryIdentity:
    """Refusals carry identity."""

    def test_sofr_refusal_is_fully_identified(self):
        payload = price_bundle(FIXTURES / "sofr" / "v2").to_dict()
        item = payload["items"][0]

        assert item["calculations"]["npv"]["status"] == "unsupported"
        assert item["itemId"]
        assert item["sourceIdentity"]["contractId"] == "SW-3"
        assert item["sourceIdentity"]["accountId"] == "22214"
        assert item["sourceIdentity"]["clusterEpoch"] == "synthetic-shared-examples-v1"

    @pytest.mark.parametrize("case", ("bill", "note", "sofr"))
    def test_v1_refusals_are_identified(self, case):
        """With no terms artifact nothing joins, yet every row is attributable."""
        payload = price_bundle(FIXTURES / case / "v1").to_dict()

        for item in payload["items"]:
            assert item["itemId"]
            assert item["sourceIdentity"]["accountId"]
            assert item["sourceIdentity"]["clusterEpoch"]

    def test_identity_is_structurally_mandatory(self):
        """An `ItemResult` cannot be built without identity."""
        from engine.integration.result import ItemResult
        with pytest.raises(TypeError):
            ItemResult(calculations={})


class TestItemOrderArtifact:
    """Ordering is published as its own hashed artifact, never inferred from array position."""

    def test_order_matches_the_items_array(self):
        payload = price_bundle(FIXTURES / "note" / "v2").to_dict()
        assert payload["itemOrder"]["itemIds"] == [i["itemId"] for i in payload["items"]]

    def test_artifact_carries_its_own_hash(self):
        payload = price_bundle(FIXTURES / "note" / "v2").to_dict()
        order = payload["itemOrder"]

        assert len(order["sha256"]) == 64
        assert order["itemCount"] == len(order["itemIds"])

    def test_hash_is_reproducible(self):
        payload = price_bundle(FIXTURES / "note" / "v2").to_dict()
        order = payload["itemOrder"]
        assert item_order_artifact(order["itemIds"])["sha256"] == order["sha256"]

    def test_reordering_changes_the_hash(self):
        """A consumer can verify the order it read is the one published."""
        ids = ["a" * 32, "b" * 32]
        assert item_order_artifact(ids)["sha256"] != item_order_artifact(list(reversed(ids)))["sha256"]

    def test_scheme_is_recorded(self):
        assert item_order_artifact([])["scheme"] == ITEM_ID_SCHEME


class TestIdentityIsNotArrayPosition:
    """I-10 regression: reconciliation works by `itemId` alone with the array shuffled."""

    def test_items_are_findable_by_id_not_index(self):
        result = price_bundle(FIXTURES / "note" / "v2")
        by_id = {item.item_id: item for item in result.items}

        for item in result.items:
            found = by_id[item.item_id]
            assert found.identity.account_id == item.identity.account_id

    def test_ids_are_reproducible_across_runs(self):
        """The same bundle yields the same ids every time."""
        first = [i.item_id for i in price_bundle(FIXTURES / "note" / "v2").items]
        second = [i.item_id for i in price_bundle(FIXTURES / "note" / "v2").items]
        assert first == second

    def test_v1_and_v2_agree_on_identity(self):
        """The bundle version changes what can be priced, never the rows' identities."""
        v1 = {i.item_id for i in price_bundle(FIXTURES / "note" / "v1").items}
        v2 = {i.item_id for i in price_bundle(FIXTURES / "note" / "v2").items}
        assert v1 == v2

"""
Row identity for the integration boundary (closes I-10 there): results are never keyed by
array position.

Every row, including a refused one, carries both an opaque `itemId` (a stable key the
coordinator need not parse) and the source identity block it was derived from
(`kind`, `accountId`, `security` or `contractId`, `clusterEpoch`), which can be reconciled
without a lookup table. Identity is built from the raw row, so it never depends on a
successful terms join or normalization.

`clusterEpoch` is part of every item's id. A `contractId` is unique only within its
epoch, so this keeps contracts from colliding across epochs; a position's id also changes
with the epoch.
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Optional

#: Bumping this changes every `itemId`, so a revised scheme cannot collide with the old.
ITEM_ID_SCHEME = "traderx-item-v1"


@dataclass(frozen=True)
class ItemIdentity:
    """One item's source identity. Exactly one of `security` (a position) and
    `contract_id` (an OTC contract) is set."""
    kind: str                      # "position" | "contract"
    account_id: str
    cluster_epoch: str
    security: Optional[str] = None
    contract_id: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.security is None) == (self.contract_id is None):
            raise ValueError(
                "ItemIdentity needs exactly one of security / contract_id; got "
                f"security={self.security!r}, contract_id={self.contract_id!r}"
            )

    @property
    def item_id(self) -> str:
        return item_id(self)

    def to_dict(self) -> Dict:
        """The source identity block echoed on every result row."""
        block = {
            "kind": self.kind,
            "accountId": self.account_id,
            "clusterEpoch": self.cluster_epoch,
        }
        if self.security is not None:
            block["security"] = self.security
        else:
            block["contractId"] = self.contract_id
        return block


def item_id(identity: ItemIdentity) -> str:
    """Stable, opaque id: the first 32 hex characters (128 bits) of the SHA-256 of the
    canonical identity JSON. Deterministic across runs and machines (no `hash()`), and the
    epoch is in the preimage."""
    preimage = json.dumps(
        {"scheme": ITEM_ID_SCHEME, **identity.to_dict()},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:32]


def position_identity(row: Dict[str, str], cluster_epoch: str) -> ItemIdentity:
    """Identity of a position row, from the raw row."""
    return ItemIdentity(
        kind="position",
        account_id=row["accountId"],
        cluster_epoch=cluster_epoch,
        security=row["security"],
    )


def contract_identity(row: Dict[str, str], cluster_epoch: str) -> ItemIdentity:
    """Identity for an OTC contract row."""
    return ItemIdentity(
        kind="contract",
        account_id=row["accountId"],
        cluster_epoch=cluster_epoch,
        contract_id=row["contractId"],
    )


def identity_for(source: str, row: Dict[str, str], cluster_epoch: str) -> ItemIdentity:
    """Dispatches on the row's source artifact."""
    if source == "positions":
        return position_identity(row, cluster_epoch)
    if source == "contracts":
        return contract_identity(row, cluster_epoch)
    raise ValueError(f"unknown row source {source!r}")


def item_order_artifact(item_ids) -> Dict:
    """The result's row order as its own hashed document, so a consumer can verify the
    order it read is the order published (array position carries no integrity)."""
    ids = list(item_ids)
    body = json.dumps(
        {"scheme": ITEM_ID_SCHEME, "itemIds": ids},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return {
        "scheme": ITEM_ID_SCHEME,
        "itemIds": ids,
        "itemCount": len(ids),
        "sha256": hashlib.sha256(body).hexdigest(),
    }

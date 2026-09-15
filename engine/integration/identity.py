"""
W0.7 -- identity plumbing. Closes **I-10** for the integration boundary.

**Identity never rides on array position** (plan working rule 7). I-10 is
exactly what happens when it does: results keyed by index, unattributable
the moment anything reorders or filters.

Two things travel together on every row:

  - an opaque **`itemId`**, stable for a given source identity, which the
    coordinator can use as a key without parsing it; and
  - the **source identity block** it was derived from -- `{kind, accountId,
    security | contractId}` plus `clusterEpoch` -- which a human can read
    and which lets the coordinator reconcile against its own records
    without a lookup table.

Both, not either. An opaque id alone is unreconcilable without a side
channel; a structured identity alone invites consumers to re-derive keys
with their own subtly different rules.

**Unsupported rows carry full identity too** (step 3). An unidentified
refusal is useless -- "something in this bundle could not be priced" is not
actionable. This is why `item_identity` takes a raw row and never depends on
a successful terms join or normalization.

**The epoch is part of contract identity, and not part of security
identity.** A `contractId` is unique only within its cluster epoch; a
security identifier is global. Making the id generation mirror that
asymmetry is what keeps a contract from silently colliding across epochs --
the same asymmetry `engine.integration.terms` joins on.
"""
import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Optional

#: Bumping this changes every `itemId`. It exists so the id scheme can be
#: revised without a new id silently colliding with an old one.
ITEM_ID_SCHEME = "traderx-item-v1"


@dataclass(frozen=True)
class ItemIdentity:
    """One item's source identity, as the coordinator would recognize it.

    `security` and `contract_id` are mutually exclusive: a row is either a
    position in a security or a booked OTC contract.
    """
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
        """The source identity block echoed on every result row, including
        refusals."""
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
    """A stable, opaque id for one item.

    Derived by hashing the canonical source identity, so it is:

      - **stable** -- the same identity always yields the same id, across
        runs, processes and machines (no randomness, no insertion order, no
        `hash()` which is PYTHONHASHSEED-salted);
      - **opaque** -- consumers key on it without parsing it, so the
        identity shape can gain fields without breaking them; and
      - **collision-resistant across epochs** -- `cluster_epoch` is inside
        the preimage, so the same `contractId` in two epochs gets two ids.

    Truncated to 32 hex characters: 128 bits, far past any collision concern
    at bundle scale, and short enough to read in a log line.
    """
    preimage = json.dumps(
        {"scheme": ITEM_ID_SCHEME, **identity.to_dict()},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:32]


def position_identity(row: Dict[str, str], cluster_epoch: str) -> ItemIdentity:
    """Identity for a position row. Takes the raw row, so an unjoined or
    unnormalizable row still gets identified (step 3)."""
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
    """The item-ordering artifact (step 4): the result's row order published
    as **its own hashed document**, never inferred from array position in
    the result payload.

    Separating it is what lets a consumer verify that the order it read is
    the order that was published -- an array's position carries no integrity
    guarantee of its own.
    """
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

from __future__ import annotations
import typing
from dataclasses import dataclass
from construct import Container
from solders.pubkey import Pubkey
from anchorpy.borsh_extension import BorshPubkey
import borsh_construct as borsh


class ObligationCollateralJSON(typing.TypedDict):
    deposit_reserve: str
    deposited_amount: int
    market_value: int
    padding: list[int]


@dataclass
class ObligationCollateral:
    layout: typing.ClassVar = borsh.CStruct(
        "deposit_reserve" / BorshPubkey,
        "deposited_amount" / borsh.U64,
        "market_value" / borsh.U128,
        "padding" / borsh.U64[4],
    )
    deposit_reserve: Pubkey
    deposited_amount: int
    market_value: int
    padding: list[int]

    @classmethod
    def from_decoded(cls, obj: Container) -> "ObligationCollateral":
        return cls(
            deposit_reserve=obj.deposit_reserve,
            deposited_amount=obj.deposited_amount,
            market_value=obj.market_value,
            padding=obj.padding,
        )

    def to_encodable(self) -> dict[str, typing.Any]:
        return {
            "deposit_reserve": self.deposit_reserve,
            "deposited_amount": self.deposited_amount,
            "market_value": self.market_value,
            "padding": self.padding,
        }

    def to_json(self) -> ObligationCollateralJSON:
        return {
            "deposit_reserve": str(self.deposit_reserve),
            "deposited_amount": self.deposited_amount,
            "market_value": self.market_value,
            "padding": self.padding,
        }

    @classmethod
    def from_json(cls, obj: ObligationCollateralJSON) -> "ObligationCollateral":
        return cls(
            deposit_reserve=Pubkey.from_string(obj["deposit_reserve"]),
            deposited_amount=obj["deposited_amount"],
            market_value=obj["market_value"],
            padding=obj["padding"],
        )

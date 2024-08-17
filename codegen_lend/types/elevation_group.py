from __future__ import annotations
import typing
from dataclasses import dataclass
from construct import Container
from solders.pubkey import Pubkey
from anchorpy.borsh_extension import BorshPubkey
import borsh_construct as borsh


class ElevationGroupJSON(typing.TypedDict):
    max_liquidation_bonus_bps: int
    id: int
    ltv_pct: int
    liquidation_threshold_pct: int
    allow_new_loans: int
    max_reserves_as_collateral: int
    padding0: int
    debt_reserve: str
    padding1: list[int]


@dataclass
class ElevationGroup:
    layout: typing.ClassVar = borsh.CStruct(
        "max_liquidation_bonus_bps" / borsh.U16,
        "id" / borsh.U8,
        "ltv_pct" / borsh.U8,
        "liquidation_threshold_pct" / borsh.U8,
        "allow_new_loans" / borsh.U8,
        "max_reserves_as_collateral" / borsh.U8,
        "padding0" / borsh.U8,
        "debt_reserve" / BorshPubkey,
        "padding1" / borsh.U64[4],
    )
    max_liquidation_bonus_bps: int
    id: int
    ltv_pct: int
    liquidation_threshold_pct: int
    allow_new_loans: int
    max_reserves_as_collateral: int
    padding0: int
    debt_reserve: Pubkey
    padding1: list[int]

    @classmethod
    def from_decoded(cls, obj: Container) -> "ElevationGroup":
        return cls(
            max_liquidation_bonus_bps=obj.max_liquidation_bonus_bps,
            id=obj.id,
            ltv_pct=obj.ltv_pct,
            liquidation_threshold_pct=obj.liquidation_threshold_pct,
            allow_new_loans=obj.allow_new_loans,
            max_reserves_as_collateral=obj.max_reserves_as_collateral,
            padding0=obj.padding0,
            debt_reserve=obj.debt_reserve,
            padding1=obj.padding1,
        )

    def to_encodable(self) -> dict[str, typing.Any]:
        return {
            "max_liquidation_bonus_bps": self.max_liquidation_bonus_bps,
            "id": self.id,
            "ltv_pct": self.ltv_pct,
            "liquidation_threshold_pct": self.liquidation_threshold_pct,
            "allow_new_loans": self.allow_new_loans,
            "max_reserves_as_collateral": self.max_reserves_as_collateral,
            "padding0": self.padding0,
            "debt_reserve": self.debt_reserve,
            "padding1": self.padding1,
        }

    def to_json(self) -> ElevationGroupJSON:
        return {
            "max_liquidation_bonus_bps": self.max_liquidation_bonus_bps,
            "id": self.id,
            "ltv_pct": self.ltv_pct,
            "liquidation_threshold_pct": self.liquidation_threshold_pct,
            "allow_new_loans": self.allow_new_loans,
            "max_reserves_as_collateral": self.max_reserves_as_collateral,
            "padding0": self.padding0,
            "debt_reserve": str(self.debt_reserve),
            "padding1": self.padding1,
        }

    @classmethod
    def from_json(cls, obj: ElevationGroupJSON) -> "ElevationGroup":
        return cls(
            max_liquidation_bonus_bps=obj["max_liquidation_bonus_bps"],
            id=obj["id"],
            ltv_pct=obj["ltv_pct"],
            liquidation_threshold_pct=obj["liquidation_threshold_pct"],
            allow_new_loans=obj["allow_new_loans"],
            max_reserves_as_collateral=obj["max_reserves_as_collateral"],
            padding0=obj["padding0"],
            debt_reserve=Pubkey.from_string(obj["debt_reserve"]),
            padding1=obj["padding1"],
        )

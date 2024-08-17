from __future__ import annotations
import typing
from dataclasses import dataclass
from construct import Container
from anchorpy.borsh_extension import BorshPubkey
import borsh_construct as borsh


# Adapting to release 1.6.2

"""
max_reserves_as_collateral: u8,
padding_0: u8
debt_reserve: Pubkey
padding_1: u64[4]
"""


class ElevationGroupJSON(typing.TypedDict):
    max_liquidation_bonus_bps: int
    id: int
    ltv_pct: int
    liquidation_threshold_pct: int
    allow_new_loans: int
    max_reserves_as_collateral: int
    padding_0: int
    padding_1: list[int]


@dataclass
class ElevationGroup:
    layout: typing.ClassVar = borsh.CStruct(
        "max_liquidation_bonus_bps" / borsh.U16,
        "id" / borsh.U8,
        "ltv_pct" / borsh.U8,
        "liquidation_threshold_pct" / borsh.U8,
        "allow_new_loans" / borsh.U8,
        "max_reserves_as_collateral" / borsh.U8,
        "padding_0" / borsh.U8,
        "debt_reserve" / BorshPubkey,
        "padding_1" / borsh.U64[4],
    )
    max_liquidation_bonus_bps: int
    id: int
    ltv_pct: int
    liquidation_threshold_pct: int
    allow_new_loans: int
    max_reserves_as_collateral: int
    padding_0: int
    debt_reserve: str
    padding_1: list[int]

    @classmethod
    def from_decoded(cls, obj: Container) -> "ElevationGroup":
        return cls(
            max_liquidation_bonus_bps=obj.max_liquidation_bonus_bps,
            id=obj.id,
            ltv_pct=obj.ltv_pct,
            liquidation_threshold_pct=obj.liquidation_threshold_pct,
            allow_new_loans=obj.allow_new_loans,
            max_reserves_as_collateral=obj.max_reserves_as_collateral,
            padding_0=obj.padding_0,
            debt_reserve=obj.debt_reserve,
            padding_1=obj.padding_1,
        )

    def to_encodable(self) -> dict[str, typing.Any]:
        return {
            "max_liquidation_bonus_bps": self.max_liquidation_bonus_bps,
            "id": self.id,
            "ltv_pct": self.ltv_pct,
            "liquidation_threshold_pct": self.liquidation_threshold_pct,
            "allow_new_loans": self.allow_new_loans,
            "max_reserves_as_collateral": self.max_reserves_as_collateral,
            "padding_0": self.padding_0,
            "debt_reserve": self.debt_reserve,
            "padding_1": self.padding_1,
        }

    def to_json(self) -> ElevationGroupJSON:
        return {
            "max_liquidation_bonus_bps": self.max_liquidation_bonus_bps,
            "id": self.id,
            "ltv_pct": self.ltv_pct,
            "liquidation_threshold_pct": self.liquidation_threshold_pct,
            "allow_new_loans": self.allow_new_loans,
            "max_reserves_as_collateral": self.max_reserves_as_collateral,
            "padding_0": self.padding_0,
            "debt_reserve": self.debt_reserve,
            "padding_1": self.padding_1,
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
            padding_0=obj["padding_0"],
            debt_reserve=obj["debt_reserve"],
            padding_1=obj["padding_1"],
        )

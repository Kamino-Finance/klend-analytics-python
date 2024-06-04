from __future__ import annotations
import typing
from dataclasses import dataclass
from construct import Container
import borsh_construct as borsh


class ElevationGroupJSON(typing.TypedDict):
    max_liquidation_bonus_bps: int
    id: int
    ltv_ratio_pct: int
    liquidation_threshold_pct: int
    allow_new_loans: int
    padding: list[int]


@dataclass
class ElevationGroup:
    layout: typing.ClassVar = borsh.CStruct(
        "max_liquidation_bonus_bps" / borsh.U16,
        "id" / borsh.U8,
        "ltv_ratio_pct" / borsh.U8,
        "liquidation_threshold_pct" / borsh.U8,
        "allow_new_loans" / borsh.U8,
        "padding" / borsh.U8[2],
    )
    max_liquidation_bonus_bps: int
    id: int
    ltv_ratio_pct: int
    liquidation_threshold_pct: int
    allow_new_loans: int
    padding: list[int]

    @classmethod
    def from_decoded(cls, obj: Container) -> "ElevationGroup":
        return cls(
            max_liquidation_bonus_bps=obj.max_liquidation_bonus_bps,
            id=obj.id,
            ltv_ratio_pct=obj.ltv_ratio_pct,
            liquidation_threshold_pct=obj.liquidation_threshold_pct,
            allow_new_loans=obj.allow_new_loans,
            padding=obj.padding,
        )

    def to_encodable(self) -> dict[str, typing.Any]:
        return {
            "max_liquidation_bonus_bps": self.max_liquidation_bonus_bps,
            "id": self.id,
            "ltv_ratio_pct": self.ltv_ratio_pct,
            "liquidation_threshold_pct": self.liquidation_threshold_pct,
            "allow_new_loans": self.allow_new_loans,
            "padding": self.padding,
        }

    def to_json(self) -> ElevationGroupJSON:
        return {
            "max_liquidation_bonus_bps": self.max_liquidation_bonus_bps,
            "id": self.id,
            "ltv_ratio_pct": self.ltv_ratio_pct,
            "liquidation_threshold_pct": self.liquidation_threshold_pct,
            "allow_new_loans": self.allow_new_loans,
            "padding": self.padding,
        }

    @classmethod
    def from_json(cls, obj: ElevationGroupJSON) -> "ElevationGroup":
        return cls(
            max_liquidation_bonus_bps=obj["max_liquidation_bonus_bps"],
            id=obj["id"],
            ltv_ratio_pct=obj["ltv_ratio_pct"],
            liquidation_threshold_pct=obj["liquidation_threshold_pct"],
            allow_new_loans=obj["allow_new_loans"],
            padding=obj["padding"],
        )

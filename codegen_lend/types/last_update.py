from __future__ import annotations
import typing
from dataclasses import dataclass
from construct import Container
import borsh_construct as borsh


class LastUpdateJSON(typing.TypedDict):
    slot: int
    stale: int
    price_status: int
    placeholder: list[int]


@dataclass
class LastUpdate:
    layout: typing.ClassVar = borsh.CStruct(
        "slot" / borsh.U64,
        "stale" / borsh.U8,
        "price_status" / borsh.U8,
        "placeholder" / borsh.U8[6],
    )
    slot: int
    stale: int
    price_status: int
    placeholder: list[int]

    @classmethod
    def from_decoded(cls, obj: Container) -> "LastUpdate":
        return cls(
            slot=obj.slot,
            stale=obj.stale,
            price_status=obj.price_status,
            placeholder=obj.placeholder,
        )

    def to_encodable(self) -> dict[str, typing.Any]:
        return {
            "slot": self.slot,
            "stale": self.stale,
            "price_status": self.price_status,
            "placeholder": self.placeholder,
        }

    def to_json(self) -> LastUpdateJSON:
        return {
            "slot": self.slot,
            "stale": self.stale,
            "price_status": self.price_status,
            "placeholder": self.placeholder,
        }

    @classmethod
    def from_json(cls, obj: LastUpdateJSON) -> "LastUpdate":
        return cls(
            slot=obj["slot"],
            stale=obj["stale"],
            price_status=obj["price_status"],
            placeholder=obj["placeholder"],
        )

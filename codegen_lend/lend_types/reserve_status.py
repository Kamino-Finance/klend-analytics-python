from __future__ import annotations
import typing
from dataclasses import dataclass
from anchorpy.borsh_extension import EnumForCodegen
import borsh_construct as borsh


class ActiveJSON(typing.TypedDict):
    kind: typing.Literal["Active"]


class ObsoleteJSON(typing.TypedDict):
    kind: typing.Literal["Obsolete"]


@dataclass
class Active:
    discriminator: typing.ClassVar = 0
    kind: typing.ClassVar = "Active"

    @classmethod
    def to_json(cls) -> ActiveJSON:
        return ActiveJSON(
            kind="Active",
        )

    @classmethod
    def to_encodable(cls) -> dict:
        return {
            "Active": {},
        }


@dataclass
class Obsolete:
    discriminator: typing.ClassVar = 1
    kind: typing.ClassVar = "Obsolete"

    @classmethod
    def to_json(cls) -> ObsoleteJSON:
        return ObsoleteJSON(
            kind="Obsolete",
        )

    @classmethod
    def to_encodable(cls) -> dict:
        return {
            "Obsolete": {},
        }


ReserveStatusKind = typing.Union[Active, Obsolete]
ReserveStatusJSON = typing.Union[ActiveJSON, ObsoleteJSON]


def from_decoded(obj: dict) -> ReserveStatusKind:
    if not isinstance(obj, dict):
        raise ValueError("Invalid enum object")
    if "Active" in obj:
        return Active()
    if "Obsolete" in obj:
        return Obsolete()
    raise ValueError("Invalid enum object")


def from_json(obj: ReserveStatusJSON) -> ReserveStatusKind:
    if obj["kind"] == "Active":
        return Active()
    if obj["kind"] == "Obsolete":
        return Obsolete()
    kind = obj["kind"]
    raise ValueError(f"Unrecognized enum kind: {kind}")


layout = EnumForCodegen("Active" / borsh.CStruct(), "Obsolete" / borsh.CStruct())

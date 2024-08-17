from __future__ import annotations
import typing
from dataclasses import dataclass
from anchorpy.borsh_extension import EnumForCodegen
import borsh_construct as borsh


# Adapted for release 1.6.2
# Adding Hidden


class ActiveJSON(typing.TypedDict):
    kind: typing.Literal["Active"]


class ObsoleteJSON(typing.TypedDict):
    kind: typing.Literal["Obsolete"]


class HiddenJSON(typing.TypedDict):
    kind: typing.Literal["Hidden"]


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


@dataclass
class Hidden:
    discriminator: typing.ClassVar = 2
    kind: typing.ClassVar = "Hidden"

    @classmethod
    def to_json(cls) -> ObsoleteJSON:
        return ObsoleteJSON(
            kind="Hidden",
        )

    @classmethod
    def to_encodable(cls) -> dict:
        return {
            "Hidden": {},
        }


ReserveStatusKind = typing.Union[Active, Obsolete, Hidden]
ReserveStatusJSON = typing.Union[ActiveJSON, ObsoleteJSON, HiddenJSON]


def from_decoded(obj: dict) -> ReserveStatusKind:
    if not isinstance(obj, dict):
        raise ValueError("Invalid enum object")
    if "Active" in obj:
        return Active()
    if "Obsolete" in obj:
        return Obsolete()
    if "Hidden" in obj:
        return Hidden()
    raise ValueError("Invalid enum object")


def from_json(obj: ReserveStatusJSON) -> ReserveStatusKind:
    if obj["kind"] == "Active":
        return Active()
    if obj["kind"] == "Obsolete":
        return Obsolete()
    if obk["kind"] == "Hidden":
        return Hidden()
    raise ValueError(f"Unrecognized enum kind: {obj["kind"]}")


layout = EnumForCodegen(
    "Active" / borsh.CStruct(),
    "Obsolete" / borsh.CStruct(),
    "Hidden" / borsh.CStruct(),
)

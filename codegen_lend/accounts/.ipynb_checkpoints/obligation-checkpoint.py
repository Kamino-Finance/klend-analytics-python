import typing
from dataclasses import dataclass
from solders.pubkey import Pubkey
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Commitment
import borsh_construct as borsh
from anchorpy.coder.accounts import ACCOUNT_DISCRIMINATOR_SIZE
from anchorpy.error import AccountInvalidDiscriminator
from anchorpy.utils.rpc import get_multiple_accounts
from anchorpy.borsh_extension import BorshPubkey
from codegen_lend.program_id import PROGRAM_ID
import codegen_lend.lend_types as types


class ObligationJSON(typing.TypedDict):
    tag: int
    last_update: types.last_update.LastUpdateJSON
    lending_market: str
    owner: str
    deposits: list[types.obligation_collateral.ObligationCollateralJSON]
    borrows: list[types.obligation_liquidity.ObligationLiquidityJSON]
    deposited_value: int
    borrow_factor_adjusted_debt_value: int
    allowed_borrow_value: int
    unhealthy_borrow_value: int
    lowest_reserve_deposit_ltv: int
    borrowed_assets_market_value: int
    deposits_asset_tiers: list[int]
    borrows_asset_tiers: list[int]
    elevation_group: int
    padding0: list[int]
    referrer: str
    padding1: list[int]


@dataclass
class Obligation:
    discriminator: typing.ClassVar = b"\xa8\xce\x8djXL\xac\xa7"
    layout: typing.ClassVar = borsh.CStruct(
        "tag" / borsh.U64,
        "last_update" / types.last_update.LastUpdate.layout,
        "lending_market" / BorshPubkey,
        "owner" / BorshPubkey,
        "deposits" / types.obligation_collateral.ObligationCollateral.layout[10],
        "borrows" / types.obligation_liquidity.ObligationLiquidity.layout[10],
        "deposited_value" / borsh.U128,
        "borrow_factor_adjusted_debt_value" / borsh.U128,
        "allowed_borrow_value" / borsh.U128,
        "unhealthy_borrow_value" / borsh.U128,
        "lowest_reserve_deposit_ltv" / borsh.U64,
        "borrowed_assets_market_value" / borsh.U128,
        "deposits_asset_tiers" / borsh.U8[10],
        "borrows_asset_tiers" / borsh.U8[10],
        "elevation_group" / borsh.U8,
        "padding0" / borsh.U8[3],
        "referrer" / BorshPubkey,
        "padding1" / borsh.U64[182],
    )
    tag: int
    last_update: types.last_update.LastUpdate
    lending_market: Pubkey
    owner: Pubkey
    deposits: list[types.obligation_collateral.ObligationCollateral]
    borrows: list[types.obligation_liquidity.ObligationLiquidity]
    deposited_value: int
    borrow_factor_adjusted_debt_value: int
    allowed_borrow_value: int
    unhealthy_borrow_value: int
    lowest_reserve_deposit_ltv: int
    borrowed_assets_market_value: int
    deposits_asset_tiers: list[int]
    borrows_asset_tiers: list[int]
    elevation_group: int
    padding0: list[int]
    referrer: Pubkey
    padding1: list[int]

    @classmethod
    async def fetch(
        cls,
        conn: AsyncClient,
        address: Pubkey,
        commitment: typing.Optional[Commitment] = None,
        program_id: Pubkey = PROGRAM_ID,
    ) -> typing.Optional["Obligation"]:
        resp = await conn.get_account_info(address, commitment=commitment)
        info = resp.value
        if info is None:
            return None
        if info.owner != program_id:
            raise ValueError("Account does not belong to this program")
        bytes_data = info.data
        return cls.decode(bytes_data)

    @classmethod
    async def fetch_multiple(
        cls,
        conn: AsyncClient,
        addresses: list[Pubkey],
        commitment: typing.Optional[Commitment] = None,
        program_id: Pubkey = PROGRAM_ID,
    ) -> typing.List[typing.Optional["Obligation"]]:
        infos = await get_multiple_accounts(conn, addresses, commitment=commitment)
        res: typing.List[typing.Optional["Obligation"]] = []
        for info in infos:
            if info is None:
                res.append(None)
                continue
            if info.account.owner != program_id:
                raise ValueError("Account does not belong to this program")
            res.append(cls.decode(info.account.data))
        return res

    @classmethod
    def decode(cls, data: bytes) -> "Obligation":
        if data[:ACCOUNT_DISCRIMINATOR_SIZE] != cls.discriminator:
            raise AccountInvalidDiscriminator(
                "The discriminator for this account is invalid"
            )
        dec = Obligation.layout.parse(data[ACCOUNT_DISCRIMINATOR_SIZE:])
        return cls(
            tag=dec.tag,
            last_update=types.last_update.LastUpdate.from_decoded(dec.last_update),
            lending_market=dec.lending_market,
            owner=dec.owner,
            deposits=list(
                map(
                    lambda item: types.obligation_collateral.ObligationCollateral.from_decoded(
                        item
                    ),
                    dec.deposits,
                )
            ),
            borrows=list(
                map(
                    lambda item: types.obligation_liquidity.ObligationLiquidity.from_decoded(
                        item
                    ),
                    dec.borrows,
                )
            ),
            deposited_value=dec.deposited_value,
            borrow_factor_adjusted_debt_value=dec.borrow_factor_adjusted_debt_value,
            allowed_borrow_value=dec.allowed_borrow_value,
            unhealthy_borrow_value=dec.unhealthy_borrow_value,
            lowest_reserve_deposit_ltv=dec.lowest_reserve_deposit_ltv,
            borrowed_assets_market_value=dec.borrowed_assets_market_value,
            deposits_asset_tiers=dec.deposits_asset_tiers,
            borrows_asset_tiers=dec.borrows_asset_tiers,
            elevation_group=dec.elevation_group,
            padding0=dec.padding0,
            referrer=dec.referrer,
            padding1=dec.padding1,
        )

    def to_json(self) -> ObligationJSON:
        return {
            "tag": self.tag,
            "last_update": self.last_update.to_json(),
            "lending_market": str(self.lending_market),
            "owner": str(self.owner),
            "deposits": list(map(lambda item: item.to_json(), self.deposits)),
            "borrows": list(map(lambda item: item.to_json(), self.borrows)),
            "deposited_value": self.deposited_value,
            "borrow_factor_adjusted_debt_value": self.borrow_factor_adjusted_debt_value,
            "allowed_borrow_value": self.allowed_borrow_value,
            "unhealthy_borrow_value": self.unhealthy_borrow_value,
            "lowest_reserve_deposit_ltv": self.lowest_reserve_deposit_ltv,
            "borrowed_assets_market_value": self.borrowed_assets_market_value,
            "deposits_asset_tiers": self.deposits_asset_tiers,
            "borrows_asset_tiers": self.borrows_asset_tiers,
            "elevation_group": self.elevation_group,
            "padding0": self.padding0,
            "referrer": str(self.referrer),
            "padding1": self.padding1,
        }

    @classmethod
    def from_json(cls, obj: ObligationJSON) -> "Obligation":
        return cls(
            tag=obj["tag"],
            last_update=types.last_update.LastUpdate.from_json(obj["last_update"]),
            lending_market=Pubkey.from_string(obj["lending_market"]),
            owner=Pubkey.from_string(obj["owner"]),
            deposits=list(
                map(
                    lambda item: types.obligation_collateral.ObligationCollateral.from_json(
                        item
                    ),
                    obj["deposits"],
                )
            ),
            borrows=list(
                map(
                    lambda item: types.obligation_liquidity.ObligationLiquidity.from_json(
                        item
                    ),
                    obj["borrows"],
                )
            ),
            deposited_value=obj["deposited_value"],
            borrow_factor_adjusted_debt_value=obj["borrow_factor_adjusted_debt_value"],
            allowed_borrow_value=obj["allowed_borrow_value"],
            unhealthy_borrow_value=obj["unhealthy_borrow_value"],
            lowest_reserve_deposit_ltv=obj["lowest_reserve_deposit_ltv"],
            borrowed_assets_market_value=obj["borrowed_assets_market_value"],
            deposits_asset_tiers=obj["deposits_asset_tiers"],
            borrows_asset_tiers=obj["borrows_asset_tiers"],
            elevation_group=obj["elevation_group"],
            padding0=obj["padding0"],
            referrer=Pubkey.from_string(obj["referrer"]),
            padding1=obj["padding1"],
        )

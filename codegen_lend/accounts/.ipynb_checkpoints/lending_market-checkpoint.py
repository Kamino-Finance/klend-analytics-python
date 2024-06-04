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


class LendingMarketJSON(typing.TypedDict):
    version: int
    bump_seed: int
    lending_market_owner: str
    quote_currency: list[int]
    lending_market_owner_cached: str
    emergency_mode: int
    liquidation_max_debt_close_factor_pct: int
    insolvency_risk_unhealthy_ltv_pct: int
    reserved0: int
    reserved1: list[int]
    max_liquidatable_debt_market_value_at_once: int
    global_unhealthy_borrow_value: int
    global_allowed_borrow_value: int
    min_full_liquidation_amount_threshold: int
    risk_council: str
    elevation_groups: list[types.elevation_group.ElevationGroupJSON]
    reserved2: list[int]
    multiplier_points_tag_boost: list[int]
    referral_fee_bps: int
    price_refresh_trigger_to_max_age_pct: int
    padding0: list[int]
    padding1: list[int]


@dataclass
class LendingMarket:
    discriminator: typing.ClassVar = b"\xf6r2bH\x9d\x1cx"
    layout: typing.ClassVar = borsh.CStruct(
        "version" / borsh.U64,
        "bump_seed" / borsh.U64,
        "lending_market_owner" / BorshPubkey,
        "quote_currency" / borsh.U8[32],
        "lending_market_owner_cached" / BorshPubkey,
        "emergency_mode" / borsh.U8,
        "liquidation_max_debt_close_factor_pct" / borsh.U8,
        "insolvency_risk_unhealthy_ltv_pct" / borsh.U8,
        "reserved0" / borsh.U8,
        "reserved1" / borsh.U8[4],
        "max_liquidatable_debt_market_value_at_once" / borsh.U64,
        "global_unhealthy_borrow_value" / borsh.U64,
        "global_allowed_borrow_value" / borsh.U64,
        "min_full_liquidation_amount_threshold" / borsh.U64,
        "risk_council" / BorshPubkey,
        "elevation_groups" / types.elevation_group.ElevationGroup.layout[10],
        "reserved2" / borsh.U64[2],
        "multiplier_points_tag_boost" / borsh.U8[8],
        "referral_fee_bps" / borsh.U16,
        "price_refresh_trigger_to_max_age_pct" / borsh.U8,
        "padding0" / borsh.U8[5],
        "padding1" / borsh.U64[45],
    )
    version: int
    bump_seed: int
    lending_market_owner: Pubkey
    quote_currency: list[int]
    lending_market_owner_cached: Pubkey
    emergency_mode: int
    liquidation_max_debt_close_factor_pct: int
    insolvency_risk_unhealthy_ltv_pct: int
    reserved0: int
    reserved1: list[int]
    max_liquidatable_debt_market_value_at_once: int
    global_unhealthy_borrow_value: int
    global_allowed_borrow_value: int
    min_full_liquidation_amount_threshold: int
    risk_council: Pubkey
    elevation_groups: list[types.elevation_group.ElevationGroup]
    reserved2: list[int]
    multiplier_points_tag_boost: list[int]
    referral_fee_bps: int
    price_refresh_trigger_to_max_age_pct: int
    padding0: list[int]
    padding1: list[int]

    @classmethod
    async def fetch(
        cls,
        conn: AsyncClient,
        address: Pubkey,
        commitment: typing.Optional[Commitment] = None,
        program_id: Pubkey = PROGRAM_ID,
    ) -> typing.Optional["LendingMarket"]:
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
    ) -> typing.List[typing.Optional["LendingMarket"]]:
        infos = await get_multiple_accounts(conn, addresses, commitment=commitment)
        res: typing.List[typing.Optional["LendingMarket"]] = []
        for info in infos:
            if info is None:
                res.append(None)
                continue
            if info.account.owner != program_id:
                raise ValueError("Account does not belong to this program")
            res.append(cls.decode(info.account.data))
        return res

    @classmethod
    def decode(cls, data: bytes) -> "LendingMarket":
        if data[:ACCOUNT_DISCRIMINATOR_SIZE] != cls.discriminator:
            raise AccountInvalidDiscriminator(
                "The discriminator for this account is invalid"
            )
        dec = LendingMarket.layout.parse(data[ACCOUNT_DISCRIMINATOR_SIZE:])
        return cls(
            version=dec.version,
            bump_seed=dec.bump_seed,
            lending_market_owner=dec.lending_market_owner,
            quote_currency=dec.quote_currency,
            lending_market_owner_cached=dec.lending_market_owner_cached,
            emergency_mode=dec.emergency_mode,
            liquidation_max_debt_close_factor_pct=dec.liquidation_max_debt_close_factor_pct,
            insolvency_risk_unhealthy_ltv_pct=dec.insolvency_risk_unhealthy_ltv_pct,
            reserved0=dec.reserved0,
            reserved1=dec.reserved1,
            max_liquidatable_debt_market_value_at_once=dec.max_liquidatable_debt_market_value_at_once,
            global_unhealthy_borrow_value=dec.global_unhealthy_borrow_value,
            global_allowed_borrow_value=dec.global_allowed_borrow_value,
            min_full_liquidation_amount_threshold=dec.min_full_liquidation_amount_threshold,
            risk_council=dec.risk_council,
            elevation_groups=list(
                map(
                    lambda item: types.elevation_group.ElevationGroup.from_decoded(
                        item
                    ),
                    dec.elevation_groups,
                )
            ),
            reserved2=dec.reserved2,
            multiplier_points_tag_boost=dec.multiplier_points_tag_boost,
            referral_fee_bps=dec.referral_fee_bps,
            price_refresh_trigger_to_max_age_pct=dec.price_refresh_trigger_to_max_age_pct,
            padding0=dec.padding0,
            padding1=dec.padding1,
        )

    def to_json(self) -> LendingMarketJSON:
        return {
            "version": self.version,
            "bump_seed": self.bump_seed,
            "lending_market_owner": str(self.lending_market_owner),
            "quote_currency": self.quote_currency,
            "lending_market_owner_cached": str(self.lending_market_owner_cached),
            "emergency_mode": self.emergency_mode,
            "liquidation_max_debt_close_factor_pct": self.liquidation_max_debt_close_factor_pct,
            "insolvency_risk_unhealthy_ltv_pct": self.insolvency_risk_unhealthy_ltv_pct,
            "reserved0": self.reserved0,
            "reserved1": self.reserved1,
            "max_liquidatable_debt_market_value_at_once": self.max_liquidatable_debt_market_value_at_once,
            "global_unhealthy_borrow_value": self.global_unhealthy_borrow_value,
            "global_allowed_borrow_value": self.global_allowed_borrow_value,
            "min_full_liquidation_amount_threshold": self.min_full_liquidation_amount_threshold,
            "risk_council": str(self.risk_council),
            "elevation_groups": list(
                map(lambda item: item.to_json(), self.elevation_groups)
            ),
            "reserved2": self.reserved2,
            "multiplier_points_tag_boost": self.multiplier_points_tag_boost,
            "referral_fee_bps": self.referral_fee_bps,
            "price_refresh_trigger_to_max_age_pct": self.price_refresh_trigger_to_max_age_pct,
            "padding0": self.padding0,
            "padding1": self.padding1,
        }

    @classmethod
    def from_json(cls, obj: LendingMarketJSON) -> "LendingMarket":
        return cls(
            version=obj["version"],
            bump_seed=obj["bump_seed"],
            lending_market_owner=Pubkey.from_string(obj["lending_market_owner"]),
            quote_currency=obj["quote_currency"],
            lending_market_owner_cached=Pubkey.from_string(
                obj["lending_market_owner_cached"]
            ),
            emergency_mode=obj["emergency_mode"],
            liquidation_max_debt_close_factor_pct=obj[
                "liquidation_max_debt_close_factor_pct"
            ],
            insolvency_risk_unhealthy_ltv_pct=obj["insolvency_risk_unhealthy_ltv_pct"],
            reserved0=obj["reserved0"],
            reserved1=obj["reserved1"],
            max_liquidatable_debt_market_value_at_once=obj[
                "max_liquidatable_debt_market_value_at_once"
            ],
            global_unhealthy_borrow_value=obj["global_unhealthy_borrow_value"],
            global_allowed_borrow_value=obj["global_allowed_borrow_value"],
            min_full_liquidation_amount_threshold=obj[
                "min_full_liquidation_amount_threshold"
            ],
            risk_council=Pubkey.from_string(obj["risk_council"]),
            elevation_groups=list(
                map(
                    lambda item: types.elevation_group.ElevationGroup.from_json(item),
                    obj["elevation_groups"],
                )
            ),
            reserved2=obj["reserved2"],
            multiplier_points_tag_boost=obj["multiplier_points_tag_boost"],
            referral_fee_bps=obj["referral_fee_bps"],
            price_refresh_trigger_to_max_age_pct=obj[
                "price_refresh_trigger_to_max_age_pct"
            ],
            padding0=obj["padding0"],
            padding1=obj["padding1"],
        )

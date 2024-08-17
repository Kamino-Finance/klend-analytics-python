from __future__ import annotations
import typing
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
import borsh_construct as borsh
from .. import types
from ..program_id import PROGRAM_ID


class IdlMissingTypesArgs(typing.TypedDict):
    reserve_farm_kind: types.reserve_farm_kind.ReserveFarmKindKind
    asset_tier: types.asset_tier.AssetTierKind
    fee_calculation: types.fee_calculation.FeeCalculationKind
    reserve_status: types.reserve_status.ReserveStatusKind
    update_config_mode: types.update_config_mode.UpdateConfigModeKind
    update_lending_market_config_value: (
        types.update_lending_market_config_value.UpdateLendingMarketConfigValueKind
    )
    update_lending_market_config_mode: (
        types.update_lending_market_mode.UpdateLendingMarketModeKind
    )


layout = borsh.CStruct(
    "reserve_farm_kind" / types.reserve_farm_kind.layout,
    "asset_tier" / types.asset_tier.layout,
    "fee_calculation" / types.fee_calculation.layout,
    "reserve_status" / types.reserve_status.layout,
    "update_config_mode" / types.update_config_mode.layout,
    "update_lending_market_config_value"
    / types.update_lending_market_config_value.layout,
    "update_lending_market_config_mode" / types.update_lending_market_mode.layout,
)


class IdlMissingTypesAccounts(typing.TypedDict):
    lending_market_owner: Pubkey
    lending_market: Pubkey
    reserve: Pubkey


def idl_missing_types(
    args: IdlMissingTypesArgs,
    accounts: IdlMissingTypesAccounts,
    program_id: Pubkey = PROGRAM_ID,
    remaining_accounts: typing.Optional[typing.List[AccountMeta]] = None,
) -> Instruction:
    keys: list[AccountMeta] = [
        AccountMeta(
            pubkey=accounts["lending_market_owner"], is_signer=True, is_writable=False
        ),
        AccountMeta(
            pubkey=accounts["lending_market"], is_signer=False, is_writable=False
        ),
        AccountMeta(pubkey=accounts["reserve"], is_signer=False, is_writable=True),
    ]
    if remaining_accounts is not None:
        keys += remaining_accounts
    identifier = b"\x82P&\x99P\xd4\xb6\xfd"
    encoded_args = layout.build(
        {
            "reserve_farm_kind": args["reserve_farm_kind"].to_encodable(),
            "asset_tier": args["asset_tier"].to_encodable(),
            "fee_calculation": args["fee_calculation"].to_encodable(),
            "reserve_status": args["reserve_status"].to_encodable(),
            "update_config_mode": args["update_config_mode"].to_encodable(),
            "update_lending_market_config_value": args[
                "update_lending_market_config_value"
            ].to_encodable(),
            "update_lending_market_config_mode": args[
                "update_lending_market_config_mode"
            ].to_encodable(),
        }
    )
    data = identifier + encoded_args
    return Instruction(program_id, data, keys)

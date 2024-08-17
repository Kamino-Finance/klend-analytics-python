from __future__ import annotations
import typing
from solders.pubkey import Pubkey
from solders.instruction import Instruction, AccountMeta
import borsh_construct as borsh
from ..program_id import PROGRAM_ID


class WithdrawObligationCollateralAndRedeemReserveCollateralArgs(typing.TypedDict):
    collateral_amount: int


layout = borsh.CStruct("collateral_amount" / borsh.U64)


class WithdrawObligationCollateralAndRedeemReserveCollateralAccounts(typing.TypedDict):
    owner: Pubkey
    obligation: Pubkey
    lending_market: Pubkey
    lending_market_authority: Pubkey
    withdraw_reserve: Pubkey
    reserve_liquidity_mint: Pubkey
    reserve_source_collateral: Pubkey
    reserve_collateral_mint: Pubkey
    reserve_liquidity_supply: Pubkey
    user_destination_liquidity: Pubkey
    placeholder_user_destination_collateral: typing.Optional[Pubkey]
    collateral_token_program: Pubkey
    liquidity_token_program: Pubkey
    instruction_sysvar_account: Pubkey


def withdraw_obligation_collateral_and_redeem_reserve_collateral(
    args: WithdrawObligationCollateralAndRedeemReserveCollateralArgs,
    accounts: WithdrawObligationCollateralAndRedeemReserveCollateralAccounts,
    program_id: Pubkey = PROGRAM_ID,
    remaining_accounts: typing.Optional[typing.List[AccountMeta]] = None,
) -> Instruction:
    keys: list[AccountMeta] = [
        AccountMeta(pubkey=accounts["owner"], is_signer=True, is_writable=True),
        AccountMeta(pubkey=accounts["obligation"], is_signer=False, is_writable=True),
        AccountMeta(
            pubkey=accounts["lending_market"], is_signer=False, is_writable=False
        ),
        AccountMeta(
            pubkey=accounts["lending_market_authority"],
            is_signer=False,
            is_writable=False,
        ),
        AccountMeta(
            pubkey=accounts["withdraw_reserve"], is_signer=False, is_writable=True
        ),
        AccountMeta(
            pubkey=accounts["reserve_liquidity_mint"], is_signer=False, is_writable=True
        ),
        AccountMeta(
            pubkey=accounts["reserve_source_collateral"],
            is_signer=False,
            is_writable=True,
        ),
        AccountMeta(
            pubkey=accounts["reserve_collateral_mint"],
            is_signer=False,
            is_writable=True,
        ),
        AccountMeta(
            pubkey=accounts["reserve_liquidity_supply"],
            is_signer=False,
            is_writable=True,
        ),
        AccountMeta(
            pubkey=accounts["user_destination_liquidity"],
            is_signer=False,
            is_writable=True,
        ),
        (
            AccountMeta(
                pubkey=accounts["placeholder_user_destination_collateral"],
                is_signer=False,
                is_writable=False,
            )
            if accounts["placeholder_user_destination_collateral"]
            else AccountMeta(pubkey=program_id, is_signer=False, is_writable=False)
        ),
        AccountMeta(
            pubkey=accounts["collateral_token_program"],
            is_signer=False,
            is_writable=False,
        ),
        AccountMeta(
            pubkey=accounts["liquidity_token_program"],
            is_signer=False,
            is_writable=False,
        ),
        AccountMeta(
            pubkey=accounts["instruction_sysvar_account"],
            is_signer=False,
            is_writable=False,
        ),
    ]
    if remaining_accounts is not None:
        keys += remaining_accounts
    identifier = b'K]]\xdc"\x96\xda\xc4'
    encoded_args = layout.build(
        {
            "collateral_amount": args["collateral_amount"],
        }
    )
    data = identifier + encoded_args
    return Instruction(program_id, data, keys)

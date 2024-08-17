from pathlib import Path
from solana.rpc.async_api import AsyncClient
from solders.pubkey import Pubkey
from anchorpy import Program, Provider, Wallet, Idl
from dataclasses import dataclass
from based58 import b58encode
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TypedDict,
    Union,
    Callable,
    Any,
    Coroutine,
    Sequence,
)
import os
import requests, json, datetime, logging
from requests.exceptions import RequestException
import time
import numpy as np
import pandas as pd
import logging

from anchorpy.program.namespace.account import ProgramAccount
from anchorpy.coder.accounts import (
    ACCOUNT_DISCRIMINATOR_SIZE,
    _account_discriminator,
)
from solana.rpc.types import DataSliceOpts, MemcmpOpts


from kamino_client.client import KaminoClient
from codegen_lend.accounts import *

MAX_SCOPE_TRIES = 5
MAX_API_TRIES = 5

DISCRIMINATOR_SIZE = 8
RESERVE_SIZE = Reserve.layout.sizeof() + DISCRIMINATOR_SIZE  # 8624
OBLIGATION_SIZE = Obligation.layout.sizeof() + DISCRIMINATOR_SIZE  # 3344

SCALE_FACTOR_60 = 2**60
LAMPORTS_MULTIPLIER = 10**-9

SLOTS_PER_SECOND = 1000 / 450  # approx 450ms per epoch
SLOTS_PER_MINUTE = SLOTS_PER_SECOND * 60
SLOTS_PER_HOUR = SLOTS_PER_MINUTE * 60
SLOTS_PER_DAY = SLOTS_PER_HOUR * 24
SLOTS_PER_YEAR = SLOTS_PER_DAY * 365

# MAINNET
KAMINOLEND_PROGRAM_ID = Pubkey.from_string(
    "KLend2g3cP87fffoy8q1mQqGKjrxjC8boSyAYavgmjD"
)

LENDING_MARKETS = {
    "main_market": "7u3HeHxYDLhnCoErrtycNokbQYbWGzLs6JSDqGAv5PfF",
    "jlp_market": "DxXdAyU3kCjnyggvHmY5nAwg5cRbbmdyX3npfDMjjMek",
    "altcoin_market": "ByYiZxp8QrdN9qbdtaAiePN8AAr3qvTPppNJDpf5DVJ5",
}

NULL_PUBKEY = Pubkey.from_string("11111111111111111111111111111111")

# NOTE: need to set SOLANARPC_HTTP_URI env var


logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(
    "kamino_risk_monitoring",
)
logger.setLevel(logging.INFO)


def get_kamino_lend_program(uri):
    client = AsyncClient(uri)
    wallet = Wallet.dummy()
    provider = Provider(client, wallet)
    idl_filename = "kamino_lending.json"
    if os.path.isfile(f"./{idl_filename}"):
        idl = Idl.from_json(Path(f"./{idl_filename}").open().read())
    else:
        raise BaseException("IDL json file not found!")
    kamino_lend_program = Program(idl, KAMINOLEND_PROGRAM_ID, provider)
    return Program(idl, KAMINOLEND_PROGRAM_ID, provider)


def object_to_dict(obj):
    if isinstance(obj, list):
        return [object_to_dict(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        result = {}
        for key, val in obj.__dict__.items():
            if key.startswith("padding"):
                continue
            element = []
            if isinstance(val, list):
                for item in val:
                    if hasattr(item, "__dict__"):
                        element.append(object_to_dict(item))
                    else:
                        element.append(item)
                result[key] = element
            else:
                if hasattr(val, "__dict__"):
                    result[key] = object_to_dict(val)
                else:
                    result[key] = val
        return result
    else:
        return obj


def get_str_from_byte_list(byte_list):
    byte_data = bytes(byte_list)
    # Decode bytes to string
    string_data = byte_data.decode("utf-8")  # specify the correct encoding
    string_before_null = string_data.split("\x00", 1)[0]

    return string_before_null


async def get_reserves_configs(kamino_lend_program, lending_market):
    """
    Fetch reserves configs from on-chain.
    """
    reserves = await kamino_lend_program.account["Reserve"].all(
        filters=[RESERVE_SIZE, MemcmpOpts(0, Reserve.discriminator)]
    )

    reserves_dict = {}
    for i in range(len(reserves)):
        if str(reserves[i].account.lending_market) == lending_market:
            # get reserve config file dict from on-chain object
            reserve_dict = object_to_dict(reserves[i])

            # convert byted to str token name
            reserve_dict["account"]["config"]["token_info"]["name"] = (
                get_str_from_byte_list(
                    reserve_dict["account"]["config"]["token_info"]["name"]
                )
            )

            reserves_dict.update(
                {str(reserve_dict["account"]["liquidity"]["mint_pubkey"]): reserve_dict}
            )
    return reserves_dict


async def get_elevation_group_dict(lending_market_pubkey):
    kamino_client = KaminoClient(os.getenv("SOLANARPC_HTTP_URI"))

    lending_market_object = await kamino_client.fetch_with_retries(
        LendingMarket.fetch, Pubkey.from_string(lending_market_pubkey), extra=None
    )

    elevation_group_dict = {}
    for group in lending_market_object.elevation_groups:
        # elevation group 0 is the default, i.e. no special elevation group benefits
        if group.id != 0:
            elevation_group_dict[group.id] = {
                "max_liquidation_bonus_bps": group.max_liquidation_bonus_bps,
                "ltv_pct": group.ltv_pct,
                "liquidation_threshold_pct": group.liquidation_threshold_pct,
                "allow_new_loans": group.allow_new_loans,
            }
    return elevation_group_dict


async def get_loans_metrics(
    reserves_configs,
    elevation_group_dict,
    return_intermediate_dfs=False,
    lending_market=LENDING_MARKETS["main_market"],
):
    # get token mint to token name map
    mint_to_str_map, str_to_mint_map = await get_scope_mints_to_str_map()

    uri = os.getenv("SOLANARPC_HTTP_URI")

    # make kamino_lend_program
    kamino_lend_program = get_kamino_lend_program(uri)

    # get list of all obligations
    obligations_list = await get_all_obligations(
        kamino_lend_program, market_pubkey=lending_market
    )
    logging.info(f"Number of obligations = {len(obligations_list)}")
    logging.info("1. Fetched obligations list from on-chain data.")

    # create obligations df (with only positive deposits?)
    obl_df = get_all_obligations_df(obligations_list, only_positive_deposit=True)
    logging.info("2. Created all obligations df.")

    # expand deposit and collateral cols
    obl_df2 = expand_borrows_deposits(
        obl_df, reserves_configs, mint_to_str_map, lending_market=lending_market
    )
    logging.info("3. Added columns from deposit and collateral dicts.")

    # convert decimals and sf to float
    (
        obl_df3,
        deposited_amount_float_cols,
        borrowed_amount_float_cols,
    ) = convert_decimals_and_sf_to_float(obl_df2, reserves_configs, mint_to_str_map)
    logging.info("4. Converted decimals and sf to floats.")

    # get scope prices
    prices = get_prices()
    logging.info("5. Got Scope prices.")

    # add scope prices and calc current usd values of deposits and borrows
    obl_df4 = calc_deposit_borrow_amounts(
        obl_df3, prices, reserves_configs, mint_to_str_map
    )
    logging.info("6. Calculated deposit and borrow amounts.")

    # calc ltvs etc
    obl_df5 = calc_ltvs(
        obl_df4,
        reserves_configs,
        str_to_mint_map,
        mint_to_str_map,
        elevation_group_dict,
    )
    logging.info("7. Calculated LTVs.")

    # drop loans with no deposits and no borrows
    obl_df5 = obl_df5[
        (obl_df5.total_deposit_usd != 0) | (obl_df5.total_borrow_usd != 0)
    ]

    # don't need this. Difficult to pickle this object column for parallelization
    cols_to_drop = ["last_update"]
    obl_df5.drop(columns=cols_to_drop, inplace=True)

    # convert from Pubkey to string
    obl_df5.referrer = obl_df5.referrer.apply(lambda x: str(x))

    if return_intermediate_dfs:
        return obl_df5, obl_df, obl_df2, obl_df3, obl_df4
    else:
        return obl_df5


def get_reserve_mint_decimals_map(reserves_configs):
    reserve_mint_decimals_map = {}
    for reserve_mint, reserve_config in reserves_configs.items():
        reserve_mint_decimals_map[
            str(reserve_config["account"]["liquidity"]["mint_pubkey"])
        ] = reserve_config["account"]["liquidity"]["mint_decimals"]
    return reserve_mint_decimals_map


async def all_pubkeys_w_data_slice(
    account_idl,
    market_pubkey=None,
    filters: Optional[Sequence[Union[int, MemcmpOpts]]] = None,
    data_slice: Optional[DataSliceOpts] = None,
) -> list[ProgramAccount]:
    """Return all pubkeys of this account type for the program with data slice options.

    Args:
        filters: (optional) Options to compare a provided series of bytes with
            program account data at a particular offset.
            Note: an int entry is converted to a `dataSize` filter.
        data_slice: (optional) Data slice options for partial account data retrieval.
    """
    all_pubkeys = []
    discriminator = _account_discriminator(account_idl._idl_account.name)
    to_encode = discriminator
    bytes_arg = b58encode(to_encode).decode("ascii")
    base_memcmp_opt = MemcmpOpts(
        offset=0,
        bytes=bytes_arg,
    )

    if market_pubkey:
        market_memcmp_opt = MemcmpOpts(offset=32, bytes=market_pubkey)

        filters_to_use = (
            [base_memcmp_opt] + [market_memcmp_opt] + (filters if filters else [])
        )
    else:
        filters_to_use = [base_memcmp_opt] + (filters if filters else [])
    resp = await account_idl._provider.connection.get_program_accounts(
        account_idl._program_id,
        encoding="base64",
        commitment=account_idl.provider.connection._commitment,
        filters=filters_to_use,
        data_slice=data_slice,
    )
    for r in resp.value:
        account_data = r.account.data
        all_pubkeys.append(r.pubkey)
    return all_pubkeys


async def fetch_obls_chunk(program, pubkeys, chunk_size):
    """Asynchronously fetch a chunk of obligations."""
    return await program.account["Obligation"].fetch_multiple(pubkeys, chunk_size)


async def fetch_all_obls(program, obl_pubkeys, step, chunk_size):
    """Asynchronously fetch all obligations sequentially in chunks."""
    all_obls = {}
    for i in range(0, len(obl_pubkeys), step):
        chunk = obl_pubkeys[i : i + step]
        obls_chunk = await fetch_obls_chunk(program, chunk, chunk_size)
        # Combine each pubkey with its corresponding obligation
        all_obls.update(dict(zip(chunk, obls_chunk)))
    return all_obls


async def get_all_obligations(kamino_lend_program, market_pubkey=None):
    start_time = time.time()

    # step 1: only fetch list of obl pubkeys
    obl_pubkeys = await all_pubkeys_w_data_slice(
        kamino_lend_program.account["Obligation"],
        market_pubkey,
        filters=[OBLIGATION_SIZE],
        data_slice=DataSliceOpts(offset=0, length=0),
    )
    logging.info(f"Number of obligations = {len(obl_pubkeys)}")
    time_taken = time.time() - start_time
    logging.info(f"get all_pubkeys_w_data_slice time_taken = {time_taken:.2f} seconds")

    # step 2: batch fetch obligations
    return await fetch_all_obls(
        kamino_lend_program, obl_pubkeys, step=1000, chunk_size=500
    )


# [DEPRECATED]: much slower
# async def get_all_obligations_one_shot(kamino_lend_program):
#     return await kamino_lend_program.account["Obligation"].all(
#         filters=[OBLIGATION_SIZE]
#     )

# def get_obligation_info_one_shot(obligation):
#     try:
#         obl_dict = {}
#         obl_dict.update(vars(obligation.account))
#         obl_dict.update({"public_key": str(obligation.public_key)})
#         obl_dict.update({"owner": str(obligation.account.owner)})
#         return obl_dict
#     except BaseException as e:
#         return None


def get_obligation_info(pubkey, obligation):
    try:
        obl_dict = {}
        obl_dict.update(vars(obligation))
        obl_dict.update({"public_key": str(pubkey)})
        obl_dict.update({"owner": str(obligation.owner)})
        return obl_dict
    except BaseException as e:
        return None


def get_all_obligations_df(obligations_list, only_positive_deposit=False):
    obl_list = [
        get_obligation_info(pubkey, obl) for pubkey, obl in obligations_list.items()
    ]
    obl_list = [x for x in obl_list if x is not None]
    obl_df = pd.DataFrame(obl_list)

    # reset index
    obl_df.reset_index(inplace=True, drop=True)

    # for display/plotting
    obl_df["pubkey_short"] = obl_df["public_key"].map(lambda x: x[:5])

    # for display/plotting
    obl_df["owner_pubkey_short"] = obl_df["owner"].map(lambda x: x[:5])

    return obl_df


def expand_borrows_deposits(
    obl_df,
    reserves_configs,
    mint_to_str_map,
    lending_market=LENDING_MARKETS["main_market"],
):
    token_reserve_to_token_mint_map = get_token_reserve_to_token_mint_map(
        reserves_configs
    )

    obl_df2 = obl_df.copy()
    obl_df2 = drop_padding_cols(obl_df2)
    pubkey_cols = ["public_key", "lending_market", "owner"]
    obl_df2 = convert_pubkey_cols_to_str(obl_df2, pubkey_cols)
    obl_df2 = obl_df2[obl_df2.lending_market == lending_market]
    obl_df2.set_index("public_key", inplace=True)

    obl_df2["borrows_list"] = [[] for _ in range(len(obl_df2))]
    obl_df2["deposits_list"] = [[] for _ in range(len(obl_df2))]

    for j in obl_df2.index:
        tmp_list = object_to_dict(obl_df2.loc[j].borrows)
        tmp_list = [
            borrow for borrow in tmp_list if borrow["borrow_reserve"] != NULL_PUBKEY
        ]

        if len(tmp_list) > 0:
            borrows_list_j = [
                {
                    f"{mint_to_str_map[token_reserve_to_token_mint_map[str(tmp_list_i['borrow_reserve'])]]}_{key}": (
                        get_borrow_cumulative_borrow_rate(value["value"])
                        if key == "cumulative_borrow_rate_bsf"
                        else str(value) if key == "borrow_reserve" else value
                    )
                    for key, value in tmp_list_i.items()
                }
                for tmp_list_i in tmp_list
            ]
            borrows_list_j = list(borrows_list_j)
            if borrows_list_j:
                obl_df2.at[j, "borrows_list"] = borrows_list_j

        tmp_list2 = object_to_dict(obl_df2.loc[j].deposits)
        tmp_list2 = [
            deposit
            for deposit in tmp_list2
            if deposit["deposit_reserve"] != NULL_PUBKEY
        ]

        if len(tmp_list2) > 0:
            deposit_list_j = [
                {
                    f"{mint_to_str_map[token_reserve_to_token_mint_map[str(tmp_list_i_2['deposit_reserve'])]]}_{key}": (
                        str(value) if key == "deposit_reserve" else value
                    )
                    for key, value in tmp_list_i_2.items()
                }
                for tmp_list_i_2 in tmp_list2
            ]
            deposit_list_j = list(deposit_list_j)
            if deposit_list_j:
                obl_df2.at[j, "deposits_list"] = deposit_list_j

    obl_df2.drop(columns=["borrows", "deposits"], inplace=True)

    # check no duplicates
    check_for_dup_borrows_deposits_keys_in_list(obl_df2)

    # Apply the merge_dicts function to convert the 'list_column' to a column of dictionaries
    obl_df2["borrows_list"] = obl_df2["borrows_list"].apply(merge_dicts)
    obl_df2["deposits_list"] = obl_df2["deposits_list"].apply(merge_dicts)

    expanded_df = pd.json_normalize(obl_df2["borrows_list"])
    columns_to_suffix = [
        col for col in expanded_df.columns if col.endswith("market_value_sf")
    ]
    expanded_df.rename(
        columns={col: col + "_borrow" for col in columns_to_suffix}, inplace=True
    )
    obl_df3 = obl_df2.reset_index().join(expanded_df)

    expanded_df = pd.json_normalize(obl_df3["deposits_list"])
    columns_to_suffix = [
        col for col in expanded_df.columns if col.endswith("market_value_sf")
    ]
    expanded_df.rename(
        columns={col: col + "_deposit" for col in columns_to_suffix}, inplace=True
    )
    obl_df4 = obl_df3.join(expanded_df)
    obl_df4.set_index("public_key", inplace=True)

    return obl_df4


def get_borrow_cumulative_borrow_rate(cumulative_borrow_rate_bsf):
    acc_sf = 0
    for value in reversed(cumulative_borrow_rate_bsf):
        acc_sf = acc_sf * 2**64  # Equivalent of right-shifting 64 bits
        acc_sf += value
    return acc_sf / 2**60


def get_token_reserve_to_token_mint_map(reserves_configs):
    return {
        str(reserves_configs[token_mint]["public_key"]): token_mint
        for token_mint in list(reserves_configs.keys())
    }


def drop_padding_cols(df):
    cols_to_drop = df.filter(like="padding", axis=1).columns
    return df.drop(columns=cols_to_drop)


def convert_pubkey_cols_to_str(df_in, pubkey_cols):
    df = df_in.copy()
    for col in pubkey_cols:
        df[col] = df[col].apply(lambda x: str(x))
    return df


def find_duplicate_keys(list_of_dicts):
    key_count = {}
    for dictionary in list_of_dicts:
        for key in dictionary.keys():
            if key in key_count:
                key_count[key] += 1
            else:
                key_count[key] = 1
    duplicate_keys = [key for key, count in key_count.items() if count > 1]
    return duplicate_keys


def check_for_dup_borrows_deposits_keys_in_list(obl_df_in):
    obl_df2 = obl_df_in.copy()

    for i in range(len(obl_df2)):
        deposits_duplicate_keys = find_duplicate_keys(obl_df2.iloc[i].deposits_list)
        if deposits_duplicate_keys:
            raise ValueError(f"deposits_duplicate_keys = {deposits_duplicate_keys}")

        borrows_duplicate_keys = find_duplicate_keys(obl_df2.iloc[i].borrows_list)
        if borrows_duplicate_keys:
            raise ValueError(f"borrows_duplicate_keys = {borrows_duplicate_keys}")


def merge_dicts(lst):
    merged_dict = {}
    for d in lst:
        merged_dict.update(d)
    return merged_dict


def get_collateral_exchange_rate(reserve_config):
    mint_total_supply = reserve_config["account"]["collateral"]["mint_total_supply"]
    new_total_supply = (
        reserve_config["account"]["liquidity"]["available_amount"]
        + (
            reserve_config["account"]["liquidity"]["borrowed_amount_sf"]
            / SCALE_FACTOR_60
        )
        - (
            reserve_config["account"]["liquidity"]["accumulated_protocol_fees_sf"]
            / SCALE_FACTOR_60
        )
        - (
            reserve_config["account"]["liquidity"]["accumulated_referrer_fees_sf"]
            / SCALE_FACTOR_60
        )
    )
    return mint_total_supply / new_total_supply


def get_reserve_cum_borrow_rate(reserve_config):
    return get_borrow_cumulative_borrow_rate(
        reserve_config["account"]["liquidity"]["cumulative_borrow_rate_bsf"]["value"]
    )


def convert_decimals_and_sf_to_float(df, reserves_configs, mint_to_str_map):
    """
    NOTE:

    market value, borrowed amount sf, market price and interest rate (cumulative borrow rate sf),
    and accumulated protocol fees sf are scaled by 2^60 to preserve precision.
    """
    obl_df3 = df.copy()

    cols_to_div_by_sf60 = [
        "deposited_value_sf",
        "borrow_factor_adjusted_debt_value_sf",
        "allowed_borrow_value_sf",
        "unhealthy_borrow_value_sf",
    ]
    for col in cols_to_div_by_sf60:
        obl_df3[col] /= SCALE_FACTOR_60

    # deposits
    deposited_amount_float_cols = []
    for token in reserves_configs.keys():
        token_str = mint_to_str_map[token]
        try:
            col_name = f"{token_str}_deposited_amount_float"
            obl_df3[col_name] = (
                obl_df3[f"{token_str}_deposited_amount"]
                / get_collateral_exchange_rate(reserves_configs[token])
                * (
                    10
                    ** -reserves_configs[token]["account"]["liquidity"]["mint_decimals"]
                )
            )
            obl_df3.drop(columns=f"{token_str}_deposited_amount", inplace=True)
            deposited_amount_float_cols.append(col_name)
        except KeyError as e:
            # logging.info(f"No column named {e} in df!")
            pass

    # borrows
    borrowed_amount_float_cols = []
    for token in reserves_configs.keys():
        token_str = mint_to_str_map[token]
        try:
            col_name = f"{token_str}_borrowed_amount_float"
            obl_df3[col_name] = (
                (obl_df3[f"{token_str}_borrowed_amount_sf"] / SCALE_FACTOR_60)
                * get_reserve_cum_borrow_rate(reserves_configs[token])
                / obl_df3[f"{token_str}_cumulative_borrow_rate_bsf"]
                * (
                    10
                    ** -reserves_configs[token]["account"]["liquidity"]["mint_decimals"]
                )
            )
            obl_df3.drop(columns=f"{token_str}_borrowed_amount_sf", inplace=True)
            borrowed_amount_float_cols.append(col_name)
        except KeyError as e:
            # logging.info(f"No column named {e} in df!")
            pass

    return obl_df3, deposited_amount_float_cols, borrowed_amount_float_cols


def calc_deposit_borrow_amounts(obl_df_in, prices, reserves_configs, mint_to_str_map):
    obl_df4 = obl_df_in.copy()

    # add deposit columns
    for token in reserves_configs.keys():
        token_str = mint_to_str_map[token]
        try:
            col_name = f"{token_str}_deposited_amount_usd_value"
            obl_df4[col_name] = (
                obl_df4[f"{token_str}_deposited_amount_float"]
                * prices[token_str.lower()]
            )
        except KeyError as e:
            pass
            # logging.info(f"No column named {e} in df!")

    # add borrow columns
    for token in reserves_configs.keys():
        token_str = mint_to_str_map[token]
        try:
            col_name = f"{token_str}_borrowed_amount_usd_value"
            obl_df4[col_name] = (
                obl_df4[f"{token_str}_borrowed_amount_float"]
                * prices[token_str.lower()]
            )
        except KeyError as e:
            pass
            # logging.info(f"No column named {e} in df!")

    return obl_df4


def calc_ltvs(
    obl_df_in, reserves_configs, str_to_mint_map, mint_to_str_map, elevation_group_dict
):
    obl_df5 = obl_df_in.copy()
    (
        deposited_amount_usd_value_cols,
        borrowed_amount_usd_value_cols,
    ) = get_deposit_borrow_amount_usd_cols(obl_df5)

    (
        deposited_amount_token_amount_cols,
        borrowed_amount_token_amount_cols,
    ) = get_deposit_borrow_token_amount_cols(obl_df5)

    # replace nan with zero
    obl_df5.fillna(0, inplace=True)

    # get tokens with 0% ltv
    reserves_configs_keys_zero_ltv = [
        mint_to_str_map[k]
        for k, r in reserves_configs.items()
        if r["account"]["config"]["loan_to_value_pct"] == 0
    ]

    deposited_amount_usd_value_store = {}
    for key_zero_ltv in reserves_configs_keys_zero_ltv:
        key_zero_ltv_upper = key_zero_ltv.upper()

        if f"{key_zero_ltv_upper}_deposited_amount_usd_value" in obl_df5.columns:
            deposited_amount_usd_value_store[key_zero_ltv_upper] = obl_df5[
                f"{key_zero_ltv_upper}_deposited_amount_usd_value"
            ]
            obl_df5[f"{key_zero_ltv_upper}_deposited_amount_usd_value"] = 0

    obl_df5["total_borrow_usd"] = obl_df5[borrowed_amount_usd_value_cols].sum(axis=1)
    obl_df5["total_deposit_usd"] = obl_df5[deposited_amount_usd_value_cols].sum(axis=1)

    # calc ltv's
    obl_df5["current_ltv_without_borrow_factor"] = (
        obl_df5["total_borrow_usd"] / obl_df5["total_deposit_usd"]
    )

    obl_df5["borrow_factor_adjusted_total_borrow_usd"] = 0
    for col in borrowed_amount_usd_value_cols:
        token = col.split("_")[0]
        obl_df5["borrow_factor_adjusted_total_borrow_usd"] += obl_df5[col] * (
            reserves_configs[str_to_mint_map[token]]["account"]["config"][
                "borrow_factor_pct"
            ]
            / 100
        )

    # overwrite using elevation groups
    obl_df5["borrow_factor_adjusted_total_borrow_usd"] = obl_df5.apply(
        lambda row: (
            row["total_borrow_usd"]
            if row["elevation_group"] != 0
            else row["borrow_factor_adjusted_total_borrow_usd"]
        ),
        axis=1,
    )

    obl_df5["current_ltv"] = (
        obl_df5["borrow_factor_adjusted_total_borrow_usd"]
        / obl_df5["total_deposit_usd"]
    )

    # overwrite using elevation groups
    obl_df5["current_ltv"] = obl_df5.apply(
        lambda row: (
            row["total_borrow_usd"] / row["total_deposit_usd"]
            if row["elevation_group"] != 0 and row["total_deposit_usd"] != 0
            else row["current_ltv"]
        ),
        axis=1,
    )

    # calc max_allowed_borrow_usd
    obl_df5["max_allowed_borrow_usd"] = 0
    for col in deposited_amount_usd_value_cols:
        token = col.split("_")[0]
        obl_df5["max_allowed_borrow_usd"] += obl_df5[col] * (
            reserves_configs[str_to_mint_map[token]]["account"]["config"][
                "loan_to_value_pct"
            ]
            / 100
        )

    # overwrite using elevation groups
    obl_df5["max_allowed_borrow_usd"] = obl_df5.apply(
        lambda row: (
            elevation_group_dict.get(row["elevation_group"], {}).get("ltv_pct")
            / 100
            * row["total_deposit_usd"]
            if row["elevation_group"] != 0
            else row["max_allowed_borrow_usd"]
        ),
        axis=1,
    )

    obl_df5["max_ltv"] = (
        obl_df5["max_allowed_borrow_usd"] / obl_df5["total_deposit_usd"]
    )

    # calc unhealthy_borrow_usd
    obl_df5["unhealthy_borrow_usd"] = 0
    for col in deposited_amount_usd_value_cols:
        token = col.split("_")[0]
        obl_df5["unhealthy_borrow_usd"] += obl_df5[col] * (
            reserves_configs[str_to_mint_map[token]]["account"]["config"][
                "liquidation_threshold_pct"
            ]
            / 100
        )

    # overwrite using elevation groups
    obl_df5["unhealthy_borrow_usd"] = obl_df5.apply(
        lambda row: (
            elevation_group_dict.get(row["elevation_group"], {}).get(
                "liquidation_threshold_pct"
            )
            / 100
            * row["total_deposit_usd"]
            if row["elevation_group"] != 0
            else row["unhealthy_borrow_usd"]
        ),
        axis=1,
    )

    obl_df5["unhealthy_ltv"] = (
        obl_df5["unhealthy_borrow_usd"] / obl_df5["total_deposit_usd"]
    )

    obl_df5["min_collateral_usd_allowed"] = (
        obl_df5["borrow_factor_adjusted_total_borrow_usd"] / obl_df5["unhealthy_ltv"]
    )

    # calc net value
    obl_df5["net_value"] = obl_df5[deposited_amount_usd_value_cols].sum(
        axis=1
    ) - obl_df5[borrowed_amount_usd_value_cols].sum(axis=1)

    # calc net value above liquidation level
    obl_df5["net_value_above_liq"] = (
        obl_df5["unhealthy_ltv"] * obl_df5[deposited_amount_usd_value_cols].sum(axis=1)
    ) - obl_df5[borrowed_amount_usd_value_cols].sum(axis=1)

    # calc distance until liquidation (0 means liquidation)
    obl_df5["dist_to_liq"] = obl_df5["unhealthy_ltv"] - obl_df5["current_ltv"]

    for key_zero_ltv in reserves_configs_keys_zero_ltv:
        key_zero_ltv_upper = key_zero_ltv.upper()

        if f"{key_zero_ltv_upper}_deposited_amount_usd_value" in obl_df5.columns:
            # restore token amounts
            obl_df5[f"{key_zero_ltv_upper}_deposited_amount_usd_value"] = (
                deposited_amount_usd_value_store[key_zero_ltv_upper]
            )

    obl_df5["total_borrow_usd"] = obl_df5[borrowed_amount_usd_value_cols].sum(axis=1)
    obl_df5["total_deposit_usd"] = obl_df5[deposited_amount_usd_value_cols].sum(axis=1)

    return obl_df5


def get_prices(return_mints_as_keys=False):
    num_scope_tries = 0

    while num_scope_tries < MAX_SCOPE_TRIES:
        logging.info(f"num_scope_tries = {num_scope_tries}")
        try:
            token_prices = prices(return_mints_as_keys=return_mints_as_keys)
            break
        except BaseException as e:
            num_scope_tries += 1

    return token_prices


def get_deposit_borrow_amount_usd_cols(loan_metrics_df):
    deposited_amount_usd_value_cols = [
        col
        for col in loan_metrics_df.columns
        if col.endswith("_deposited_amount_usd_value")
    ]
    borrowed_amount_usd_value_cols = [
        col
        for col in loan_metrics_df.columns
        if col.endswith("_borrowed_amount_usd_value")
    ]
    return deposited_amount_usd_value_cols, borrowed_amount_usd_value_cols


def get_deposit_borrow_token_amount_cols(loan_metrics_df):
    deposited_amount_token_amount_cols = [
        col
        for col in loan_metrics_df.columns
        if col.endswith("_deposited_amount_float")
    ]
    borrowed_amount_token_amount_cols = [
        col for col in loan_metrics_df.columns if col.endswith("_borrowed_amount_float")
    ]
    return deposited_amount_token_amount_cols, borrowed_amount_token_amount_cols


# IR CURVES


def get_ir_curve(reserves_configs, token):
    borrow_curve_df = pd.DataFrame(
        reserves_configs[token]["account"]["config"]["borrow_rate_curve"]["points"]
    )
    borrow_curve_df.utilization_rate_bps /= 100
    borrow_curve_df.borrow_rate_bps /= 100

    borrow_curve_df.rename(
        columns={
            "utilization_rate_bps": "utilization_rate",
            "borrow_rate_bps": "borrow_rate",
        },
        inplace=True,
    )
    borrow_curve_df = borrow_curve_df[borrow_curve_df.utilization_rate <= 100]
    borrow_curve_df.drop_duplicates(inplace=True)
    borrow_curve_df["supply_rate"] = (
        borrow_curve_df.borrow_rate
        * (
            1
            - reserves_configs[token]["account"]["config"]["protocol_take_rate_pct"]
            / 100
        )
    ) * (borrow_curve_df.utilization_rate / 100)

    new_utilization_rate = np.arange(0, 101)
    df_interpolated = pd.DataFrame({"utilization_rate": new_utilization_rate})

    df_interpolated["borrow_rate"] = np.interp(
        df_interpolated["utilization_rate"],
        borrow_curve_df["utilization_rate"],
        borrow_curve_df["borrow_rate"],
    )

    df_interpolated["supply_rate"] = (
        df_interpolated.borrow_rate
        * (
            1
            - reserves_configs[token]["account"]["config"]["protocol_take_rate_pct"]
            / 100
        )
    ) * (df_interpolated.utilization_rate / 100)

    current_util = get_curr_util(reserves_configs[token])

    if current_util:
        current_borrow_rate = np.interp(
            current_util,
            df_interpolated["utilization_rate"],
            df_interpolated["borrow_rate"],
        )
        current_borrow_rate_apy = 100 * calculate_apy_from_apr(
            current_borrow_rate / 100, SLOTS_PER_YEAR
        )

        current_supply_rate = (
            current_borrow_rate
            * (
                1
                - reserves_configs[token]["account"]["config"]["protocol_take_rate_pct"]
                / 100
            )
        ) * (current_util / 100)
        current_supply_rate_apy = 100 * calculate_apy_from_apr(
            current_supply_rate / 100, SLOTS_PER_YEAR
        )

        # calc apys
        df_interpolated["borrow_rate_apy"] = 100 * df_interpolated.borrow_rate.apply(
            lambda x: calculate_apy_from_apr(x / 100, SLOTS_PER_YEAR)
        )

        df_interpolated["supply_rate_apy"] = 100 * df_interpolated.supply_rate.apply(
            lambda x: calculate_apy_from_apr(x / 100, SLOTS_PER_YEAR)
        )
    else:
        df_interpolated["borrow_rate_apy"] = 0
        df_interpolated["supply_rate_apy"] = 0
        (
            current_util,
            current_borrow_rate,
            current_borrow_rate_apy,
            current_supply_rate,
            current_supply_rate_apy,
        ) = (0, 0, 0, 0, 0)

    return (
        df_interpolated,
        borrow_curve_df,
        current_util,
        current_borrow_rate,
        current_borrow_rate_apy,
        current_supply_rate,
        current_supply_rate_apy,
    )


def plot_ir_curve(
    df_interpolated,
    borrow_curve_df,
    current_util,
    current_borrow_rate,
    current_borrow_rate_apy,
    current_supply_rate,
    current_supply_rate_apy,
):
    fig = px.line(
        df_interpolated,
        x="utilization_rate",
        y=["borrow_rate", "supply_rate"],
        line_shape="linear",
        line_dash_sequence=["solid", "solid"],
        color_discrete_sequence=["red", "blue"],
        labels={
            "utilization_rate": "Utilisation Rate",
            "borrow_rate": "Borrow Rate",
            "supply_rate": "Supply Rate",
        },
    )
    fig.add_scatter(
        x=borrow_curve_df["utilization_rate"],
        y=borrow_curve_df["borrow_rate"],
        mode="markers",
        marker_symbol="diamond",
        marker_size=10,
        marker_color="magenta",
        showlegend=False,
    )
    fig.add_scatter(
        x=borrow_curve_df["utilization_rate"],
        y=borrow_curve_df["supply_rate"],
        mode="markers",
        marker_symbol="diamond",
        marker_size=10,
        marker_color="magenta",
        showlegend=False,
    )
    fig.add_vline(current_util, line_dash="dash", line_color="green")
    fig.update_layout(
        title=f"Current Utilisation = {current_util:,.2f}%, Current Supply Rate = {current_supply_rate:,.2f}%, Current Borrow Rate = {current_borrow_rate:,.2f}%",
        title_x=0.35,
        yaxis_title="Interest Rate (%)",
        legend_title_text="",
    )
    fig.for_each_trace(
        lambda trace: (
            trace.update(name=trace.name.title().replace("_", " "))
            if trace.name
            else None
        )
    )

    fig_apy = px.line(
        df_interpolated,
        x="utilization_rate",
        y=["borrow_rate_apy", "supply_rate_apy"],
        line_shape="linear",
        line_dash_sequence=["solid", "solid"],
        color_discrete_sequence=["red", "blue"],
        labels={
            "utilization_rate": "Utilisation Rate",
            "borrow_rate_apy": "Borrow Rate APY",
            "supply_rate_apy": "Supply Rate APY",
        },
    )
    fig_apy.add_vline(current_util, line_dash="dash", line_color="green")
    fig_apy.update_layout(
        title=f"Current Utilisation = {max(0, current_util):,.2f}%, Current Supply Rate APY = {max(0, current_supply_rate_apy):,.2f}%, Current Borrow Rate APY = {max(0, current_borrow_rate_apy):,.2f}%",
        title_x=0.35,
        yaxis_title="Interest Rate (%)",
        legend_title_text="",
    )
    fig_apy.for_each_trace(
        lambda trace: (
            trace.update(name=trace.name.title().replace("_", " "))
            if trace.name
            else None
        )
    )

    return fig, fig_apy


def calculate_apy_from_apr(apr, slots_per_year):
    apy = (1 + apr / slots_per_year) ** slots_per_year - 1
    return apy


def get_curr_util(reserve_config):
    numerator = reserve_config["account"]["liquidity"]["borrowed_amount_sf"] / (
        SCALE_FACTOR_60
    )

    denominator = (
        (
            reserve_config["account"]["liquidity"]["borrowed_amount_sf"]
            / (SCALE_FACTOR_60)
        )
        + (reserve_config["account"]["liquidity"]["available_amount"])
        + (
            reserve_config["account"]["liquidity"]["accumulated_protocol_fees_sf"]
            / (SCALE_FACTOR_60)
        )
        + (
            reserve_config["account"]["liquidity"]["accumulated_referrer_fees_sf"]
            / (SCALE_FACTOR_60)
        )
    )

    return 100 * numerator / denominator if denominator else None


# SCOPE DATA FETCHING


def price_history_point(timestamp, token):
    st = (timestamp - datetime.timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")
    en = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return price_history(token, start=st, end=en)[-1]


def prices_history_point(timestamp, tokens):
    tokens = [t.upper() for t in tokens]
    return {token.lower(): price_history_point(timestamp, token) for token in tokens}


def price_history(token, start="2022-11-06", end="2022-11-07"):
    url = f"https://api.hubbleprotocol.io/prices/history?env=mainnet-beta&start={start}&end={end}&token={token}"
    resp = requests.get(url)
    prices = json.loads(resp.content)
    prices = [(datetime.datetime.fromtimestamp(p[1]), float(p[0])) for p in prices]
    return prices


def prices(return_mints_as_keys=False):
    url = "https://api.hubbleprotocol.io/prices?env=mainnet-beta"

    for _ in range(MAX_API_TRIES):  # retry logic
        try:
            response = requests.get(url)
            response.raise_for_status()  # This will check if the status code is 200
            prices = json.loads(response.content)
            if return_mints_as_keys:
                return {
                    p["mint"] if p["mint"] is not None else p["token"].lower(): float(
                        p["usdPrice"]
                    )
                    for p in prices
                }
            else:
                return {p["token"].lower(): float(p["usdPrice"]) for p in prices}
        except RequestException as err:
            logging.info(f"An error occurred: {err}. Retrying...")
    raise Exception("API Error: Unable to retrieve data after 5 attempts")


async def get_scope_mints_to_str_map():
    url = "https://api.hubbleprotocol.io/prices?env=mainnet-beta"

    for _ in range(MAX_API_TRIES):  # retry logic
        try:
            response = requests.get(url)
            response.raise_for_status()  # This will check if the status code is 200
            prices = json.loads(response.content)
            mint_to_str_map = {
                p["mint"] if p["mint"] is not None else p["token"]: p["token"].upper()
                for p in prices
            }
            # make reverse dict
            str_to_mint_map = {value: key for key, value in mint_to_str_map.items()}

            return mint_to_str_map, str_to_mint_map
        except RequestException as err:
            logging.info(f"An error occurred: {err}. Retrying...")
    raise Exception("API Error: Unable to retrieve data after 5 attempts")

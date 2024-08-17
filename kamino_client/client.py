#!/usr/bin/env python3
from dataclasses import dataclass
import os
import json
import struct
import asyncio
import borsh_construct as borsh
from functools import partial
from typing import Optional, TypedDict, Union, Callable, Any, Coroutine
from time import time
from enum import Enum
from solana.rpc.types import TxOpts
from solana.rpc.commitment import Confirmed
from solana.rpc.async_api import AsyncClient
from solders.account import Account
from solders.account_decoder import UiTokenAmount
from solders.pubkey import Pubkey as PublicKey
from solders.keypair import Keypair
from solders.instruction import Instruction, AccountMeta
from solders.hash import Hash
from solders.message import MessageV0
from solders.transaction import VersionedTransaction
from solders.address_lookup_table_account import AddressLookupTableAccount
from solders.rpc.responses import GetAccountInfoResp, GetTokenAccountBalanceResp
from solders.signature import Signature
from solders.transaction_status import TransactionConfirmationStatus
from anchorpy import Provider, Wallet
from spl.token.instructions import get_associated_token_address

FETCH_RETRIES = 5


class KaminoClient:
    def __init__(self, rpc_url: str, wallet_path: Optional[str] = None):
        # Create client and wallet
        self.client = AsyncClient(rpc_url)
        if wallet_path is None:
            self.wallet_kp = Keypair()
        else:
            with open(wallet_path, "r") as f:
                self.wallet_kp = Keypair.from_bytes(bytes(json.load(f)))
        self.wallet = Wallet(self.wallet_kp)
        self.provider = Provider(self.client, self.wallet)

    async def fetch_with_retries(
        self,
        fetch: Callable[..., Any],
        address: PublicKey,
        extra: None,
        program_id: Optional[PublicKey] = None,
    ) -> Optional[Any]:
        last_err = None
        for _ in range(FETCH_RETRIES):
            try:
                if program_id is not None:
                    account = await fetch(self.client, address, program_id=program_id)
                else:
                    account = await fetch(self.client, address)
                if account is not None:
                    return account
            except Exception as err:
                last_err = err
        error(
            f"Could not fetch account {address} with fetch function {fetch} after {FETCH_RETRIES} retries! Got last err {last_err}",
            extra,
        )
        return None

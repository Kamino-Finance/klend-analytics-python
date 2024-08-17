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
from ..program_id import PROGRAM_ID


class UserStateJSON(typing.TypedDict):
    user_id: int
    farm_state: str
    owner: str
    is_farm_delegated: int
    padding0: list[int]
    rewards_tally_scaled: list[int]
    rewards_issued_unclaimed: list[int]
    last_claim_ts: list[int]
    active_stake_scaled: int
    pending_deposit_stake_scaled: int
    pending_deposit_stake_ts: int
    pending_withdrawal_unstake_scaled: int
    pending_withdrawal_unstake_ts: int
    bump: int
    delegatee: str
    last_stake_ts: int
    padding1: list[int]


@dataclass
class UserState:
    discriminator: typing.ClassVar = b"H\xb1U\xf9L\xa7\xba~"
    layout: typing.ClassVar = borsh.CStruct(
        "user_id" / borsh.U64,
        "farm_state" / BorshPubkey,
        "owner" / BorshPubkey,
        "is_farm_delegated" / borsh.U8,
        "padding0" / borsh.U8[7],
        "rewards_tally_scaled" / borsh.U128[10],
        "rewards_issued_unclaimed" / borsh.U64[10],
        "last_claim_ts" / borsh.U64[10],
        "active_stake_scaled" / borsh.U128,
        "pending_deposit_stake_scaled" / borsh.U128,
        "pending_deposit_stake_ts" / borsh.U64,
        "pending_withdrawal_unstake_scaled" / borsh.U128,
        "pending_withdrawal_unstake_ts" / borsh.U64,
        "bump" / borsh.U64,
        "delegatee" / BorshPubkey,
        "last_stake_ts" / borsh.U64,
        "padding1" / borsh.U64[50],
    )
    user_id: int
    farm_state: Pubkey
    owner: Pubkey
    is_farm_delegated: int
    padding0: list[int]
    rewards_tally_scaled: list[int]
    rewards_issued_unclaimed: list[int]
    last_claim_ts: list[int]
    active_stake_scaled: int
    pending_deposit_stake_scaled: int
    pending_deposit_stake_ts: int
    pending_withdrawal_unstake_scaled: int
    pending_withdrawal_unstake_ts: int
    bump: int
    delegatee: Pubkey
    last_stake_ts: int
    padding1: list[int]

    @classmethod
    async def fetch(
        cls,
        conn: AsyncClient,
        address: Pubkey,
        commitment: typing.Optional[Commitment] = None,
        program_id: Pubkey = PROGRAM_ID,
    ) -> typing.Optional["UserState"]:
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
    ) -> typing.List[typing.Optional["UserState"]]:
        infos = await get_multiple_accounts(conn, addresses, commitment=commitment)
        res: typing.List[typing.Optional["UserState"]] = []
        for info in infos:
            if info is None:
                res.append(None)
                continue
            if info.account.owner != program_id:
                raise ValueError("Account does not belong to this program")
            res.append(cls.decode(info.account.data))
        return res

    @classmethod
    def decode(cls, data: bytes) -> "UserState":
        if data[:ACCOUNT_DISCRIMINATOR_SIZE] != cls.discriminator:
            raise AccountInvalidDiscriminator(
                "The discriminator for this account is invalid"
            )
        dec = UserState.layout.parse(data[ACCOUNT_DISCRIMINATOR_SIZE:])
        return cls(
            user_id=dec.user_id,
            farm_state=dec.farm_state,
            owner=dec.owner,
            is_farm_delegated=dec.is_farm_delegated,
            padding0=dec.padding0,
            rewards_tally_scaled=dec.rewards_tally_scaled,
            rewards_issued_unclaimed=dec.rewards_issued_unclaimed,
            last_claim_ts=dec.last_claim_ts,
            active_stake_scaled=dec.active_stake_scaled,
            pending_deposit_stake_scaled=dec.pending_deposit_stake_scaled,
            pending_deposit_stake_ts=dec.pending_deposit_stake_ts,
            pending_withdrawal_unstake_scaled=dec.pending_withdrawal_unstake_scaled,
            pending_withdrawal_unstake_ts=dec.pending_withdrawal_unstake_ts,
            bump=dec.bump,
            delegatee=dec.delegatee,
            last_stake_ts=dec.last_stake_ts,
            padding1=dec.padding1,
        )

    def to_json(self) -> UserStateJSON:
        return {
            "user_id": self.user_id,
            "farm_state": str(self.farm_state),
            "owner": str(self.owner),
            "is_farm_delegated": self.is_farm_delegated,
            "padding0": self.padding0,
            "rewards_tally_scaled": self.rewards_tally_scaled,
            "rewards_issued_unclaimed": self.rewards_issued_unclaimed,
            "last_claim_ts": self.last_claim_ts,
            "active_stake_scaled": self.active_stake_scaled,
            "pending_deposit_stake_scaled": self.pending_deposit_stake_scaled,
            "pending_deposit_stake_ts": self.pending_deposit_stake_ts,
            "pending_withdrawal_unstake_scaled": self.pending_withdrawal_unstake_scaled,
            "pending_withdrawal_unstake_ts": self.pending_withdrawal_unstake_ts,
            "bump": self.bump,
            "delegatee": str(self.delegatee),
            "last_stake_ts": self.last_stake_ts,
            "padding1": self.padding1,
        }

    @classmethod
    def from_json(cls, obj: UserStateJSON) -> "UserState":
        return cls(
            user_id=obj["user_id"],
            farm_state=Pubkey.from_string(obj["farm_state"]),
            owner=Pubkey.from_string(obj["owner"]),
            is_farm_delegated=obj["is_farm_delegated"],
            padding0=obj["padding0"],
            rewards_tally_scaled=obj["rewards_tally_scaled"],
            rewards_issued_unclaimed=obj["rewards_issued_unclaimed"],
            last_claim_ts=obj["last_claim_ts"],
            active_stake_scaled=obj["active_stake_scaled"],
            pending_deposit_stake_scaled=obj["pending_deposit_stake_scaled"],
            pending_deposit_stake_ts=obj["pending_deposit_stake_ts"],
            pending_withdrawal_unstake_scaled=obj["pending_withdrawal_unstake_scaled"],
            pending_withdrawal_unstake_ts=obj["pending_withdrawal_unstake_ts"],
            bump=obj["bump"],
            delegatee=Pubkey.from_string(obj["delegatee"]),
            last_stake_ts=obj["last_stake_ts"],
            padding1=obj["padding1"],
        )

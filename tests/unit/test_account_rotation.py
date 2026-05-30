import asyncio

from aistudio_api.application.account_rotator import AccountRotator, RotationMode
from aistudio_api.infrastructure.account.account_store import AccountStore


def _store_with_accounts(tmp_path, n: int) -> AccountStore:
    store = AccountStore(accounts_dir=tmp_path)
    for i in range(n):
        store.save_account(
            name=f"acc{i}",
            email=f"acc{i}@gmail.com",
            storage_state={"cookies": [], "origins": []},
            account_id=f"acc{i}",
        )
    return store


def test_record_auth_failure_puts_account_in_long_cooldown(tmp_path):
    store = _store_with_accounts(tmp_path, 1)
    rotator = AccountRotator(store, mode=RotationMode.ROUND_ROBIN)

    stats = rotator.get_all_stats()["acc0"]
    assert stats["is_available"] is True

    rotator.record_auth_failure("acc0", cooldown_seconds=600)
    stats = rotator.get_all_stats()["acc0"]
    assert stats["is_available"] is False
    assert stats["cooldown_remaining"] > 0
    assert stats["errors"] == 1


def test_get_next_account_skips_auth_failed_account(tmp_path):
    store = _store_with_accounts(tmp_path, 2)
    rotator = AccountRotator(store, mode=RotationMode.ROUND_ROBIN)

    rotator.record_auth_failure("acc0", cooldown_seconds=600)
    picked = asyncio.run(rotator.get_next_account())
    assert picked is not None
    assert picked.id == "acc1"


def test_success_clears_after_cooldown_naturally(tmp_path):
    store = _store_with_accounts(tmp_path, 1)
    rotator = AccountRotator(store, mode=RotationMode.ROUND_ROBIN)

    # A very short cooldown should free the account up again.
    rotator.record_auth_failure("acc0", cooldown_seconds=0)
    stats = rotator.get_all_stats()["acc0"]
    assert stats["is_available"] is True

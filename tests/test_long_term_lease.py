import asyncio
import unittest

from feedsummary_core.long_term import LeaseLostError, LongTermLeaseHeartbeat


class FakeStore:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def renew_long_term_lease(
        self, profile_id, owner_id, *, now_ts, lease_seconds
    ):
        self.calls.append((profile_id, owner_id, now_ts, lease_seconds))
        outcome = self.outcomes.pop(0) if self.outcomes else True
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class LongTermLeaseHeartbeatTests(unittest.IsolatedAsyncioTestCase):
    async def test_background_heartbeat_renews_claimed_lease(self):
        store = FakeStore([True, True])
        heartbeat = LongTermLeaseHeartbeat(
            store,
            profile_id="profile",
            owner_id="worker",
            lease_seconds=1,
            interval_seconds=0.01,
            clock=lambda: 123,
        )

        await heartbeat.start()
        await asyncio.sleep(0.025)
        await heartbeat.stop()

        self.assertGreaterEqual(len(store.calls), 1)
        self.assertEqual(("profile", "worker", 123, 1), store.calls[0])

    async def test_failed_renewal_is_reported_before_write_guard_returns(self):
        heartbeat = LongTermLeaseHeartbeat(
            FakeStore([False]),
            profile_id="profile",
            owner_id="worker",
            lease_seconds=60,
            interval_seconds=10,
            clock=lambda: 123,
        )

        await heartbeat.start()
        with self.assertRaisesRegex(LeaseLostError, "lease was lost"):
            await heartbeat.ensure_owned()
        await heartbeat.stop()

    async def test_database_error_is_preserved_as_lease_loss_cause(self):
        heartbeat = LongTermLeaseHeartbeat(
            FakeStore([RuntimeError("database unavailable")]),
            profile_id="profile",
            owner_id="worker",
            lease_seconds=60,
            interval_seconds=10,
            clock=lambda: 123,
        )

        await heartbeat.start()
        with self.assertRaises(LeaseLostError) as raised:
            await heartbeat.ensure_owned()
        await heartbeat.stop()

        self.assertIsInstance(raised.exception.__cause__, RuntimeError)


if __name__ == "__main__":
    unittest.main()

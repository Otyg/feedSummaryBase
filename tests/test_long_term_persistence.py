import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tinydb import TinyDB

try:
    import mongomock
except ImportError:  # pragma: no cover - optional test dependency
    mongomock = None

from feedsummary_core.persistence import (
    CleanupPolicy,
    MongoDBStore,
    SqliteStore,
    TinyDBStore,
)


class LongTermStoreContract:
    def make_store(self, directory):
        raise NotImplementedError

    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = self.make_store(Path(self.directory.name))

    def test_article_cursor_uses_fetched_at_then_article_id(self):
        for article_id, fetched_at, source in (
            ("b", 10, "included"),
            ("a", 10, "included"),
            ("c", 20, "other"),
            ("d", 30, "included"),
        ):
            self.store.upsert_article(
                {
                    "id": article_id,
                    "fetched_at": fetched_at,
                    "published_ts": fetched_at - 100,
                    "source": source,
                }
            )

        rows = self.store.list_articles_for_long_term(
            after_fetched_at=10,
            after_article_id="a",
            until_fetched_at=25,
        )
        self.assertEqual(["b", "c"], [row["id"] for row in rows])
        filtered = self.store.list_articles_for_long_term(
            after_fetched_at=0,
            until_fetched_at=30,
            sources=["included"],
        )
        self.assertEqual(["a", "b", "d"], [row["id"] for row in filtered])

    def test_lease_and_cursor_are_owner_checked_and_optimistic(self):
        self.assertTrue(
            self.store.claim_long_term_lease(
                "profile", "worker-a", now_ts=100, lease_seconds=50
            )
        )
        self.assertFalse(
            self.store.claim_long_term_lease(
                "profile", "worker-b", now_ts=120, lease_seconds=50
            )
        )
        self.assertTrue(
            self.store.advance_long_term_cursor(
                "profile",
                "worker-a",
                expected_fetched_at=0,
                expected_article_id="",
                fetched_at=10,
                article_id="article-a",
                now_ts=120,
            )
        )
        self.assertFalse(
            self.store.advance_long_term_cursor(
                "profile",
                "worker-a",
                expected_fetched_at=0,
                expected_article_id="",
                fetched_at=20,
                article_id="article-b",
                now_ts=121,
            )
        )
        cursor = self.store.get_long_term_cursor("profile")
        self.assertEqual((10, "article-a"), self._cursor(cursor))
        self.assertTrue(self.store.release_long_term_lease("profile", "worker-a"))
        self.assertFalse(self.store.release_long_term_lease("profile", "worker-a"))
        self.assertTrue(
            self.store.claim_long_term_lease(
                "profile", "worker-b", now_ts=151, lease_seconds=50
            )
        )

    def test_lease_renewal_requires_current_owner_and_unexpired_lease(self):
        self.assertTrue(
            self.store.claim_long_term_lease(
                "renew-profile", "worker-a", now_ts=100, lease_seconds=50
            )
        )
        self.assertFalse(
            self.store.renew_long_term_lease(
                "renew-profile", "worker-b", now_ts=120, lease_seconds=50
            )
        )
        self.assertTrue(
            self.store.renew_long_term_lease(
                "renew-profile", "worker-a", now_ts=120, lease_seconds=50
            )
        )
        self.assertEqual(
            170, self.store.get_long_term_cursor("renew-profile")["lease_until"]
        )
        self.assertFalse(
            self.store.claim_long_term_lease(
                "renew-profile", "worker-b", now_ts=160, lease_seconds=50
            )
        )
        self.assertFalse(
            self.store.renew_long_term_lease(
                "renew-profile", "worker-a", now_ts=170, lease_seconds=50
            )
        )
        self.assertTrue(
            self.store.claim_long_term_lease(
                "renew-profile", "worker-b", now_ts=170, lease_seconds=50
            )
        )
        self.assertFalse(
            self.store.renew_long_term_lease(
                "renew-profile", "worker-a", now_ts=171, lease_seconds=50
            )
        )

    def test_cluster_revision_and_membership_are_idempotent(self):
        cluster = self.cluster_doc()
        self.assertTrue(self.store.save_threat_cluster(cluster))
        self.assertFalse(self.store.save_threat_cluster(cluster))

        updated = {**cluster, "membership_revision": 2, "member_count": 2}
        self.assertTrue(
            self.store.save_threat_cluster(updated, expected_membership_revision=1)
        )
        self.assertFalse(
            self.store.save_threat_cluster(
                {**updated, "membership_revision": 3},
                expected_membership_revision=1,
            )
        )
        self.assertEqual(2, self.store.get_threat_cluster("cluster-1")["member_count"])
        self.assertEqual(
            ["cluster-1"],
            [
                row["id"]
                for row in self.store.list_threat_clusters(
                    "profile",
                    statuses=["active"],
                    min_last_seen_ts=900,
                    embedding_model="model",
                    embedding_dimension=2,
                    embedding_instruction="instruction",
                )
            ],
        )

        membership = {
            "profile_id": "profile",
            "article_id": "article-a",
            "cluster_id": "cluster-1",
            "assigned_at": 1010,
        }
        self.assertTrue(self.store.save_cluster_membership(membership))
        self.assertFalse(self.store.save_cluster_membership(membership))
        self.assertEqual(
            "cluster-1",
            self.store.get_cluster_membership("profile", "article-a")["cluster_id"],
        )
        self.assertEqual(
            ["article-a"],
            [
                row["article_id"]
                for row in self.store.list_cluster_memberships("cluster-1")
            ],
        )

    def test_cluster_assignment_updates_cluster_and_membership_together(self):
        initial = self.cluster_doc()
        initial["id"] = "cluster-atomic"
        initial["member_count"] = 1
        initial["membership_revision"] = 1
        first_membership = {
            "profile_id": "profile",
            "article_id": "article-first",
            "cluster_id": "cluster-atomic",
            "assigned_at": 1000,
        }
        self.assertTrue(
            self.store.save_cluster_assignment(initial, first_membership)
        )

        updated = {
            **initial,
            "member_count": 2,
            "membership_revision": 2,
            "last_seen_ts": 1100,
        }
        second_membership = {
            "profile_id": "profile",
            "article_id": "article-second",
            "cluster_id": "cluster-atomic",
            "assigned_at": 1100,
        }
        self.assertTrue(
            self.store.save_cluster_assignment(
                updated,
                second_membership,
                expected_membership_revision=1,
            )
        )
        self.assertFalse(
            self.store.save_cluster_assignment(
                updated,
                second_membership,
                expected_membership_revision=1,
            )
        )
        self.assertEqual(
            2,
            self.store.get_threat_cluster("cluster-atomic")["membership_revision"],
        )
        self.assertEqual(
            ["article-first", "article-second"],
            [
                row["article_id"]
                for row in self.store.list_cluster_memberships("cluster-atomic")
            ],
        )

        stale_membership = {
            **second_membership,
            "article_id": "article-stale",
        }
        self.assertFalse(
            self.store.save_cluster_assignment(
                {**updated, "membership_revision": 3, "member_count": 3},
                stale_membership,
                expected_membership_revision=1,
            )
        )
        self.assertIsNone(
            self.store.get_cluster_membership("profile", "article-stale")
        )

    def test_snapshots_reports_and_runs_are_insert_only(self):
        snapshot = {
            "id": "snapshot-1",
            "profile_id": "profile",
            "cluster_id": "cluster-1",
            "membership_revision": 1,
            "prompt_version": "cluster-v1",
            "created_at": 1000,
        }
        self.assertTrue(self.store.save_cluster_snapshot(snapshot))
        self.assertFalse(self.store.save_cluster_snapshot(snapshot))
        self.assertEqual(
            ["snapshot-1"],
            [row["id"] for row in self.store.list_cluster_snapshots("profile")],
        )

        report = {
            "id": "report-1",
            "profile_id": "profile",
            "period_end_ts": 2000,
            "created_at": 2010,
        }
        self.assertTrue(self.store.save_threat_landscape_report(report))
        self.assertFalse(self.store.save_threat_landscape_report(report))
        self.assertEqual(
            "report-1", self.store.get_threat_landscape_report("report-1")["id"]
        )
        self.assertEqual(
            ["report-1"],
            [row["id"] for row in self.store.list_threat_landscape_reports("profile")],
        )

        run = {
            "id": "run-1",
            "profile_id": "profile",
            "run_type": "incremental",
            "started_at": 3000,
            "status": "running",
        }
        self.assertTrue(self.store.create_long_term_run(run))
        self.assertFalse(self.store.create_long_term_run(run))
        self.assertTrue(
            self.store.update_long_term_run(
                "run-1", expected_status="running", fields={"status": "done"}
            )
        )
        self.assertFalse(
            self.store.update_long_term_run(
                "run-1", expected_status="running", fields={"status": "failed"}
            )
        )
        self.assertEqual("done", self.store.get_long_term_run("run-1")["status"])

    def test_summary_cleanup_never_deletes_authoritative_landscape_report(self):
        report = {
            "id": "report-retained",
            "profile_id": "profile",
            "period_end_ts": 1,
            "created_at": 1,
        }
        mirror = {
            "id": "summary_report-retained",
            "created": 1,
            "kind": "threat_landscape",
            "source_report_id": report["id"],
            "summary": "Mirror",
        }
        self.assertTrue(self.store.save_threat_landscape_report(report))
        self.store.save_summary_doc(mirror)

        removed = self.store.run_cleanup(
            CleanupPolicy(daily_summaries_days=1, weekly_summaries_days=1)
        )

        self.assertEqual(1, removed["summary_docs"])
        self.assertIsNone(self.store.get_summary_doc(mirror["id"]))
        self.assertEqual(
            report["id"],
            self.store.get_threat_landscape_report(report["id"])["id"],
        )

    def test_article_quarantine_tracks_retries_and_resolution(self):
        quarantine = {
            "profile_id": "profile",
            "article_id": "article-bad",
            "reason": "similarity_embedding_missing",
            "observed_at": 1000,
            "fetched_at": 900,
        }
        self.assertTrue(self.store.save_long_term_quarantine(quarantine))
        self.assertTrue(
            self.store.save_long_term_quarantine(
                {**quarantine, "observed_at": 1100}
            )
        )
        stored = self.store.get_long_term_quarantine("profile", "article-bad")
        self.assertEqual("open", stored["status"])
        self.assertEqual(1000, stored["first_seen_at"])
        self.assertEqual(1100, stored["last_seen_at"])
        self.assertEqual(2, stored["attempt_count"])
        self.assertEqual(
            ["article-bad"],
            [row["article_id"] for row in self.store.list_long_term_quarantine("profile")],
        )
        self.assertTrue(
            self.store.resolve_long_term_quarantine(
                "profile", "article-bad", resolved_at=1200
            )
        )
        self.assertFalse(
            self.store.resolve_long_term_quarantine(
                "profile", "article-bad", resolved_at=1300
            )
        )
        self.assertEqual(
            [], self.store.list_long_term_quarantine("profile", status="open")
        )
        self.assertEqual(
            ["article-bad"],
            [
                row["article_id"]
                for row in self.store.list_long_term_quarantine(
                    "profile", status="resolved"
                )
            ],
        )

    def test_snapshot_and_summarized_revision_advance_together(self):
        cluster = self.cluster_doc()
        cluster["id"] = "cluster-snapshot"
        cluster["summarized_revision"] = 0
        self.assertTrue(self.store.save_threat_cluster(cluster))
        snapshot = {
            "id": "snapshot-atomic",
            "profile_id": "profile",
            "cluster_id": "cluster-snapshot",
            "membership_revision": 1,
            "prompt_version": "cluster-update-v1",
            "created_at": 1200,
            "payload": {"summary": "Validated"},
        }
        summarized = {
            **cluster,
            "summarized_revision": 1,
            "latest_snapshot_id": "snapshot-atomic",
            "last_summarized_at": 1200,
        }
        self.assertTrue(
            self.store.save_cluster_snapshot_revision(
                summarized,
                snapshot,
                expected_membership_revision=1,
                expected_summarized_revision=0,
            )
        )
        self.assertFalse(
            self.store.save_cluster_snapshot_revision(
                summarized,
                snapshot,
                expected_membership_revision=1,
                expected_summarized_revision=0,
            )
        )
        stored = self.store.get_threat_cluster("cluster-snapshot")
        self.assertEqual(1, stored["summarized_revision"])
        self.assertEqual("snapshot-atomic", stored["latest_snapshot_id"])
        self.assertEqual(
            ["snapshot-atomic"],
            [
                row["id"]
                for row in self.store.list_cluster_snapshots(
                    "profile", cluster_id="cluster-snapshot"
                )
            ],
        )

    @staticmethod
    def _cursor(cursor):
        return cursor["cursor_fetched_at"], cursor["cursor_article_id"]

    @staticmethod
    def cluster_doc():
        return {
            "id": "cluster-1",
            "profile_id": "profile",
            "status": "active",
            "first_seen_ts": 900,
            "last_seen_ts": 1000,
            "embedding_model": "model",
            "embedding_dimension": 2,
            "embedding_instruction": "instruction",
            "centroid": [1.0, 0.0],
            "vector_sum": [1.0, 0.0],
            "member_count": 1,
            "membership_revision": 1,
            "updated_at": 1000,
        }


class SqliteLongTermStoreTests(LongTermStoreContract, unittest.TestCase):
    def make_store(self, directory):
        return SqliteStore(str(directory / "store.sqlite"))


class TinyDBLongTermStoreTests(LongTermStoreContract, unittest.TestCase):
    def make_store(self, directory):
        return TinyDBStore(str(directory / "store.json"))

    def test_pending_assignment_journal_is_replayed(self):
        initial = self.cluster_doc()
        initial["id"] = "cluster-recovery"
        self.assertTrue(self.store.save_threat_cluster(initial))
        target = {
            **initial,
            "membership_revision": 2,
            "member_count": 2,
        }
        membership = {
            "profile_id": "profile",
            "article_id": "article-recovery",
            "cluster_id": "cluster-recovery",
            "assigned_at": 1100,
        }
        self.assertTrue(
            self.store.save_threat_cluster(
                target,
                expected_membership_revision=1,
            )
        )

        database = TinyDB(self.store.path)
        database.table("long_term_assignment_journal").insert(
            {
                "id": "profile:article-recovery",
                "cluster": target,
                "membership": membership,
                "expected_membership_revision": 1,
                "created_at": 1100,
            }
        )
        database.close()

        self.assertFalse(
            self.store.save_cluster_assignment(
                target,
                membership,
                expected_membership_revision=1,
            )
        )
        self.assertEqual(
            "cluster-recovery",
            self.store.get_cluster_membership("profile", "article-recovery")[
                "cluster_id"
            ],
        )
        database = TinyDB(self.store.path)
        self.assertEqual([], database.table("long_term_assignment_journal").all())
        database.close()


@unittest.skipIf(mongomock is None, "mongomock is not installed")
class MongoDBLongTermStoreTests(LongTermStoreContract, unittest.TestCase):
    def make_store(self, directory):
        del directory
        return MongoDBStore(
            database="long_term_contract",
            client=mongomock.MongoClient(),
        )


if __name__ == "__main__":
    unittest.main()

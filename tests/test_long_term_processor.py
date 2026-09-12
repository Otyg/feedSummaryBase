import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

from feedsummary_core.long_term import (
    ClusteringSettings,
    EmbeddingSignature,
    IncrementalSettings,
    LeaseLostError,
    LeaseUnavailableError,
    create_cluster,
    run_incremental_clustering,
)
from feedsummary_core.persistence import SqliteStore


class LongTermProcessorTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SqliteStore(str(Path(self.directory.name) / "store.sqlite"))

    @staticmethod
    def article(article_id, fetched_at, vector):
        return {
            "id": article_id,
            "source": "source-a",
            "fetched_at": fetched_at,
            "published_ts": fetched_at,
            "similarity_embedding_vector": vector,
            "similarity_embedding_model": "embedding-model",
            "similarity_embedding_instruction": "cluster incidents",
        }

    @staticmethod
    def settings():
        return IncrementalSettings(
            batch_size=20,
            lease_seconds=60,
            clustering=ClusteringSettings(
                similarity_threshold=0.90,
                ambiguity_margin=0.02,
                candidate_window_days=60,
            ),
        )

    def test_incremental_batch_clusters_and_advances_cursor(self):
        for article in (
            self.article("article-a", 100, [1.0, 0.0]),
            self.article("article-b", 200, [0.99, 0.01]),
            self.article("article-c", 300, [0.0, 1.0]),
        ):
            self.store.upsert_article(article)

        result = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=500,
            now_ts=500,
            settings=self.settings(),
        )

        self.assertEqual(3, result.assigned)
        self.assertEqual(2, result.created)
        self.assertEqual(1, result.matched)
        self.assertEqual((300, "article-c"), (result.cursor_fetched_at, result.cursor_article_id))
        clusters = self.store.list_threat_clusters("profile")
        self.assertEqual([1, 2], sorted(cluster["member_count"] for cluster in clusters))
        membership = self.store.get_cluster_membership("profile", "article-a")
        self.assertEqual("source-a", membership["evidence"]["source"])
        self.assertEqual(100, membership["evidence"]["published_ts"])
        self.assertEqual("done", self.store.get_long_term_run(result.run_id)["status"])

        replay = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=500,
            now_ts=501,
            settings=self.settings(),
        )
        self.assertEqual(0, replay.fetched)
        self.assertEqual(0, replay.assigned)
        self.assertEqual(2, len(self.store.list_threat_clusters("profile")))

    def test_ambiguous_membership_retains_both_review_candidate_ids(self):
        signature = EmbeddingSignature(
            "embedding-model", 2, "cluster incidents"
        )
        first = create_cluster(
            profile_id="profile",
            article_id="seed-a",
            article_ts=100,
            embedding=[1.0, 0.0],
            signature=signature,
        )
        second = create_cluster(
            profile_id="profile",
            article_id="seed-b",
            article_ts=100,
            embedding=[0.999, 0.045],
            signature=signature,
        )
        self.assertTrue(self.store.save_threat_cluster(first.to_document()))
        self.assertTrue(self.store.save_threat_cluster(second.to_document()))
        self.store.upsert_article(self.article("ambiguous", 200, [1.0, 0.0]))

        result = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=500,
            now_ts=500,
            settings=IncrementalSettings(
                batch_size=20,
                lease_seconds=60,
                clustering=ClusteringSettings(
                    similarity_threshold=0.8,
                    ambiguity_margin=0.01,
                    candidate_window_days=60,
                ),
            ),
        )

        self.assertEqual(1, result.needs_review)
        membership = self.store.get_cluster_membership("profile", "ambiguous")
        self.assertEqual(first.id, membership["best_candidate_cluster_id"])
        self.assertEqual(second.id, membership["second_candidate_cluster_id"])

    def test_clustering_renews_lease_before_cursor_commit(self):
        self.store.upsert_article(self.article("article-a", 100, [1.0, 0.0]))

        with patch.object(
            self.store,
            "renew_long_term_lease",
            wraps=self.store.renew_long_term_lease,
        ) as renew:
            result = run_incremental_clustering(
                self.store,
                profile_id="profile",
                owner_id="worker",
                until_fetched_at=500,
                now_ts=500,
                settings=self.settings(),
            )

        self.assertEqual((100, "article-a"), (result.cursor_fetched_at, result.cursor_article_id))
        self.assertGreaterEqual(renew.call_count, 2)

    def test_lost_lease_blocks_cursor_commit(self):
        self.store.upsert_article(self.article("article-a", 100, [1.0, 0.0]))

        with (
            patch.object(self.store, "renew_long_term_lease", return_value=False),
            self.assertRaisesRegex(LeaseLostError, "lease was lost"),
        ):
            run_incremental_clustering(
                self.store,
                profile_id="profile",
                owner_id="worker",
                until_fetched_at=500,
                now_ts=500,
                settings=self.settings(),
            )

        cursor = self.store.get_long_term_cursor("profile")
        self.assertEqual((0, ""), (cursor["cursor_fetched_at"], cursor["cursor_article_id"]))

    def test_two_concurrent_starts_allow_exactly_one_profile_owner(self):
        entered = Event()
        release = Event()
        original_create_run = self.store.create_long_term_run

        def hold_first_run(document):
            created = original_create_run(document)
            entered.set()
            if not release.wait(2):
                raise RuntimeError("concurrency test timed out")
            return created

        def run(owner):
            return run_incremental_clustering(
                self.store,
                profile_id="profile",
                owner_id=owner,
                until_fetched_at=500,
                now_ts=500,
                settings=self.settings(),
            )

        with (
            patch.object(
                self.store,
                "create_long_term_run",
                side_effect=hold_first_run,
            ),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            first = pool.submit(run, "worker-a")
            self.assertTrue(entered.wait(1))
            second = pool.submit(run, "worker-b")
            with self.assertRaises(LeaseUnavailableError):
                second.result(timeout=1)
            release.set()
            winner = first.result(timeout=1)

        self.assertEqual("profile", winner.profile_id)

    def test_cursor_write_failure_replays_existing_assignment_without_duplicate(self):
        self.store.upsert_article(self.article("article-a", 100, [1.0, 0.0]))

        with (
            patch.object(
                self.store,
                "advance_long_term_cursor",
                side_effect=RuntimeError("cursor database failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "cursor database failure"),
        ):
            run_incremental_clustering(
                self.store,
                profile_id="profile",
                owner_id="worker-a",
                until_fetched_at=500,
                now_ts=500,
                settings=self.settings(),
            )

        self.assertIsNotNone(self.store.get_cluster_membership("profile", "article-a"))
        self.assertEqual(0, self.store.get_long_term_cursor("profile")["cursor_fetched_at"])

        replay = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker-b",
            until_fetched_at=500,
            now_ts=501,
            settings=self.settings(),
        )

        self.assertEqual(0, replay.assigned)
        self.assertEqual(1, replay.existing)
        self.assertEqual((100, "article-a"), (replay.cursor_fetched_at, replay.cursor_article_id))
        clusters = self.store.list_threat_clusters("profile")
        self.assertEqual(1, len(clusters))
        self.assertEqual(1, clusters[0]["member_count"])

    def test_failure_after_cursor_commit_does_not_reprocess_or_duplicate(self):
        self.store.upsert_article(self.article("article-a", 100, [1.0, 0.0]))
        original_update_run = self.store.update_long_term_run
        failed_once = False

        def fail_after_cursor(run_id, *, expected_status, fields):
            nonlocal failed_once
            if fields.get("status") == "done" and not failed_once:
                failed_once = True
                raise RuntimeError("run status database failure")
            return original_update_run(
                run_id,
                expected_status=expected_status,
                fields=fields,
            )

        with (
            patch.object(
                self.store,
                "update_long_term_run",
                side_effect=fail_after_cursor,
            ),
            self.assertRaisesRegex(RuntimeError, "run status database failure"),
        ):
            run_incremental_clustering(
                self.store,
                profile_id="profile",
                owner_id="worker-a",
                until_fetched_at=500,
                now_ts=500,
                settings=self.settings(),
            )

        cursor = self.store.get_long_term_cursor("profile")
        self.assertEqual((100, "article-a"), (cursor["cursor_fetched_at"], cursor["cursor_article_id"]))

        replay = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker-b",
            until_fetched_at=500,
            now_ts=501,
            settings=self.settings(),
        )

        self.assertEqual(0, replay.fetched)
        self.assertEqual(0, replay.assigned)
        clusters = self.store.list_threat_clusters("profile")
        self.assertEqual(1, len(clusters))
        self.assertEqual(1, clusters[0]["member_count"])

    def test_blocked_identity_can_be_corrected_and_resumed_without_cursor_gap(self):
        identityless = self.article("", 100, [1.0, 0.0])

        with patch.object(
            self.store,
            "list_articles_for_long_term",
            return_value=[identityless],
        ):
            blocked = run_incremental_clustering(
                self.store,
                profile_id="profile",
                owner_id="worker-a",
                until_fetched_at=500,
                now_ts=500,
                settings=self.settings(),
            )

        self.assertEqual("article_id_missing", blocked.blocked_reason)
        self.assertEqual((0, ""), (blocked.cursor_fetched_at, blocked.cursor_article_id))
        self.assertEqual("blocked", self.store.get_long_term_run(blocked.run_id)["status"])

        self.store.upsert_article(self.article("article-a", 100, [1.0, 0.0]))
        resumed = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker-b",
            until_fetched_at=500,
            now_ts=501,
            settings=self.settings(),
        )

        self.assertIsNone(resumed.blocked_reason)
        self.assertEqual(1, resumed.assigned)
        self.assertEqual((100, "article-a"), (resumed.cursor_fetched_at, resumed.cursor_article_id))
        self.assertEqual("done", self.store.get_long_term_run(resumed.run_id)["status"])

    def test_embedding_signature_changes_never_mix_cluster_memberships(self):
        original = self.article("article-a", 100, [1.0, 0.0])
        changed_model = self.article("article-b", 200, [1.0, 0.0])
        changed_model["similarity_embedding_model"] = "embedding-model-v2"
        changed_dimension = self.article("article-c", 300, [1.0, 0.0, 0.0])
        changed_instruction = self.article("article-d", 400, [1.0, 0.0])
        changed_instruction["similarity_embedding_instruction"] = "new incident instruction"
        for article in (
            original,
            changed_model,
            changed_dimension,
            changed_instruction,
        ):
            self.store.upsert_article(article)

        result = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=500,
            now_ts=500,
            settings=self.settings(),
        )

        self.assertEqual(4, result.created)
        clusters = self.store.list_threat_clusters("profile")
        signatures = {
            (
                cluster["embedding_model"],
                cluster["embedding_dimension"],
                cluster["embedding_instruction"],
            )
            for cluster in clusters
        }
        self.assertEqual(
            {
                ("embedding-model", 2, "cluster incidents"),
                ("embedding-model-v2", 2, "cluster incidents"),
                ("embedding-model", 3, "cluster incidents"),
                ("embedding-model", 2, "new incident instruction"),
            },
            signatures,
        )
        self.assertEqual([1, 1, 1, 1], sorted(row["member_count"] for row in clusters))

    def test_dry_run_predicts_without_writes_or_cursor_movement(self):
        self.store.upsert_article(self.article("article-a", 100, [1.0, 0.0]))

        result = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="preview",
            until_fetched_at=500,
            now_ts=500,
            settings=self.settings(),
            dry_run=True,
        )

        self.assertEqual(1, result.assigned)
        self.assertEqual(1, result.created)
        self.assertEqual((0, ""), (result.cursor_fetched_at, result.cursor_article_id))
        self.assertEqual([], self.store.list_threat_clusters("profile"))
        self.assertIsNone(self.store.get_cluster_membership("profile", "article-a"))

    def test_invalid_embedding_is_quarantined_and_retried_after_backfill(self):
        self.store.upsert_article(self.article("article-a", 100, [1.0, 0.0]))
        self.store.upsert_article(self.article("article-b", 200, [0.0, 0.0]))

        result = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=500,
            now_ts=500,
            settings=self.settings(),
        )

        self.assertIsNone(result.blocked_article_id)
        self.assertIsNone(result.blocked_reason)
        self.assertEqual(1, result.quarantined)
        self.assertEqual((200, "article-b"), (result.cursor_fetched_at, result.cursor_article_id))
        self.assertIsNotNone(self.store.get_cluster_membership("profile", "article-a"))
        self.assertIsNone(self.store.get_cluster_membership("profile", "article-b"))
        self.assertEqual(
            "open",
            self.store.get_long_term_quarantine("profile", "article-b")["status"],
        )
        self.assertEqual("done", self.store.get_long_term_run(result.run_id)["status"])

        self.store.upsert_article(self.article("article-b", 200, [0.0, 1.0]))
        retry = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=500,
            now_ts=600,
            settings=self.settings(),
        )
        self.assertEqual(1, retry.quarantine_retried)
        self.assertEqual(1, retry.assigned)
        self.assertIsNotNone(self.store.get_cluster_membership("profile", "article-b"))
        self.assertEqual(
            "resolved",
            self.store.get_long_term_quarantine("profile", "article-b")["status"],
        )

    def test_required_tags_filter_articles_without_blocking_cursor(self):
        included = self.article("article-a", 100, [1.0, 0.0])
        excluded = self.article("article-b", 200, [0.0, 1.0])
        self.store.upsert_article(included)
        self.store.upsert_article(excluded)
        tag_id = self.store.add_tag("Healthcare", "GENERAL")
        self.store.add_article_tags("article-a", [tag_id])

        result = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            required_tags=["healthcare"],
            until_fetched_at=500,
            now_ts=500,
            settings=self.settings(),
        )

        self.assertEqual(1, result.assigned)
        self.assertEqual(1, result.filtered)
        self.assertEqual("filtered", result.assignments[1].action)
        self.assertEqual((200, "article-b"), (result.cursor_fetched_at, result.cursor_article_id))
        self.assertIsNone(self.store.get_cluster_membership("profile", "article-b"))

    def test_lifecycle_moves_inactive_clusters_to_dormant_then_closed(self):
        signature = EmbeddingSignature("embedding-model", 2, "cluster incidents")
        cluster = create_cluster(
            profile_id="profile",
            article_id="seed",
            article_ts=100,
            embedding=[1.0, 0.0],
            signature=signature,
        )
        self.assertTrue(self.store.save_threat_cluster(cluster.to_document()))
        settings = IncrementalSettings(
            batch_size=20,
            lease_seconds=60,
            dormant_after_days=1,
            close_after_days=3,
            clustering=self.settings().clustering,
        )

        dormant = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=200_000,
            now_ts=100 + 2 * 86400,
            settings=settings,
        )
        self.assertEqual(1, dormant.lifecycle_updated)
        self.assertEqual(
            "dormant", self.store.get_threat_cluster(cluster.id)["status"]
        )

        closed = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=400_000,
            now_ts=100 + 4 * 86400,
            settings=settings,
        )
        self.assertEqual(1, closed.lifecycle_updated)
        self.assertEqual("closed", self.store.get_threat_cluster(cluster.id)["status"])

    def test_matching_new_evidence_records_dormant_cluster_reopening(self):
        self.store.upsert_article(self.article("article-a", 100, [1.0, 0.0]))
        settings = IncrementalSettings(
            batch_size=20,
            lease_seconds=60,
            dormant_after_days=1,
            close_after_days=30,
            clustering=self.settings().clustering,
        )
        run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=100,
            now_ts=100,
            settings=settings,
        )
        run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=200_000,
            now_ts=100 + 2 * 86400,
            settings=settings,
        )
        dormant = self.store.list_threat_clusters("profile")[0]
        self.assertEqual("dormant", dormant["status"])

        self.store.upsert_article(
            self.article("article-b", 100 + 3 * 86400, [1.0, 0.0])
        )
        run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=100 + 3 * 86400,
            now_ts=100 + 3 * 86400,
            settings=settings,
        )
        reopened = self.store.get_threat_cluster(dormant["id"])
        self.assertEqual("active", reopened["status"])
        self.assertEqual(1, reopened["reopened_count"])
        self.assertEqual(100 + 3 * 86400, reopened["last_reopened_at"])

    def test_conflicting_cves_create_separate_clusters(self):
        first = self.article("article-a", 100, [1.0, 0.0])
        first["title"] = "CVE-2026-1000 - Product vulnerability"
        second = self.article("article-b", 200, [1.0, 0.0])
        second["title"] = "CVE-2026-2000 - Product vulnerability"
        self.store.upsert_article(first)
        self.store.upsert_article(second)

        result = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=500,
            now_ts=500,
            settings=self.settings(),
        )

        self.assertEqual(2, result.created)
        self.assertEqual(0, result.matched)
        clusters = self.store.list_threat_clusters("profile")
        self.assertEqual(2, len(clusters))
        self.assertEqual(
            [["cve:cve-2026-1000"], ["cve:cve-2026-2000"]],
            sorted(cluster["strong_indicators"] for cluster in clusters),
        )

    def test_narrative_event_articles_with_different_cves_can_merge(self):
        first = self.article("article-a", 100, [1.0, 0.0])
        first["title"] = "PaperCut vulnerability actively exploited"
        first["text"] = "PaperCut incident involving CVE-2023-27350"
        second = self.article("article-b", 200, [1.0, 0.0])
        second["title"] = "PaperCut warns of zero-day attacks"
        second["text"] = "The same PaperCut campaign mentions CVE-2023-27351"
        self.store.upsert_article(first)
        self.store.upsert_article(second)

        result = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=500,
            now_ts=500,
            settings=self.settings(),
        )

        self.assertEqual(1, result.created)
        self.assertEqual(1, result.matched)
        cluster = self.store.list_threat_clusters("profile")[0]
        self.assertFalse(cluster["strict_cve_identity"])
        self.assertEqual(2, cluster["member_count"])


if __name__ == "__main__":
    unittest.main()

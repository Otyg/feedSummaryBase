import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tinydb import Query, TinyDB

try:
    import mongomock
except ImportError:  # pragma: no cover - optional test dependency
    mongomock = None

from feedsummary_core.long_term import (
    build_cluster_membership_edit,
    build_cluster_merge_operation,
    build_cluster_review_merge_operation,
    build_cluster_review_resolution,
)
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

    def test_cluster_listing_supports_pagination_without_vectors(self):
        older = {**self.cluster_doc(), "id": "cluster-older"}
        newer = {
            **self.cluster_doc(),
            "id": "cluster-newer",
            "last_seen_ts": 1100,
        }
        self.assertTrue(self.store.save_threat_cluster(older))
        self.assertTrue(self.store.save_threat_cluster(newer))

        rows = self.store.list_threat_clusters(
            "profile",
            limit=1,
            offset=1,
            include_vectors=False,
        )

        self.assertEqual(["cluster-older"], [row["id"] for row in rows])
        self.assertNotIn("centroid", rows[0])
        self.assertNotIn("vector_sum", rows[0])

    def test_cluster_listing_filters_on_minimum_member_count(self):
        single = {**self.cluster_doc(), "id": "cluster-single", "last_seen_ts": 1200}
        multiple = {
            **self.cluster_doc(),
            "id": "cluster-multiple",
            "member_count": 2,
            "membership_revision": 2,
        }
        self.assertTrue(self.store.save_threat_cluster(single))
        self.assertTrue(self.store.save_threat_cluster(multiple))
        rows = self.store.list_threat_clusters(
            "profile", min_member_count=2, limit=1, include_vectors=False
        )
        self.assertEqual(["cluster-multiple"], [row["id"] for row in rows])

    def test_membership_edit_moves_article_atomically_and_is_audited(self):
        source = {
            **self.cluster_doc(), "id": "cluster-source-edit",
            "member_count": 2, "membership_revision": 2,
            "vector_sum": [1.0, 1.0], "centroid": [0.5, 0.5],
            "summarized_revision": 2, "latest_snapshot_id": "old-snapshot",
        }
        target = {**self.cluster_doc(), "id": "cluster-target-edit"}
        source_rows = [
            {"profile_id": "profile", "article_id": "article-stay",
             "cluster_id": source["id"], "assigned_at": 900, "article_ts": 900,
             "cluster_membership_revision": 1, "strict_cve_identity": False},
            {"profile_id": "profile", "article_id": "article-move",
             "cluster_id": source["id"], "assigned_at": 1000, "article_ts": 1000,
             "cluster_membership_revision": 2, "strict_cve_identity": False},
        ]
        target_rows = [
            {"profile_id": "profile", "article_id": "article-target",
             "cluster_id": target["id"], "assigned_at": 950, "article_ts": 950,
             "cluster_membership_revision": 1, "strict_cve_identity": False}
        ]
        self.assertTrue(self.store.save_threat_cluster(source))
        self.assertTrue(self.store.save_threat_cluster(target))
        for row in source_rows + target_rows:
            self.assertTrue(self.store.save_cluster_membership(row))
        operation = build_cluster_membership_edit(
            source, source_rows, target_cluster=target,
            target_memberships=target_rows, article_id="article-move",
            article={
                "id": "article-move", "similarity_embedding_vector": [0.0, 1.0],
                "similarity_embedding_model": "model",
                "similarity_embedding_instruction": "instruction",
            },
            edited_at=1300, edited_by="analyst@example", comment="Same incident.",
        )
        self.assertTrue(self.store.apply_cluster_membership_edit(operation))
        self.assertTrue(self.store.apply_cluster_membership_edit(operation))
        self.assertEqual(
            target["id"],
            self.store.get_cluster_membership("profile", "article-move")["cluster_id"],
        )
        self.assertEqual(1, self.store.get_threat_cluster(source["id"])["member_count"])
        self.assertEqual(2, self.store.get_threat_cluster(target["id"])["member_count"])
        self.assertIsNone(self.store.get_threat_cluster(source["id"])["latest_snapshot_id"])
        self.assertEqual(
            "applied", self.store.get_cluster_membership_edit(operation["id"])["status"]
        )

    def test_cluster_review_resolution_is_optimistic_and_audited(self):
        review = {**self.cluster_doc(), "id": "cluster-review", "status": "needs_review"}
        self.assertTrue(self.store.save_threat_cluster(review))
        resolution = build_cluster_review_resolution(
            review,
            decision="keep_separate",
            reviewed_at=1200,
            reviewed_by="analyst@example",
            comment="Distinct victim and incident window.",
        )

        self.assertTrue(
            self.store.resolve_threat_cluster_review(
                resolution,
                expected_membership_revision=1,
            )
        )
        self.assertFalse(
            self.store.resolve_threat_cluster_review(
                resolution,
                expected_membership_revision=1,
            )
        )
        stored = self.store.get_threat_cluster("cluster-review")
        self.assertEqual("active", stored["status"])
        self.assertEqual("keep_separate", stored["review_decision"])
        self.assertEqual("analyst@example", stored["reviewed_by"])

    def test_review_merge_preserves_decision_on_tombstone_and_operation(self):
        target = {**self.cluster_doc(), "id": "cluster-review-target"}
        review = {
            **self.cluster_doc(),
            "id": "cluster-review-source",
            "status": "needs_review",
        }
        memberships = {
            target["id"]: {
                "profile_id": "profile",
                "article_id": "article-review-target",
                "cluster_id": target["id"],
                "assigned_at": 1000,
                "evidence": {"title": "Target event"},
            },
            review["id"]: {
                "profile_id": "profile",
                "article_id": "article-review-source",
                "cluster_id": review["id"],
                "assigned_at": 1001,
                "evidence": {"title": "Ambiguous event"},
            },
        }
        self.assertTrue(self.store.save_cluster_assignment(target, memberships[target["id"]]))
        self.assertTrue(self.store.save_cluster_assignment(review, memberships[review["id"]]))
        operation = build_cluster_review_merge_operation(
            review,
            target,
            {cluster_id: [row] for cluster_id, row in memberships.items()},
            allowed_candidate_cluster_ids=[target["id"]],
            reviewed_at=1200,
            reviewed_by="analyst@example",
        )

        self.assertTrue(self.store.apply_cluster_reconciliation(operation))
        tombstone = self.store.get_threat_cluster(review["id"])
        self.assertEqual("merge", tombstone["review_decision"])
        self.assertEqual(target["id"], tombstone["reviewed_target_cluster_id"])
        stored = self.store.get_cluster_reconciliation(operation["id"])
        self.assertEqual("analyst@example", stored["review_resolution"]["reviewed_by"])

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

    def test_cluster_reconciliation_is_atomic_idempotent_and_preserves_snapshots(self):
        first = {**self.cluster_doc(), "id": "cluster-primary"}
        second = {
            **self.cluster_doc(),
            "id": "cluster-secondary",
            "first_seen_ts": 950,
            "last_seen_ts": 1050,
            "centroid": [0.0, 1.0],
            "vector_sum": [0.0, 1.0],
        }
        first_membership = {
            "profile_id": "profile",
            "article_id": "article-primary",
            "cluster_id": first["id"],
            "assigned_at": 1000,
            "strict_cve_identity": False,
            "strong_indicators": ["organization:papercut"],
            "evidence": {"title": "PaperCut incident"},
        }
        second_membership = {
            "profile_id": "profile",
            "article_id": "article-secondary",
            "cluster_id": second["id"],
            "assigned_at": 1050,
            "strict_cve_identity": False,
            "strong_indicators": ["product:papercut mf/ng"],
            "evidence": {"title": "PaperCut exploit"},
        }
        self.assertTrue(self.store.save_cluster_assignment(first, first_membership))
        self.assertTrue(self.store.save_cluster_assignment(second, second_membership))
        self.assertTrue(
            self.store.save_cluster_snapshot(
                {
                    "id": "snapshot-secondary",
                    "profile_id": "profile",
                    "cluster_id": second["id"],
                    "membership_revision": 1,
                    "prompt_version": "cluster-update-v1",
                    "created_at": 1100,
                }
            )
        )
        operation = build_cluster_merge_operation(
            [first, second],
            {
                first["id"]: [first_membership],
                second["id"]: [second_membership],
            },
            primary_cluster_id=first["id"],
            reconciliation_id="reconciliation-1",
            reconciled_at=1200,
        )

        self.assertTrue(self.store.apply_cluster_reconciliation(operation))
        self.assertTrue(self.store.apply_cluster_reconciliation(operation))
        primary = self.store.get_threat_cluster(first["id"])
        secondary = self.store.get_threat_cluster(second["id"])
        self.assertEqual(2, primary["member_count"])
        self.assertEqual(3, primary["membership_revision"])
        self.assertIsNone(primary["latest_snapshot_id"])
        self.assertEqual(0, secondary["member_count"])
        self.assertEqual(first["id"], secondary["superseded_by_cluster_id"])
        self.assertEqual(
            ["article-primary", "article-secondary"],
            [
                row["article_id"]
                for row in self.store.list_cluster_memberships(first["id"])
            ],
        )
        moved = self.store.get_cluster_membership("profile", "article-secondary")
        self.assertEqual(second["id"], moved["previous_cluster_id"])
        self.assertEqual("reconciliation-1", moved["reconciliation_id"])
        revisions = sorted(
            row["cluster_membership_revision"]
            for row in self.store.list_cluster_memberships(first["id"])
        )
        self.assertEqual([2, 3], revisions)
        self.assertIsNotNone(self.store.get_cluster_snapshot("snapshot-secondary"))
        stored_operation = self.store.get_cluster_reconciliation("reconciliation-1")
        self.assertEqual("applied", stored_operation["status"])
        self.assertEqual(2, stored_operation["membership_count"])

    def test_cluster_reconciliation_rejects_a_stale_source_revision(self):
        operation = self.reconciliation_fixture("stale")
        secondary_id = next(
            value
            for value in operation["source_cluster_ids"]
            if value != operation["primary_cluster_id"]
        )
        secondary = self.store.get_threat_cluster(secondary_id)
        secondary["membership_revision"] = 2
        self.assertTrue(
            self.store.save_threat_cluster(
                secondary, expected_membership_revision=1
            )
        )

        self.assertFalse(self.store.apply_cluster_reconciliation(operation))
        self.assertEqual(
            secondary_id,
            self.store.get_cluster_membership(
                "profile", "article-secondary-stale"
            )["cluster_id"],
        )
        self.assertEqual(
            1,
            self.store.get_threat_cluster(operation["primary_cluster_id"])[
                "member_count"
            ],
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
        self.assertEqual(
            "snapshot-1",
            self.store.get_cluster_snapshot("snapshot-1")["id"],
        )
        self.assertIsNone(self.store.get_cluster_snapshot("missing-snapshot"))

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
        self.assertEqual(
            ["run-1"],
            [row["id"] for row in self.store.list_long_term_runs("profile")],
        )

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

    def test_long_term_cleanup_preserves_live_and_referenced_provenance(self):
        now = int(time.time())
        old = now - 800 * 86400

        def save_cluster(cluster_id, *, status="closed"):
            cluster = {
                **self.cluster_doc(),
                "id": cluster_id,
                "status": status,
                "first_seen_ts": old,
                "last_seen_ts": old,
                "created_at": old,
                "updated_at": old,
            }
            membership = {
                "profile_id": "profile",
                "article_id": f"article-{cluster_id}",
                "cluster_id": cluster_id,
                "assigned_at": old,
            }
            self.assertTrue(self.store.save_cluster_assignment(cluster, membership))

        save_cluster("cluster-expired")
        save_cluster("cluster-referenced")
        save_cluster("cluster-active", status="active")
        for cluster_id in ("cluster-expired", "cluster-referenced"):
            self.assertTrue(
                self.store.save_cluster_snapshot(
                    {
                        "id": f"snapshot-{cluster_id}",
                        "profile_id": "profile",
                        "cluster_id": cluster_id,
                        "membership_revision": 1,
                        "prompt_version": "cluster-v1",
                        "created_at": old,
                    }
                )
            )
        self.assertTrue(
            self.store.save_threat_landscape_report(
                {
                    "id": "report-expired",
                    "profile_id": "profile",
                    "period_end_ts": old,
                    "created_at": old,
                    "input_snapshot_ids": ["snapshot-cluster-expired"],
                }
            )
        )
        self.assertTrue(
            self.store.save_threat_landscape_report(
                {
                    "id": "report-current",
                    "profile_id": "profile",
                    "period_end_ts": now,
                    "created_at": now,
                    "input_snapshot_ids": ["snapshot-cluster-referenced"],
                }
            )
        )

        for run_id, status in (("run-expired", "done"), ("run-running", "running")):
            self.assertTrue(
                self.store.create_long_term_run(
                    {
                        "id": run_id,
                        "profile_id": "profile",
                        "run_type": "incremental",
                        "started_at": old,
                        "status": "running",
                    }
                )
            )
            if status == "done":
                self.assertTrue(
                    self.store.update_long_term_run(
                        run_id,
                        expected_status="running",
                        fields={"status": status, "finished_at": old},
                    )
                )

        for article_id in ("quarantine-expired", "quarantine-open"):
            self.assertTrue(
                self.store.save_long_term_quarantine(
                    {
                        "profile_id": "profile",
                        "article_id": article_id,
                        "reason": "missing_embedding",
                        "observed_at": old,
                    }
                )
            )
        self.assertTrue(
            self.store.resolve_long_term_quarantine(
                "profile", "quarantine-expired", resolved_at=old
            )
        )

        removed = self.store.run_cleanup(CleanupPolicy(long_term_days=730))

        self.assertEqual(1, removed["long_term_reports"])
        self.assertEqual(1, removed["long_term_snapshots"])
        self.assertEqual(1, removed["long_term_clusters"])
        self.assertEqual(1, removed["long_term_memberships"])
        self.assertEqual(1, removed["long_term_runs"])
        self.assertEqual(1, removed["long_term_quarantine"])
        self.assertIsNone(self.store.get_threat_cluster("cluster-expired"))
        self.assertIsNotNone(self.store.get_threat_cluster("cluster-referenced"))
        self.assertIsNotNone(self.store.get_threat_cluster("cluster-active"))
        self.assertIsNotNone(
            self.store.get_threat_landscape_report("report-current")
        )
        self.assertEqual(
            ["snapshot-cluster-referenced"],
            [
                row["id"]
                for row in self.store.list_cluster_snapshots("profile", limit=10)
            ],
        )
        self.assertIsNotNone(self.store.get_long_term_run("run-running"))
        self.assertIsNotNone(
            self.store.get_long_term_quarantine("profile", "quarantine-open")
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

    def reconciliation_fixture(self, suffix):
        first = {**self.cluster_doc(), "id": f"cluster-primary-{suffix}"}
        second = {
            **self.cluster_doc(),
            "id": f"cluster-secondary-{suffix}",
            "centroid": [0.0, 1.0],
            "vector_sum": [0.0, 1.0],
        }
        first_membership = {
            "profile_id": "profile",
            "article_id": f"article-primary-{suffix}",
            "cluster_id": first["id"],
            "assigned_at": 1000,
            "strict_cve_identity": False,
            "evidence": {"title": "PaperCut incident"},
        }
        second_membership = {
            "profile_id": "profile",
            "article_id": f"article-secondary-{suffix}",
            "cluster_id": second["id"],
            "assigned_at": 1001,
            "strict_cve_identity": False,
            "evidence": {"title": "PaperCut exploit"},
        }
        self.assertTrue(
            self.store.save_cluster_assignment(first, first_membership)
        )
        self.assertTrue(
            self.store.save_cluster_assignment(second, second_membership)
        )
        return build_cluster_merge_operation(
            [first, second],
            {
                first["id"]: [first_membership],
                second["id"]: [second_membership],
            },
            primary_cluster_id=first["id"],
            reconciliation_id=f"reconciliation-{suffix}",
            reconciled_at=1200,
        )


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
        self.assertIsNone(
            self.store.get_cluster_membership("profile", "article-recovery")
        )
        database = TinyDB(self.store.path)
        self.assertEqual(1, len(database.table("long_term_assignment_journal").all()))
        database.table("threat_clusters").update(
            {"last_assignment_operation_id": "profile:article-recovery"},
            Query().id == "cluster-recovery",
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

    def test_pending_snapshot_is_hidden_until_revision_commits(self):
        database = TinyDB(self.store.path)
        database.table("threat_cluster_snapshots").insert(
            {
                "id": "snapshot-pending",
                "profile_id": "profile",
                "cluster_id": "cluster-pending",
                "membership_revision": 1,
                "prompt_version": "cluster-update-v1",
                "created_at": 1200,
                "pending_operation_id": "cluster-pending:1:cluster-update-v1",
            }
        )
        database.close()

        self.assertEqual([], self.store.list_cluster_snapshots("profile"))
        self.assertIsNone(self.store.get_cluster_snapshot("snapshot-pending"))

    def test_partial_reconciliation_journal_is_replayed(self):
        operation = self.reconciliation_fixture("tiny-recovery")
        database = TinyDB(self.store.path)
        query = Query()
        database.table("long_term_reconciliation_journal").insert(operation)
        targets = [
            operation["merged_cluster"],
            *operation["superseded_clusters"],
        ]
        for target in targets:
            def replace_document(row, replacement=target):
                row.clear()
                row.update(replacement)

            database.table("threat_clusters").update(
                replace_document, query.id == target["id"]
            )
        database.close()

        self.assertTrue(self.store.apply_cluster_reconciliation(operation))
        database = TinyDB(self.store.path)
        self.assertEqual([], database.table("long_term_reconciliation_journal").all())
        database.close()
        moved = self.store.get_cluster_membership(
            "profile", "article-secondary-tiny-recovery"
        )
        self.assertEqual(operation["primary_cluster_id"], moved["cluster_id"])


@unittest.skipIf(mongomock is None, "mongomock is not installed")
class MongoDBLongTermStoreTests(LongTermStoreContract, unittest.TestCase):
    def make_store(self, directory):
        del directory
        return MongoDBStore(
            database="long_term_contract",
            client=mongomock.MongoClient(),
        )

    def test_partial_reconciliation_journal_is_replayed(self):
        operation = self.reconciliation_fixture("mongo-recovery")
        journal = {
            **operation,
            "_id": operation["id"],
            "created_at": operation["reconciled_at"],
        }
        self.store.db.long_term_reconciliation_journal.insert_one(journal)
        for target in [
            operation["merged_cluster"],
            *operation["superseded_clusters"],
        ]:
            self.store.db.threat_clusters.replace_one(
                {"_id": target["id"]}, {**target, "_id": target["id"]}
            )

        self.assertTrue(self.store.apply_cluster_reconciliation(operation))
        self.assertIsNone(
            self.store.db.long_term_reconciliation_journal.find_one(
                {"_id": operation["id"]}
            )
        )
        moved = self.store.get_cluster_membership(
            "profile", "article-secondary-mongo-recovery"
        )
        self.assertEqual(operation["primary_cluster_id"], moved["cluster_id"])

    def test_pending_assignment_journal_requires_matching_replay_marker(self):
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

        self.store.db.long_term_assignment_journal.insert_one(
            {
                "_id": "profile:article-recovery",
                "id": "profile:article-recovery",
                "cluster": target,
                "membership": membership,
                "expected_membership_revision": 1,
                "created_at": 1100,
            }
        )

        self.assertFalse(
            self.store.save_cluster_assignment(
                target,
                membership,
                expected_membership_revision=1,
            )
        )
        self.assertIsNone(
            self.store.get_cluster_membership("profile", "article-recovery")
        )
        self.assertIsNotNone(
            self.store.db.long_term_assignment_journal.find_one(
                {"_id": "profile:article-recovery"}
            )
        )

        self.store.db.threat_clusters.update_one(
            {"_id": "cluster-recovery"},
            {"$set": {"last_assignment_operation_id": "profile:article-recovery"}},
        )
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
        self.assertIsNone(
            self.store.db.long_term_assignment_journal.find_one(
                {"_id": "profile:article-recovery"}
            )
        )

    def test_pending_snapshot_is_hidden_until_revision_commits(self):
        self.store.db.threat_cluster_snapshots.insert_one(
            {
                "_id": "snapshot-pending",
                "id": "snapshot-pending",
                "profile_id": "profile",
                "cluster_id": "cluster-pending",
                "membership_revision": 1,
                "prompt_version": "cluster-update-v1",
                "created_at": 1200,
                "pending_operation_id": "cluster-pending:1:cluster-update-v1",
            }
        )

        self.assertEqual([], self.store.list_cluster_snapshots("profile"))
        self.assertIsNone(self.store.get_cluster_snapshot("snapshot-pending"))


if __name__ == "__main__":
    unittest.main()

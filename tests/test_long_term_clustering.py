import math
import unittest
from dataclasses import replace

from feedsummary_core.long_term import (
    AssignmentAction,
    ClusteringSettings,
    ClusterStatus,
    EmbeddingSignature,
    VectorValidationError,
    add_cluster_member,
    assign_article,
    cluster_status_at,
    cosine_similarity,
    create_cluster,
    stable_cluster_id,
)


class LongTermClusteringTests(unittest.TestCase):
    def setUp(self):
        self.signature = EmbeddingSignature("embedding-model", 2, "Represent the event")

    def cluster(self, article_id, vector, ts=1_000_000, profile="healthcare_europe"):
        return create_cluster(
            profile_id=profile,
            article_id=article_id,
            article_ts=ts,
            embedding=vector,
            signature=self.signature,
        )

    def test_cosine_similarity_and_vector_validation(self):
        self.assertAlmostEqual(1.0, cosine_similarity([2, 0], [4, 0]))
        self.assertAlmostEqual(0.0, cosine_similarity([1, 0], [0, 1]))
        with self.assertRaises(VectorValidationError):
            cosine_similarity([0, 0], [1, 0])
        with self.assertRaises(VectorValidationError):
            cosine_similarity([math.nan, 0], [1, 0])
        with self.assertRaises(VectorValidationError):
            cosine_similarity([1, 0], [1, 0, 0])

    def test_cluster_id_is_stable_and_scoped(self):
        first = stable_cluster_id("profile", "article")
        self.assertEqual(first, stable_cluster_id("profile", "article"))
        self.assertNotEqual(first, stable_cluster_id("other", "article"))

    def test_member_update_keeps_exact_vector_sum_and_centroid(self):
        cluster = self.cluster("a", [2, 0])
        updated = add_cluster_member(cluster, article_ts=1_000_100, embedding=[0, 5])
        self.assertEqual((1.0, 1.0), updated.vector_sum)
        self.assertEqual((0.5, 0.5), updated.centroid)
        self.assertEqual(2, updated.member_count)
        self.assertEqual(2, updated.membership_revision)
        self.assertEqual(1_000_100, updated.last_seen_ts)

    def test_adding_to_a_dormant_cluster_records_reopening(self):
        dormant = replace(
            self.cluster("a", [1, 0]),
            status=ClusterStatus.DORMANT,
        )
        reopened = add_cluster_member(
            dormant,
            article_ts=1_000_100,
            embedding=[1, 0],
        )

        self.assertEqual(ClusterStatus.ACTIVE, reopened.status)
        self.assertEqual(1, reopened.reopened_count)
        self.assertEqual(1_000_100, reopened.last_reopened_at)

    def test_lifecycle_transitions_are_time_based_and_terminal(self):
        cluster = self.cluster("seed", [1.0, 0.0], ts=1_000_000)
        self.assertEqual(
            ClusterStatus.ACTIVE,
            cluster_status_at(
                cluster,
                now_ts=1_000_000 + 29 * 86400,
                dormant_after_days=30,
                close_after_days=180,
            ),
        )
        self.assertEqual(
            ClusterStatus.DORMANT,
            cluster_status_at(
                cluster,
                now_ts=1_000_000 + 30 * 86400,
                dormant_after_days=30,
                close_after_days=180,
            ),
        )
        self.assertEqual(
            ClusterStatus.CLOSED,
            cluster_status_at(
                cluster,
                now_ts=1_000_000 + 180 * 86400,
                dormant_after_days=30,
                close_after_days=180,
            ),
        )
        review = replace(cluster, status=ClusterStatus.NEEDS_REVIEW)
        self.assertEqual(
            ClusterStatus.NEEDS_REVIEW,
            cluster_status_at(
                review,
                now_ts=1_000_000 + 365 * 86400,
                dormant_after_days=30,
                close_after_days=180,
            ),
        )

    def test_cluster_document_round_trip_flattens_embedding_signature(self):
        cluster = self.cluster("a", [1, 0])
        document = cluster.to_document()

        self.assertEqual("active", document["status"])
        self.assertEqual("embedding-model", document["embedding_model"])
        self.assertEqual(cluster, cluster.from_document(document))

    def test_assignment_matches_best_compatible_cluster(self):
        best = self.cluster("best", [1, 0])
        other = self.cluster("other", [0, 1])
        decision = assign_article(
            profile_id="healthcare_europe",
            article_ts=1_000_010,
            embedding=[0.99, 0.01],
            signature=self.signature,
            candidates=[other, best],
            settings=ClusteringSettings(similarity_threshold=0.8),
        )
        self.assertEqual(AssignmentAction.MATCH, decision.action)
        self.assertEqual(best.id, decision.cluster_id)

    def test_assignment_rejects_wrong_signature_profile_status_and_age(self):
        wrong_profile = self.cluster("wrong-profile", [1, 0], profile="other")
        wrong_signature = create_cluster(
            profile_id="healthcare_europe",
            article_id="wrong-signature",
            article_ts=1_000_000,
            embedding=[1, 0, 0],
            signature=EmbeddingSignature("other-model", 3, "Other"),
        )
        closed = self.cluster("closed", [1, 0])
        closed = closed.__class__(**{**closed.__dict__, "status": ClusterStatus.CLOSED})
        old = self.cluster("old", [1, 0], ts=1)
        decision = assign_article(
            profile_id="healthcare_europe",
            article_ts=10_000_000,
            embedding=[1, 0],
            signature=self.signature,
            candidates=[wrong_profile, wrong_signature, closed, old],
            settings=ClusteringSettings(candidate_window_days=1),
        )
        self.assertEqual(AssignmentAction.NEW_CLUSTER, decision.action)
        self.assertEqual("no_compatible_candidate", decision.reason)

    def test_close_candidates_are_quarantined_as_ambiguous(self):
        first = self.cluster("first", [1.0, 0.0])
        second = self.cluster("second", [0.999, 0.045])
        decision = assign_article(
            profile_id="healthcare_europe",
            article_ts=1_000_010,
            embedding=[1, 0],
            signature=self.signature,
            candidates=[first, second],
            settings=ClusteringSettings(
                similarity_threshold=0.8,
                ambiguity_margin=0.01,
            ),
        )
        self.assertEqual(AssignmentAction.NEEDS_REVIEW, decision.action)
        self.assertEqual("ambiguous_candidates", decision.reason)
        self.assertEqual(first.id, decision.best_candidate_cluster_id)
        self.assertEqual(second.id, decision.second_candidate_cluster_id)

    def test_conflicting_cves_prevent_high_similarity_merge(self):
        candidate = create_cluster(
            profile_id="healthcare_europe",
            article_id="seed",
            article_ts=1_000_000,
            embedding=[1.0, 0.0],
            signature=self.signature,
            strong_indicators=["cve:cve-2026-1000"],
        )
        conflicting = assign_article(
            profile_id="healthcare_europe",
            article_ts=1_000_100,
            embedding=[1.0, 0.0],
            signature=self.signature,
            candidates=[candidate],
            strong_indicators=["cve:cve-2026-2000"],
        )
        self.assertEqual(AssignmentAction.NEW_CLUSTER, conflicting.action)
        self.assertEqual("conflicting_cve_indicators", conflicting.reason)

        supported = assign_article(
            profile_id="healthcare_europe",
            article_ts=1_000_100,
            embedding=[1.0, 0.0],
            signature=self.signature,
            candidates=[candidate],
            strong_indicators=["cve:cve-2026-1000"],
        )
        self.assertEqual(AssignmentAction.MATCH, supported.action)
        self.assertEqual(
            "similarity_match_with_indicator_support", supported.reason
        )

    def test_narrative_articles_can_match_despite_disjoint_cves(self):
        candidate = create_cluster(
            profile_id="healthcare_europe",
            article_id="papercut-seed",
            article_ts=1_000_000,
            embedding=[1.0, 0.0],
            signature=self.signature,
            strong_indicators=["cve:cve-2023-27350", "organization:papercut"],
            strict_cve_identity=False,
        )
        decision = assign_article(
            profile_id="healthcare_europe",
            article_ts=1_000_100,
            embedding=[1.0, 0.0],
            signature=self.signature,
            candidates=[candidate],
            strong_indicators=["cve:cve-2023-27351", "organization:papercut"],
            strict_cve_identity=False,
        )
        self.assertEqual(AssignmentAction.MATCH, decision.action)
        self.assertEqual(
            "similarity_match_with_indicator_support", decision.reason
        )


if __name__ == "__main__":
    unittest.main()

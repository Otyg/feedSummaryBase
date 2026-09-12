import unittest

from feedsummary_core.long_term import (
    build_cluster_review_merge_operation,
    build_cluster_review_resolution,
)


def _cluster(cluster_id, *, status, vector):
    return {
        "id": cluster_id,
        "profile_id": "profile",
        "status": status,
        "first_seen_ts": 1000,
        "last_seen_ts": 1100,
        "embedding_model": "model",
        "embedding_dimension": 2,
        "embedding_instruction": "event",
        "centroid": vector,
        "vector_sum": vector,
        "member_count": 1,
        "membership_revision": 1,
        "algorithm_version": "online-centroid-v1",
    }


def _membership(cluster_id, article_id):
    return {
        "profile_id": "profile",
        "cluster_id": cluster_id,
        "article_id": article_id,
        "article_ts": 1000,
        "assigned_at": 1100,
        "evidence": {"title": article_id},
    }


class LongTermReviewTests(unittest.TestCase):
    def test_keep_and_exclude_build_audited_terminal_decisions(self):
        review = _cluster("review", status="needs_review", vector=[1.0, 0.0])

        kept = build_cluster_review_resolution(
            review,
            decision="keep_separate",
            reviewed_at=1200,
            reviewed_by="analyst",
            comment="Different victim",
        )
        excluded = build_cluster_review_resolution(
            review,
            decision="exclude",
            reviewed_at=1201,
            reviewed_by="analyst",
        )

        self.assertEqual("active", kept["status"])
        self.assertEqual("keep_separate", kept["review_decision"])
        self.assertEqual("Different victim", kept["review_comment"])
        self.assertEqual("closed", excluded["status"])
        self.assertEqual("exclude", excluded["review_decision"])

    def test_merge_requires_an_allowed_open_candidate_and_records_audit(self):
        review = _cluster("review", status="needs_review", vector=[1.0, 0.0])
        target = _cluster("target", status="active", vector=[0.9, 0.1])
        memberships = {
            "review": [_membership("review", "article-review")],
            "target": [_membership("target", "article-target")],
        }

        operation = build_cluster_review_merge_operation(
            review,
            target,
            memberships,
            allowed_candidate_cluster_ids=["target"],
            reviewed_at=1200,
            reviewed_by="analyst",
            comment="Same incident",
        )

        self.assertEqual("target", operation["primary_cluster_id"])
        self.assertEqual("review", operation["review_resolution"]["review_cluster_id"])
        tombstone = operation["superseded_clusters"][0]
        self.assertEqual("merge", tombstone["review_decision"])
        self.assertEqual("target", tombstone["reviewed_target_cluster_id"])

        with self.assertRaisesRegex(ValueError, "approved review candidate"):
            build_cluster_review_merge_operation(
                review,
                target,
                memberships,
                allowed_candidate_cluster_ids=["different"],
                reviewed_at=1200,
                reviewed_by="analyst",
            )


if __name__ == "__main__":
    unittest.main()

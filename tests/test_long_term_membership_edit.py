import unittest

from feedsummary_core.long_term import build_cluster_membership_edit


class ClusterMembershipEditTests(unittest.TestCase):
    def setUp(self):
        self.cluster = {
            "id": "cluster-source",
            "profile_id": "profile",
            "status": "active",
            "first_seen_ts": 100,
            "last_seen_ts": 200,
            "embedding_model": "model",
            "embedding_dimension": 2,
            "embedding_instruction": "instruction",
            "centroid": [0.5, 0.5],
            "vector_sum": [1.0, 1.0],
            "member_count": 2,
            "membership_revision": 2,
            "summarized_revision": 2,
            "latest_snapshot_id": "snapshot-before-edit",
        }
        self.memberships = [
            {
                "profile_id": "profile", "article_id": "first",
                "cluster_id": "cluster-source", "article_ts": 100,
                "assigned_at": 100, "cluster_membership_revision": 1,
            },
            {
                "profile_id": "profile", "article_id": "second",
                "cluster_id": "cluster-source", "article_ts": 200,
                "assigned_at": 200, "cluster_membership_revision": 2,
            },
        ]
        self.article = {
            "id": "second",
            "similarity_embedding_vector": [0.0, 1.0],
            "similarity_embedding_model": "model",
            "similarity_embedding_instruction": "instruction",
        }

    def test_split_creates_audited_singleton_and_rebuilds_source(self):
        operation = build_cluster_membership_edit(
            self.cluster,
            self.memberships,
            article=self.article,
            article_id="second",
            edited_at=300,
            edited_by="Analyst",
            comment="Different event",
        )

        self.assertEqual("split", operation["action"])
        self.assertEqual(1, operation["source_cluster"]["member_count"])
        self.assertIsNone(operation["source_cluster"]["latest_snapshot_id"])
        self.assertEqual(1, operation["target_cluster"]["member_count"])
        self.assertEqual(
            operation["target_cluster_id"],
            next(row for row in operation["memberships"] if row["article_id"] == "second")[
                "cluster_id"
            ],
        )
        self.assertEqual("Analyst", operation["edited_by"])

    def test_singleton_cannot_be_split_again(self):
        singleton = {
            **self.cluster,
            "member_count": 1,
            "membership_revision": 1,
            "summarized_revision": 1,
            "vector_sum": [1.0, 0.0],
            "centroid": [1.0, 0.0],
        }
        with self.assertRaisesRegex(ValueError, "already separate"):
            build_cluster_membership_edit(
                singleton,
                self.memberships[:1],
                article={**self.article, "id": "first", "similarity_embedding_vector": [1.0, 0.0]},
                article_id="first",
                edited_at=300,
                edited_by="Analyst",
            )

    def test_split_preserves_tag_scope_and_move_rejects_another_scope(self):
        source = {**self.cluster, "tag_scope": "MALWARE:malware a"}
        memberships = [
            {**row, "tag_scope": "MALWARE:malware a"} for row in self.memberships
        ]
        operation = build_cluster_membership_edit(
            source, memberships, article=self.article, article_id="second",
            edited_at=300, edited_by="Analyst",
        )
        self.assertEqual("MALWARE:malware a", operation["target_cluster"]["tag_scope"])

        other = {
            **self.cluster, "id": "other", "tag_scope": "MALWARE:malware b",
            "member_count": 1, "membership_revision": 1,
            "summarized_revision": 1,
            "vector_sum": [1.0, 0.0], "centroid": [1.0, 0.0],
        }
        with self.assertRaisesRegex(ValueError, "tag scope"):
            build_cluster_membership_edit(
                source, memberships, target_cluster=other,
                target_memberships=[{"article_id": "other", "article_ts": 100}],
                article=self.article, article_id="second",
                edited_at=300, edited_by="Analyst",
            )


if __name__ == "__main__":
    unittest.main()

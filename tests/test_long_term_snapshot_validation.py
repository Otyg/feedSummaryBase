import json
import unittest

from feedsummary_core.long_term import (
    SnapshotValidationError,
    parse_snapshot_json,
    validate_cluster_snapshot,
)


class ClusterSnapshotValidationTests(unittest.TestCase):
    @staticmethod
    def payload():
        return {
            "schema_version": 1,
            "profile_id": "profile",
            "cluster_id": "cluster-1",
            "membership_revision": 2,
            "title": "Aktiv exploatering",
            "summary": "Två källor beskriver samma händelse.",
            "confidence": "medium",
            "insufficient_evidence": False,
            "facts": [
                {
                    "statement": "Sårbarheten exploateras aktivt.",
                    "status": "active",
                    "evidence_article_ids": ["article-a", "article-b"],
                }
            ],
            "timeline": [
                {
                    "event_time": "2026-09-01",
                    "statement": "Exploatering observerades.",
                    "evidence_article_ids": ["article-a"],
                }
            ],
            "mitre_techniques": [
                {
                    "technique_id": "T1190",
                    "statement": "Exploit Public-Facing Application anges i källan.",
                    "evidence_article_ids": ["article-b"],
                }
            ],
            "uncertainties": ["Kampanjens omfattning är okänd."],
        }

    def test_valid_snapshot_is_accepted(self):
        payload = self.payload()
        result = validate_cluster_snapshot(
            payload,
            profile_id="profile",
            cluster_id="cluster-1",
            membership_revision=2,
            allowed_article_ids={"article-a", "article-b"},
        )
        self.assertEqual(payload, result)
        self.assertEqual(payload, parse_snapshot_json(json.dumps(payload)))

    def test_unknown_evidence_is_rejected(self):
        payload = self.payload()
        payload["facts"][0]["evidence_article_ids"] = ["article-x"]
        with self.assertRaisesRegex(SnapshotValidationError, "unknown article IDs"):
            validate_cluster_snapshot(
                payload,
                profile_id="profile",
                cluster_id="cluster-1",
                membership_revision=2,
                allowed_article_ids={"article-a", "article-b"},
            )

    def test_identity_unknown_fields_and_markdown_wrappers_are_rejected(self):
        payload = self.payload()
        with self.assertRaisesRegex(SnapshotValidationError, "identity"):
            validate_cluster_snapshot(
                payload,
                profile_id="other",
                cluster_id="cluster-1",
                membership_revision=2,
                allowed_article_ids={"article-a", "article-b"},
            )
        payload["invented"] = True
        with self.assertRaisesRegex(SnapshotValidationError, "unknown fields"):
            validate_cluster_snapshot(
                payload,
                profile_id="profile",
                cluster_id="cluster-1",
                membership_revision=2,
                allowed_article_ids={"article-a", "article-b"},
            )
        with self.assertRaises(SnapshotValidationError):
            parse_snapshot_json("```json\n{}\n```")


if __name__ == "__main__":
    unittest.main()

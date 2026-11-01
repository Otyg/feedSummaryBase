import unittest

from feedsummary_core.long_term import (
    EmbeddingSignature,
    ReconciliationSettings,
    build_cluster_merge_operation,
    create_cluster,
    propose_cluster_reconciliation,
)


class LongTermReconciliationTests(unittest.TestCase):
    signature = EmbeddingSignature("model", 2, "instruction")

    def cluster(self, article_id, ts, vector, indicators=(), strict=True):
        return create_cluster(
            profile_id="profile",
            article_id=article_id,
            article_ts=ts,
            embedding=vector,
            signature=self.signature,
            strong_indicators=indicators,
            strict_cve_identity=strict,
        ).to_document()

    @staticmethod
    def membership(cluster, article_id, title, indicators, strict):
        return {
            "cluster_id": cluster["id"],
            "article_id": article_id,
            "strict_cve_identity": strict,
            "strong_indicators": list(indicators),
            "evidence": {"title": title},
        }

    def test_papercut_narratives_form_one_merge_group_despite_different_cves(self):
        first = self.cluster(
            "a", 100, [1.0, 0.0],
            ["cve:cve-2023-27350", "organization:papercut"], False,
        )
        second = self.cluster(
            "b", 200, [0.99, 0.05],
            ["cve:cve-2023-27351", "organization:papercut"], False,
        )
        memberships = {
            first["id"]: [
                self.membership(
                    first, "a", "PaperCut zero-day exploited",
                    first["strong_indicators"], False,
                )
            ],
            second["id"]: [
                self.membership(
                    second, "b", "PaperCut warns about attacks",
                    second["strong_indicators"], False,
                )
            ],
        }
        result = propose_cluster_reconciliation(
            [first, second], memberships,
            settings=ReconciliationSettings(auto_merge_similarity=0.88),
        )
        self.assertEqual(1, len(result.auto_merge_edges))
        self.assertEqual(("papercut",), result.auto_merge_edges[0].shared_title_terms)
        self.assertEqual(1, len(result.groups))
        self.assertEqual(2, len(result.groups[0].cluster_ids))

    def test_disjoint_pure_cve_records_are_blocked(self):
        first = self.cluster("a", 100, [1.0, 0.0], ["cve:cve-2026-1000"])
        second = self.cluster("b", 200, [1.0, 0.0], ["cve:cve-2026-2000"])
        memberships = {
            first["id"]: [
                self.membership(
                    first, "a", "CVE-2026-1000 - Product flaw",
                    first["strong_indicators"], True,
                )
            ],
            second["id"]: [
                self.membership(
                    second, "b", "CVE-2026-2000 - Product flaw",
                    second["strong_indicators"], True,
                )
            ],
        }
        result = propose_cluster_reconciliation([first, second], memberships)
        self.assertEqual(1, result.blocked_strict_cve_pair_count)
        self.assertFalse(result.auto_merge_edges)

    def test_narrative_to_pure_cve_is_sent_to_review(self):
        overview = self.cluster("overview", 100, [1.0, 0.0], ["product:hpe"], False)
        record = self.cluster("record", 200, [1.0, 0.0], ["product:hpe", "cve:cve-2026-1000"])
        memberships = {
            overview["id"]: [
                self.membership(
                    overview, "overview", "HPE security bulletin",
                    overview["strong_indicators"], False,
                )
            ],
            record["id"]: [
                self.membership(
                    record, "record", "CVE-2026-1000 - HPE flaw",
                    record["strong_indicators"], True,
                )
            ],
        }
        result = propose_cluster_reconciliation([overview, record], memberships)
        self.assertFalse(result.auto_merge_edges)
        self.assertEqual("narrative_to_strict_cve_requires_review", result.review_edges[0].reason)

    def test_recurring_bulletins_are_not_auto_merged(self):
        first = self.cluster("a", 100, [1.0, 0.0], ["organization:cisa"], False)
        second = self.cluster("b", 200, [1.0, 0.0], ["organization:cisa"], False)
        memberships = {
            first["id"]: [
                self.membership(
                    first, "a", "CISA Adds Six Vulnerabilities to Catalog",
                    first["strong_indicators"], False,
                )
            ],
            second["id"]: [
                self.membership(
                    second, "b", "CISA Adds One Vulnerability to Catalog",
                    second["strong_indicators"], False,
                )
            ],
        }
        result = propose_cluster_reconciliation([first, second], memberships)
        self.assertFalse(result.auto_merge_edges)
        self.assertEqual(
            "summary_or_recurring_bulletin_requires_review",
            result.review_edges[0].reason,
        )

    def test_build_merge_operation_recomputes_cluster_and_lineage(self):
        first = self.cluster(
            "a", 100, [1.0, 0.0], ["organization:papercut"], False
        )
        second = self.cluster(
            "b", 200, [0.0, 1.0], ["product:papercut mf/ng"], False
        )
        memberships = {
            first["id"]: [
                self.membership(
                    first, "a", "PaperCut incident", first["strong_indicators"], False
                )
            ],
            second["id"]: [
                self.membership(
                    second, "b", "PaperCut exploit", second["strong_indicators"], False
                )
            ],
        }
        operation = build_cluster_merge_operation(
            [first, second],
            memberships,
            primary_cluster_id=first["id"],
            reconciliation_id="reconciliation-1",
            reconciled_at=300,
        )
        merged = operation["merged_cluster"]
        self.assertEqual(2, merged["member_count"])
        self.assertEqual(3, merged["membership_revision"])
        self.assertEqual([0.5, 0.5], merged["centroid"])
        self.assertEqual(1, merged["summarized_revision"])
        self.assertIsNone(merged["latest_snapshot_id"])
        self.assertEqual(
            {"a": 2, "b": 3},
            operation["membership_revision_by_article_id"],
        )
        tombstone = operation["superseded_clusters"][0]
        self.assertEqual(0, tombstone["member_count"])
        self.assertEqual("closed", tombstone["status"])
        self.assertEqual(first["id"], tombstone["superseded_by_cluster_id"])

    def test_proposals_exclude_superseded_cluster_tombstones(self):
        active = self.cluster(
            "active", 100, [1.0, 0.0], ["organization:papercut"], False
        )
        tombstone = self.cluster(
            "old", 100, [1.0, 0.0], ["organization:papercut"], False
        )
        tombstone.update(
            {
                "status": "closed",
                "member_count": 0,
                "membership_revision": 2,
                "vector_sum": [0.0, 0.0],
                "superseded_by_cluster_id": active["id"],
                "reconciliation_id": "reconciliation-1",
                "reconciled_at": 200,
            }
        )
        memberships = {
            active["id"]: [
                self.membership(
                    active, "active", "PaperCut incident",
                    active["strong_indicators"], False,
                )
            ],
            tombstone["id"]: [],
        }

        result = propose_cluster_reconciliation([active, tombstone], memberships)

        self.assertEqual(1, result.cluster_count)
        self.assertFalse(result.auto_merge_edges)
        self.assertFalse(result.review_edges)


if __name__ == "__main__":
    unittest.main()

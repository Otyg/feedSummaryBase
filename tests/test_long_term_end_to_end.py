import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from feedsummary_core.long_term import (
    ClusteringSettings,
    IncrementalSettings,
    LandscapeMetricSettings,
    MapSettings,
    ReduceSettings,
    build_landscape_metrics,
    run_incremental_clustering,
    run_landscape_reduce,
    update_cluster_map_snapshot,
)
from feedsummary_core.persistence import SqliteStore


class StaticLLM:
    def __init__(self, response):
        self.response = json.dumps(response)
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.response


class LongTermEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SqliteStore(str(Path(self.directory.name) / "store.sqlite"))

    def test_fixture_articles_flow_to_validated_published_report_without_network(self):
        period_end = 10_000_000
        fixtures = (
            ("article-1", period_end - 1 * 86400, "source-a", [1.0, 0.0, 0.0]),
            ("article-2", period_end - 10 * 86400, "source-b", [0.0, 1.0, 0.0]),
            ("article-3", period_end - 20 * 86400, "source-c", [0.0, 0.0, 1.0]),
        )
        for article_id, timestamp, source, vector in fixtures:
            self.store.upsert_article(
                {
                    "id": article_id,
                    "title": f"Independent incident {article_id}",
                    "text": "Fixture evidence for a distinct cyber incident.",
                    "source": source,
                    "fetched_at": timestamp,
                    "published_ts": timestamp,
                    "similarity_embedding_vector": vector,
                    "similarity_embedding_model": "fixture-model",
                    "similarity_embedding_instruction": "represent incident identity",
                }
            )

        clustered = run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="cluster-worker",
            until_fetched_at=period_end,
            now_ts=period_end,
            settings=IncrementalSettings(
                clustering=ClusteringSettings(similarity_threshold=0.9)
            ),
        )
        self.assertEqual(3, clustered.created)

        map_prompt = {
            "prompt_version": "cluster-update-v1",
            "temperature": 0.0,
            "system": "Return evidence-bound JSON.",
            "user_template": (
                "profile={profile_id} cluster={cluster_id} revision={membership_revision} "
                "allowed={allowed_article_ids} previous={previous_snapshot} "
                "articles={new_articles}"
            ),
            "output_schema": {"type": "object"},
        }
        clusters = self.store.list_threat_clusters("profile")
        for cluster in clusters:
            membership = self.store.list_cluster_memberships(cluster["id"])[0]
            article_id = membership["article_id"]
            snapshot_response = {
                "schema_version": 1,
                "profile_id": "profile",
                "cluster_id": cluster["id"],
                "membership_revision": 1,
                "title": "Independent incident",
                "summary": "A fixture-backed incident was observed.",
                "confidence": "medium",
                "insufficient_evidence": False,
                "facts": [
                    {
                        "statement": "The fixture incident was reported.",
                        "status": "active",
                        "evidence_article_ids": [article_id],
                    }
                ],
                "timeline": [],
                "mitre_techniques": [],
                "uncertainties": [],
            }
            result = asyncio.run(
                update_cluster_map_snapshot(
                    self.store,
                    StaticLLM(snapshot_response),
                    cluster_id=cluster["id"],
                    prompt_package=map_prompt,
                    now_ts=period_end,
                    settings=MapSettings(min_pending_articles=1),
                )
            )
            self.assertEqual("saved", result.action)

        metrics = build_landscape_metrics(
            self.store,
            profile_id="profile",
            period_end_ts=period_end + 1,
            settings=LandscapeMetricSettings(),
        )
        cluster_ids = sorted(cluster["id"] for cluster in clusters)
        report_response = {
            "schema_version": 1,
            "profile_id": "profile",
            "period_start_ts": min(
                window["period_start_ts"] for window in metrics["windows"]
            ),
            "period_end_ts": period_end + 1,
            "executive_summary": "Tre separata fixture-händelser analyserades.",
            "observations": [
                {
                    "statement": "Tre oberoende händelser finns i underlaget.",
                    "confidence": "medium",
                    "evidence_cluster_ids": cluster_ids,
                }
            ],
            "changes": [],
            "alternative_explanations": [],
            "data_gaps": ["Fixture-underlaget är avsiktligt litet."],
            "forecast": [],
        }
        final_prompt = {
            "prompt_version": "landscape-report-v1",
            "temperature": 0.0,
            "system": "Return only JSON.",
            "user_template": "metrics={metrics} snapshots={cluster_snapshots}",
            "output_schema": {"type": "object"},
        }
        segment_prompt = {
            "prompt_version": "landscape-segment-v1",
            "temperature": 0.0,
            "system": "Compress snapshots.",
            "user_template": (
                "profile={profile_id} segment={segment_id} snapshots={cluster_snapshots}"
            ),
            "output_schema": {"type": "object"},
        }

        reduced = asyncio.run(
            run_landscape_reduce(
                self.store,
                StaticLLM(report_response),
                profile_id="profile",
                profile_context={"name": "Fixture profile"},
                metrics=metrics,
                final_prompt_package=final_prompt,
                segment_prompt_package=segment_prompt,
                model="fixture-llm",
                now_ts=period_end + 2,
                settings=ReduceSettings(max_context_tokens=16000),
                mirror_to_summary_docs=False,
            )
        )

        self.assertEqual("saved", reduced.action)
        self.assertEqual(3, len(self.store.list_cluster_snapshots("profile")))
        report = self.store.get_threat_landscape_report(reduced.report_id)
        self.assertEqual("published", report["status"])
        self.assertEqual(3, len(report["input_snapshot_ids"]))
        self.assertIn("Tre separata fixture-händelser", report["markdown"])


if __name__ == "__main__":
    unittest.main()

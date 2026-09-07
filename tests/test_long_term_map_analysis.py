import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from feedsummary_core.long_term import (
    ClusteringSettings,
    IncrementalSettings,
    MapSettings,
    PromptBudgetError,
    SnapshotValidationError,
    run_incremental_clustering,
    update_cluster_map_snapshot,
)
from feedsummary_core.persistence import SqliteStore


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, *, temperature=0.0):
        self.calls.append((messages, temperature))
        return self.responses.pop(0)


class LongTermMapAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SqliteStore(str(Path(self.directory.name) / "store.sqlite"))
        for index in range(1, 4):
            self.store.upsert_article(
                {
                    "id": f"article-{index}",
                    "source": "source",
                    "title": f"Incident update {index}",
                    "text": "Evidence text",
                    "fetched_at": index * 100,
                    "published_ts": index * 100,
                    "similarity_embedding_vector": [1.0, 0.0],
                    "similarity_embedding_model": "model",
                    "similarity_embedding_instruction": "event",
                }
            )
        run_incremental_clustering(
            self.store,
            profile_id="profile",
            owner_id="worker",
            until_fetched_at=500,
            now_ts=500,
            settings=IncrementalSettings(
                clustering=ClusteringSettings(similarity_threshold=0.9)
            ),
        )
        self.cluster = self.store.list_threat_clusters("profile")[0]
        self.prompt = {
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

    @staticmethod
    def response(revision):
        return json.dumps(
            {
                "schema_version": 1,
                "profile_id": "profile",
                "cluster_id": "threat_cluster_"
                "PLACEHOLDER",
                "membership_revision": revision,
                "title": "Incident",
                "summary": "Evidence-bound summary.",
                "confidence": "medium",
                "insufficient_evidence": False,
                "facts": [
                    {
                        "statement": "The incident was reported.",
                        "status": "active",
                        "evidence_article_ids": ["article-1"],
                    }
                ],
                "timeline": [],
                "mitre_techniques": [],
                "uncertainties": [],
            }
        )

    def valid_response(self, revision):
        payload = json.loads(self.response(revision))
        payload["cluster_id"] = self.cluster["id"]
        return json.dumps(payload)

    def test_map_update_is_chunked_and_revision_coupled(self):
        llm = FakeLLM([self.valid_response(2), self.valid_response(3)])
        settings = MapSettings(min_pending_articles=1, max_articles_per_call=2)
        first = asyncio.run(
            update_cluster_map_snapshot(
                self.store,
                llm,
                cluster_id=self.cluster["id"],
                prompt_package=self.prompt,
                now_ts=600,
                settings=settings,
            )
        )
        self.assertEqual(2, first.summarized_revision)
        self.assertEqual(("article-1", "article-2"), first.input_article_ids)
        self.assertEqual(
            2, self.store.get_threat_cluster(self.cluster["id"])["summarized_revision"]
        )

        second = asyncio.run(
            update_cluster_map_snapshot(
                self.store,
                llm,
                cluster_id=self.cluster["id"],
                prompt_package=self.prompt,
                now_ts=700,
                settings=settings,
            )
        )
        self.assertEqual(3, second.summarized_revision)
        self.assertEqual(("article-3",), second.input_article_ids)
        self.assertEqual(
            [2, 3],
            sorted(
                snapshot["membership_revision"]
                for snapshot in self.store.list_cluster_snapshots("profile")
            ),
        )

    def test_chunk_cannot_cite_an_article_from_a_later_revision(self):
        payload = json.loads(self.valid_response(2))
        payload["facts"][0]["evidence_article_ids"] = ["article-3"]
        llm = FakeLLM([json.dumps(payload)])
        with self.assertRaises(SnapshotValidationError):
            asyncio.run(
                update_cluster_map_snapshot(
                    self.store,
                    llm,
                    cluster_id=self.cluster["id"],
                    prompt_package=self.prompt,
                    now_ts=600,
                    settings=MapSettings(
                        min_pending_articles=1,
                        max_articles_per_call=2,
                        format_repair_attempts=0,
                    ),
                )
            )

    def test_budget_shrink_also_shrinks_allowed_evidence(self):
        payload = json.loads(self.valid_response(1))
        payload["facts"][0]["evidence_article_ids"] = ["article-2"]
        llm = FakeLLM([json.dumps(payload)])
        with (
            patch(
                "feedsummary_core.long_term.map_analysis.estimate_tokens",
                side_effect=[999, 100],
            ),
            self.assertRaises(SnapshotValidationError),
        ):
            asyncio.run(
                update_cluster_map_snapshot(
                    self.store,
                    llm,
                    cluster_id=self.cluster["id"],
                    prompt_package=self.prompt,
                    now_ts=600,
                    settings=MapSettings(
                        min_pending_articles=1,
                        max_articles_per_call=2,
                        max_context_tokens=1000,
                        max_output_tokens=100,
                        format_repair_attempts=0,
                    ),
                )
            )

    def test_one_format_repair_is_allowed(self):
        llm = FakeLLM(["```json\n{}\n```", self.valid_response(3)])
        result = asyncio.run(
            update_cluster_map_snapshot(
                self.store,
                llm,
                cluster_id=self.cluster["id"],
                prompt_package=self.prompt,
                now_ts=600,
                settings=MapSettings(min_pending_articles=1),
            )
        )
        self.assertTrue(result.repair_attempted)
        self.assertEqual(2, len(llm.calls))

    def test_too_small_budget_fails_before_llm_call(self):
        llm = FakeLLM([])
        with self.assertRaises(PromptBudgetError):
            asyncio.run(
                update_cluster_map_snapshot(
                    self.store,
                    llm,
                    cluster_id=self.cluster["id"],
                    prompt_package=self.prompt,
                    now_ts=600,
                    settings=MapSettings(
                        min_pending_articles=1,
                        max_context_tokens=300,
                        max_output_tokens=200,
                        safety_margin_tokens=99,
                    ),
                )
            )
        self.assertEqual([], llm.calls)


if __name__ == "__main__":
    unittest.main()

import asyncio
import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from feedsummary_core.llm_client.fallback_client import FallbackLLMClient, FallbackPolicy
from feedsummary_core.llm_client.ollama_cloud import LLMUnavailableError
from feedsummary_core.long_term import (
    ClusteringSettings,
    IncrementalSettings,
    LeaseLostError,
    LongTermLeaseHeartbeat,
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

    async def chat(self, messages, *, temperature=0.0, max_output_tokens=None):
        self.calls.append((messages, temperature, max_output_tokens))
        return self.responses.pop(0)


class LostLeaseGuard:
    async def ensure_owned(self):
        raise RuntimeError("lease lost")


class UnavailableProvider:
    def __init__(self):
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        raise LLMUnavailableError("provider timeout")


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
        self.assertEqual([1200, 1200], [call[2] for call in llm.calls])

    def test_configured_output_limit_matches_reserved_budget(self):
        llm = FakeLLM([self.valid_response(3)])
        settings = MapSettings(
            min_pending_articles=1,
            max_context_tokens=2000,
            max_output_tokens=321,
            safety_margin_tokens=100,
        )

        with patch(
            "feedsummary_core.long_term.map_analysis.estimate_tokens",
            return_value=1579,
        ):
            asyncio.run(
                update_cluster_map_snapshot(
                    self.store,
                    llm,
                    cluster_id=self.cluster["id"],
                    prompt_package=self.prompt,
                    now_ts=600,
                    settings=settings,
                )
            )

        self.assertEqual(321, llm.calls[0][2])

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

    def test_lost_lease_blocks_snapshot_persistence(self):
        llm = FakeLLM([self.valid_response(3)])

        with self.assertRaisesRegex(RuntimeError, "lease lost"):
            asyncio.run(
                update_cluster_map_snapshot(
                    self.store,
                    llm,
                    cluster_id=self.cluster["id"],
                    prompt_package=self.prompt,
                    now_ts=600,
                    settings=MapSettings(min_pending_articles=1),
                    lease_guard=LostLeaseGuard(),
                )
            )

        self.assertEqual([], self.store.list_cluster_snapshots("profile"))
        self.assertEqual(
            0,
            self.store.get_threat_cluster(self.cluster["id"])["summarized_revision"],
        )

    def test_heartbeat_loss_during_slow_llm_blocks_snapshot_persistence(self):
        renewal_attempted = threading.Event()

        def fail_renewal(*_args, **_kwargs):
            renewal_attempted.set()
            return False

        class SlowLLM(FakeLLM):
            async def chat(self, messages, **kwargs):
                completed = await asyncio.to_thread(renewal_attempted.wait, 1)
                if not completed:
                    raise RuntimeError("heartbeat did not run")
                return await super().chat(messages, **kwargs)

        async def run_scenario():
            owner = "map-worker"
            self.assertTrue(
                self.store.claim_long_term_lease(
                    "profile", owner, now_ts=600, lease_seconds=60
                )
            )
            heartbeat = LongTermLeaseHeartbeat(
                self.store,
                profile_id="profile",
                owner_id=owner,
                lease_seconds=60,
                interval_seconds=0.01,
                clock=lambda: 601,
            )
            await heartbeat.start()
            try:
                with patch.object(
                    self.store,
                    "renew_long_term_lease",
                    side_effect=fail_renewal,
                ):
                    await update_cluster_map_snapshot(
                        self.store,
                        SlowLLM([self.valid_response(3)]),
                        cluster_id=self.cluster["id"],
                        prompt_package=self.prompt,
                        now_ts=601,
                        settings=MapSettings(min_pending_articles=1),
                        lease_guard=heartbeat,
                    )
            finally:
                await heartbeat.stop()
                self.store.release_long_term_lease("profile", owner)

        with self.assertRaisesRegex(LeaseLostError, "lease was lost"):
            asyncio.run(run_scenario())

        self.assertEqual([], self.store.list_cluster_snapshots("profile"))

    def test_exhausted_timeout_fallback_never_persists_snapshot(self):
        primary = UnavailableProvider()
        fallback = UnavailableProvider()
        llm = FallbackLLMClient(
            [primary, fallback],
            policy=FallbackPolicy(max_quota_retries=0, default_wait_s=0),
        )

        with self.assertRaisesRegex(LLMUnavailableError, "timeout"):
            asyncio.run(
                update_cluster_map_snapshot(
                    self.store,
                    llm,
                    cluster_id=self.cluster["id"],
                    prompt_package=self.prompt,
                    now_ts=600,
                    settings=MapSettings(min_pending_articles=1),
                )
            )

        self.assertEqual(1, len(primary.calls))
        self.assertEqual(1, len(fallback.calls))
        self.assertEqual(1200, primary.calls[0][1]["max_output_tokens"])
        self.assertEqual(1200, fallback.calls[0][1]["max_output_tokens"])
        self.assertEqual([], self.store.list_cluster_snapshots("profile"))


if __name__ == "__main__":
    unittest.main()

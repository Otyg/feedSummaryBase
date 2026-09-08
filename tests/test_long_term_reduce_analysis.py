import asyncio
import copy
import json
import unittest

from feedsummary_core.llm_client.fallback_client import FallbackLLMClient, FallbackPolicy
from feedsummary_core.llm_client.ollama_cloud import LLMUnavailableError
from feedsummary_core.long_term import (
    ReduceResult,
    ReduceSettings,
    ReportMirrorError,
    ReportValidationError,
    run_landscape_reduce,
    select_report_snapshots,
)
from feedsummary_core.long_term.reduce_analysis import _segment_id


class FakeStore:
    def __init__(self, snapshots):
        self.snapshots = list(snapshots)
        self.reports = []
        self.summaries = {}
        self.fail_summary_writes = 0

    def list_cluster_snapshots(self, profile_id, *, cluster_id=None, limit=1000):
        rows = [row for row in self.snapshots if row["profile_id"] == profile_id]
        if cluster_id is not None:
            rows = [row for row in rows if row["cluster_id"] == cluster_id]
        return rows[:limit]

    def list_threat_landscape_reports(self, profile_id, *, limit=100):
        rows = [row for row in self.reports if row["profile_id"] == profile_id]
        return rows[:limit]

    def get_threat_landscape_report(self, report_id):
        return next((row for row in self.reports if row["id"] == report_id), None)

    def save_threat_landscape_report(self, report_doc):
        if self.get_threat_landscape_report(report_doc["id"]):
            return False
        self.reports.append(copy.deepcopy(report_doc))
        return True

    def get_summary_doc(self, summary_doc_id):
        row = self.summaries.get(summary_doc_id)
        return copy.deepcopy(row) if row is not None else None

    def save_summary_doc(self, summary_doc):
        if self.fail_summary_writes:
            self.fail_summary_writes -= 1
            raise RuntimeError("summary store unavailable")
        summary_id = str(summary_doc["id"])
        self.summaries[summary_id] = copy.deepcopy(summary_doc)
        return summary_id


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


class LongTermReduceAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {
            "schema_version": 1,
            "profile_id": "profile",
            "period_end_ts": 2_000_000,
            "settings": {"min_independent_events_for_trend": 3},
            "windows": [
                {
                    "period_start_ts": 1_000_000,
                    "event_cluster_ids": ["cluster-1", "cluster-2", "cluster-3"],
                    "weekly_buckets": [
                        {"event_cluster_ids": ["cluster-1", "cluster-2"]},
                        {"event_cluster_ids": ["cluster-2", "cluster-3"]},
                    ],
                    "coverage": {"trend_eligible": True, "warnings": []},
                }
            ],
            "quality": {"warning_codes": []},
        }
        self.snapshots = [
            {
                "id": f"snapshot-{index}",
                "profile_id": "profile",
                "cluster_id": f"cluster-{index}",
                "membership_revision": 1,
                "created_at": 1_900_000,
                "payload": {
                    "profile_id": "profile",
                    "cluster_id": f"cluster-{index}",
                    "summary": f"Incident {index}",
                },
            }
            for index in range(1, 4)
        ]
        self.final_prompt = {
            "prompt_version": "landscape-report-v1",
            "temperature": 0.1,
            "system": "Return only JSON.",
            "user_template": "metrics={metrics} snapshots={cluster_snapshots}",
            "output_schema": {"type": "object"},
        }
        self.segment_prompt = {
            "prompt_version": "landscape-segment-v1",
            "temperature": 0.0,
            "system": "Compress snapshots.",
            "user_template": (
                "profile={profile_id} segment={segment_id} snapshots={cluster_snapshots}"
            ),
            "output_schema": {"type": "object"},
        }

    @staticmethod
    def report(*, sparse=False):
        return {
            "schema_version": 1,
            "profile_id": "profile",
            "period_start_ts": 1_000_000,
            "period_end_ts": 2_000_000,
            "executive_summary": "Strukturerad analys.",
            "observations": [
                {
                    "statement": "En observerad händelse.",
                    "confidence": "medium",
                    "evidence_cluster_ids": ["cluster-1"],
                }
            ],
            "changes": []
            if sparse
            else [
                {
                    "statement": "Flera händelser visar en förändring.",
                    "direction": "shifting",
                    "confidence": "medium",
                    "evidence_cluster_ids": ["cluster-1", "cluster-2", "cluster-3"],
                }
            ],
            "alternative_explanations": [],
            "data_gaps": ["En snapshot saknas."] if sparse else [],
            "forecast": []
            if sparse
            else [
                {
                    "hypothesis": "Mönstret kan fortsätta.",
                    "horizon_days": 30,
                    "confidence": "low",
                    "rationale": "Tre oberoende kluster stödjer hypotesen.",
                    "evidence_cluster_ids": ["cluster-1", "cluster-2", "cluster-3"],
                    "leading_indicators": ["Fler oberoende händelser"],
                    "invalidation_conditions": ["Inga nya händelser"],
                }
            ],
        }

    @staticmethod
    def segment_payload(group):
        segment_id = _segment_id("profile", group)
        return {
            "schema_version": 1,
            "profile_id": "profile",
            "segment_id": segment_id,
            "snapshots": [
                {
                    "id": snapshot["id"],
                    "profile_id": "profile",
                    "cluster_id": snapshot["cluster_id"],
                    "summary": "Kort sammanfattning.",
                    "key_facts": ["Belagd uppgift."],
                    "uncertainties": [],
                }
                for snapshot in group
            ],
        }

    def run_reduce(self, store, llm, *, settings=None, lease_guard=None):
        return asyncio.run(
            run_landscape_reduce(
                store,
                llm,
                profile_id="profile",
                profile_context={"name": "Vård"},
                metrics=self.metrics,
                final_prompt_package=self.final_prompt,
                segment_prompt_package=self.segment_prompt,
                model="test-model",
                now_ts=2_000_100,
                settings=settings,
                mirror_to_summary_docs=True,
                lease_guard=lease_guard,
            )
        )

    def test_direct_reduce_saves_once_and_is_idempotent(self):
        store = FakeStore(self.snapshots)
        first = self.run_reduce(store, FakeLLM([json.dumps(self.report())]))
        second = self.run_reduce(store, FakeLLM([json.dumps(self.report())]))

        self.assertIsInstance(first, ReduceResult)
        self.assertEqual("saved", first.action)
        self.assertEqual("existing", second.action)
        self.assertEqual(first.report_id, second.report_id)
        self.assertEqual(1, len(store.reports))
        self.assertEqual("saved", first.mirror_action)
        self.assertEqual("existing", second.mirror_action)
        self.assertEqual(1, len(store.summaries))
        self.assertEqual(0, first.segment_count)

    def test_failed_mirror_preserves_original_and_retry_uses_canonical_report(self):
        store = FakeStore(self.snapshots)
        store.fail_summary_writes = 1
        with self.assertRaisesRegex(ReportMirrorError, "summary mirror write failed"):
            self.run_reduce(store, FakeLLM([json.dumps(self.report())]))

        self.assertEqual(1, len(store.reports))
        self.assertEqual({}, store.summaries)

        changed = self.report()
        changed["executive_summary"] = "Ett annat men giltigt svar."
        retried = self.run_reduce(store, FakeLLM([json.dumps(changed)]))

        mirror = next(iter(store.summaries.values()))
        self.assertEqual("existing", retried.action)
        self.assertEqual("saved", retried.mirror_action)
        self.assertEqual(store.reports[0]["markdown"], mirror["summary"])
        self.assertNotIn("Ett annat men giltigt svar.", mirror["summary"])

    def test_conflicting_existing_mirror_is_not_overwritten(self):
        store = FakeStore(self.snapshots)
        first = self.run_reduce(store, FakeLLM([json.dumps(self.report())]))
        mirror_id = f"summary_{first.report_id}"
        store.summaries[mirror_id]["source_report_id"] = "another-report"

        with self.assertRaisesRegex(ReportMirrorError, "conflicting summary mirror"):
            self.run_reduce(store, FakeLLM([json.dumps(self.report())]))

        self.assertEqual(
            "another-report", store.summaries[mirror_id]["source_report_id"]
        )

    def test_final_response_gets_one_repair_before_save(self):
        store = FakeStore(self.snapshots)
        llm = FakeLLM(["```json\n{}\n```", json.dumps(self.report())])
        result = self.run_reduce(store, llm)

        self.assertTrue(result.repair_attempted)
        self.assertEqual(2, result.llm_call_count)
        self.assertEqual(1, len(store.reports))
        self.assertEqual([2500, 2500], [call[2] for call in llm.calls])

    def test_invalid_final_response_never_creates_report(self):
        store = FakeStore(self.snapshots)
        invalid = self.report()
        invalid["observations"][0]["evidence_cluster_ids"] = ["unknown"]
        with self.assertRaises(ReportValidationError):
            self.run_reduce(
                store,
                FakeLLM([json.dumps(invalid)]),
                settings=ReduceSettings(format_repair_attempts=0),
            )
        self.assertEqual([], store.reports)

    def test_lost_lease_blocks_report_and_mirror_persistence(self):
        store = FakeStore(self.snapshots)

        with self.assertRaisesRegex(RuntimeError, "lease lost"):
            self.run_reduce(
                store,
                FakeLLM([json.dumps(self.report())]),
                lease_guard=LostLeaseGuard(),
            )

        self.assertEqual([], store.reports)
        self.assertEqual({}, store.summaries)

    def test_exhausted_timeout_fallback_never_persists_report(self):
        store = FakeStore(self.snapshots)
        primary = UnavailableProvider()
        fallback = UnavailableProvider()
        llm = FallbackLLMClient(
            [primary, fallback],
            policy=FallbackPolicy(max_quota_retries=0, default_wait_s=0),
        )

        with self.assertRaisesRegex(LLMUnavailableError, "timeout"):
            self.run_reduce(store, llm)

        self.assertEqual(1, len(primary.calls))
        self.assertEqual(1, len(fallback.calls))
        self.assertEqual(2500, primary.calls[0][1]["max_output_tokens"])
        self.assertEqual(2500, fallback.calls[0][1]["max_output_tokens"])
        self.assertEqual([], store.reports)
        self.assertEqual({}, store.summaries)

    def test_large_input_is_segmented_before_final_reduce(self):
        snapshots = copy.deepcopy(self.snapshots)
        for snapshot in snapshots:
            snapshot["payload"]["summary"] = "x" * 3000
        groups = [[snapshot] for snapshot in snapshots]
        responses = [json.dumps(self.segment_payload(group)) for group in groups]
        responses.append(json.dumps(self.report()))
        llm = FakeLLM(responses)
        store = FakeStore(snapshots)

        result = self.run_reduce(
            store,
            llm,
            settings=ReduceSettings(
                max_context_tokens=2000,
                max_output_tokens=200,
                safety_margin_tokens=100,
                max_snapshots_per_segment=1,
            ),
        )

        self.assertEqual(3, result.segment_count)
        self.assertEqual(4, result.llm_call_count)
        self.assertEqual("landscape-segment-v1", store.reports[0]["segment_prompt_version"])
        self.assertEqual([200, 200, 200, 200], [call[2] for call in llm.calls])

    def test_missing_snapshot_disables_trend_claims_and_is_recorded(self):
        store = FakeStore(self.snapshots[:2])
        result = self.run_reduce(store, FakeLLM([json.dumps(self.report(sparse=True))]))

        self.assertEqual("saved", result.action)
        report = store.reports[0]
        self.assertIn("missing_report_snapshots", report["quality"]["warning_codes"])
        self.assertEqual(
            ["cluster-3"],
            report["metrics"]["windows"][0]["coverage"][
                "missing_snapshot_cluster_ids"
            ],
        )

    def test_snapshot_selection_uses_latest_revision_before_cutoff(self):
        rows = [
            {**self.snapshots[0], "id": "old", "created_at": 1_800_000},
            {**self.snapshots[0], "id": "latest", "created_at": 1_950_000},
            {**self.snapshots[0], "id": "future", "created_at": 2_100_000},
            *self.snapshots[1:],
        ]
        selected = select_report_snapshots(
            FakeStore(rows),
            profile_id="profile",
            metrics=self.metrics,
            settings=ReduceSettings(),
        )

        self.assertEqual(
            ["latest", "snapshot-2", "snapshot-3"],
            [snapshot["id"] for snapshot in selected],
        )


if __name__ == "__main__":
    unittest.main()

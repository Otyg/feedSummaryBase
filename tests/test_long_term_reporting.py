import copy
import json
import unittest

from feedsummary_core.long_term import (
    ReportValidationError,
    build_landscape_report_document,
    build_landscape_summary_document,
    parse_landscape_report_json,
    render_landscape_markdown,
    render_landscape_report_messages,
    validate_landscape_report,
)


class LongTermReportingTests(unittest.TestCase):
    def setUp(self):
        self.metrics = {
            "schema_version": 1,
            "profile_id": "profile",
            "period_end_ts": 2_000_000,
            "settings": {"min_independent_events_for_trend": 3},
            "windows": [
                {
                    "period_start_ts": 1_000_000,
                    "coverage": {"trend_eligible": True, "warnings": []},
                    "weekly_buckets": [
                        {"event_cluster_ids": ["cluster-1", "cluster-2"]},
                        {"event_cluster_ids": ["cluster-2", "cluster-3"]},
                    ],
                }
            ],
            "quality": {"warning_codes": []},
        }
        self.snapshots = [
            {
                "id": f"snapshot-{index}",
                "profile_id": "profile",
                "cluster_id": f"cluster-{index}",
                "payload": {
                    "profile_id": "profile",
                    "cluster_id": f"cluster-{index}",
                    "summary": f"Incident {index}",
                },
            }
            for index in range(1, 4)
        ]
        self.report = {
            "schema_version": 1,
            "profile_id": "profile",
            "period_start_ts": 1_000_000,
            "period_end_ts": 2_000_000,
            "executive_summary": "Tre oberoende händelser visar ett mönster.",
            "observations": [
                {
                    "statement": "En händelse berörde vårdsektorn.",
                    "confidence": "high",
                    "evidence_cluster_ids": ["cluster-1"],
                }
            ],
            "changes": [
                {
                    "statement": "Mönstret ökade över flera tidsperioder.",
                    "direction": "increasing",
                    "confidence": "medium",
                    "evidence_cluster_ids": ["cluster-1", "cluster-2", "cluster-3"],
                }
            ],
            "alternative_explanations": [
                {
                    "statement": "Ökad rapportering kan bidra till utfallet.",
                    "confidence": "low",
                    "evidence_cluster_ids": ["cluster-2"],
                }
            ],
            "data_gaps": [],
            "forecast": [
                {
                    "hypothesis": "Liknande aktivitet kan fortsätta.",
                    "horizon_days": 30,
                    "confidence": "low",
                    "rationale": "Flera oberoende händelser stödjer hypotesen.",
                    "evidence_cluster_ids": ["cluster-1", "cluster-2", "cluster-3"],
                    "leading_indicators": ["Nya oberoende incidenter"],
                    "invalidation_conditions": ["Inga nya incidenter observeras"],
                }
            ],
        }

    def validate(self, report=None, metrics=None):
        return validate_landscape_report(
            report or self.report,
            profile_id="profile",
            metrics=metrics or self.metrics,
            snapshots=self.snapshots,
        )

    def test_valid_report_preserves_evidence_bound_assessments(self):
        validated = self.validate()

        self.assertEqual(
            ["cluster-1", "cluster-2", "cluster-3"],
            validated["changes"][0]["evidence_cluster_ids"],
        )
        self.assertEqual("low", validated["forecast"][0]["confidence"])

    def test_unknown_evidence_and_high_forecast_confidence_are_rejected(self):
        unknown = copy.deepcopy(self.report)
        unknown["observations"][0]["evidence_cluster_ids"] = ["unknown"]
        with self.assertRaises(ReportValidationError):
            self.validate(unknown)

        high = copy.deepcopy(self.report)
        high["forecast"][0]["confidence"] = "high"
        with self.assertRaises(ReportValidationError):
            self.validate(high)

    def test_sparse_metrics_forbid_changes_and_forecasts_and_require_gap(self):
        sparse_metrics = copy.deepcopy(self.metrics)
        sparse_metrics["windows"][0]["coverage"] = {
            "trend_eligible": False,
            "warnings": ["insufficient_independent_events"],
        }
        with self.assertRaises(ReportValidationError):
            self.validate(metrics=sparse_metrics)

        sparse_report = copy.deepcopy(self.report)
        sparse_report["changes"] = []
        sparse_report["forecast"] = []
        sparse_report["data_gaps"] = ["För få oberoende händelser."]
        validated = self.validate(sparse_report, sparse_metrics)
        self.assertEqual([], validated["changes"])
        self.assertEqual([], validated["forecast"])

    def test_change_evidence_must_span_two_time_buckets(self):
        metrics = copy.deepcopy(self.metrics)
        metrics["windows"][0]["weekly_buckets"] = [
            {"event_cluster_ids": ["cluster-1", "cluster-2", "cluster-3"]},
            {"event_cluster_ids": []},
        ]

        with self.assertRaises(ReportValidationError):
            self.validate(metrics=metrics)

    def test_bare_json_is_required(self):
        self.assertEqual(self.report, parse_landscape_report_json(json.dumps(self.report)))
        with self.assertRaises(ReportValidationError):
            parse_landscape_report_json(f"```json\n{json.dumps(self.report)}\n```")

    def test_prompt_input_and_markdown_are_deterministic(self):
        prompt = {
            "system": "Returnera endast JSON.",
            "user_template": (
                "profile={profile_context} period={analysis_period} metrics={metrics} "
                "warnings={coverage_warnings} snapshots={cluster_snapshots} "
                "previous={previous_report}"
            ),
            "output_schema": {"type": "object"},
        }
        first = render_landscape_report_messages(
            prompt,
            profile_context={"name": "Vård"},
            metrics=self.metrics,
            snapshots=list(reversed(self.snapshots)),
            previous_report=None,
        )
        second = render_landscape_report_messages(
            prompt,
            profile_context={"name": "Vård"},
            metrics=self.metrics,
            snapshots=self.snapshots,
            previous_report=None,
        )
        self.assertEqual(first, second)
        self.assertLess(
            first[1]["content"].index("snapshot-1"),
            first[1]["content"].index("snapshot-2"),
        )

        validated = self.validate()
        markdown = render_landscape_markdown(validated, metrics=self.metrics)
        self.assertIn("# Långtidsanalys", markdown)
        self.assertIn("`cluster-1`", markdown)
        self.assertIn("## Prognoshypoteser", markdown)

    def test_report_document_has_stable_identity_and_snapshot_provenance(self):
        validated = self.validate()
        first = build_landscape_report_document(
            validated,
            metrics=self.metrics,
            snapshots=list(reversed(self.snapshots)),
            prompt_version="landscape-report-v1",
            model="test-model",
            created_at=2_000_100,
        )
        second = build_landscape_report_document(
            validated,
            metrics=self.metrics,
            snapshots=self.snapshots,
            prompt_version="landscape-report-v1",
            model="test-model",
            created_at=2_000_200,
        )

        self.assertEqual(first["id"], second["id"])
        self.assertEqual(
            ["snapshot-1", "snapshot-2", "snapshot-3"],
            first["input_snapshot_ids"],
        )
        self.assertEqual("published", first["status"])
        self.assertNotIn("forecast", first["analysis"])
        self.assertEqual(64, len(first["input_signature"]))

    def test_published_report_builds_deterministic_summary_mirror(self):
        report = build_landscape_report_document(
            self.validate(),
            metrics=self.metrics,
            snapshots=self.snapshots,
            prompt_version="landscape-report-v1",
            model="test-model",
            created_at=2_000_100,
        )

        mirror = build_landscape_summary_document(report)

        self.assertEqual(f"summary_{report['id']}", mirror["id"])
        self.assertEqual("threat_landscape", mirror["kind"])
        self.assertEqual(report["markdown"], mirror["summary"])
        self.assertEqual(report["id"], mirror["source_report_id"])
        self.assertEqual(
            "threat_landscape_reports",
            mirror["meta"]["authoritative_collection"],
        )
        with self.assertRaises(ReportValidationError):
            build_landscape_summary_document({**report, "status": "draft"})


if __name__ == "__main__":
    unittest.main()

# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse, promote, or sell copies of
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import json
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from feedsummary_core.tagging_ml.benchmark import (
    BenchmarkSettings,
    _aggregate_metrics,
    _combination_ranking,
    _ml_imports,
    _per_label_metrics,
    _tune_threshold,
    build_article_text,
    make_splits,
    prepare_training_examples,
    run_benchmark,
)
from feedsummary_core.tagging_ml.cli import main as benchmark_main

try:
    import joblib
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    joblib = None


def _training_rows(count=30):
    rows = []
    for index in range(count):
        phishing = index % 2 == 0
        label = "phishing" if phishing else "ransomware"
        keyword = "credential email login" if phishing else "encrypted files extortion"
        rows.append(
            {
                "article": {
                    "id": f"article-{index:03d}",
                    "title": f"{label} report",
                    "content": f"{keyword} {label} incident number {index}",
                    "published_ts": 1_000 + index,
                    "content_hash": f"hash-{index}",
                },
                "tags": [{"id": 1 if phishing else 2, "name": label, "category": "GENERAL"}],
            }
        )
    return rows


class TrainingDataTests(unittest.TestCase):
    def test_article_text_prefers_content_and_weights_title(self):
        text = build_article_text(
            {"title": "Heading", "content": "Full body", "summary": "Fallback"},
            max_chars=100,
        )

        self.assertEqual("Heading\nHeading\nFull body", text)

    def test_article_text_supports_persisted_text_field(self):
        text = build_article_text(
            {"title": "Heading", "text": "Stored article text", "summary": "Fallback"},
            max_chars=100,
        )

        self.assertEqual("Heading\nHeading\nStored article text", text)

    def test_preparation_keeps_completed_negative_examples_and_filters_rare_labels(self):
        rows = _training_rows(12)
        rows[0]["tags"] = []
        rows[1]["tags"].append({"name": "rare", "category": "GENERAL"})
        for row in rows:
            row["tags"].append({"name": "acme", "category": "ORGANIZATION"})
        settings = BenchmarkSettings(min_label_support=3)

        examples, diagnostics = prepare_training_examples(rows, settings)

        self.assertEqual(12, len(examples))
        self.assertEqual(1, diagnostics["articles_without_eligible_labels"])
        self.assertEqual({"rare": 1}, diagnostics["filtered_labels"])
        self.assertEqual(3, diagnostics["observed_label_count"])
        self.assertEqual(2, diagnostics["eligible_label_count"])
        self.assertAlmostEqual(2 / 3, diagnostics["label_vocabulary_coverage"])
        self.assertEqual(12, diagnostics["observed_tag_assignments"])
        self.assertEqual(11, diagnostics["eligible_tag_assignments"])
        self.assertAlmostEqual(11 / 12, diagnostics["tag_assignment_coverage"])
        self.assertEqual(11, diagnostics["articles_with_category_labels"])
        self.assertEqual(11, diagnostics["articles_with_eligible_labels"])
        self.assertEqual(1.0, diagnostics["eligible_article_coverage"])
        self.assertNotIn("acme", diagnostics["label_support"])
        self.assertEqual((), examples[0].labels)

    def test_grouped_splits_are_reproducible_and_never_split_content_hashes(self):
        rows = _training_rows(30)
        rows[1]["article"]["content_hash"] = rows[0]["article"]["content_hash"]
        examples, _ = prepare_training_examples(rows, BenchmarkSettings(min_label_support=5))

        first = make_splits(examples, seed=17)
        second = make_splits(examples, seed=17)

        self.assertEqual(first, second)
        for split in first.values():
            memberships = {}
            for part, indices in split.items():
                for index in indices:
                    key = examples[index].group_key
                    if key in memberships:
                        self.assertEqual(part, memberships[key])
                    else:
                        memberships[key] = part
        chronological_memberships = {
            examples[index].group_key: part
            for part, indices in first["chronological"].items()
            for index in indices
        }
        self.assertEqual("train", chronological_memberships[examples[0].group_key])


class BenchmarkCliTests(unittest.TestCase):
    def test_cli_rejects_non_mongodb_store_without_writing_output(self):
        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.write_text("store:\n  provider: sqlite\n", encoding="utf-8")
            output_dir = Path(directory) / "output"
            stderr = StringIO()

            with redirect_stderr(stderr):
                exit_code = benchmark_main(
                    ["--config", str(config_path), "--output-dir", str(output_dir)]
                )

            self.assertEqual(2, exit_code)
            self.assertIn("supports MongoDB only", stderr.getvalue())
            self.assertFalse(output_dir.exists())


@unittest.skipIf(joblib is None, "scikit-learn benchmark dependencies are not installed")
class BenchmarkRunTests(unittest.TestCase):
    def test_single_label_metrics_do_not_count_true_negatives_as_tag_matches(self):
        ml = _ml_imports()
        y_true = ml["np"].array([[1], [0], [0], [0]])
        y_pred = ml["np"].array([[0], [0], [0], [0]])

        metrics = _aggregate_metrics(y_true, y_pred, ml)
        per_label = _per_label_metrics(y_true, y_pred, ["rare"], ml)

        self.assertEqual(0.0, metrics["micro_f1"])
        self.assertEqual(0.0, metrics["macro_f1"])
        self.assertEqual(1, per_label["rare"]["support"])

    def test_threshold_tuning_prefers_no_predictions_when_validation_has_no_tags(self):
        ml = _ml_imports()
        scores = ml["np"].array([[-1.0], [0.0], [1.0]])
        y_true = ml["np"].zeros((3, 1), dtype=int)

        _, predictions, metrics = _tune_threshold(
            scores, y_true, max_tags=1, method="decision_function", ml=ml
        )

        self.assertEqual(0, int(predictions.sum()))
        self.assertEqual(0.0, metrics["hamming_loss"])

    def test_category_ranking_prefers_coverage_adjusted_score(self):
        sparse = {
            "categories": ["GENERAL"],
            "selection_score": 0.2,
            "validation": {"micro_f1": 0.95, "macro_f1": 0.95},
            "tag_assignment_coverage": 0.1,
            "eligible_label_count": 1,
        }
        representative = {
            "categories": ["DOMAIN_ENTITY"],
            "selection_score": 0.7,
            "validation": {"micro_f1": 0.8, "macro_f1": 0.8},
            "tag_assignment_coverage": 0.9,
            "eligible_label_count": 3,
        }

        self.assertGreater(
            _combination_ranking(representative), _combination_ranking(sparse)
        )

    def test_run_writes_reports_and_loadable_winner(self):
        with TemporaryDirectory() as directory:
            rows = _training_rows(30)
            for index, row in enumerate(rows):
                row["tags"] = (
                    [{"id": 1, "name": "phishing", "category": "GENERAL"}] if index % 2 == 0 else []
                )
            report = run_benchmark(
                rows,
                BenchmarkSettings(min_label_support=5, max_features=1_000),
                directory,
            )
            run_dir = Path(report["output_directory"])

            self.assertTrue((run_dir / "report.json").is_file())
            self.assertTrue((run_dir / "report.md").is_file())
            self.assertTrue((run_dir / "best_model.joblib").is_file())
            persisted_report = json.loads((run_dir / "report.json").read_text("utf-8"))
            artifact = joblib.load(run_dir / "best_model.joblib")
            self.assertEqual(
                report["winner"]["candidate_id"], persisted_report["winner"]["candidate_id"]
            )
            self.assertEqual(["phishing"], artifact["classes"])
            self.assertEqual(5, artifact["max_tags_per_article"])
            self.assertEqual(3, report["winner"]["chronological"]["test_per_label"]["phishing"]["support"])

    def test_multiple_categories_are_compared_and_the_winner_is_persisted(self):
        rows = _training_rows(24)
        for index, row in enumerate(rows):
            if index % 3 == 0:
                row["tags"].append({"id": 3, "name": "credential theft", "category": "THREAT"})
        candidate = {
            "id": "logistic-regression-c0.5",
            "family": "logistic_regression",
            "C": 0.5,
        }

        with TemporaryDirectory() as directory:
            with patch(
                "feedsummary_core.tagging_ml.benchmark._candidate_specs",
                return_value=[candidate],
            ):
                report = run_benchmark(
                    rows,
                    BenchmarkSettings(
                        categories=("GENERAL", "THREAT"),
                        min_label_support=5,
                        max_features=1_000,
                    ),
                    directory,
                )

            comparison = report["category_comparison"]
            successful = [
                item for item in comparison["evaluated_combinations"] if item["status"] == "ok"
            ]
            artifact = joblib.load(Path(report["output_directory"]) / "best_model.joblib")
            markdown = (Path(report["output_directory"]) / "report.md").read_text("utf-8")

            self.assertEqual(3, len(successful))
            self.assertEqual([1, 2, 3], [item["rank"] for item in successful])
            self.assertEqual(report["winner"]["categories"], artifact["categories"])
            self.assertEqual(report["winner"]["categories"], comparison["winner_categories"])
            self.assertIn("Category combination comparison", markdown)


if __name__ == "__main__":
    unittest.main()

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

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from itertools import combinations
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkSettings:
    """Configuration for one reproducible benchmark run."""

    categories: tuple[str, ...] = ("GENERAL",)
    min_label_support: int = 10
    max_tags_per_article: int = 5
    max_text_chars: int = 20_000
    random_seed: int = 42
    max_articles: int | None = None
    n_jobs: int = 1
    max_features: int = 100_000
    max_category_combinations: int = 63

    def __post_init__(self) -> None:
        categories = tuple(
            dict.fromkeys(
                str(category).strip() for category in self.categories if str(category).strip()
            )
        )
        if not categories:
            raise ValueError("At least one ML tag category must be configured")
        object.__setattr__(self, "categories", categories)
        if self.min_label_support < 2:
            raise ValueError("min_label_support must be at least 2")
        if self.max_tags_per_article < 1:
            raise ValueError("max_tags_per_article must be positive")
        if self.max_text_chars < 100:
            raise ValueError("max_text_chars must be at least 100")
        if self.max_articles is not None and self.max_articles < 1:
            raise ValueError("max_articles must be positive when set")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")
        if self.max_category_combinations < 1:
            raise ValueError("max_category_combinations must be positive")


@dataclass(frozen=True)
class TrainingExample:
    article_id: str
    text: str
    labels: tuple[str, ...]
    timestamp: int
    group_key: str


def _ml_imports() -> dict[str, Any]:
    """Import optional ML dependencies only when the benchmark is invoked."""
    try:
        import joblib
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.metrics import accuracy_score, hamming_loss
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.naive_bayes import ComplementNB
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.svm import LinearSVC
    except ImportError as error:  # pragma: no cover - depends on installation extras
        raise RuntimeError("ML dependencies are missing; install feedsummary-core[ml]") from error

    return {
        "joblib": joblib,
        "np": np,
        "TfidfVectorizer": TfidfVectorizer,
        "LogisticRegression": LogisticRegression,
        "SGDClassifier": SGDClassifier,
        "accuracy_score": accuracy_score,
        "hamming_loss": hamming_loss,
        "OneVsRestClassifier": OneVsRestClassifier,
        "ComplementNB": ComplementNB,
        "Pipeline": Pipeline,
        "MultiLabelBinarizer": MultiLabelBinarizer,
        "LinearSVC": LinearSVC,
    }


def build_article_text(article: dict[str, Any], max_chars: int = 20_000) -> str:
    """Build the same kind of text that will be available during inference."""
    title = str(article.get("title") or "").strip()
    content = str(article.get("content") or "").strip()
    text = str(article.get("text") or "").strip()
    summary = str(article.get("summary") or "").strip()
    body = content or text or summary
    parts = [part for part in (title, title, body) if part]
    return "\n".join(parts)[: max(1, int(max_chars))].strip()


def prepare_training_examples(
    rows: Iterable[dict[str, Any]],
    settings: BenchmarkSettings,
) -> tuple[list[TrainingExample], dict[str, Any]]:
    """Normalize MongoDB export rows and define the benchmark label vocabulary."""
    staged: list[TrainingExample] = []
    label_support: Counter[str] = Counter()
    diagnostics: dict[str, Any] = {
        "exported_rows": 0,
        "empty_text_rows": 0,
        "missing_timestamp_rows": 0,
        "duplicate_article_rows": 0,
    }
    seen_ids = set()
    allowed_categories = {category.casefold() for category in settings.categories}

    for row in rows:
        diagnostics["exported_rows"] += 1
        article = row.get("article") if isinstance(row, dict) else None
        if not isinstance(article, dict):
            continue
        article_id = str(article.get("id") or "").strip()
        if not article_id:
            continue
        if article_id in seen_ids:
            diagnostics["duplicate_article_rows"] += 1
            continue
        seen_ids.add(article_id)

        text = build_article_text(article, settings.max_text_chars)
        if not text:
            diagnostics["empty_text_rows"] += 1
            continue

        try:
            timestamp = int(article.get("published_ts") or article.get("fetched_at") or 0)
        except (TypeError, ValueError):
            timestamp = 0
        if timestamp <= 0:
            diagnostics["missing_timestamp_rows"] += 1

        labels = tuple(
            sorted(
                {
                    str(tag.get("name") or "").strip().casefold()
                    for tag in row.get("tags") or []
                    if isinstance(tag, dict)
                    and str(tag.get("name") or "").strip()
                    and str(tag.get("category") or "GENERAL").casefold() in allowed_categories
                }
            )
        )
        label_support.update(labels)
        content_hash = str(article.get("content_hash") or "").strip()
        group_key = f"hash:{content_hash}" if content_hash else f"article:{article_id}"
        staged.append(TrainingExample(article_id, text, labels, timestamp, group_key))

    eligible_labels = sorted(
        label for label, support in label_support.items() if support >= settings.min_label_support
    )
    eligible_set = set(eligible_labels)
    examples = [
        TrainingExample(
            item.article_id,
            item.text,
            tuple(label for label in item.labels if label in eligible_set),
            item.timestamp,
            item.group_key,
        )
        for item in staged
    ]
    observed_label_count = len(label_support)
    observed_tag_assignments = sum(label_support.values())
    eligible_tag_assignments = sum(label_support[label] for label in eligible_labels)
    articles_with_category_labels = sum(bool(item.labels) for item in staged)
    articles_with_eligible_labels = sum(bool(item.labels) for item in examples)
    diagnostics.update(
        {
            "usable_articles": len(examples),
            "observed_label_count": observed_label_count,
            "eligible_label_count": len(eligible_labels),
            "label_vocabulary_coverage": (
                len(eligible_labels) / observed_label_count if observed_label_count else 0.0
            ),
            "observed_tag_assignments": observed_tag_assignments,
            "eligible_tag_assignments": eligible_tag_assignments,
            "tag_assignment_coverage": (
                eligible_tag_assignments / observed_tag_assignments
                if observed_tag_assignments
                else 0.0
            ),
            "articles_with_category_labels": articles_with_category_labels,
            "articles_with_eligible_labels": articles_with_eligible_labels,
            "eligible_article_coverage": (
                articles_with_eligible_labels / articles_with_category_labels
                if articles_with_category_labels
                else 0.0
            ),
            "eligible_labels": eligible_labels,
            "label_support": dict(sorted(label_support.items())),
            "filtered_labels": {
                label: support
                for label, support in sorted(label_support.items())
                if label not in eligible_set
            },
            "articles_without_eligible_labels": len(examples) - articles_with_eligible_labels,
        }
    )
    if len(examples) < 10:
        raise ValueError("At least 10 tagged articles with usable text are required")
    if not eligible_labels:
        raise ValueError(
            "No labels meet the configured minimum support; lower min_label_support or add data"
        )
    return examples, diagnostics


def _grouped_split(
    examples: Sequence[TrainingExample],
    *,
    chronological: bool,
    seed: int,
) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        groups[example.group_key].append(index)

    ordered_groups = list(groups.items())
    if chronological:
        ordered_groups.sort(
            key=lambda item: (
                min(examples[index].timestamp for index in item[1]),
                item[0],
            )
        )
    else:
        random.Random(seed).shuffle(ordered_groups)

    total = len(examples)
    train_target = max(1, math.floor(total * 0.64))
    validation_target = max(1, math.floor(total * 0.16))
    split = {"train": [], "validation": [], "test": []}
    for _, indices in ordered_groups:
        if len(split["train"]) < train_target:
            destination = "train"
        elif len(split["validation"]) < validation_target:
            destination = "validation"
        else:
            destination = "test"
        split[destination].extend(indices)

    if any(not split[name] for name in split):
        raise ValueError("The grouped data cannot produce non-empty train/validation/test splits")
    return split


def make_splits(
    examples: Sequence[TrainingExample], seed: int = 42
) -> dict[str, dict[str, list[int]]]:
    return {
        "chronological": _grouped_split(examples, chronological=True, seed=seed),
        "random": _grouped_split(examples, chronological=False, seed=seed),
    }


def _candidate_specs() -> list[dict[str, Any]]:
    return (
        [
            {"id": f"logistic-regression-c{value}", "family": "logistic_regression", "C": value}
            for value in (0.5, 2.0)
        ]
        + [
            {"id": f"linear-svc-c{value}", "family": "linear_svc", "C": value}
            for value in (0.5, 1.5)
        ]
        + [
            {"id": f"sgd-log-alpha{value:g}", "family": "sgd", "alpha": value}
            for value in (1e-5, 1e-4)
        ]
        + [
            {"id": f"complement-nb-alpha{value}", "family": "complement_nb", "alpha": value}
            for value in (0.1, 1.0)
        ]
    )


def _build_pipeline(spec: dict[str, Any], settings: BenchmarkSettings, ml: dict[str, Any]) -> Any:
    family = spec["family"]
    if family == "logistic_regression":
        estimator = ml["LogisticRegression"](
            C=float(spec["C"]),
            class_weight="balanced",
            max_iter=1_000,
            solver="liblinear",
            random_state=settings.random_seed,
        )
    elif family == "linear_svc":
        estimator = ml["LinearSVC"](
            C=float(spec["C"]),
            class_weight="balanced",
            random_state=settings.random_seed,
        )
    elif family == "sgd":
        estimator = ml["SGDClassifier"](
            loss="log_loss",
            alpha=float(spec["alpha"]),
            class_weight="balanced",
            max_iter=1_000,
            tol=1e-3,
            random_state=settings.random_seed,
        )
    elif family == "complement_nb":
        estimator = ml["ComplementNB"](alpha=float(spec["alpha"]))
    else:  # pragma: no cover - candidate registry is internal
        raise ValueError(f"Unknown candidate family: {family}")

    classifier = ml["OneVsRestClassifier"](estimator, n_jobs=settings.n_jobs)
    vectorizer = ml["TfidfVectorizer"](
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=settings.max_features,
        sublinear_tf=True,
    )
    return ml["Pipeline"]([("tfidf", vectorizer), ("classifier", classifier)])


def _score_matrix(
    pipeline: Any,
    texts: Sequence[str],
    label_count: int,
    ml: dict[str, Any],
) -> tuple[Any, str]:
    np = ml["np"]
    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(texts)
        method = "decision_function"
    else:
        scores = pipeline.predict_proba(texts)
        method = "predict_proba"
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    elif label_count == 1 and matrix.shape[1] == 2:
        # In the single-label case, probability-based binary estimators expose
        # [negative, positive] columns instead of one multilabel-positive column.
        matrix = matrix[:, 1:2]
    if matrix.shape[1] != label_count:
        raise ValueError(
            f"Estimator returned {matrix.shape[1]} score columns for {label_count} labels"
        )
    return matrix, method


def _threshold_predictions(scores: Any, threshold: float, max_tags: int, ml: dict[str, Any]) -> Any:
    np = ml["np"]
    predictions = np.asarray(scores >= threshold, dtype=int)
    for row_index in range(predictions.shape[0]):
        positive = np.flatnonzero(predictions[row_index])
        if len(positive) <= max_tags:
            continue
        ranked = positive[np.argsort(scores[row_index, positive])[::-1]]
        predictions[row_index, :] = 0
        predictions[row_index, ranked[:max_tags]] = 1
    return predictions


def _aggregate_metrics(y_true: Any, y_pred: Any, ml: dict[str, Any]) -> dict[str, float]:
    np = ml["np"]
    true_matrix = np.asarray(y_true, dtype=bool)
    predicted_matrix = np.asarray(y_pred, dtype=bool)
    if true_matrix.ndim != 2 or predicted_matrix.shape != true_matrix.shape:
        raise ValueError("Multilabel metrics require equally shaped two-dimensional matrices")

    true_positives = np.logical_and(true_matrix, predicted_matrix).sum(axis=0)
    false_positives = np.logical_and(~true_matrix, predicted_matrix).sum(axis=0)
    false_negatives = np.logical_and(true_matrix, ~predicted_matrix).sum(axis=0)
    support = true_positives + false_negatives

    def divide(numerator: Any, denominator: Any) -> Any:
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(denominator, dtype=float),
            where=denominator != 0,
        )

    per_label_precision = divide(true_positives, true_positives + false_positives)
    per_label_recall = divide(true_positives, support)
    per_label_f1 = divide(
        2.0 * per_label_precision * per_label_recall,
        per_label_precision + per_label_recall,
    )
    micro_precision = float(divide(true_positives.sum(), true_positives.sum() + false_positives.sum()))
    micro_recall = float(divide(true_positives.sum(), support.sum()))
    micro_f1 = float(
        divide(2.0 * micro_precision * micro_recall, micro_precision + micro_recall)
    )
    support_total = int(support.sum())
    weights = support / support_total if support_total else np.zeros_like(support, dtype=float)
    metrics = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": float(per_label_precision.mean()),
        "macro_recall": float(per_label_recall.mean()),
        "macro_f1": float(per_label_f1.mean()),
        "weighted_precision": float(np.dot(per_label_precision, weights)),
        "weighted_recall": float(np.dot(per_label_recall, weights)),
        "weighted_f1": float(np.dot(per_label_f1, weights)),
    }

    true_counts = true_matrix.sum(axis=1)
    predicted_counts = predicted_matrix.sum(axis=1)
    intersections = np.logical_and(true_matrix, predicted_matrix).sum(axis=1)
    denominators = true_counts + predicted_counts
    sample_scores = divide(2.0 * intersections, denominators)
    metrics["samples_f1"] = float(sample_scores.mean())
    metrics["hamming_loss"] = float(ml["hamming_loss"](true_matrix, predicted_matrix))
    metrics["subset_accuracy"] = float(ml["accuracy_score"](true_matrix, predicted_matrix))
    metrics["average_true_tags"] = float(true_counts.mean())
    metrics["average_predicted_tags"] = float(predicted_counts.mean())
    return metrics


def _per_label_metrics(
    y_true: Any, y_pred: Any, label_names: Sequence[str], ml: dict[str, Any]
) -> dict[str, dict[str, float | int]]:
    np = ml["np"]
    true_matrix = np.asarray(y_true, dtype=bool)
    predicted_matrix = np.asarray(y_pred, dtype=bool)
    true_positives = np.logical_and(true_matrix, predicted_matrix).sum(axis=0)
    false_positives = np.logical_and(~true_matrix, predicted_matrix).sum(axis=0)
    support = true_matrix.sum(axis=0)

    precision = np.divide(
        true_positives,
        true_positives + false_positives,
        out=np.zeros_like(true_positives, dtype=float),
        where=(true_positives + false_positives) != 0,
    )
    recall = np.divide(
        true_positives,
        support,
        out=np.zeros_like(true_positives, dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision, dtype=float),
        where=(precision + recall) != 0,
    )
    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(label_names)
    }


def _coverage_adjusted_f1(metrics: dict[str, float], coverage: float) -> float:
    """Treat tag assignments filtered by minimum support as additional false negatives."""
    precision = metrics["micro_precision"]
    adjusted_recall = metrics["micro_recall"] * coverage
    denominator = precision + adjusted_recall
    return 2.0 * precision * adjusted_recall / denominator if denominator else 0.0


def _tune_threshold(
    scores: Any, y_true: Any, max_tags: int, method: str, ml: dict[str, Any]
) -> tuple[float, Any, dict[str, float]]:
    np = ml["np"]
    flattened = np.asarray(scores, dtype=float).ravel()
    quantiles = np.quantile(flattened, np.linspace(0.01, 0.99, 99))
    default = 0.0 if method == "decision_function" else 0.5
    predict_none = float(np.nextafter(flattened.max(), np.inf))
    thresholds = sorted({float(value) for value in quantiles} | {default, predict_none})
    best_ranking: tuple[float, ...] | None = None
    best_predictions: Any = None
    best_metrics: dict[str, float] | None = None
    best_threshold = default
    for threshold in thresholds:
        predictions = _threshold_predictions(scores, threshold, max_tags, ml)
        metrics = _aggregate_metrics(y_true, predictions, ml)
        ranking = (
            metrics["micro_f1"],
            metrics["macro_f1"],
            -metrics["hamming_loss"],
            -metrics["average_predicted_tags"],
        )
        if best_ranking is None or ranking > best_ranking:
            best_ranking = ranking
            best_predictions = predictions
            best_metrics = metrics
            best_threshold = threshold
    assert best_metrics is not None
    return best_threshold, best_predictions, best_metrics


def _subset(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices]


def _run_candidate(
    spec: dict[str, Any],
    settings: BenchmarkSettings,
    texts: Sequence[str],
    targets: Any,
    split: dict[str, list[int]],
    label_names: Sequence[str],
    ml: dict[str, Any],
) -> dict[str, Any]:
    np = ml["np"]
    pipeline = _build_pipeline(spec, settings, ml)
    started = time.perf_counter()
    pipeline.fit(_subset(texts, split["train"]), targets[split["train"]])
    fit_seconds = time.perf_counter() - started

    validation_scores, method = _score_matrix(
        pipeline,
        _subset(texts, split["validation"]),
        len(label_names),
        ml,
    )
    threshold, _, validation_metrics = _tune_threshold(
        validation_scores,
        targets[split["validation"]],
        settings.max_tags_per_article,
        method,
        ml,
    )
    predict_started = time.perf_counter()
    test_scores, _ = _score_matrix(
        pipeline,
        _subset(texts, split["test"]),
        len(label_names),
        ml,
    )
    test_predictions = _threshold_predictions(
        test_scores, threshold, settings.max_tags_per_article, ml
    )
    predict_seconds = time.perf_counter() - predict_started
    test_targets = np.asarray(targets[split["test"]])
    test_metrics = _aggregate_metrics(test_targets, test_predictions, ml)

    return {
        "candidate": dict(spec),
        "status": "ok",
        "score_method": method,
        "threshold": float(threshold),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "validation": validation_metrics,
        "test": test_metrics,
        "test_per_label": _per_label_metrics(test_targets, test_predictions, label_names, ml),
    }


def _corpus_fingerprint(examples: Sequence[TrainingExample], settings: BenchmarkSettings) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(settings), sort_keys=True).encode("utf-8"))
    for item in sorted(examples, key=lambda example: example.article_id):
        digest.update(item.article_id.encode("utf-8"))
        digest.update(item.group_key.encode("utf-8"))
        digest.update(hashlib.sha256(item.text.encode("utf-8")).digest())
        digest.update(str(item.timestamp).encode("ascii"))
        digest.update("\0".join(item.labels).encode("utf-8"))
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("feedsummary-core", "scikit-learn", "numpy", "scipy", "joblib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _category_combinations(settings: BenchmarkSettings) -> list[tuple[str, ...]]:
    categories = settings.categories
    combination_count = (2 ** len(categories)) - 1
    if combination_count > settings.max_category_combinations:
        raise ValueError(
            f"{len(categories)} categories produce {combination_count} combinations, "
            f"above max_category_combinations={settings.max_category_combinations}"
        )
    return [
        combination
        for size in range(1, len(categories) + 1)
        for combination in combinations(categories, size)
    ]


def _evaluate_benchmark(
    rows: Sequence[dict[str, Any]],
    settings: BenchmarkSettings,
    ml: dict[str, Any],
) -> dict[str, Any]:
    examples, diagnostics = prepare_training_examples(rows, settings)
    label_names = diagnostics["eligible_labels"]
    texts = [example.text for example in examples]
    mlb = ml["MultiLabelBinarizer"](classes=label_names)
    targets = mlb.fit_transform([example.labels for example in examples])
    splits = make_splits(examples, settings.random_seed)
    results: dict[str, list[dict[str, Any]]] = {}

    for split_name, split in splits.items():
        split_results = []
        for spec in _candidate_specs():
            try:
                result = _run_candidate(spec, settings, texts, targets, split, label_names, ml)
            except Exception as error:  # noqa: BLE001 - one model must not abort the benchmark
                result = {
                    "candidate": dict(spec),
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            split_results.append(result)
        results[split_name] = split_results

    successful_chronological = [
        result for result in results["chronological"] if result["status"] == "ok"
    ]
    if not successful_chronological:
        failure_counts = Counter(
            result.get("error", "unknown error") for result in results["chronological"]
        )
        details = "; ".join(f"{count}x {error}" for error, count in failure_counts.most_common(3))
        raise RuntimeError(f"Every benchmark candidate failed. Causes: {details}")

    chronological_winner = max(
        successful_chronological,
        key=lambda result: (
            result["validation"]["micro_f1"],
            result["validation"]["macro_f1"],
            -result["predict_seconds"],
            result["candidate"]["id"],
        ),
    )
    candidate_id = chronological_winner["candidate"]["id"]
    random_winner_result = next(
        (
            result
            for result in results["random"]
            if result["status"] == "ok" and result["candidate"]["id"] == candidate_id
        ),
        None,
    )
    return {
        "settings": settings,
        "examples": examples,
        "diagnostics": diagnostics,
        "texts": texts,
        "targets": targets,
        "splits": splits,
        "results": results,
        "chronological_winner": chronological_winner,
        "random_winner": random_winner_result,
    }


def _combination_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    settings = evaluation["settings"]
    diagnostics = evaluation["diagnostics"]
    chronological = evaluation["chronological_winner"]
    random_result = evaluation["random_winner"]
    assignment_coverage = diagnostics["tag_assignment_coverage"]
    return {
        "categories": list(settings.categories),
        "status": "ok",
        "observed_label_count": diagnostics["observed_label_count"],
        "eligible_label_count": diagnostics["eligible_label_count"],
        "label_vocabulary_coverage": diagnostics["label_vocabulary_coverage"],
        "observed_tag_assignments": diagnostics["observed_tag_assignments"],
        "eligible_tag_assignments": diagnostics["eligible_tag_assignments"],
        "tag_assignment_coverage": assignment_coverage,
        "eligible_article_coverage": diagnostics["eligible_article_coverage"],
        "selection_score": _coverage_adjusted_f1(
            chronological["validation"], assignment_coverage
        ),
        "candidate_id": chronological["candidate"]["id"],
        "validation": chronological["validation"],
        "chronological_test": chronological["test"],
        "random_test": random_result["test"] if random_result else None,
    }


def _combination_ranking(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        summary["selection_score"],
        summary["validation"]["micro_f1"],
        summary["validation"]["macro_f1"],
        summary["tag_assignment_coverage"],
        summary["eligible_label_count"],
        -len(summary["categories"]),
        tuple(summary["categories"]),
    )


def _markdown_report(report: dict[str, Any]) -> str:
    winner = report["winner"]
    chronological = winner["chronological"]
    random_result = winner.get("random")
    lines = [
        "# ML tagging benchmark",
        "",
        f"Generated: {report['generated_at']}",
        f"Corpus fingerprint: `{report['corpus_fingerprint']}`",
        f"Articles: {report['dataset']['usable_articles']}",
        (
            "Eligible labels: "
            f"{report['dataset']['eligible_label_count']}/"
            f"{report['dataset']['observed_label_count']} "
            f"({report['dataset']['label_vocabulary_coverage']:.1%})"
        ),
        f"Eligible tag assignments: {report['dataset']['tag_assignment_coverage']:.1%}",
        "",
        "## Winner",
        "",
        f"- Categories: `{', '.join(winner['categories'])}`",
        f"- Candidate: `{winner['candidate_id']}`",
        f"- Coverage-adjusted validation micro-F1: `{winner['selection_score']:.4f}`",
        f"- Threshold: `{chronological['threshold']:.6g}`",
        f"- Chronological test micro-F1: `{chronological['test']['micro_f1']:.4f}`",
        f"- Chronological test macro-F1: `{chronological['test']['macro_f1']:.4f}`",
    ]
    if random_result:
        lines.append(f"- Random test micro-F1: `{random_result['test']['micro_f1']:.4f}`")
    comparison = report.get("category_comparison") or {}
    evaluated = comparison.get("evaluated_combinations") or []
    if len(evaluated) > 1:
        lines.extend(
            [
                "",
                "## Category combination comparison",
                "",
                (
                    "Selection uses coverage-adjusted chronological validation micro-F1; "
                    "test scores are reported only. Rare tag assignments excluded by the "
                    "minimum-support threshold count as missed tags in the selection score."
                ),
                "",
                "| Rank | Categories | Eligible labels | Tag assignment coverage | Selection score | Validation micro-F1 | Validation macro-F1 | Chronological test micro-F1 | Random test micro-F1 | Candidate |",
                "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        rank = 0
        for item in evaluated:
            if item["status"] != "ok":
                continue
            rank += 1
            random_test = item.get("random_test")
            random_score = f"{random_test['micro_f1']:.4f}" if random_test else "n/a"
            lines.append(
                "| "
                f"{rank} | {', '.join(item['categories'])} | "
                f"{item['eligible_label_count']}/{item['observed_label_count']} | "
                f"{item['tag_assignment_coverage']:.1%} | "
                f"{item['selection_score']:.4f} | "
                f"{item['validation']['micro_f1']:.4f} | "
                f"{item['validation']['macro_f1']:.4f} | "
                f"{item['chronological_test']['micro_f1']:.4f} | "
                f"{random_score} | "
                f"{item['candidate_id']} |"
            )
        skipped = [item for item in evaluated if item["status"] != "ok"]
        if skipped:
            lines.extend(
                [
                    "",
                    f"Skipped combinations: {len(skipped)}. See `report.json` for reasons.",
                ]
            )
    lines.extend(
        [
            "",
            "## Data warnings",
            "",
            "Historical database tags are treated as labels and may contain LLM-generated noise.",
            f"Filtered rare labels: {len(report['dataset']['filtered_labels'])}.",
            f"Articles without eligible labels: {report['dataset']['articles_without_eligible_labels']}.",
            (
                "Coverage-adjusted selection scores estimate performance when filtered rare "
                "tag assignments are treated as unpredicted."
            ),
            "",
            "The Joblib artifact must only be loaded from a trusted source.",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(
    rows: Iterable[dict[str, Any]],
    settings: BenchmarkSettings,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compare category subsets, persist reports, and retrain the overall winner."""
    ml = _ml_imports()
    exported_rows = list(rows)
    evaluated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    skipped: list[dict[str, Any]] = []
    category_sets = _category_combinations(settings)
    for index, categories in enumerate(category_sets, start=1):
        logger.info(
            "Benchmarking category combination %d/%d: %s",
            index,
            len(category_sets),
            ", ".join(categories),
        )
        combination_settings = BenchmarkSettings(**{**asdict(settings), "categories": categories})
        try:
            evaluation = _evaluate_benchmark(exported_rows, combination_settings, ml)
        except (RuntimeError, ValueError) as error:
            skipped.append(
                {
                    "categories": list(categories),
                    "status": "skipped",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        summary = _combination_summary(evaluation)
        evaluated.append((summary, evaluation))
        logger.info(
            "Completed %s with selection score %.4f and validation micro-F1 %.4f",
            ", ".join(categories),
            summary["selection_score"],
            evaluation["chronological_winner"]["validation"]["micro_f1"],
        )

    if not evaluated:
        reasons = "; ".join(
            f"{'+'.join(item['categories'])}: {item['error']}" for item in skipped[:3]
        )
        raise RuntimeError(f"Every category combination failed. Causes: {reasons}")

    evaluated.sort(key=lambda item: _combination_ranking(item[0]), reverse=True)
    for rank, (summary, _) in enumerate(evaluated, start=1):
        summary["rank"] = rank
    best_summary, best_evaluation = evaluated[0]
    selected_settings: BenchmarkSettings = best_evaluation["settings"]
    examples = best_evaluation["examples"]
    diagnostics = best_evaluation["diagnostics"]
    texts = best_evaluation["texts"]
    targets = best_evaluation["targets"]
    splits = best_evaluation["splits"]
    results = best_evaluation["results"]
    chronological_winner = best_evaluation["chronological_winner"]
    random_winner_result = best_evaluation["random_winner"]
    candidate_id = chronological_winner["candidate"]["id"]
    label_names = diagnostics["eligible_labels"]

    final_pipeline = _build_pipeline(chronological_winner["candidate"], selected_settings, ml)
    final_pipeline.fit(texts, targets)
    fingerprint = _corpus_fingerprint(examples, selected_settings)
    generated_at = datetime.now(timezone.utc).isoformat()
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"-{fingerprint[:8]}"
    run_dir = Path(output_dir).expanduser().resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    artifact = {
        "pipeline": final_pipeline,
        "classes": list(label_names),
        "threshold": chronological_winner["threshold"],
        "score_method": chronological_winner["score_method"],
        "max_tags_per_article": selected_settings.max_tags_per_article,
        "categories": list(selected_settings.categories),
        "max_text_chars": selected_settings.max_text_chars,
        "text_builder": "title_twice_then_content_text_or_summary_v2",
        "tfidf": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.98,
            "max_features": selected_settings.max_features,
            "sublinear_tf": True,
        },
        "corpus_fingerprint": fingerprint,
        "trained_at": generated_at,
        "versions": _package_versions(),
    }
    model_path = run_dir / "best_model.joblib"
    ml["joblib"].dump(artifact, model_path)

    report = {
        "generated_at": generated_at,
        "corpus_fingerprint": fingerprint,
        "settings": asdict(settings),
        "selected_settings": asdict(selected_settings),
        "dataset": diagnostics,
        "splits": {
            name: {part: len(indices) for part, indices in split.items()}
            for name, split in splits.items()
        },
        "results": results,
        "winner": {
            "categories": list(selected_settings.categories),
            "candidate_id": candidate_id,
            "selection_score": best_summary["selection_score"],
            "selection_rule": (
                "coverage-adjusted chronological validation micro-F1 across category combinations"
            ),
            "chronological": chronological_winner,
            "random": random_winner_result,
        },
        "category_comparison": {
            "requested_categories": list(settings.categories),
            "selection_rule": (
                "coverage-adjusted chronological validation micro-F1, then raw micro-F1, "
                "macro-F1, tag assignment coverage, label coverage, and fewer categories"
            ),
            "winner_categories": list(selected_settings.categories),
            "evaluated_combinations": [summary for summary, _ in evaluated] + skipped,
        },
        "artifact": {
            "model": model_path.name,
            "trusted_source_only": True,
            "versions": artifact["versions"],
        },
        "output_directory": str(run_dir),
    }
    (run_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(_markdown_report(report), encoding="utf-8")
    return report

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
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES ARISING FROM USE.

"""Configurable multilabel tagging based on persisted article embeddings."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _ml_imports() -> dict[str, Any]:
    try:
        import joblib
        import numpy as np
        from sklearn.linear_model import LogisticRegression, SGDClassifier
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer, Normalizer
    except ImportError as error:  # pragma: no cover - installation dependent
        raise RuntimeError("ML dependencies are missing; install feedsummary-core[ml]") from error
    return {
        "joblib": joblib,
        "np": np,
        "LogisticRegression": LogisticRegression,
        "SGDClassifier": SGDClassifier,
        "OneVsRestClassifier": OneVsRestClassifier,
        "MultiLabelBinarizer": MultiLabelBinarizer,
        "Normalizer": Normalizer,
    }


@dataclass(frozen=True)
class EmbeddingClassifierSettings:
    enabled: bool = False
    algorithm: str = ""
    categories: tuple[str, ...] = ()
    model_path: str = "data/tagging_ml/embedding_classifier.joblib"
    embedding_model: str = ""
    min_label_support: int = 10
    min_training_articles: int = 30
    max_tags_per_article: int = 5
    threshold: float = 0.5
    regularization_c: float = 1.0
    alpha: float = 0.0001
    max_iter: int = 2_000
    tolerance: float = 1e-4
    random_state: int = 42
    auto_retrain: bool = True

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        embedding_model: str = "",
    ) -> EmbeddingClassifierSettings:
        tagging = config.get("tagging") if isinstance(config.get("tagging"), dict) else {}
        raw = tagging.get("ml") if isinstance(tagging.get("ml"), dict) else {}
        categories = raw.get("categories") or []
        if isinstance(categories, str):
            categories = [categories]
        elif not isinstance(categories, (list, tuple)):
            raise TypeError("tagging.ml.categories must be a list of category names")
        algorithm = str(raw.get("classifier") or raw.get("algorithm") or "").strip().lower()
        representation = str(raw.get("representation") or "embedding").strip().lower()
        if algorithm and algorithm not in {"logistic_regression", "sgd"}:
            raise ValueError(
                "tagging.ml.classifier must be 'logistic_regression' or 'sgd'"
            )
        if representation not in {"embedding", "embeddings"}:
            raise ValueError(
                "tagging.ml.representation must be 'embedding' in this implementation"
            )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            algorithm=algorithm,
            categories=tuple(
                dict.fromkeys(
                    str(category).strip().upper()
                    for category in categories
                    if str(category).strip()
                )
            ),
            model_path=str(
                raw.get("model_path")
                or "data/tagging_ml/embedding_classifier.joblib"
            ),
            embedding_model=str(raw.get("embedding_model") or embedding_model or "").strip(),
            min_label_support=max(2, int(raw.get("min_label_support", 10))),
            min_training_articles=max(10, int(raw.get("min_training_articles", 30))),
            max_tags_per_article=max(1, int(raw.get("max_tags_per_article", 5))),
            threshold=float(raw.get("threshold", 0.5)),
            regularization_c=float(raw.get("regularization_c", raw.get("c", 1.0))),
            alpha=float(raw.get("alpha", 0.0001)),
            max_iter=max(1, int(raw.get("max_iter", 2_000))),
            tolerance=float(raw.get("tolerance", raw.get("tol", 1e-4))),
            random_state=int(raw.get("random_state", 42)),
            auto_retrain=bool(raw.get("auto_retrain", True)),
        )

    def __post_init__(self) -> None:
        if self.enabled and not self.categories:
            raise ValueError("tagging.ml.categories is required when ML tagging is enabled")
        if self.enabled and not self.algorithm:
            raise ValueError("tagging.ml.classifier is required when ML tagging is enabled")
        if self.algorithm and self.algorithm not in {"logistic_regression", "sgd"}:
            raise ValueError("Unsupported tagging.ml.classifier")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("tagging.ml.threshold must be between 0 and 1")
        if self.regularization_c <= 0:
            raise ValueError("tagging.ml.regularization_c must be positive")
        if self.alpha <= 0:
            raise ValueError("tagging.ml.alpha must be positive")
        if self.tolerance <= 0:
            raise ValueError("tagging.ml.tolerance must be positive")


@dataclass(frozen=True)
class _TrainingRow:
    article_id: str
    embedding: tuple[float, ...]
    labels: tuple[str, ...]
    source_hash: str


class EmbeddingClassifierTagger:
    """Train, persist, refresh, and use a configured embedding classifier."""

    ARTIFACT_VERSION = 2

    def __init__(self, settings: EmbeddingClassifierSettings):
        self.settings = settings
        self._artifact: dict[str, Any] | None = None

    @property
    def ready(self) -> bool:
        return self._artifact is not None

    @property
    def model_metadata(self) -> dict[str, Any]:
        """Return non-sensitive model metadata suitable for diagnostics and logs."""
        artifact = self._artifact or {}
        return {
            "artifact_version": artifact.get("artifact_version"),
            "classifier": self.settings.algorithm,
            "categories": list(self.settings.categories),
            "threshold": self.settings.threshold,
            "embedding_model": str(
                artifact.get("embedding_model") or self.settings.embedding_model
            ),
            "embedding_dimension": artifact.get("embedding_dimension"),
            "training_articles": artifact.get("training_articles"),
            "label_count": len(artifact.get("classes") or ()),
            "trained_at": artifact.get("trained_at"),
            "corpus_fingerprint": artifact.get("corpus_fingerprint"),
        }

    @property
    def label_names(self) -> tuple[str, ...]:
        """Return the label vocabulary stored in the loaded artifact."""
        return tuple(str(name) for name in (self._artifact or {}).get("classes") or ())

    def _model_path(self) -> Path:
        return Path(os.path.expandvars(os.path.expanduser(self.settings.model_path))).resolve()

    def load(self) -> bool:
        """Load a trusted local artifact and reject incompatible configuration."""
        path = self._model_path()
        if not path.is_file():
            return False
        artifact = _ml_imports()["joblib"].load(path)
        if (
            not isinstance(artifact, dict)
            or artifact.get("artifact_version") != self.ARTIFACT_VERSION
        ):
            raise ValueError(f"Unsupported ML tag artifact at {path}")
        if tuple(artifact.get("categories") or ()) != self.settings.categories:
            logger.info("ML tag artifact categories changed; retraining is required")
            return False
        if str(artifact.get("algorithm") or "") != self.settings.algorithm:
            logger.info("ML tag artifact algorithm changed; retraining is required")
            return False
        if self.settings.embedding_model and (
            str(artifact.get("embedding_model") or "") != self.settings.embedding_model
        ):
            logger.info("ML tag artifact embedding model changed; retraining is required")
            return False
        self._artifact = artifact
        return True

    def _prepare_rows(
        self,
        exported_rows: Iterable[dict[str, Any]],
    ) -> tuple[list[_TrainingRow], str, int, str]:
        staged: list[_TrainingRow] = []
        support: Counter[str] = Counter()
        expected_dimension = 0
        observed_model = self.settings.embedding_model
        allowed_categories = set(self.settings.categories)

        for exported in exported_rows:
            article = exported.get("article") if isinstance(exported, dict) else None
            if not isinstance(article, dict):
                continue
            article_id = str(article.get("id") or "").strip()
            vector = article.get("embedding_vector")
            model = str(article.get("embedding_model") or "").strip()
            if not article_id or not isinstance(vector, list) or not vector:
                continue
            if not all(isinstance(value, (int, float)) for value in vector):
                continue
            if self.settings.embedding_model and model != self.settings.embedding_model:
                continue
            if not observed_model:
                observed_model = model
            if model != observed_model:
                continue
            if not expected_dimension:
                expected_dimension = len(vector)
            if len(vector) != expected_dimension:
                continue
            labels = tuple(
                sorted(
                    {
                        str(tag.get("name") or "").strip().casefold()
                        for tag in exported.get("tags") or []
                        if isinstance(tag, dict)
                        and str(tag.get("name") or "").strip()
                        and str(tag.get("category") or "GENERAL").strip().upper()
                        in allowed_categories
                    }
                )
            )
            support.update(labels)
            staged.append(
                _TrainingRow(
                    article_id=article_id,
                    embedding=tuple(float(value) for value in vector),
                    labels=labels,
                    source_hash=str(article.get("embedding_source_hash") or ""),
                )
            )

        eligible = {
            label
            for label, count in support.items()
            if count >= self.settings.min_label_support and count < len(staged)
        }
        rows = [
            _TrainingRow(
                row.article_id,
                row.embedding,
                tuple(label for label in row.labels if label in eligible),
                row.source_hash,
            )
            for row in staged
        ]
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                {
                    "algorithm": self.settings.algorithm,
                    "categories": self.settings.categories,
                    "min_label_support": self.settings.min_label_support,
                    "regularization_c": self.settings.regularization_c,
                    "alpha": self.settings.alpha,
                    "max_iter": self.settings.max_iter,
                    "tolerance": self.settings.tolerance,
                    "random_state": self.settings.random_state,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        for row in sorted(rows, key=lambda item: item.article_id):
            digest.update(row.article_id.encode("utf-8"))
            digest.update(row.source_hash.encode("utf-8"))
            digest.update(json.dumps(row.embedding).encode("utf-8"))
            digest.update("\0".join(row.labels).encode("utf-8"))
        return rows, digest.hexdigest(), expected_dimension, observed_model

    def refresh_from_store(self, store: Any, *, force: bool = False) -> bool:
        """Load or retrain from persisted embeddings and tag assignments.

        ``force`` bypasses both the corpus-fingerprint shortcut and
        ``auto_retrain``.  The existing artifact is still kept if the current
        training corpus cannot produce a replacement model.
        """
        if not self.settings.enabled:
            return False
        iterator = getattr(store, "iter_articles_with_tags", None)
        if not callable(iterator):
            logger.warning("ML tagging requires a store with iter_articles_with_tags()")
            return False if force else self.load()

        rows, fingerprint, dimension, embedding_model = self._prepare_rows(
            iterator(categories=list(self.settings.categories))
        )
        loaded = self.load()
        if (
            not force
            and loaded
            and self._artifact
            and self._artifact.get("corpus_fingerprint") == fingerprint
        ):
            return True
        if not force and not self.settings.auto_retrain and loaded:
            logger.warning("ML tag data changed but tagging.ml.auto_retrain is disabled")
            return True
        if len(rows) < self.settings.min_training_articles:
            logger.warning(
                "ML tagging needs at least %d embedding-covered articles; found %d",
                self.settings.min_training_articles,
                len(rows),
            )
            return False if force else loaded

        label_names = sorted({label for row in rows for label in row.labels})
        if not label_names:
            logger.warning(
                "No ML tag labels meet min_label_support=%d",
                self.settings.min_label_support,
            )
            return False if force else loaded

        ml = _ml_imports()
        np = ml["np"]
        embeddings = np.asarray([row.embedding for row in rows], dtype=float)
        feature_transformer = ml["Normalizer"](norm="l2")
        features = feature_transformer.fit_transform(embeddings)
        binarizer = ml["MultiLabelBinarizer"](classes=label_names)
        targets = binarizer.fit_transform([row.labels for row in rows])
        if self.settings.algorithm == "logistic_regression":
            estimator = ml["LogisticRegression"](
                C=self.settings.regularization_c,
                class_weight="balanced",
                max_iter=self.settings.max_iter,
                tol=self.settings.tolerance,
                random_state=self.settings.random_state,
            )
        else:
            estimator = ml["SGDClassifier"](
                loss="log_loss",
                alpha=self.settings.alpha,
                class_weight="balanced",
                max_iter=self.settings.max_iter,
                tol=self.settings.tolerance,
                random_state=self.settings.random_state,
            )
        if targets.shape[1] == 1:
            classifier = estimator.fit(features, targets[:, 0])
            classifier_kind = "binary"
        else:
            classifier = ml["OneVsRestClassifier"](estimator, n_jobs=1)
            classifier.fit(features, targets)
            classifier_kind = "multilabel"
        artifact = {
            "artifact_version": self.ARTIFACT_VERSION,
            "algorithm": self.settings.algorithm,
            "representation": "embedding",
            "classifier": classifier,
            "classifier_kind": classifier_kind,
            "feature_transformer": feature_transformer,
            "classes": label_names,
            "categories": list(self.settings.categories),
            "embedding_model": embedding_model,
            "embedding_dimension": dimension,
            "threshold": self.settings.threshold,
            "max_tags_per_article": self.settings.max_tags_per_article,
            "corpus_fingerprint": fingerprint,
            "training_articles": len(rows),
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save_atomic(artifact, ml["joblib"])
        self._artifact = artifact
        logger.info(
            "Trained embedding %s tagger on %d articles and %d labels",
            self.settings.algorithm,
            len(rows),
            len(label_names),
        )
        return True

    def _save_atomic(self, artifact: dict[str, Any], joblib: Any) -> None:
        path = self._model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            joblib.dump(artifact, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def score_names(self, article: dict[str, Any]) -> list[tuple[str, float]]:
        """Return every model label and probability, ordered by descending score."""
        if not self.can_predict(article):
            return []
        assert self._artifact is not None
        vector = article.get("embedding_vector")
        assert isinstance(vector, list)

        ml = _ml_imports()
        features = self._artifact["feature_transformer"].transform(
            ml["np"].asarray([vector], dtype=float)
        )
        raw_probabilities = self._artifact["classifier"].predict_proba(features)
        if self._artifact.get("classifier_kind") == "binary":
            probabilities = ml["np"].asarray([raw_probabilities[0, 1]], dtype=float)
        else:
            probabilities = ml["np"].asarray(raw_probabilities[0], dtype=float)
        classes = list(self._artifact["classes"])
        scored = [
            (classes[index], float(score))
            for index, score in enumerate(probabilities)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    def predict_names(self, article: dict[str, Any]) -> list[tuple[str, float]]:
        selected = [
            item for item in self.score_names(article) if item[1] >= self.settings.threshold
        ]
        return selected[: self.settings.max_tags_per_article]

    def can_predict(self, article: dict[str, Any]) -> bool:
        """Return whether an article has an embedding compatible with the artifact."""
        return self.embedding_incompatibility_reason(article) is None

    def embedding_incompatibility_reason(self, article: dict[str, Any]) -> str | None:
        """Explain why an article cannot be scored, or return ``None`` when compatible."""
        if not self._artifact:
            return "model_not_ready"
        vector = article.get("embedding_vector")
        if not isinstance(vector, list) or not vector:
            return "missing_embedding"
        if not all(isinstance(value, (int, float)) for value in vector):
            return "invalid_embedding"
        if len(vector) != int(self._artifact["embedding_dimension"]):
            return "embedding_dimension_mismatch"
        expected_model = str(self._artifact.get("embedding_model") or "")
        if expected_model and str(article.get("embedding_model") or "") != expected_model:
            return "embedding_model_mismatch"
        return None

    def predict_tags(
        self,
        article: dict[str, Any],
        store: Any,
        *,
        scores: list[tuple[str, float]] | None = None,
    ) -> list[dict[str, Any]]:
        predictions = []
        selected = [
            item
            for item in (scores if scores is not None else self.score_names(article))
            if item[1] >= self.settings.threshold
        ][: self.settings.max_tags_per_article]
        for name, probability in selected:
            tag = store.get_tag_by_name(name)
            if not isinstance(tag, dict):
                continue
            if str(tag.get("category") or "").strip().upper() not in self.settings.categories:
                continue
            predicted = dict(tag)
            predicted["reasoning"] = (
                "Predicted by embedding "
                f"{self.settings.algorithm} model (probability={probability:.3f})."
            )
            predicted["ml_probability"] = probability
            predictions.append(predicted)
        return predictions


# Backwards-compatible names for callers using the original production API.
EmbeddingSGDSettings = EmbeddingClassifierSettings
EmbeddingSGDTagger = EmbeddingClassifierTagger

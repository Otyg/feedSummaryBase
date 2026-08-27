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

"""Production SGD multilabel tagging based on persisted article embeddings."""

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
        from sklearn.linear_model import SGDClassifier
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.preprocessing import MultiLabelBinarizer
    except ImportError as error:  # pragma: no cover - installation dependent
        raise RuntimeError("ML dependencies are missing; install feedsummary-core[ml]") from error
    return {
        "joblib": joblib,
        "np": np,
        "SGDClassifier": SGDClassifier,
        "OneVsRestClassifier": OneVsRestClassifier,
        "MultiLabelBinarizer": MultiLabelBinarizer,
    }


@dataclass(frozen=True)
class EmbeddingSGDSettings:
    enabled: bool = False
    categories: tuple[str, ...] = ("DOMAIN_ENTITY",)
    model_path: str = "data/tagging_ml/domain_entity_sgd_embeddings.joblib"
    embedding_model: str = ""
    min_label_support: int = 10
    min_training_articles: int = 30
    max_tags_per_article: int = 5
    threshold: float = 0.5
    alpha: float = 0.0001
    random_state: int = 42
    auto_retrain: bool = True

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        *,
        embedding_model: str = "",
    ) -> "EmbeddingSGDSettings":
        tagging = config.get("tagging") if isinstance(config.get("tagging"), dict) else {}
        raw = tagging.get("ml") if isinstance(tagging.get("ml"), dict) else {}
        categories = raw.get("categories") or ["DOMAIN_ENTITY"]
        if isinstance(categories, str):
            categories = [categories]
        algorithm = str(raw.get("algorithm") or "sgd").strip().lower()
        representation = str(raw.get("representation") or "embedding").strip().lower()
        if algorithm != "sgd":
            raise ValueError("tagging.ml.algorithm must be 'sgd' in this implementation")
        if representation not in {"embedding", "embeddings"}:
            raise ValueError(
                "tagging.ml.representation must be 'embedding' in this implementation"
            )
        return cls(
            enabled=bool(raw.get("enabled", False)),
            categories=tuple(
                dict.fromkeys(
                    str(category).strip().upper()
                    for category in categories
                    if str(category).strip()
                )
            ),
            model_path=str(
                raw.get("model_path")
                or "data/tagging_ml/domain_entity_sgd_embeddings.joblib"
            ),
            embedding_model=str(raw.get("embedding_model") or embedding_model or "").strip(),
            min_label_support=max(2, int(raw.get("min_label_support", 10))),
            min_training_articles=max(10, int(raw.get("min_training_articles", 30))),
            max_tags_per_article=max(1, int(raw.get("max_tags_per_article", 5))),
            threshold=float(raw.get("threshold", 0.5)),
            alpha=float(raw.get("alpha", 0.0001)),
            random_state=int(raw.get("random_state", 42)),
            auto_retrain=bool(raw.get("auto_retrain", True)),
        )

    def __post_init__(self) -> None:
        if not self.categories:
            raise ValueError("At least one tagging.ml category is required")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("tagging.ml.threshold must be between 0 and 1")
        if self.alpha <= 0:
            raise ValueError("tagging.ml.alpha must be positive")


@dataclass(frozen=True)
class _TrainingRow:
    article_id: str
    embedding: tuple[float, ...]
    labels: tuple[str, ...]
    source_hash: str


class EmbeddingSGDTagger:
    """Train, persist, refresh, and use an embedding-based SGD tagger."""

    ARTIFACT_VERSION = 1

    def __init__(self, settings: EmbeddingSGDSettings):
        self.settings = settings
        self._artifact: dict[str, Any] | None = None

    @property
    def ready(self) -> bool:
        return self._artifact is not None

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
        digest.update(json.dumps(self.settings.categories).encode("utf-8"))
        digest.update(str(self.settings.min_label_support).encode("ascii"))
        for row in sorted(rows, key=lambda item: item.article_id):
            digest.update(row.article_id.encode("utf-8"))
            digest.update(row.source_hash.encode("utf-8"))
            digest.update(json.dumps(row.embedding).encode("utf-8"))
            digest.update("\0".join(row.labels).encode("utf-8"))
        return rows, digest.hexdigest(), expected_dimension, observed_model

    def refresh_from_store(self, store: Any) -> bool:
        """Load or retrain when persisted embeddings or tag assignments changed."""
        if not self.settings.enabled:
            return False
        iterator = getattr(store, "iter_articles_with_tags", None)
        if not callable(iterator):
            logger.warning("ML tagging requires a store with iter_articles_with_tags()")
            return self.load()

        rows, fingerprint, dimension, embedding_model = self._prepare_rows(
            iterator(categories=list(self.settings.categories))
        )
        loaded = self.load()
        if loaded and self._artifact and self._artifact.get("corpus_fingerprint") == fingerprint:
            return True
        if not self.settings.auto_retrain and loaded:
            logger.warning("ML tag data changed but tagging.ml.auto_retrain is disabled")
            return True
        if len(rows) < self.settings.min_training_articles:
            logger.warning(
                "ML tagging needs at least %d embedding-covered articles; found %d",
                self.settings.min_training_articles,
                len(rows),
            )
            return loaded

        label_names = sorted({label for row in rows for label in row.labels})
        if not label_names:
            logger.warning(
                "No ML tag labels meet min_label_support=%d",
                self.settings.min_label_support,
            )
            return loaded

        ml = _ml_imports()
        np = ml["np"]
        embeddings = np.asarray([row.embedding for row in rows], dtype=float)
        binarizer = ml["MultiLabelBinarizer"](classes=label_names)
        targets = binarizer.fit_transform([row.labels for row in rows])
        estimator = ml["SGDClassifier"](
            loss="log_loss",
            alpha=self.settings.alpha,
            class_weight="balanced",
            max_iter=1_000,
            tol=1e-3,
            random_state=self.settings.random_state,
        )
        classifier = ml["OneVsRestClassifier"](estimator, n_jobs=1)
        classifier.fit(embeddings, targets)
        artifact = {
            "artifact_version": self.ARTIFACT_VERSION,
            "algorithm": "sgd",
            "representation": "embedding",
            "classifier": classifier,
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
            "Trained embedding SGD tagger on %d articles and %d labels",
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

    def predict_names(self, article: dict[str, Any]) -> list[tuple[str, float]]:
        if not self.can_predict(article):
            return []
        assert self._artifact is not None
        vector = article.get("embedding_vector")
        assert isinstance(vector, list)

        ml = _ml_imports()
        probabilities = ml["np"].asarray(
            self._artifact["classifier"].predict_proba(
                ml["np"].asarray([vector], dtype=float)
            )[0],
            dtype=float,
        )
        threshold = float(self._artifact.get("threshold", self.settings.threshold))
        classes = list(self._artifact["classes"])
        selected = [
            (classes[index], float(score))
            for index, score in enumerate(probabilities)
            if score >= threshold
        ]
        selected.sort(key=lambda item: (-item[1], item[0]))
        maximum = int(
            self._artifact.get("max_tags_per_article", self.settings.max_tags_per_article)
        )
        return selected[:maximum]

    def can_predict(self, article: dict[str, Any]) -> bool:
        """Return whether an article has an embedding compatible with the artifact."""
        if not self._artifact:
            return False
        vector = article.get("embedding_vector")
        if not isinstance(vector, list) or not vector:
            return False
        if not all(isinstance(value, (int, float)) for value in vector):
            return False
        if len(vector) != int(self._artifact["embedding_dimension"]):
            return False
        expected_model = str(self._artifact.get("embedding_model") or "")
        return not expected_model or str(article.get("embedding_model") or "") == expected_model

    def predict_tags(self, article: dict[str, Any], store: Any) -> list[dict[str, Any]]:
        predictions = []
        for name, probability in self.predict_names(article):
            tag = store.get_tag_by_name(name)
            if not isinstance(tag, dict):
                continue
            if str(tag.get("category") or "").strip().upper() not in self.settings.categories:
                continue
            predicted = dict(tag)
            predicted["reasoning"] = (
                f"Predicted by embedding SGD model (probability={probability:.3f})."
            )
            predicted["ml_probability"] = probability
            predictions.append(predicted)
        return predictions

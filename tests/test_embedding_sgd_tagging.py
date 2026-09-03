import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from feedsummary_core.tagging_ml.embedding_sgd import (
    EmbeddingClassifierSettings,
    EmbeddingClassifierTagger,
)

try:
    import sklearn
except ImportError:  # pragma: no cover - optional dependency
    sklearn = None


class FakeTrainingStore:
    def __init__(self):
        self.rows = []
        self.requested_categories = None
        self.tags = {
            "acme": {"id": 1, "name": "acme", "category": "DOMAIN_ENTITY"},
            "globex": {"id": 2, "name": "globex", "category": "DOMAIN_ENTITY"},
        }
        for index in range(40):
            label = "acme" if index % 2 == 0 else "globex"
            vector = [3.0, 0.0] if label == "acme" else [0.0, 3.0]
            self.rows.append(
                {
                    "article": {
                        "id": f"article-{index}",
                        "embedding_vector": vector,
                        "embedding_model": "test-embedding",
                        "embedding_source_hash": f"hash-{index}",
                    },
                    "tags": [self.tags[label].copy()],
                }
            )

    def iter_articles_with_tags(self, *, categories=None, limit=None):
        self.requested_categories = list(categories or [])
        allowed = {str(category).upper() for category in categories or []}
        rows = self.rows[:limit] if limit else self.rows
        for row in rows:
            yield {
                "article": row["article"],
                "tags": [
                    tag
                    for tag in row["tags"]
                    if not allowed or str(tag.get("category") or "GENERAL").upper() in allowed
                ],
            }

    def get_tag_by_name(self, name):
        tag = self.tags.get(name)
        return tag.copy() if tag else None


class EmbeddingClassifierSettingsTests(unittest.TestCase):
    def test_reads_production_configuration(self):
        settings = EmbeddingClassifierSettings.from_config(
            {
                "tagging": {
                    "ml": {
                        "enabled": True,
                        "classifier": "logistic_regression",
                        "categories": ["domain_entity", "threat"],
                        "threshold": 0.7,
                    }
                }
            },
            embedding_model="configured-model",
        )

        self.assertTrue(settings.enabled)
        self.assertEqual("logistic_regression", settings.algorithm)
        self.assertEqual(("DOMAIN_ENTITY", "THREAT"), settings.categories)
        self.assertEqual("configured-model", settings.embedding_model)
        self.assertEqual(0.7, settings.threshold)

    def test_enabled_classifier_requires_configured_categories(self):
        with self.assertRaisesRegex(ValueError, "tagging.ml.categories"):
            EmbeddingClassifierSettings.from_config(
                {
                    "tagging": {
                        "ml": {"enabled": True, "classifier": "logistic_regression"}
                    }
                }
            )

    def test_enabled_classifier_requires_configured_classifier(self):
        with self.assertRaisesRegex(ValueError, "tagging.ml.classifier"):
            EmbeddingClassifierSettings.from_config(
                {"tagging": {"ml": {"enabled": True, "categories": ["THREAT"]}}}
            )

    def test_legacy_algorithm_key_remains_supported(self):
        settings = EmbeddingClassifierSettings.from_config(
            {
                "tagging": {
                    "ml": {
                        "enabled": True,
                        "algorithm": "sgd",
                        "categories": ["sector"],
                    }
                }
            }
        )

        self.assertEqual("sgd", settings.algorithm)
        self.assertEqual(("SECTOR",), settings.categories)


@unittest.skipIf(sklearn is None, "scikit-learn ML dependencies are not installed")
class EmbeddingClassifierTaggerTests(unittest.TestCase):
    def _settings(self, directory):
        return EmbeddingClassifierSettings(
            enabled=True,
            algorithm="logistic_regression",
            categories=("DOMAIN_ENTITY",),
            model_path=str(Path(directory) / "model.joblib"),
            embedding_model="test-embedding",
            min_label_support=2,
            min_training_articles=10,
            threshold=0.5,
        )

    def test_trains_persists_and_predicts_existing_domain_entity_tags(self):
        with TemporaryDirectory() as directory:
            store = FakeTrainingStore()
            tagger = EmbeddingClassifierTagger(self._settings(directory))

            self.assertTrue(tagger.refresh_from_store(store))
            predictions = tagger.predict_tags(
                {
                    "embedding_vector": [3.0, 0.0],
                    "embedding_model": "test-embedding",
                },
                store,
            )

            self.assertTrue((Path(directory) / "model.joblib").is_file())
            self.assertEqual("logistic_regression", tagger._artifact["algorithm"])
            self.assertEqual(
                "LogisticRegression",
                type(tagger._artifact["classifier"].estimators_[0]).__name__,
            )
            self.assertEqual("acme", predictions[0]["name"])
            self.assertIn("probability=", predictions[0]["reasoning"])

    def test_training_and_prediction_use_only_configured_categories(self):
        with TemporaryDirectory() as directory:
            store = FakeTrainingStore()
            for tag in store.tags.values():
                tag["category"] = "THREAT"
            for row in store.rows:
                row["tags"][0]["category"] = "THREAT"
            settings = EmbeddingClassifierSettings(
                **{**self._settings(directory).__dict__, "categories": ("THREAT",)}
            )
            tagger = EmbeddingClassifierTagger(settings)

            self.assertTrue(tagger.refresh_from_store(store))
            predictions = tagger.predict_tags(
                {
                    "embedding_vector": [3.0, 0.0],
                    "embedding_model": "test-embedding",
                },
                store,
            )

            self.assertEqual(["THREAT"], store.requested_categories)
            self.assertEqual("THREAT", predictions[0]["category"])

    def test_changed_article_tags_change_fingerprint_and_retrain(self):
        with TemporaryDirectory() as directory:
            store = FakeTrainingStore()
            tagger = EmbeddingClassifierTagger(self._settings(directory))
            tagger.refresh_from_store(store)
            first_fingerprint = tagger._artifact["corpus_fingerprint"]

            store.rows[0]["tags"] = [store.tags["globex"].copy()]
            tagger.refresh_from_store(store)

            self.assertNotEqual(
                first_fingerprint,
                tagger._artifact["corpus_fingerprint"],
            )

    def test_force_retrains_an_unchanged_corpus(self):
        with TemporaryDirectory() as directory:
            store = FakeTrainingStore()
            settings = EmbeddingClassifierSettings(
                **{
                    **self._settings(directory).__dict__,
                    "auto_retrain": False,
                }
            )
            tagger = EmbeddingClassifierTagger(settings)
            self.assertTrue(tagger.refresh_from_store(store))
            first_classifier = tagger._artifact["classifier"]

            self.assertTrue(tagger.refresh_from_store(store, force=True))

            self.assertIsNot(first_classifier, tagger._artifact["classifier"])

    def test_force_keeps_existing_artifact_when_corpus_is_too_small(self):
        with TemporaryDirectory() as directory:
            store = FakeTrainingStore()
            tagger = EmbeddingClassifierTagger(self._settings(directory))
            self.assertTrue(tagger.refresh_from_store(store))
            model_path = Path(directory) / "model.joblib"
            original_bytes = model_path.read_bytes()
            store.rows = store.rows[:5]

            self.assertFalse(tagger.refresh_from_store(store, force=True))

            self.assertEqual(original_bytes, model_path.read_bytes())

    def test_incompatible_embedding_falls_back_without_prediction(self):
        with TemporaryDirectory() as directory:
            store = FakeTrainingStore()
            tagger = EmbeddingClassifierTagger(self._settings(directory))
            tagger.refresh_from_store(store)

            self.assertFalse(
                tagger.can_predict(
                    {
                        "embedding_vector": [3.0, 0.0],
                        "embedding_model": "different-model",
                    }
                )
            )
            self.assertEqual([], tagger.predict_names({"embedding_vector": [3.0]}))


if __name__ == "__main__":
    unittest.main()

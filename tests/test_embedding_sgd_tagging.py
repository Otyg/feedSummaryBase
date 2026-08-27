import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from feedsummary_core.tagging_ml.embedding_sgd import (
    EmbeddingSGDSettings,
    EmbeddingSGDTagger,
)

try:
    import sklearn  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    sklearn = None


class FakeTrainingStore:
    def __init__(self):
        self.rows = []
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
        yield from self.rows

    def get_tag_by_name(self, name):
        tag = self.tags.get(name)
        return tag.copy() if tag else None


class EmbeddingSGDSettingsTests(unittest.TestCase):
    def test_reads_production_configuration(self):
        settings = EmbeddingSGDSettings.from_config(
            {
                "tagging": {
                    "ml": {
                        "enabled": True,
                        "categories": ["domain_entity"],
                        "threshold": 0.7,
                    }
                }
            },
            embedding_model="configured-model",
        )

        self.assertTrue(settings.enabled)
        self.assertEqual(("DOMAIN_ENTITY",), settings.categories)
        self.assertEqual("configured-model", settings.embedding_model)
        self.assertEqual(0.7, settings.threshold)


@unittest.skipIf(sklearn is None, "scikit-learn ML dependencies are not installed")
class EmbeddingSGDTaggerTests(unittest.TestCase):
    def _settings(self, directory):
        return EmbeddingSGDSettings(
            enabled=True,
            model_path=str(Path(directory) / "model.joblib"),
            embedding_model="test-embedding",
            min_label_support=2,
            min_training_articles=10,
            threshold=0.5,
        )

    def test_trains_persists_and_predicts_existing_domain_entity_tags(self):
        with TemporaryDirectory() as directory:
            store = FakeTrainingStore()
            tagger = EmbeddingSGDTagger(self._settings(directory))

            self.assertTrue(tagger.refresh_from_store(store))
            predictions = tagger.predict_tags(
                {
                    "embedding_vector": [3.0, 0.0],
                    "embedding_model": "test-embedding",
                },
                store,
            )

            self.assertTrue((Path(directory) / "model.joblib").is_file())
            self.assertEqual("acme", predictions[0]["name"])
            self.assertIn("probability=", predictions[0]["reasoning"])

    def test_changed_article_tags_change_fingerprint_and_retrain(self):
        with TemporaryDirectory() as directory:
            store = FakeTrainingStore()
            tagger = EmbeddingSGDTagger(self._settings(directory))
            tagger.refresh_from_store(store)
            first_fingerprint = tagger._artifact["corpus_fingerprint"]

            store.rows[0]["tags"] = [store.tags["globex"].copy()]
            tagger.refresh_from_store(store)

            self.assertNotEqual(
                first_fingerprint,
                tagger._artifact["corpus_fingerprint"],
            )

    def test_incompatible_embedding_falls_back_without_prediction(self):
        with TemporaryDirectory() as directory:
            store = FakeTrainingStore()
            tagger = EmbeddingSGDTagger(self._settings(directory))
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

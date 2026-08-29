import unittest
from unittest.mock import AsyncMock, patch

from feedsummary_core.summarizer.tagging_integration import tag_articles
from feedsummary_core.tagging_ml.embedding_sgd import EmbeddingClassifierSettings


class FakeStore:
    def __init__(self):
        self.article = {
            "id": "article-1",
            "title": "Acme incident",
            "text": "Acme disclosed an incident.",
            "embedding_vector": [1.0, 0.0],
            "embedding_model": "test-model",
        }
        self.persisted_entries = []

    def get_article_tags(self, article_id):
        return []

    def get_article(self, article_id):
        return self.article.copy()

    def add_article_tags(self, article_id, entries):
        self.persisted_entries = list(entries)

    def upsert_article(self, article):
        self.article = article.copy()


class MlTaggingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = FakeStore()
        self.settings = EmbeddingClassifierSettings(
            enabled=True,
            algorithm="logistic_regression",
            categories=("DOMAIN_ENTITY",),
            embedding_model="test-model",
            threshold=0.8,
        )
        self.ml_tag = {
            "id": 1,
            "name": "acme",
            "category": "DOMAIN_ENTITY",
            "reasoning": "ML prediction",
            "ml_probability": 0.95,
        }
        self.llm_tag = {
            "id": 2,
            "name": "incident",
            "category": "GENERAL",
            "reasoning": "LLM prediction",
        }

    @patch(
        "feedsummary_core.summarizer.tagging_integration.TagManager.generate_tags_for_article",
        new_callable=AsyncMock,
    )
    @patch(
        "feedsummary_core.summarizer.tagging_integration.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch(
        "feedsummary_core.summarizer.tagging_integration.EmbeddingClassifierSettings.from_config"
    )
    @patch(
        "feedsummary_core.summarizer.tagging_integration.EmbeddingClassifierTagger.refresh_from_store",
        return_value=True,
    )
    @patch(
        "feedsummary_core.summarizer.tagging_integration.EmbeddingClassifierTagger.can_predict",
        return_value=True,
    )
    @patch(
        "feedsummary_core.summarizer.tagging_integration.EmbeddingClassifierTagger.score_names",
        return_value=[("acme", 0.95), ("globex", 0.7)],
    )
    @patch(
        "feedsummary_core.summarizer.tagging_integration.EmbeddingClassifierTagger.predict_tags"
    )
    async def test_ml_category_is_merged_and_excluded_from_llm(
        self,
        predict_tags,
        score_names,
        can_predict,
        refresh,
        from_config,
        to_thread,
        generate_tags,
    ):
        from_config.return_value = self.settings
        predict_tags.return_value = [self.ml_tag]
        generate_tags.return_value = [self.llm_tag]

        with self.assertLogs(
            "feedsummary_core.summarizer.tagging_integration", level="DEBUG"
        ) as captured:
            result = await tag_articles(
                self.store,
                llm_client=object(),
                article_ids=["article-1"],
                config={
                    "tagging": {
                        "ml": {"enabled": True},
                        "similarity_consistency": {"enabled": False},
                    }
                },
                max_tags_per_article=5,
            )

        self.assertEqual([self.ml_tag, self.llm_tag], result["article-1"])
        self.assertEqual(
            [
                {"tag_id": 1, "reasoning": "ML prediction"},
                {"tag_id": 2, "reasoning": "LLM prediction"},
            ],
            self.store.persisted_entries,
        )
        call = generate_tags.await_args.kwargs
        self.assertEqual(4, call["max_tags"])
        self.assertEqual({"DOMAIN_ENTITY"}, call["excluded_categories"])
        combined_logs = "\n".join(captured.output)
        self.assertIn("ml_tagging.model_ready", combined_logs)
        self.assertIn("ml_tagging.predictions", combined_logs)
        self.assertIn('"article_id": "article-1"', combined_logs)
        self.assertIn('"probability": 0.95', combined_logs)
        self.assertIn("ml_tagging.below_threshold", combined_logs)
        self.assertIn('"tag": "globex"', combined_logs)

    @patch(
        "feedsummary_core.summarizer.tagging_integration.TagManager.generate_tags_for_article",
        new_callable=AsyncMock,
    )
    @patch(
        "feedsummary_core.summarizer.tagging_integration.asyncio.to_thread",
        new_callable=AsyncMock,
        return_value=True,
    )
    @patch(
        "feedsummary_core.summarizer.tagging_integration.EmbeddingClassifierSettings.from_config"
    )
    @patch(
        "feedsummary_core.summarizer.tagging_integration.EmbeddingClassifierTagger.refresh_from_store",
        return_value=True,
    )
    @patch(
        "feedsummary_core.summarizer.tagging_integration.EmbeddingClassifierTagger.can_predict",
        return_value=False,
    )
    async def test_incompatible_embedding_leaves_category_to_llm(
        self,
        can_predict,
        refresh,
        from_config,
        to_thread,
        generate_tags,
    ):
        from_config.return_value = self.settings
        generate_tags.return_value = [self.llm_tag]

        with self.assertLogs(
            "feedsummary_core.summarizer.tagging_integration", level="INFO"
        ) as captured:
            await tag_articles(
                self.store,
                llm_client=object(),
                article_ids=["article-1"],
                config={
                    "tagging": {
                        "ml": {"enabled": True},
                        "similarity_consistency": {"enabled": False},
                    }
                },
                max_tags_per_article=5,
            )

        call = generate_tags.await_args.kwargs
        self.assertEqual(5, call["max_tags"])
        self.assertEqual(set(), call["excluded_categories"])
        combined_logs = "\n".join(captured.output)
        self.assertIn("ml_tagging.skipped", combined_logs)
        self.assertIn('"reason": "incompatible_embedding"', combined_logs)


if __name__ == "__main__":
    unittest.main()

import asyncio
import unittest

from feedsummary_core.summarizer.batching import (
    batch_articles_by_similarity,
    embedding_source_hash,
)


class SimilarityBatchingTests(unittest.TestCase):
    def test_persisted_article_embedding_is_reused(self):
        articles = [
            {
                "id": "cached",
                "title": "Cached",
                "text": "first",
                "embedding_vector": [1.0, 0.0],
                "embedding_model": "embedding-model",
                "embedding_source_hash": embedding_source_hash("Cached\n\nfirst"),
            },
            {"id": "new", "title": "New", "text": "second"},
        ]
        embed_calls = []

        async def embed(text):
            embed_calls.append(text)
            return [0.99, 0.01]

        class Store:
            def __init__(self):
                self.updates = []

            def update_article_embedding(self, article_id, vector, **metadata):
                self.updates.append((article_id, vector, metadata))
                return True

        store = Store()
        asyncio.run(
            batch_articles_by_similarity(
                articles,
                embed,
                max_chars_per_batch=10000,
                max_articles_per_batch=2,
                similarity_threshold=0.9,
                store=store,
                embedding_model="embedding-model",
            )
        )

        self.assertEqual(["New\n\nsecond"], embed_calls)
        self.assertEqual("new", store.updates[0][0])
        self.assertEqual("embedding-model", store.updates[0][2]["model"])

    def test_similar_articles_are_kept_in_the_same_batch(self):
        articles = [
            {"id": "a", "title": "Alpha", "text": "first"},
            {"id": "b", "title": "Unrelated", "text": "second"},
            {"id": "c", "title": "Alpha follow-up", "text": "third"},
        ]
        vectors = {
            "Alpha\n\nfirst": [1.0, 0.0],
            "Unrelated\n\nsecond": [0.0, 1.0],
            "Alpha follow-up\n\nthird": [0.99, 0.01],
        }

        async def embed(text):
            return vectors[text]

        batches = asyncio.run(
            batch_articles_by_similarity(
                articles,
                embed,
                max_chars_per_batch=10000,
                max_articles_per_batch=2,
                similarity_threshold=0.9,
            )
        )

        self.assertEqual([["a", "c"], ["b"]], [[a["id"] for a in b] for b in batches])

    def test_hard_batch_limit_splits_large_similarity_group(self):
        articles = [
            {"id": str(index), "title": f"Story {index}", "text": "same"} for index in range(3)
        ]

        async def embed(_text):
            return [1.0, 0.0]

        batches = asyncio.run(
            batch_articles_by_similarity(
                articles,
                embed,
                max_chars_per_batch=10000,
                max_articles_per_batch=2,
                similarity_threshold=0.9,
            )
        )

        self.assertEqual([["0", "1"], ["2"]], [[a["id"] for a in b] for b in batches])

    def test_embedding_failure_falls_back_to_original_batching(self):
        articles = [
            {"id": "a", "title": "One", "text": "first"},
            {"id": "b", "title": "Two", "text": "second"},
        ]

        async def embed(_text):
            return []

        batches = asyncio.run(
            batch_articles_by_similarity(
                articles,
                embed,
                max_chars_per_batch=10000,
                max_articles_per_batch=1,
            )
        )

        self.assertEqual([["a"], ["b"]], [[a["id"] for a in b] for b in batches])


if __name__ == "__main__":
    unittest.main()

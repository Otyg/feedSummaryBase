import asyncio
import unittest

from feedsummary_core.summarizer.batching import (
    SIMILARITY_EMBEDDING_INSTRUCTION,
    TAGGING_EMBEDDING_INSTRUCTION,
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
                "similarity_embedding_vector": [1.0, 0.0],
                "similarity_embedding_model": "embedding-model",
                "similarity_embedding_instruction": SIMILARITY_EMBEDDING_INSTRUCTION,
                "similarity_embedding_source_hash": embedding_source_hash(
                    "Cached\n\nfirst", SIMILARITY_EMBEDDING_INSTRUCTION
                ),
            },
            {"id": "new", "title": "New", "text": "second"},
        ]
        embed_calls = []

        async def embed(text, *, instruction, dimensions):
            embed_calls.append((text, instruction, dimensions))
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
                embedding_dimensions=2,
            )
        )

        self.assertEqual(3, len(embed_calls))
        self.assertEqual(
            {SIMILARITY_EMBEDDING_INSTRUCTION, TAGGING_EMBEDDING_INSTRUCTION},
            {call[1] for call in embed_calls},
        )
        self.assertTrue(all(call[2] == 2 for call in embed_calls))
        self.assertEqual({"cached", "new"}, {update[0] for update in store.updates})
        self.assertEqual(
            {"similarity", "tagging"},
            {update[2]["purpose"] for update in store.updates},
        )

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

        async def embed(text, **_kwargs):
            return vectors[text]

        batches = asyncio.run(
            batch_articles_by_similarity(
                articles,
                embed,
                max_chars_per_batch=10000,
                max_articles_per_batch=2,
                similarity_threshold=0.9,
                embedding_dimensions=2,
            )
        )

        self.assertEqual([["a", "c"], ["b"]], [[a["id"] for a in b] for b in batches])

    def test_hard_batch_limit_splits_large_similarity_group(self):
        articles = [
            {"id": str(index), "title": f"Story {index}", "text": "same"} for index in range(3)
        ]

        async def embed(_text, **_kwargs):
            return [1.0, 0.0]

        batches = asyncio.run(
            batch_articles_by_similarity(
                articles,
                embed,
                max_chars_per_batch=10000,
                max_articles_per_batch=2,
                similarity_threshold=0.9,
                embedding_dimensions=2,
            )
        )

        self.assertEqual([["0", "1"], ["2"]], [[a["id"] for a in b] for b in batches])

    def test_embedding_failure_falls_back_to_original_batching(self):
        articles = [
            {"id": "a", "title": "One", "text": "first"},
            {"id": "b", "title": "Two", "text": "second"},
        ]

        async def embed(_text, **_kwargs):
            return []

        batches = asyncio.run(
            batch_articles_by_similarity(
                articles,
                embed,
                max_chars_per_batch=10000,
                max_articles_per_batch=1,
                embedding_dimensions=2,
            )
        )

        self.assertEqual([["a"], ["b"]], [[a["id"] for a in b] for b in batches])


if __name__ == "__main__":
    unittest.main()

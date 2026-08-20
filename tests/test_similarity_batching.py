import asyncio
import unittest

from feedsummary_core.summarizer.batching import batch_articles_by_similarity


class SimilarityBatchingTests(unittest.TestCase):
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
            {"id": str(index), "title": f"Story {index}", "text": "same"}
            for index in range(3)
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

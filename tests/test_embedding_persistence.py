import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from feedsummary_core.persistence import SqliteStore, TinyDBStore
from feedsummary_core.summarizer.batching import (
    SIMILARITY_EMBEDDING_INSTRUCTION,
    TAGGING_EMBEDDING_INSTRUCTION,
    cached_embedding,
    embedding_source_hash,
    ensure_article_embedding,
)
from feedsummary_core.summarizer.tagging import TagManager


class EmbeddingPersistenceTests(unittest.TestCase):
    def test_ensure_article_embedding_uses_canonical_text_and_persists_metadata(self):
        article = {"id": "article-1", "title": "Title", "text": "Article body"}

        class Store:
            def __init__(self):
                self.update = None

            def update_article_embedding(self, article_id, vector, **metadata):
                self.update = (article_id, vector, metadata)
                return True

        async def embed(text, *, instruction, dimensions):
            self.assertEqual("Title\n\nArticle body", text)
            self.assertEqual(TAGGING_EMBEDDING_INSTRUCTION, instruction)
            self.assertEqual(2, dimensions)
            return [1.0, 0.0]

        store = Store()
        result = asyncio.run(
            ensure_article_embedding(
                article,
                embed,
                store=store,
                embedding_model="embedding-model",
                dimensions=2,
            )
        )

        self.assertEqual([1.0, 0.0], result)
        self.assertEqual("article-1", store.update[0])
        self.assertEqual("embedding-model", store.update[2]["model"])
        self.assertEqual("tagging", store.update[2]["purpose"])
        self.assertEqual([1.0, 0.0], article["tagging_embedding_vector"])

    def test_tag_manager_reuses_persisted_tag_embedding(self):
        class Config:
            embedding_model = "embedding-model"

        class Client:
            cfg = Config()

            def __init__(self):
                self.calls = []

            async def embed(self, text):
                self.calls.append(text)
                return [1.0, 0.0]

        with TemporaryDirectory() as directory:
            store = TinyDBStore(str(Path(directory) / "tags.json"))
            store.add_tag("security")
            client = Client()

            self.assertEqual(
                1,
                asyncio.run(TagManager(store, client).generate_embeddings_for_all_tags()),
            )
            self.assertEqual(
                0,
                asyncio.run(TagManager(store, client).generate_embeddings_for_all_tags()),
            )
            self.assertEqual(["security"], client.calls)

    def test_sqlite_persists_article_and_tag_embeddings(self):
        with TemporaryDirectory() as directory:
            self._assert_embedding_round_trip(
                SqliteStore(str(Path(directory) / "embeddings.sqlite"))
            )

    def test_tinydb_persists_article_and_tag_embeddings(self):
        with TemporaryDirectory() as directory:
            self._assert_embedding_round_trip(TinyDBStore(str(Path(directory) / "embeddings.json")))

    def _assert_embedding_round_trip(self, store):
        article_text = "Title\n\nArticle body"
        similarity_hash = embedding_source_hash(
            article_text, SIMILARITY_EMBEDDING_INSTRUCTION
        )
        tagging_hash = embedding_source_hash(article_text, TAGGING_EMBEDDING_INSTRUCTION)
        store.upsert_article(
            {
                "id": "article-1",
                "title": "Title",
                "text": "Article body",
                "embedding_vector": [9.0, 9.0],
                "embedding_model": "legacy-model",
            }
        )

        self.assertTrue(
            store.update_article_embedding(
                "article-1",
                [1.0, 0.0],
                model="embedding-model",
                source_hash=similarity_hash,
                purpose="similarity",
                instruction=SIMILARITY_EMBEDDING_INSTRUCTION,
            )
        )
        self.assertTrue(
            store.update_article_embedding(
                "article-1",
                [0.0, 1.0],
                model="embedding-model",
                source_hash=tagging_hash,
                purpose="tagging",
                instruction=TAGGING_EMBEDDING_INSTRUCTION,
            )
        )
        # A normal ingest upsert must not discard an already persisted cache entry.
        store.upsert_article({"id": "article-1", "title": "Title", "text": "Article body"})
        article = store.get_article("article-1")
        self.assertNotIn("embedding_vector", article)
        self.assertEqual(
            [1.0, 0.0],
            cached_embedding(
                article,
                article_text,
                "embedding-model",
                purpose="similarity",
                instruction=SIMILARITY_EMBEDDING_INSTRUCTION,
                dimensions=2,
            ),
        )
        self.assertEqual(
            [0.0, 1.0],
            cached_embedding(
                article,
                article_text,
                "embedding-model",
                purpose="tagging",
                instruction=TAGGING_EMBEDDING_INSTRUCTION,
                dimensions=2,
            ),
        )
        self.assertIsNone(
            cached_embedding(
                article,
                article_text,
                "different-model",
                purpose="tagging",
                instruction=TAGGING_EMBEDDING_INSTRUCTION,
            )
        )

        tag_id = store.add_tag("security")
        self.assertTrue(
            store.update_tag_embedding(
                tag_id,
                [0.0, 1.0],
                model="embedding-model",
                source_hash=embedding_source_hash("security"),
            )
        )
        tag = store.get_tag_by_name("security")
        self.assertEqual([0.0, 1.0], cached_embedding(tag, "security", "embedding-model"))
        self.assertIsInstance(tag["embedding_updated_at"], int)

        legacy_text = "Legacy content"
        store.upsert_article({"id": "legacy-article"})
        self.assertTrue(
            store.update_article_embedding(
                "legacy-article",
                [0.5, 0.5],
                model="legacy-model",
                source_hash=embedding_source_hash(legacy_text),
            )
        )
        legacy_article = store.get_article("legacy-article")
        self.assertEqual(
            [0.5, 0.5],
            cached_embedding(legacy_article, legacy_text, "legacy-model"),
        )


if __name__ == "__main__":
    unittest.main()

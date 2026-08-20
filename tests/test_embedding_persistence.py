import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from feedsummary_core.persistence import SqliteStore, TinyDBStore
from feedsummary_core.summarizer.batching import cached_embedding, embedding_source_hash
from feedsummary_core.summarizer.tagging import TagManager


class EmbeddingPersistenceTests(unittest.TestCase):
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
        article_hash = embedding_source_hash(article_text)
        store.upsert_article({"id": "article-1", "title": "Title", "text": "Article body"})

        self.assertTrue(
            store.update_article_embedding(
                "article-1",
                [1.0, 0.0],
                model="embedding-model",
                source_hash=article_hash,
            )
        )
        # A normal ingest upsert must not discard an already persisted cache entry.
        store.upsert_article({"id": "article-1", "title": "Title", "text": "Article body"})
        article = store.get_article("article-1")
        self.assertEqual([1.0, 0.0], cached_embedding(article, article_text, "embedding-model"))
        self.assertIsNone(cached_embedding(article, article_text, "different-model"))

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


if __name__ == "__main__":
    unittest.main()

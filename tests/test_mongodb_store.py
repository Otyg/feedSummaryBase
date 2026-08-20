import time
import unittest

try:
    import mongomock
except ImportError:  # pragma: no cover - optional test dependency
    mongomock = None

from feedsummary_core.persistence import CleanupPolicy, MongoDBStore, create_store
from feedsummary_core.summarizer.batching import embedding_source_hash


@unittest.skipIf(mongomock is None, "mongomock is not installed")
class MongoDBStoreTests(unittest.TestCase):
    def setUp(self):
        self.client = mongomock.MongoClient()
        self.store = create_store(
            {
                "provider": "mongodb",
                "database": "feedsummary_test",
                "client": self.client,
            }
        )

    def test_factory_and_article_summary_job_flows(self):
        self.assertIsInstance(self.store, MongoDBStore)
        now = int(time.time())
        self.store.upsert_article(
            {
                "id": "newer",
                "source": "source-b",
                "published_ts": now,
                "url": "https://example.test/newer",
            }
        )
        self.store.upsert_article(
            {
                "id": "older",
                "source": "source-a",
                "published_ts": now - 10,
                "url": "https://example.test/older",
            }
        )

        self.assertEqual(["older", "newer"], [doc["id"] for doc in self.store.list_articles()])
        filtered = self.store.list_articles_by_filter(
            sources=["source-a"], since_ts=now - 20, until_ts=now
        )
        self.assertEqual(["older"], [doc["id"] for doc in filtered])

        self.store.mark_articles_summarized(["older"])
        self.assertTrue(self.store.get_article("older")["summarized"])
        self.assertEqual(["newer"], [doc["id"] for doc in self.store.list_unsummarized_articles()])
        self.assertEqual(
            ["newer", "older"],
            [doc["id"] for doc in self.store.get_articles_by_ids(["newer", "older"])],
        )

        summary_id = self.store.save_summary_doc({"kind": "summary", "body": "text"})
        self.assertEqual(summary_id, self.store.get_latest_summary_doc()["id"])

        job_id = self.store.create_job()
        self.store.update_job(job_id, status="running", progress=50, summary_id="None")
        job = self.store.get_job(job_id)
        self.assertEqual(50, job["progress"])
        self.assertIsNone(job["summary_id"])
        self.store.save_temp_summary(job_id, "draft", {"batch": 1})
        self.assertEqual("draft", self.store.get_temp_summary(job_id)["summary"])

    def test_tags_embeddings_synonyms_and_categories(self):
        now = int(time.time())
        self.store.upsert_article({"id": "article-1", "published_ts": now})
        self.store.upsert_article({"id": "article-2", "published_ts": now})

        security_id = self.store.add_tag("Security")
        cve_id = self.store.add_tag("CVE-2026-12345", "DOMAIN_ENTITY")
        self.assertEqual(security_id, self.store.add_tag("security"))

        self.store.add_article_tags("article-1", [security_id, cve_id])
        self.store.add_article_tags("article-2", [security_id])
        self.assertEqual(2, len(self.store.get_article_tags("article-1")))
        self.assertEqual(
            {"article-1", "article-2"},
            {doc["id"] for doc in self.store.get_articles_by_tags(["security"])},
        )
        self.assertEqual(
            ["article-1"],
            [
                doc["id"]
                for doc in self.store.get_articles_by_tags(
                    ["security", "CVE-2026-12345"], match_mode="all"
                )
            ],
        )

        synonym = self.store.create_tag("infosec", synonyms=["information security"])
        self.store.add_tag_to_article("article-2", synonym["id"])
        self.assertEqual(
            (1, 1), self.store.migrate_synonym_to_main_tag(security_id, [synonym["id"]])
        )
        self.assertIsNone(self.store.get_tag_by_name("infosec"))

        self.assertTrue(self.store.update_tag_embedding(security_id, [1.0, 0.0]))
        similar = self.store.get_tags_by_embedding_similarity([1.0, 0.0], limit=1)
        self.assertEqual(security_id, similar[0]["id"])

        self.store.initialize_default_categories()
        self.assertIn("VULNERABILITY", {doc["name"] for doc in self.store.get_all_categories()})
        self.assertEqual("VULNERABILITY", self.store.get_tag_by_name("CVE-2026-12345")["category"])

    def test_article_and_tag_embedding_metadata_is_persisted(self):
        self.store.upsert_article({"id": "article-1", "title": "Title"})
        self.assertTrue(
            self.store.update_article_embedding(
                "article-1",
                [1.0, 0.0],
                model="embedding-model",
                source_hash=embedding_source_hash("Title"),
            )
        )
        article = self.store.get_article("article-1")
        self.assertEqual("embedding-model", article["embedding_model"])
        self.assertEqual([1.0, 0.0], article["embedding_vector"])
        self.store.upsert_article({"id": "article-1", "title": "Title"})
        self.assertEqual([1.0, 0.0], self.store.get_article("article-1")["embedding_vector"])

        tag_id = self.store.add_tag("security")
        self.assertTrue(
            self.store.update_tag_embedding(
                tag_id,
                [0.0, 1.0],
                model="embedding-model",
                source_hash=embedding_source_hash("security"),
            )
        )
        tag = self.store.get_tag_by_name("security")
        self.assertEqual("embedding-model", tag["embedding_model"])
        self.assertEqual(embedding_source_hash("security"), tag["embedding_source_hash"])

    def test_cleanup_honors_each_retention_window(self):
        now = int(time.time())
        old = now - 100 * 86400
        eight_days_old = now - 8 * 86400
        self.store.upsert_article({"id": "old", "published_ts": old})
        self.store.upsert_article({"id": "current", "published_ts": now})
        self.store.save_summary_doc(
            {
                "id": "daily-old",
                "created": eight_days_old,
                "selection": {"prompt_package": "daily_news"},
            }
        )
        self.store.save_summary_doc(
            {
                "id": "weekly-current",
                "created": eight_days_old,
                "selection": {"prompt_package": "weekly_news"},
            }
        )
        job_id = self.store.create_job()
        self.store.update_job(job_id, status="done", finished_at=old)
        self.store.save_temp_summary(job_id, "old", {})
        self.store.db.temp_summaries.update_one({"_id": job_id}, {"$set": {"created_at": old}})

        removed = self.store.run_cleanup(CleanupPolicy(articles_days=30))

        self.assertEqual(
            {"articles": 1, "summary_docs": 1, "temp_summaries": 1, "jobs": 1},
            removed,
        )
        self.assertIsNotNone(self.store.get_summary_doc("weekly-current"))


if __name__ == "__main__":
    unittest.main()

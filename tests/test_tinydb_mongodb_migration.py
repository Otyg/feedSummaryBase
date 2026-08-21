import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import mongomock
except ImportError:  # pragma: no cover - optional test dependency
    mongomock = None

from tinydb import TinyDB

from feedsummary_core.persistence import MongoDBStore, TinyDBStore
from feedsummary_core.persistence.migrate_tinydb_to_mongodb import (
    MigrationConflict,
    migrate_tinydb_to_mongodb,
)


@unittest.skipIf(mongomock is None, "mongomock is not installed")
class TinyDBToMongoDBMigrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source_path = Path(self.directory.name) / "source.json"
        self.client = mongomock.MongoClient()

    def _populate_source(self):
        source = TinyDBStore(str(self.source_path))
        source.upsert_article(
            {
                "id": "article-1",
                "url": "https://example.test/article-1",
                "title": "Article",
                "published_ts": 100,
                "embedding_vector": [1.0, 0.0],
                "embedding_model": "embedding-model",
                "embedding_source_hash": "article-hash",
            }
        )
        summary_id = source.save_summary_doc({"id": "summary-1", "body": "Summary"})
        job_id = source.create_job()
        source.update_job(job_id, status="done", summary_id=summary_id)
        source.save_temp_summary(job_id, "Draft", {"batch": 1})
        tag_id = source.add_tag("security")
        source.update_tag_embedding(
            tag_id,
            [0.0, 1.0],
            model="embedding-model",
            source_hash="tag-hash",
        )
        source.add_article_tags("article-1", [{"tag_id": tag_id, "reasoning": "test"}])
        category = source.create_category("CUSTOM", "Custom")
        return job_id, tag_id, category["id"]

    def test_migrates_every_table_and_preserves_ids_and_embeddings(self):
        job_id, tag_id, category_id = self._populate_source()

        report = migrate_tinydb_to_mongodb(
            str(self.source_path),
            database="migration_test",
            client=self.client,
            batch_size=2,
        )

        self.assertFalse(report["dry_run"])
        for collection in (
            "articles",
            "summary_docs",
            "jobs",
            "temp_summaries",
            "tags",
            "article_tags",
            "tag_categories",
        ):
            self.assertEqual(
                report["collections"][collection]["prepared"],
                report["collections"][collection]["verified"],
            )

        target = MongoDBStore(database="migration_test", client=self.client)
        article = target.get_article("article-1")
        self.assertEqual([1.0, 0.0], article["embedding_vector"])
        self.assertEqual("summary-1", target.get_summary_doc("summary-1")["id"])
        self.assertEqual(job_id, target.get_job(job_id)["id"])
        self.assertEqual("Draft", target.get_temp_summary(job_id)["summary"])
        self.assertEqual(tag_id, target.get_tag_by_name("security")["id"])
        self.assertEqual("embedding-model", target.get_tag_by_name("security")["embedding_model"])
        self.assertEqual([tag_id], [tag["id"] for tag in target.get_article_tags("article-1")])
        self.assertEqual(category_id, target.get_category(category_id)["id"])
        self.assertGreater(target.create_job(), job_id)

        # The migration is safe to resume or rerun.
        second = migrate_tinydb_to_mongodb(
            str(self.source_path), database="migration_test", client=self.client
        )
        self.assertEqual(1, second["collections"]["articles"]["verified"])

    def test_dry_run_does_not_touch_mongodb(self):
        self._populate_source()

        report = migrate_tinydb_to_mongodb(
            str(self.source_path),
            database="dry_run_test",
            client=self.client,
            dry_run=True,
        )

        self.assertTrue(report["dry_run"])
        self.assertGreater(report["collections"]["articles"]["prepared"], 0)
        self.assertEqual([], self.client.list_database_names())

    def test_duplicate_tags_can_be_remapped_or_rejected(self):
        db = TinyDB(str(self.source_path))
        try:
            first_id = db.table("tags").insert({"name": "Security"})
            second_id = db.table("tags").insert({"name": "security"})
            db.table("articles").insert({"id": "article-1"})
            db.table("article_tags").insert({"article_id": "article-1", "tag_id": second_id})
        finally:
            db.close()

        with self.assertRaises(MigrationConflict):
            migrate_tinydb_to_mongodb(
                str(self.source_path),
                database="strict_test",
                client=self.client,
                dry_run=True,
            )

        report = migrate_tinydb_to_mongodb(
            str(self.source_path),
            database="keep_test",
            client=self.client,
            conflict_policy="keep-existing",
        )
        target = MongoDBStore(database="keep_test", client=self.client)
        self.assertEqual(first_id, target.get_article_tags("article-1")[0]["id"])
        self.assertEqual(1, report["id_remaps"]["tags"])


if __name__ == "__main__":
    unittest.main()

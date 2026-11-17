import os
import unittest
import uuid

from feedsummary_core.persistence import CouchDBStore, create_store
from feedsummary_core.persistence.CouchDBCollection import (
    CouchDBClient,
    _apply_update,
    _matches,
    _plan,
)

from tests.test_mongodb_store import MongoDBStoreTests

COUCHDB_URL = os.environ.get("COUCHDB_TEST_URL")
COUCHDB_USER = os.environ.get("COUCHDB_TEST_USER")
COUCHDB_PASSWORD = os.environ.get("COUCHDB_TEST_PASSWORD")


def _drop_databases(store: CouchDBStore) -> None:
    status, names = store.client.request("GET", "/_all_dbs")
    prefix = f"{store.database_name}_".lower()
    for name in names if status == 200 else []:
        if name.startswith(prefix):
            store.client.request("DELETE", f"/{name}")


class FilterSemanticsTests(unittest.TestCase):
    def test_mongo_filter_semantics(self):
        doc = {"a": 1, "tags": ["x", "y"], "n": None}
        self.assertTrue(_matches(doc, {"tags": "x"}))
        self.assertTrue(_matches(doc, {"missing": None, "n": None}))
        self.assertTrue(_matches(doc, {"missing": {"$ne": True}}))
        self.assertFalse(_matches(doc, {"a": {"$gt": "0"}}))
        self.assertTrue(_matches(doc, {"$or": [{"a": 2}, {"a": {"$lte": 1}}]}))
        self.assertTrue(_matches(doc, {"missing": {"$exists": False}}))
        self.assertFalse(_matches(doc, {"a": {"$nin": [1]}}))

    def test_update_operators(self):
        doc = _apply_update(
            {"_id": 1, "seq": 2, "gone": 1},
            {"$inc": {"seq": 1}, "$max": {"seq": 10}, "$unset": {"gone": ""}, "$set": {"a.b": 1}},
            inserting=False,
        )
        self.assertEqual({"_id": 1, "seq": 10, "a": {"b": 1}}, doc)
        self.assertNotIn("x", _apply_update({}, {"$setOnInsert": {"x": 1}}, inserting=False))

    def test_plan_marks_exact_pushdown_only_for_supported_filters(self):
        selector, ids, exact = _plan({"source": {"$in": ["a"]}, "_mongo_sort_ts": {"$gte": 3}})
        self.assertTrue(exact)
        self.assertIsNone(ids)
        self.assertIn("mongo_mongo_sort_ts", selector)
        self.assertEqual(["7"], _plan({"_id": 7})[1])
        self.assertFalse(_plan({"$or": [{"a": 1}]})[2])
        self.assertFalse(_plan({"summarized": {"$ne": True}})[2])

    def test_client_rejects_bad_url(self):
        with self.assertRaises(ValueError):
            CouchDBClient("ftp://example.test")


@unittest.skipUnless(COUCHDB_URL, "COUCHDB_TEST_URL is not set")
class CouchDBStoreTests(MongoDBStoreTests):
    def setUp(self):
        self.store = create_store(
            {
                "provider": "couchdb",
                "url": COUCHDB_URL,
                "username": COUCHDB_USER,
                "password": COUCHDB_PASSWORD,
                "database": f"fs_test_{uuid.uuid4().hex[:12]}",
            }
        )
        self.addCleanup(_drop_databases, self.store)
        self.addCleanup(self.store.close)

    def test_factory_and_article_summary_job_flows(self):
        self.assertIsInstance(self.store, CouchDBStore)
        super().test_factory_and_article_summary_job_flows()

    def test_unique_claims_reject_duplicates(self):
        from feedsummary_core.persistence.MongoDBStore import DuplicateKeyError

        self.store.upsert_article({"id": "one", "url": "https://example.test/u"})
        with self.assertRaises(DuplicateKeyError):
            self.store.upsert_article({"id": "two", "url": "https://example.test/u"})
        self.store.upsert_article({"id": "one", "url": "https://example.test/changed"})
        self.store.upsert_article({"id": "two", "url": "https://example.test/u"})




if __name__ == "__main__":
    unittest.main()

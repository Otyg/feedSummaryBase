import tempfile
import unittest
from pathlib import Path

try:
    import mongomock
except ImportError:  # pragma: no cover - optional dependency
    mongomock = None

from feedsummary_core.persistence import (
    MongoDBStore,
    SqliteStore,
    TagRelationError,
    TinyDBStore,
)


class TagRelationStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _stores(self):
        root = Path(self.temp_dir.name)
        stores = [
            SqliteStore(str(root / "store.sqlite")),
            TinyDBStore(str(root / "store.json")),
        ]
        if mongomock is not None:
            stores.append(
                MongoDBStore(
                    database="tag_relation_test",
                    client=mongomock.MongoClient(),
                )
            )
        return stores

    def test_parent_and_child_are_inverse_views_of_one_relation(self):
        for store in self._stores():
            with self.subTest(store=type(store).__name__):
                root = store.create_tag("root", "THREAT")["id"]
                branch = store.create_tag("branch", "THREAT")["id"]
                leaf = store.create_tag("leaf", "THREAT")["id"]

                store.set_tag_relations(branch, parent_ids=[root], child_ids=[leaf])

                self.assertEqual(
                    ["root"],
                    [tag["name"] for tag in store.get_tag_relations(branch)["parents"]],
                )
                self.assertEqual(
                    ["branch"],
                    [tag["name"] for tag in store.get_tag_relations(root)["children"]],
                )
                self.assertEqual(
                    ["branch"],
                    [tag["name"] for tag in store.get_tag_relations(leaf)["parents"]],
                )

    def test_relations_cannot_cross_categories_or_form_cycles(self):
        for store in self._stores():
            with self.subTest(store=type(store).__name__):
                first = store.create_tag("first", "THREAT")["id"]
                second = store.create_tag("second", "THREAT")["id"]
                other = store.create_tag("other", "GENERAL")["id"]

                with self.assertRaisesRegex(TagRelationError, "cross categories"):
                    store.set_tag_relations(first, child_ids=[other])

                store.set_tag_relations(first, child_ids=[second])
                with self.assertRaisesRegex(TagRelationError, "cycles"):
                    store.set_tag_relations(second, child_ids=[first])

                self.assertEqual([], store.get_tag_relations(second)["children"])

    def test_category_change_and_delete_remove_relations(self):
        for store in self._stores():
            with self.subTest(store=type(store).__name__):
                parent = store.create_tag("parent", "THREAT")["id"]
                child = store.create_tag("child", "THREAT")["id"]
                store.set_tag_relations(parent, child_ids=[child])

                store.update_tag(child, category="GENERAL")
                self.assertEqual([], store.get_tag_relations(parent)["children"])

                store.update_tag(child, category="THREAT")
                store.set_tag_relations(parent, child_ids=[child])
                self.assertTrue(store.delete_tag(child))
                self.assertEqual([], store.get_tag_relations(parent)["children"])


if __name__ == "__main__":
    unittest.main()

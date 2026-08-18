import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from feedsummary_core.persistence.SqliteStore import SqliteStore
from feedsummary_core.persistence.TinyDbStore import TinyDBStore
from feedsummary_core.summarizer.tagging import TagManager


class MemoryTagStore:
    def __init__(self):
        self.tags = []

    def add_tag(self, name, category="GENERAL", description=None):
        existing = self.get_tag_by_name(name)
        if existing:
            return existing["id"]
        tag = {
            "id": len(self.tags) + 1,
            "name": name,
            "category": category,
            "description": description,
        }
        self.tags.append(tag)
        return tag["id"]

    def get_tag_by_name(self, name):
        normalized = name.strip().lower()
        return next(
            (tag.copy() for tag in self.tags if tag["name"].lower() == normalized),
            None,
        )

    def get_all_tags(self):
        return [tag.copy() for tag in self.tags]

    def update_tag(self, tag_id, category=None, **_kwargs):
        tag = next(tag for tag in self.tags if tag["id"] == tag_id)
        if category is not None:
            tag["category"] = category
        return tag.copy()


class CveCategoryTests(unittest.TestCase):
    def setUp(self):
        self.store = MemoryTagStore()
        self.manager = TagManager(self.store)

    def test_add_tag_assigns_vulnerability_to_uppercase_cve(self):
        tag_id = self.manager.add_tag("CVE-2026-12345", category="DOMAIN_ENTITY")

        tag = self.store.get_tag_by_name("CVE-2026-12345")
        self.assertEqual(tag_id, tag["id"])
        self.assertEqual("VULNERABILITY", tag["category"])

    def test_select_tag_assigns_vulnerability_to_lowercase_six_digit_cve(self):
        selected = self.manager.select_tags_for_article(
            "article-1",
            [{"name": "cve-2026-123456", "type": "NAMED_ENTITY"}],
        )

        self.assertEqual(1, len(selected))
        self.assertEqual("cve-2026-123456", selected[0]["name"])
        self.assertEqual("VULNERABILITY", selected[0]["category"])

    def test_existing_cve_is_moved_to_vulnerability(self):
        tag_id = self.store.add_tag("cve-2025-1234", category="DOMAIN_ENTITY")

        selected = self.manager.select_tags_for_article(
            "article-1",
            ["CVE-2025-1234"],
        )

        self.assertEqual("VULNERABILITY", selected[0]["category"])
        self.assertEqual(
            "VULNERABILITY",
            self.store.get_tag_by_name("cve-2025-1234")["category"],
        )
        self.assertEqual(tag_id, selected[0]["id"])

    def test_distinct_cves_are_not_fuzzy_matched_to_each_other(self):
        first = self.manager.select_tags_for_article(
            "article-1",
            ["CVE-2026-12345"],
        )
        second = self.manager.select_tags_for_article(
            "article-2",
            ["CVE-2026-12346"],
        )

        self.assertNotEqual(first[0]["id"], second[0]["id"])
        self.assertEqual(2, len(self.store.tags))
        self.assertTrue(
            all(tag["category"] == "VULNERABILITY" for tag in self.store.tags)
        )

    def test_non_cve_tag_keeps_requested_category(self):
        self.manager.add_tag("cve-2026-123", category="DOMAIN_ENTITY")

        tag = self.store.get_tag_by_name("cve-2026-123")
        self.assertEqual("DOMAIN_ENTITY", tag["category"])


class VulnerabilityCategoryInitializationTests(unittest.TestCase):
    def test_tinydb_initializes_vulnerability_category(self):
        with TemporaryDirectory() as directory:
            store = TinyDBStore(str(Path(directory) / "tags.json"))
            tag_id = store.add_tag("cve-2024-12345", category="DOMAIN_ENTITY")
            store.initialize_default_categories()

            categories = {category["name"] for category in store.get_all_categories()}
            self.assertIn("VULNERABILITY", categories)
            stored_tag = store.get_tag_by_name("cve-2024-12345")
            self.assertEqual("VULNERABILITY", stored_tag["category"])
            self.assertEqual(tag_id, stored_tag["id"])

    def test_sqlite_initializes_vulnerability_category(self):
        with TemporaryDirectory() as directory:
            store = SqliteStore(str(Path(directory) / "tags.sqlite"))
            tag_id = store.add_tag("CVE-2024-123456", category="DOMAIN_ENTITY")
            store.initialize_default_categories()

            categories = {category["name"] for category in store.get_all_categories()}
            self.assertIn("VULNERABILITY", categories)
            stored_tag = store.get_tag_by_name("CVE-2024-123456")
            self.assertEqual("VULNERABILITY", stored_tag["category"])
            self.assertEqual(tag_id, stored_tag["id"])


if __name__ == "__main__":
    unittest.main()

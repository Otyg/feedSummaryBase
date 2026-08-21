import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from feedsummary_core.persistence.SqliteStore import SqliteStore
from feedsummary_core.persistence.TinyDbStore import TinyDBStore
from feedsummary_core.summarizer.tagging import TagManager
from feedsummary_core.tagging_rules import extract_cve_ids


class MemoryTagStore:
    def __init__(self, categories=None):
        self.tags = []
        self.categories = categories or []

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

    def get_all_categories(self):
        return [category.copy() for category in self.categories]

    def update_tag(self, tag_id, category=None, **_kwargs):
        tag = next(tag for tag in self.tags if tag["id"] == tag_id)
        if category is not None:
            tag["category"] = category
        return tag.copy()


class FakeTaggingClient:
    def __init__(self, tags=None, error=None):
        self.tags = tags
        self.error = error
        self.messages = []

    async def chat(self, messages, temperature=0.3):
        self.messages = messages
        if self.error:
            raise self.error
        return json.dumps({"tags": self.tags})


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

    def test_extracts_only_complete_cves_with_four_to_nineteen_digits(self):
        nineteen_digits = "1234567890123456789"
        text = (
            "CVE-2024-1234, cve-2025-123456 and "
            f"CVE-2026-{nineteen_digits}; repeated CVE-2024-1234. "
            "Ignore CVE-2024-123, CVE-2024-12345678901234567890 and "
            "prefixCVE-2024-9999suffix."
        )

        self.assertEqual(
            [
                "CVE-2024-1234",
                "CVE-2025-123456",
                f"CVE-2026-{nineteen_digits}",
            ],
            extract_cve_ids(text),
        )

    def test_generated_cves_do_not_count_toward_regular_tag_limit(self):
        client = FakeTaggingClient(
            [
                {"tag": "CVE-2025-1234", "type": "NAMED_ENTITY"},
                {"tag": "CVE-2026-123456", "type": "NAMED_ENTITY"},
                {"tag": "CVE-2025-1234", "type": "NAMED_ENTITY"},
                {"tag": "Acme", "type": "NAMED_ENTITY"},
                {"tag": "Example Product", "type": "NAMED_ENTITY"},
                {"tag": "Third Entity", "type": "NAMED_ENTITY"},
            ]
        )
        article = {
            "id": "article-1",
            "title": "CVE-2025-1234 fixed by Acme",
            "content": "The update also fixes cve-2026-123456.",
            "summary": "CVE-2025-1234 is mentioned again.",
        }

        selected = asyncio.run(
            self.manager.generate_tags_for_article(
                llm_client=client,
                article=article,
                config={},
                max_tags=2,
            )
        )

        cve_tags = [tag for tag in selected if tag["category"] == "VULNERABILITY"]
        regular_tags = [tag for tag in selected if tag["category"] != "VULNERABILITY"]
        self.assertEqual(
            {"cve-2025-1234", "cve-2026-123456"},
            {tag["name"] for tag in cve_tags},
        )
        self.assertEqual(2, len(regular_tags))
        self.assertEqual(4, len(selected))

    def test_direct_cve_extraction_survives_llm_failure_and_zero_limit(self):
        selected = asyncio.run(
            self.manager.generate_tags_for_article(
                llm_client=FakeTaggingClient(error=RuntimeError("LLM unavailable")),
                article={
                    "id": "article-2",
                    "title": "Advisory for CVE-2026-9876",
                    "content": "",
                },
                config={},
                max_tags=0,
            )
        )

        self.assertEqual(["cve-2026-9876"], [tag["name"] for tag in selected])
        self.assertEqual("VULNERABILITY", selected[0]["category"])

    def test_llm_uses_database_defined_category_for_new_tag(self):
        store = MemoryTagStore(
            categories=[
                {"name": "GENERAL", "label": "Allmän", "description": "Fallback"},
                {
                    "name": "RISK_THEME",
                    "label": "Risktema",
                    "description": "Strategic and emerging risks",
                },
            ]
        )
        manager = TagManager(store)
        client = FakeTaggingClient(
            [
                {"tag": "quux", "type": "CATEGORY", "category": "risk_theme"},
                {"tag": "ransomware", "type": "CATEGORY", "category": "NOT_REAL"},
            ]
        )

        selected = asyncio.run(
            manager.generate_tags_for_article(
                llm_client=client,
                article={"id": "article-3", "title": "Quux ransomware risk"},
                config={},
                max_tags=2,
            )
        )

        self.assertEqual(
            {"quux": "RISK_THEME", "ransomware": "GENERAL"},
            {tag["name"]: tag["category"] for tag in selected},
        )
        prompt = client.messages[0]["content"]
        self.assertIn("RISK_THEME: Risktema: Strategic and emerging risks", prompt)
        self.assertIn('Set "category" to exactly one name', prompt)


class VulnerabilityCategoryInitializationTests(unittest.TestCase):
    def test_sqlite_custom_category_is_used_for_new_tag(self):
        with TemporaryDirectory() as directory:
            store = SqliteStore(str(Path(directory) / "tags.sqlite"))
            store.initialize_default_categories()
            store.create_category(
                "RISK_THEME",
                "Risktema",
                description="Strategic and emerging risks",
            )

            tag_id = TagManager(store).add_tag("quantum risk", category="risk_theme")

            stored_tag = store.get_tag_by_name("quantum risk")
            self.assertEqual(tag_id, stored_tag["id"])
            self.assertEqual("RISK_THEME", stored_tag["category"])

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

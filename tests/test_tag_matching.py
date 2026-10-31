import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from feedsummary_core.summarizer.tagging import TagManager


class MemoryTagStore:
    def __init__(self, tags, embedding_matches=None):
        self.tags = tags
        self.embedding_matches = embedding_matches or []
        self.similarity_calls = []

    def get_all_tags(self):
        return [tag.copy() for tag in self.tags]

    def get_tag_by_name(self, name):
        normalized = str(name or "").strip().casefold()
        return next(
            (
                tag.copy()
                for tag in self.tags
                if str(tag.get("name") or "").strip().casefold() == normalized
            ),
            None,
        )

    def get_all_categories(self):
        return [
            {"name": category}
            for category in (
                "GENERAL",
                "DOMAIN_ENTITY",
                "VULNERABILITY",
                "ORGANIZATION",
                "PRODUCT",
            )
        ]

    def add_tag(self, name, category, description=None):
        existing = self.get_tag_by_name(name)
        if existing:
            return existing["id"]
        tag_id = max((int(tag.get("id") or 0) for tag in self.tags), default=0) + 1
        self.tags.append({"id": tag_id, "name": name, "category": category})
        return tag_id

    def get_tags_by_embedding_similarity(
        self,
        embedding,
        similarity_threshold=0.6,
        limit=5,
    ):
        self.similarity_calls.append((embedding, similarity_threshold, limit))
        return [tag.copy() for tag in self.embedding_matches[:limit]]


class FakeEmbeddingClient:
    def __init__(self, embedding=None, error=None):
        self.embedding = embedding or [0.1, 0.2]
        self.error = error
        self.embed_calls = []

    async def embed(self, text):
        self.embed_calls.append(text)
        if self.error:
            raise self.error
        return self.embedding


class FakeTaggingClient:
    def __init__(self, response, *, allow_new_tags=True):
        self.response = response
        self.messages = []
        self.tagging_allow_new_tags = allow_new_tags

    async def chat(self, messages, temperature=0.3):
        self.messages.append(messages)
        return self.response


class TagMatchingTests(unittest.TestCase):
    def test_existing_only_provider_reviews_ml_and_cannot_create_tags(self):
        store = MemoryTagStore(
            [
                {"id": 1, "name": "acme", "category": "ORGANIZATION"},
                {"id": 2, "name": "incident", "category": "GENERAL"},
            ]
        )
        client = FakeTaggingClient(
            '{"suggested_tags": ['
            '{"tag": "incident", "reasoning": "Central topic"}, '
            '{"tag": "invented", "reasoning": "Not in database"}]}',
            allow_new_tags=False,
        )
        manager = TagManager(store)

        selected = asyncio.run(
            manager.generate_tags_for_article(
                client,
                {
                    "id": "article-1",
                    "title": "Acme incident",
                    "text": "Acme disclosed an incident.",
                },
                {},
                review_tags=[
                    {
                        "id": 1,
                        "name": "acme",
                        "category": "ORGANIZATION",
                        "reasoning": "ML prediction",
                        "ml_probability": 0.95,
                    }
                ],
            )
        )

        self.assertEqual(["incident"], [tag["name"] for tag in selected])
        self.assertFalse(manager.last_generation_allowed_new_tags)
        prompt = client.messages[0][0]["content"]
        self.assertIn("MUST ONLY suggest tags from the provided list", prompt)
        self.assertIn("UNTRUSTED ML SUGGESTIONS TO REVIEW", prompt)
        self.assertIn("acme (ORGANIZATION, confidence=0.950)", prompt)
        self.assertIn("Omitting a suggestion rejects it", prompt)

    def test_substring_inside_word_does_not_match(self):
        store = MemoryTagStore(
            [
                {"id": 1, "name": "ray", "category": "GENERAL"},
                {"id": 2, "name": "comfast", "category": "DOMAIN_ENTITY"},
            ]
        )
        manager = TagManager(store)

        self.assertEqual([], manager._find_similar_existing_tags("password spraying"))
        self.assertEqual([], manager._find_similar_existing_tags("mfa"))

    def test_complete_term_still_matches_longer_phrase(self):
        password = {"id": 1, "name": "password", "category": "GENERAL"}
        manager = TagManager(MemoryTagStore([password]))

        matches = manager._find_similar_existing_tags("password spraying")

        self.assertEqual([password], matches)

    def test_async_selection_awaits_and_uses_candidate_embedding(self):
        semantic_match = {
            "id": 1,
            "name": "credential attack",
            "category": "GENERAL",
            "_similarity_score": 0.87,
        }
        store = MemoryTagStore([semantic_match], embedding_matches=[semantic_match])
        client = FakeEmbeddingClient([0.4, 0.6])
        manager = TagManager(store, llm_client=client)

        selected = asyncio.run(
            manager.select_tags_for_article_async(
                "article-1",
                [{"name": "password spraying", "type": "CATEGORY"}],
                allow_new_tags=False,
            )
        )

        self.assertEqual(["password spraying"], client.embed_calls)
        self.assertEqual([0.4, 0.6], store.similarity_calls[0][0])
        self.assertEqual(0.8, store.similarity_calls[0][1])
        self.assertEqual("credential attack", selected[0]["name"])

    def test_async_selection_forwards_configured_embedding_replacement_threshold(self):
        semantic_match = {
            "id": 1,
            "name": "credential attack",
            "category": "GENERAL",
            "_similarity_score": 0.91,
        }
        store = MemoryTagStore([semantic_match], embedding_matches=[semantic_match])
        client = FakeEmbeddingClient([0.4, 0.6])
        manager = TagManager(store, llm_client=client)

        selected = asyncio.run(
            manager.select_tags_for_article_async(
                "article-1",
                [{"name": "password spraying", "type": "CATEGORY"}],
                allow_new_tags=False,
                embedding_replacement_threshold=0.87,
            )
        )

        self.assertEqual(0.87, store.similarity_calls[0][1])
        self.assertEqual("credential attack", selected[0]["name"])

    @patch.object(
        TagManager,
        "select_tags_for_article_async",
        new_callable=AsyncMock,
        return_value=[],
    )
    def test_generation_reads_embedding_replacement_threshold_from_config(self, select):
        manager = TagManager(MemoryTagStore([]))
        client = FakeTaggingClient('{"tags": [{"tag": "security", "type": "CATEGORY"}]}')

        asyncio.run(
            manager.generate_tags_for_article(
                client,
                {"id": "article-1", "title": "Security update"},
                {"tagging": {"embedding_replacement_threshold": 0.8}},
            )
        )

        self.assertEqual(
            0.8,
            select.await_args.kwargs["embedding_replacement_threshold"],
        )

    def test_embedding_failure_uses_safe_string_fallback(self):
        store = MemoryTagStore([{"id": 1, "name": "ray", "category": "GENERAL"}])
        client = FakeEmbeddingClient(error=RuntimeError("embedding unavailable"))
        manager = TagManager(store, llm_client=client)

        selected = asyncio.run(
            manager.select_tags_for_article_async(
                "article-1",
                [{"name": "password spraying", "type": "CATEGORY"}],
                allow_new_tags=False,
            )
        )

        self.assertEqual(["password spraying"], client.embed_calls)
        self.assertEqual([], selected)

    def test_exact_match_skips_embedding(self):
        exact = {"id": 1, "name": "mfa", "category": "GENERAL"}
        client = FakeEmbeddingClient()
        manager = TagManager(MemoryTagStore([exact]), llm_client=client)

        selected = asyncio.run(
            manager.select_tags_for_article_async(
                "article-1",
                [{"name": "mfa", "type": "CATEGORY"}],
                allow_new_tags=False,
            )
        )

        self.assertEqual([], client.embed_calls)
        self.assertEqual("mfa", selected[0]["name"])

    def test_named_entity_does_not_fuzzy_match_another_organization(self):
        ibm = {
            "id": 1,
            "name": "ibm",
            "category": "ORGANIZATION",
            "_similarity_score": 0.91,
        }
        store = MemoryTagStore([ibm], embedding_matches=[ibm])
        client = FakeEmbeddingClient([0.4, 0.6])
        manager = TagManager(store, llm_client=client)

        selected = asyncio.run(
            manager.select_tags_for_article_async(
                "article-1",
                [{"name": "acme", "type": "NAMED_ENTITY"}],
                allow_new_tags=False,
            )
        )

        self.assertEqual([], client.embed_calls)
        self.assertEqual([], selected)

    def test_hallucinated_ibm_entity_is_rejected_and_text_field_is_prompted(self):
        ibm = {"id": 1, "name": "ibm", "category": "ORGANIZATION"}
        store = MemoryTagStore([ibm])
        client = FakeTaggingClient(
            '{"tags": [{"tag": "ibm", "type": "CATEGORY", '
            '"category": "GENERAL", "reasoning": '
            '"The specific software product affected by the vulnerability."}]}'
        )
        manager = TagManager(store)

        selected = asyncio.run(
            manager.generate_tags_for_article(
                client,
                {
                    "id": "article-1",
                    "title": "Critical vulnerability in Acme Server",
                    "text": "The flaw affects Acme Server installations before version 4.2.",
                    "url": "https://news.example/acme-flaw",
                    "source": "Security News",
                },
                {},
            )
        )

        prompt = client.messages[0][0]["content"]
        self.assertIn("The flaw affects Acme Server", prompt)
        self.assertEqual([], selected)

    def test_category_candidate_cannot_fuzzy_match_unmentioned_ibm_entity(self):
        ibm = {
            "id": 1,
            "name": "ibm",
            "category": "ORGANIZATION",
            "_similarity_score": 0.91,
        }
        store = MemoryTagStore([ibm], embedding_matches=[ibm])
        client = FakeEmbeddingClient([0.4, 0.6])
        manager = TagManager(store, llm_client=client)

        selected = asyncio.run(
            manager.select_tags_for_article_async(
                "article-1",
                [{"name": "acme", "type": "CATEGORY"}],
                allow_new_tags=False,
                article_text="Acme Server is affected by the vulnerability.",
            )
        )

        self.assertEqual(["acme"], client.embed_calls)
        self.assertEqual([], selected)

    def test_mentioned_ibm_entity_is_retained(self):
        ibm = {"id": 1, "name": "ibm", "category": "ORGANIZATION"}
        store = MemoryTagStore([ibm])
        client = FakeTaggingClient(
            '{"tags": [{"tag": "ibm", "type": "NAMED_ENTITY", '
            '"category": "ORGANIZATION", "reasoning": "IBM is named in the article."}]}'
        )
        manager = TagManager(store)

        selected = asyncio.run(
            manager.generate_tags_for_article(
                client,
                {
                    "id": "article-1",
                    "title": "IBM patches a vulnerability",
                    "text": "IBM released an update for the affected product.",
                },
                {},
            )
        )

        self.assertEqual(["ibm"], [tag["name"] for tag in selected])

    def test_reclassification_rejects_unmentioned_ibm_entity(self):
        ibm = {"id": 1, "name": "ibm", "category": "ORGANIZATION"}
        store = MemoryTagStore([ibm])
        client = FakeTaggingClient(
            '{"suggested_tags": [{"tag": "ibm", "reasoning": "The affected software product."}]}'
        )
        manager = TagManager(store)

        selected = asyncio.run(
            manager.reclassify_article_with_existing_tags(
                client,
                {
                    "id": "article-1",
                    "title": "Acme Server vulnerability",
                    "text": "Acme released a patch for its server product.",
                },
            )
        )

        self.assertEqual([], selected)

    def test_existing_only_review_accepts_product_alias_as_evidence(self):
        product = {
            "id": 1,
            "name": "microsoft windows",
            "category": "PRODUCT",
            "synonyms": ["MS Windows"],
        }
        store = MemoryTagStore([product])
        client = FakeTaggingClient(
            '{"suggested_tags": [{"tag": "microsoft windows", '
            '"reasoning": "The product is affected."}]}',
            allow_new_tags=False,
        )
        manager = TagManager(store)

        selected = asyncio.run(
            manager.generate_tags_for_article(
                client,
                {
                    "id": "article-1",
                    "title": "MS Windows security update",
                    "text": "The update fixes an issue in MS Windows.",
                },
                {},
                review_tags=[product],
            )
        )

        self.assertEqual(["microsoft windows"], [tag["name"] for tag in selected])
        prompt = client.messages[0][0]["content"]
        self.assertIn("aliases: MS Windows", prompt)

    def test_new_tag_provider_accepts_product_alias_as_evidence(self):
        product = {
            "id": 1,
            "name": "microsoft windows",
            "category": "PRODUCT",
            "synonyms": ["MS Windows"],
        }
        store = MemoryTagStore([product])
        client = FakeTaggingClient(
            '{"tags": [{"tag": "microsoft windows", "type": "NAMED_ENTITY", '
            '"category": "PRODUCT"}]}'
        )
        manager = TagManager(store)

        selected = asyncio.run(
            manager.generate_tags_for_article(
                client,
                {
                    "id": "article-1",
                    "title": "MS Windows security update",
                    "text": "The update fixes an issue in MS Windows.",
                },
                {},
            )
        )

        self.assertEqual(["microsoft windows"], [tag["name"] for tag in selected])

    def test_cve_automatically_adds_vulnerability_tag(self):
        store = MemoryTagStore([])
        client = FakeTaggingClient(
            '{"suggested_tags": []}',
            allow_new_tags=False,
        )
        manager = TagManager(store)

        selected = asyncio.run(
            manager.generate_tags_for_article(
                client,
                {
                    "id": "article-1",
                    "title": "Critical flaw CVE-2026-12345",
                    "text": "A fix for CVE-2026-12345 is available.",
                },
                {},
                max_tags=0,
            )
        )

        self.assertEqual(
            {"cve-2026-12345", "vulnerability"},
            {tag["name"] for tag in selected},
        )
        self.assertTrue(all(tag["category"] == "VULNERABILITY" for tag in selected))


if __name__ == "__main__":
    unittest.main()

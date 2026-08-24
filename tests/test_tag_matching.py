import asyncio
import unittest

from feedsummary_core.summarizer.tagging import TagManager


class MemoryTagStore:
    def __init__(self, tags, embedding_matches=None):
        self.tags = tags
        self.embedding_matches = embedding_matches or []
        self.similarity_calls = []

    def get_all_tags(self):
        return [tag.copy() for tag in self.tags]

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
    def __init__(self, response):
        self.response = response
        self.messages = []

    async def chat(self, messages, temperature=0.3):
        self.messages.append(messages)
        return self.response


class TagMatchingTests(unittest.TestCase):
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
        self.assertEqual("credential attack", selected[0]["name"])

    def test_embedding_failure_uses_safe_string_fallback(self):
        store = MemoryTagStore(
            [{"id": 1, "name": "ray", "category": "GENERAL"}]
        )
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
            '{"suggested_tags": [{"tag": "ibm", "reasoning": '
            '"The affected software product."}]}'
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


if __name__ == "__main__":
    unittest.main()

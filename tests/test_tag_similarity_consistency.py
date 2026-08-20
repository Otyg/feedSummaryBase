import asyncio
import unittest

from feedsummary_core.summarizer.tagging import TagManager


class FakeStore:
    def __init__(self, articles, tags):
        self.articles = {article["id"]: dict(article) for article in articles}
        self.tags = {
            article_id: [dict(tag) for tag in values]
            for article_id, values in tags.items()
        }
        self.added = []

    def get_article_tags(self, article_id):
        return [dict(tag) for tag in self.tags.get(article_id, [])]

    def add_tag_to_article(self, article_id, tag_id):
        tag = next(
            (
                candidate
                for values in self.tags.values()
                for candidate in values
                if candidate["id"] == tag_id
            ),
            None,
        )
        if tag is None or any(existing["id"] == tag_id for existing in self.tags[article_id]):
            return False
        self.tags[article_id].append(dict(tag))
        self.added.append((article_id, tag_id))
        return True

    def upsert_article(self, article):
        self.articles[article["id"]] = dict(article)


class FakeEmbeddingClient:
    def __init__(self, vectors):
        self.vectors = vectors

    async def embed(self, text):
        return self.vectors[text]


class TagSimilarityConsistencyTests(unittest.TestCase):
    def test_disjoint_tags_get_a_shared_tag_for_similar_articles(self):
        articles = [
            {"id": "a", "title": "Alpha", "content": "incident", "tags": ["ransomware"]},
            {"id": "b", "title": "Other", "content": "sports", "tags": ["football"]},
            {"id": "c", "title": "Alpha update", "content": "incident", "tags": ["microsoft"]},
        ]
        tags = {
            "a": [{"id": 1, "name": "ransomware", "category": "GENERAL"}],
            "b": [{"id": 2, "name": "football", "category": "GENERAL"}],
            "c": [{"id": 3, "name": "microsoft", "category": "DOMAIN_ENTITY"}],
        }
        vectors = {
            "Alpha\n\nincident": [1.0, 0.0],
            "Other\n\nsports": [0.0, 1.0],
            "Alpha update\n\nincident": [0.99, 0.01],
        }
        store = FakeStore(articles, tags)
        manager = TagManager(store, llm_client=FakeEmbeddingClient(vectors))

        additions = asyncio.run(
            manager.ensure_similar_articles_share_tags(
                articles,
                similarity_threshold=0.9,
            )
        )

        self.assertEqual([("c", 1)], store.added)
        self.assertEqual(["ransomware"], [tag["name"] for tag in additions["c"]])
        self.assertEqual({1, 3}, {tag["id"] for tag in store.tags["c"]})
        self.assertEqual({2}, {tag["id"] for tag in store.tags["b"]})

    def test_existing_overlap_is_left_unchanged(self):
        articles = [
            {"id": "a", "title": "Alpha", "content": "incident"},
            {"id": "b", "title": "Alpha update", "content": "incident"},
        ]
        shared = {"id": 1, "name": "ransomware", "category": "GENERAL"}
        store = FakeStore(articles, {"a": [shared], "b": [shared]})
        vectors = {
            "Alpha\n\nincident": [1.0, 0.0],
            "Alpha update\n\nincident": [1.0, 0.0],
        }
        manager = TagManager(store, llm_client=FakeEmbeddingClient(vectors))

        additions = asyncio.run(manager.ensure_similar_articles_share_tags(articles))

        self.assertEqual({}, additions)
        self.assertEqual([], store.added)


if __name__ == "__main__":
    unittest.main()

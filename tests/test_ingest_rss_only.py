import unittest
from unittest.mock import AsyncMock, patch

import feedparser

from feedsummary_core.summarizer import ingest


class _MemoryArticleStore:
    def __init__(self):
        self.articles = []

    def get_article(self, _article_id):
        return None

    def upsert_article(self, article):
        self.articles.append(article)


def _feed_with_article(rss_text):
    entry = feedparser.FeedParserDict(
        {
            "link": "https://example.com/article",
            "title": "RSS article",
            "published": "Sun, 30 Aug 2026 12:00:00 GMT",
            "summary": f"<p>{rss_text}</p>",
        }
    )
    return feedparser.FeedParserDict({"entries": [entry]})


class FeedArticleFetchConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_article_false_uses_rss_without_requesting_article_page(self):
        rss_text = "Text embedded in the RSS entry. " * 8
        store = _MemoryArticleStore()
        article_fetch = AsyncMock(return_value="unused article HTML")
        config = {
            "feeds": [
                {
                    "name": "RSS-only feed",
                    "url": "https://example.com/feed.xml",
                    "fetch_article": False,
                }
            ],
            "ingest": {"max_items_per_feed": 8},
        }

        with (
            patch.object(
                ingest,
                "fetch_rss",
                new=AsyncMock(return_value=_feed_with_article(rss_text)),
            ),
            patch.object(ingest, "guarded_fetch_article", new=article_fetch),
        ):
            inserted, updated = await ingest.gather_articles_to_store(config, store)

        article_fetch.assert_not_awaited()
        self.assertEqual((1, 0), (inserted, updated))
        self.assertEqual(rss_text.strip(), store.articles[0]["text"])

    async def test_article_page_is_still_fetched_by_default(self):
        rss_text = "RSS fallback text. " * 8
        web_text = "Extracted article page text. " * 8
        store = _MemoryArticleStore()
        article_fetch = AsyncMock(return_value="article HTML")
        config = {
            "feeds": [
                {
                    "name": "Default feed",
                    "url": "https://example.com/feed.xml",
                }
            ],
            "ingest": {"max_items_per_feed": 8},
        }

        with (
            patch.object(
                ingest,
                "fetch_rss",
                new=AsyncMock(return_value=_feed_with_article(rss_text)),
            ),
            patch.object(ingest, "guarded_fetch_article", new=article_fetch),
            patch.object(ingest, "extract_text_from_html", return_value=web_text),
        ):
            inserted, updated = await ingest.gather_articles_to_store(config, store)

        article_fetch.assert_awaited_once()
        self.assertEqual((1, 0), (inserted, updated))
        self.assertEqual(web_text, store.articles[0]["text"])

    def test_fetch_article_requires_a_boolean(self):
        with self.assertRaisesRegex(TypeError, "fetch_article"):
            ingest._should_fetch_article_page({"fetch_article": "false"})


if __name__ == "__main__":
    unittest.main()

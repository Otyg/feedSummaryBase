import unittest
from unittest.mock import AsyncMock, patch

from feedsummary_core.summarizer import main


class TagStore:
    def __init__(self, articles):
        self.articles = articles
        self.queries = []

    def get_articles_by_tags(self, tag_names, match_mode="any"):
        self.queries.append((tag_names, match_mode))
        return list(self.articles)

    def list_unsummarized_articles(self, limit=5000):
        raise AssertionError("tag-based summaries must not be limited to unsummarized articles")


class EnrichmentStore:
    def __init__(self, articles, tags_by_article):
        self.articles = {article["id"]: article for article in articles}
        self.tags_by_article = tags_by_article
        self.queries = []

    def get_articles_by_tags(self, tag_names, match_mode="any"):
        self.queries.append((list(tag_names), match_mode))
        wanted = {name.casefold() for name in tag_names}
        return [
            article
            for article_id, article in self.articles.items()
            if wanted.intersection(
                str(tag.get("name") or "").casefold()
                for tag in self.tags_by_article.get(article_id, [])
            )
        ]

    def get_article_tags(self, article_id):
        return list(self.tags_by_article.get(article_id, []))


class TagBasedSummaryTests(unittest.IsolatedAsyncioTestCase):
    def test_regular_summary_override_retains_vulnerability_sources_for_enrichment(self):
        config = {
            "feeds": [
                {"name": "Threat News", "topics": ["Cybersecurity"]},
                {"name": "CVE Feed", "topics": ["Sårbarheter"]},
            ]
        }

        overridden = main._apply_overrides(
            config,
            {"topics": ["Cybersecurity"], "enrich": True},
        )

        self.assertEqual(["Threat News"], main._selected_source_names(overridden))
        self.assertEqual({"CVE Feed"}, main._vulnerability_source_names(overridden))

    async def test_uses_store_tag_query_for_already_summarized_articles(self):
        now = 2_000_000_000
        article = {
            "id": "article-1",
            "published_ts": now - 60,
            "summary_ids": ["previous-summary"],
        }
        store = TagStore([article])
        summarize = AsyncMock(return_value="new-summary")

        with (
            patch.object(main, "create_store", return_value=store),
            patch.object(main, "load_feeds_into_config", side_effect=lambda config, **_: config),
            patch.object(main.time, "time", return_value=now),
            patch.object(main, "_summarize_and_persist_like_refresh", summarize),
        ):
            summary_id = await main.run_tag_based_summary(
                config_dict={"store": {}, "ingest": {}},
                llm=object(),
                tag_names=["vulnerability"],
                lookback="1w",
                prompt_package="vuln_report_running_text",
            )

        self.assertEqual("new-summary", summary_id)
        self.assertEqual([(["vulnerability"], "any")], store.queries)
        self.assertEqual([article], summarize.await_args.kwargs["articles"])
        self.assertEqual(
            "vuln_report_running_text",
            summarize.await_args.kwargs["config"]["prompts"]["selected"],
        )

    async def test_enriches_only_matching_vulnerability_articles_even_before_lookback(self):
        now = 2_000_000_000
        articles = [
            {
                "id": "primary",
                "source": "Threat News",
                "published_ts": now - 60,
            },
            {
                "id": "standalone-current",
                "source": "CVE Feed",
                "published_ts": now - 120,
            },
            {
                "id": "matching-old",
                "source": "CVE Feed",
                "published_ts": now - (30 * 86400),
            },
            {
                "id": "unmatched-old",
                "source": "CVE Feed",
                "published_ts": now - (40 * 86400),
            },
        ]
        tags = {
            "primary": [
                {"name": "incident", "category": "GENERAL"},
                {"name": "CVE-2026-1234", "category": "VULNERABILITY"},
                # The generic tag must never expand to all vulnerability material.
                {"name": "vulnerability", "category": "VULNERABILITY"},
            ],
            "standalone-current": [
                {"name": "vulnerability", "category": "VULNERABILITY"},
                {"name": "CVE-2026-9999", "category": "VULNERABILITY"},
            ],
            "matching-old": [
                {"name": "vulnerability", "category": "VULNERABILITY"},
                {"name": "CVE-2026-1234", "category": "VULNERABILITY"},
            ],
            "unmatched-old": [
                {"name": "vulnerability", "category": "VULNERABILITY"},
                {"name": "CVE-2026-8888", "category": "VULNERABILITY"},
            ],
        }
        store = EnrichmentStore(articles, tags)
        summarize = AsyncMock(return_value="enriched-summary")
        config = {
            "store": {},
            "ingest": {},
            "feeds": [
                {"name": "Threat News", "topics": ["Cybersecurity"]},
                {"name": "CVE Feed", "topics": ["Sårbarheter"]},
            ],
        }

        with (
            patch.object(main, "create_store", return_value=store),
            patch.object(main, "load_feeds_into_config", side_effect=lambda config, **_: config),
            patch.object(main.time, "time", return_value=now),
            patch.object(main, "_summarize_and_persist_like_refresh", summarize),
        ):
            summary_id = await main.run_tag_based_summary(
                config_dict=config,
                llm=object(),
                tag_names=["incident", "vulnerability"],
                lookback="1d",
                enrich=True,
            )

        self.assertEqual("enriched-summary", summary_id)
        selected = summarize.await_args.kwargs["articles"]
        self.assertEqual(["primary", "matching-old"], [article["id"] for article in selected])
        self.assertEqual(
            ["CVE-2026-1234"],
            selected[1]["_summary_enrichment"]["matched_tags"],
        )
        selection = summarize.await_args.kwargs["selection"]
        self.assertTrue(selection["enrich"])
        self.assertEqual(1, selection["enriched_article_count"])
        self.assertEqual(["CVE-2026-1234"], selection["enrichment_vulnerability_tags"])
        self.assertNotIn(
            "vulnerability",
            [name.casefold() for query, _mode in store.queries[1:] for name in query],
        )


if __name__ == "__main__":
    unittest.main()

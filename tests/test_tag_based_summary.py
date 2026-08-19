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


class TagBasedSummaryTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()

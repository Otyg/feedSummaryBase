import unittest

from feedsummary_core.summarizer.batching import build_messages_for_batch


class EnrichmentPromptMaterialTests(unittest.TestCase):
    def test_enrichment_article_is_marked_with_role_date_and_tag(self):
        messages = build_messages_for_batch(
            prompts={
                "batch_system": "system",
                "batch_user_template": "{articles_corpus}",
            },
            batch_index=1,
            batch_total=1,
            batch_items=[
                {
                    "title": "Old advisory",
                    "source": "CVE Feed",
                    "published_ts": 1_700_000_000,
                    "text": "Details",
                    "_summary_enrichment": {
                        "kind": "vulnerability",
                        "matched_tags": ["CVE-2026-1234"],
                    },
                }
            ],
        )

        user_prompt = messages[1]["content"]
        self.assertIn("BERIKANDE SÅRBARHETSUNDERLAG", user_prompt)
        self.assertIn("publicerad 2023-11-14", user_prompt)
        self.assertIn("matchande tagg: CVE-2026-1234", user_prompt)


if __name__ == "__main__":
    unittest.main()

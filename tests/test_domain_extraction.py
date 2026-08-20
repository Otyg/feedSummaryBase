import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from feedsummary_core.summarizer.helpers import load_extraction_rules_into_config
from feedsummary_core.summarizer.ingest import (
    _domain_extraction_rule,
    extract_text_from_html,
)


class ExtractionRulesConfigTests(unittest.TestCase):
    def test_loads_rules_relative_to_main_config(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rules_path = root / "config" / "extraction.yaml"
            rules_path.parent.mkdir()
            rules_path.write_text(
                "domains:\n"
                "  theregister.com:\n"
                "    content_xpath: //main/article/section[1]\n",
                encoding="utf-8",
            )
            config = {
                "ingest": {
                    "extraction": {
                        "path": "config/extraction.yaml",
                    }
                }
            }

            loaded = load_extraction_rules_into_config(
                config,
                base_config_path=str(root / "config.yaml"),
            )

            extraction = loaded["ingest"]["extraction"]
            self.assertEqual(
                "//main/article/section[1]",
                extraction["domains"]["theregister.com"]["content_xpath"],
            )
            self.assertEqual(str(rules_path), extraction["_source_path"])

    def test_rejects_rule_without_content_xpath(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rules_path = root / "extraction.yaml"
            rules_path.write_text(
                "domains:\n  example.com:\n    include_tables: true\n",
                encoding="utf-8",
            )
            config = {
                "ingest": {
                    "extraction": {
                        "path": "extraction.yaml",
                    }
                }
            }

            with self.assertRaisesRegex(ValueError, "content_xpath"):
                load_extraction_rules_into_config(
                    config,
                    base_config_path=str(root / "config.yaml"),
                )


class DomainExtractionTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "domains": {
                "theregister.com": {
                    "content_xpath": "//main/article/section[1]",
                }
            }
        }
        self.html = (
            "<html><body><section><main><article>"
            "<section><p>ARTICLE BODY</p></section>"
            "</article></main><footer>UNRELATED FOOTER</footer></section></body></html>"
        )

    def test_www_hostname_matches_bare_configured_domain(self):
        rule = _domain_extraction_rule(
            "https://www.theregister.com/2026/08/20/example/",
            self.config,
        )

        self.assertIs(rule, self.config["domains"]["theregister.com"])

    def test_unrelated_subdomain_does_not_match(self):
        rule = _domain_extraction_rule(
            "https://vendorvoice.theregister.com/example/",
            self.config,
        )

        self.assertIsNone(rule)

    @patch("feedsummary_core.summarizer.ingest.trafilatura.extract")
    def test_extracts_only_selected_fragment(self, extract):
        extract.return_value = "ARTICLE BODY"

        result = extract_text_from_html(
            self.html,
            "https://www.theregister.com/2026/08/20/example/",
            self.config,
        )

        self.assertEqual("ARTICLE BODY", result)
        selected_html = extract.call_args.args[0]
        self.assertIn("ARTICLE BODY", selected_html)
        self.assertNotIn("UNRELATED FOOTER", selected_html)
        extract.assert_called_once()

    @patch("feedsummary_core.summarizer.ingest.trafilatura.extract")
    def test_falls_back_to_full_page_when_selector_does_not_match(self, extract):
        extract.return_value = "FULL PAGE"
        config = {
            "domains": {
                "theregister.com": {
                    "content_xpath": "//does-not-exist",
                }
            }
        }

        result = extract_text_from_html(
            self.html,
            "https://theregister.com/example/",
            config,
        )

        self.assertEqual("FULL PAGE", result)
        extract.assert_called_once()
        self.assertEqual(self.html, extract.call_args.args[0])

    @patch("feedsummary_core.summarizer.ingest.trafilatura.extract")
    def test_falls_back_when_selected_fragment_produces_no_text(self, extract):
        extract.side_effect = [None, "FULL PAGE"]

        result = extract_text_from_html(
            self.html,
            "https://theregister.com/example/",
            self.config,
        )

        self.assertEqual("FULL PAGE", result)
        self.assertEqual(2, extract.call_count)
        self.assertEqual(self.html, extract.call_args_list[1].args[0])

    @patch("feedsummary_core.summarizer.ingest.trafilatura.extract")
    def test_passes_supported_domain_options_to_trafilatura(self, extract):
        extract.return_value = "ARTICLE BODY"
        self.config["domains"]["theregister.com"]["include_tables"] = True
        self.config["domains"]["theregister.com"]["favor_precision"] = True

        extract_text_from_html(
            self.html,
            "https://theregister.com/example/",
            self.config,
        )

        kwargs = extract.call_args.kwargs
        self.assertTrue(kwargs["include_tables"])
        self.assertTrue(kwargs["favor_precision"])
        self.assertFalse(kwargs["include_comments"])


if __name__ == "__main__":
    unittest.main()

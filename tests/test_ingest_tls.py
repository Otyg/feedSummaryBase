import asyncio
import ssl
import unittest
from unittest.mock import AsyncMock, patch

import aiohttp

from feedsummary_core.summarizer.ingest import (
    _feed_http_client,
    _feed_ssl_context,
    fetch_article_html,
    fetch_rss,
)


class FakeResponse:
    def __init__(self, body: bytes, content_type: str, status: int = 200):
        self._body = body
        self._content_type = content_type
        self.status = status
        self.headers = {}
        self.history = ()
        self.request_info = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=self.request_info,
                history=self.history,
                status=self.status,
                headers=self.headers,
            )

    async def read(self):
        return self._body

    async def text(self, errors="strict"):
        return self._body.decode("utf-8", errors=errors)


class FakeSession:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class FeedTlsConfigTests(unittest.TestCase):
    def test_no_setting_uses_aiohttp_default(self):
        self.assertIsNone(_feed_ssl_context({}))

    def test_tls_1_3_builds_context_with_minimum_version(self):
        context = _feed_ssl_context({"tls_min_version": "1.3"})

        self.assertIsInstance(context, ssl.SSLContext)
        self.assertEqual(ssl.TLSVersion.TLSv1_3, context.minimum_version)

    def test_tls_version_alias_is_supported(self):
        context = _feed_ssl_context({"tls_min_version": "TLSv1.2"})

        self.assertEqual(ssl.TLSVersion.TLSv1_2, context.minimum_version)

    def test_invalid_version_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tls_min_version"):
            _feed_ssl_context({"tls_min_version": "1.1"})

    def test_curl_http_client_is_supported(self):
        self.assertEqual("curl", _feed_http_client({"http_client": "curl"}))

    def test_invalid_http_client_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "http_client"):
            _feed_http_client({"http_client": "requests"})


class FeedTlsRequestTests(unittest.TestCase):
    def test_rss_request_receives_configured_context(self):
        response = FakeResponse(
            b"<rss version='2.0'><channel><title>Test</title></channel></rss>",
            "application/rss+xml",
        )
        session = FakeSession(response)
        context = _feed_ssl_context({"tls_min_version": "1.3"})

        asyncio.run(fetch_rss("https://example.com/feed.xml", session, context))

        self.assertIs(context, session.calls[0][1]["ssl"])

    def test_article_request_receives_same_context(self):
        response = FakeResponse(b"<html><body>Article</body></html>", "text/html")
        session = FakeSession(response)
        context = _feed_ssl_context({"tls_min_version": "1.3"})

        result = asyncio.run(
            fetch_article_html("https://example.com/article", session, 20, context)
        )

        self.assertEqual("<html><body>Article</body></html>", result)
        self.assertIs(context, session.calls[0][1]["ssl"])

    def test_unconfigured_request_does_not_override_aiohttp_ssl_default(self):
        response = FakeResponse(
            b"<rss version='2.0'><channel><title>Test</title></channel></rss>",
            "application/rss+xml",
        )
        session = FakeSession(response)

        asyncio.run(fetch_rss("https://example.com/feed.xml", session))

        self.assertNotIn("ssl", session.calls[0][1])

    @patch(
        "feedsummary_core.summarizer.ingest._fetch_with_curl",
        new_callable=AsyncMock,
    )
    def test_rss_403_retries_with_curl_tls_1_3_and_logs_config_warning(self, curl_fetch):
        forbidden = FakeResponse(b"Forbidden", "text/plain", status=403)
        curl_fetch.return_value = (
            b"<rss version='2.0'><channel><title>Test</title></channel></rss>"
        )
        session = FakeSession(forbidden)

        with self.assertLogs("feedsummary_core.summarizer.ingest", level="WARNING") as logs:
            asyncio.run(fetch_rss("https://example.com/feed.xml", session))

        self.assertEqual(1, len(session.calls))
        self.assertNotIn("ssl", session.calls[0][1])
        retry_context = curl_fetch.await_args.args[2]
        self.assertEqual(ssl.TLSVersion.TLSv1_3, retry_context.minimum_version)
        self.assertIn('http_client: "curl"', " ".join(logs.output))
        self.assertIn('tls_min_version: "1.3"', " ".join(logs.output))

    @patch(
        "feedsummary_core.summarizer.ingest._fetch_with_curl",
        new_callable=AsyncMock,
    )
    def test_article_403_retries_with_curl_tls_1_3_and_logs_config_warning(
        self,
        curl_fetch,
    ):
        forbidden = FakeResponse(b"Forbidden", "text/plain", status=403)
        curl_fetch.return_value = b"<html><body>Article</body></html>"
        session = FakeSession(forbidden)

        with self.assertLogs("feedsummary_core.summarizer.ingest", level="WARNING") as logs:
            result = asyncio.run(
                fetch_article_html("https://example.com/article", session, 20)
            )

        self.assertEqual("<html><body>Article</body></html>", result)
        retry_context = curl_fetch.await_args.args[2]
        self.assertEqual(ssl.TLSVersion.TLSv1_3, retry_context.minimum_version)
        self.assertIn('http_client: "curl"', " ".join(logs.output))
        self.assertIn('tls_min_version: "1.3"', " ".join(logs.output))

    @patch(
        "feedsummary_core.summarizer.ingest._fetch_with_curl",
        new_callable=AsyncMock,
    )
    def test_configured_curl_skips_aiohttp_request(self, curl_fetch):
        curl_fetch.return_value = (
            b"<rss version='2.0'><channel><title>Test</title></channel></rss>"
        )
        session = FakeSession()
        context = _feed_ssl_context({"tls_min_version": "1.3"})

        asyncio.run(
            fetch_rss(
                "https://example.com/feed.xml",
                session,
                context,
                http_client="curl",
            )
        )

        self.assertEqual(0, len(session.calls))
        curl_fetch.assert_awaited_once_with(
            "https://example.com/feed.xml",
            20,
            context,
        )


if __name__ == "__main__":
    unittest.main()

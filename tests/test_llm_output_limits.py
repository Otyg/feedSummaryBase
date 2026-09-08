import asyncio
import json
import logging
import unittest
from unittest.mock import AsyncMock

from feedsummary_core.llm_client.fallback_client import FallbackLLMClient, FallbackPolicy
from feedsummary_core.llm_client.ollama_cloud import (
    LLMUnavailableError,
    OllamaCloudClient,
)
from feedsummary_core.llm_client.ollama_local import OllamaConfig, OllamaLocalClient


class RecordingClient:
    def __init__(self, response="ok", error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def chat(self, messages, *, temperature=0.2, max_output_tokens=None):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


class LegacyClient:
    def __init__(self):
        self.calls = []

    async def chat(self, messages, *, temperature=0.2):
        self.calls.append((messages, temperature))
        return "legacy"


class LLMOutputLimitTests(unittest.TestCase):
    def test_fallback_forwards_same_limit_to_next_provider(self):
        primary = RecordingClient(error=LLMUnavailableError("unavailable"))
        fallback = RecordingClient(response="fallback")
        client = FallbackLLMClient(
            [primary, fallback],
            policy=FallbackPolicy(max_quota_retries=0, default_wait_s=0),
        )

        result = asyncio.run(
            client.chat(
                [{"role": "user", "content": "test"}],
                temperature=0.1,
                max_output_tokens=777,
            )
        )

        self.assertEqual("fallback", result)
        self.assertEqual(777, primary.calls[0]["max_output_tokens"])
        self.assertEqual(777, fallback.calls[0]["max_output_tokens"])

    def test_call_without_limit_remains_compatible_with_legacy_client(self):
        legacy = LegacyClient()
        client = FallbackLLMClient([legacy])

        result = asyncio.run(client.chat([{"role": "user", "content": "test"}], temperature=0.3))

        self.assertEqual("legacy", result)
        self.assertEqual(0.3, legacy.calls[0][1])

    def test_local_ollama_maps_limit_to_num_predict_and_omits_it_by_default(self):
        class Content:
            def __init__(self):
                self._lines = [json.dumps({"message": {"content": "ok"}, "done": True}).encode()]

            async def readline(self):
                return self._lines.pop(0) if self._lines else b""

        class Response:
            status = 200

            def __init__(self):
                self.content = Content()

        class RequestContext:
            async def __aenter__(self):
                return Response()

            async def __aexit__(self, *_args):
                return None

        class Session:
            def __init__(self):
                self.requests = []

            def post(self, url, *, json):
                self.requests.append((url, json))
                return RequestContext()

        session = Session()
        client = OllamaLocalClient(OllamaConfig(max_rps=0))
        client._get_session = AsyncMock(return_value=session)

        async def run_calls():
            await client.chat(
                [{"role": "user", "content": "bounded"}],
                max_output_tokens=456,
            )
            await client.chat([{"role": "user", "content": "default"}])

        asyncio.run(run_calls())

        self.assertEqual(456, session.requests[0][1]["options"]["num_predict"])
        self.assertNotIn("num_predict", session.requests[1][1]["options"])

    def test_cloud_ollama_maps_limit_to_num_predict(self):
        client = OllamaCloudClient(
            {
                "api_key": "test-key",
                "quota": {"preflight": False, "min_interval_seconds": 0},
            }
        )
        client._client = AsyncMock()
        client._client.chat.return_value = {"message": {"content": "ok"}}
        client._throttle = AsyncMock()
        client._preflight_quota_check = AsyncMock()
        client.log = logging.getLogger(__name__)

        result = asyncio.run(
            client.chat(
                [{"role": "user", "content": "bounded"}],
                max_output_tokens=654,
            )
        )

        self.assertEqual("ok", result)
        self.assertEqual(
            654,
            client._client.chat.await_args.kwargs["options"]["num_predict"],
        )


if __name__ == "__main__":
    unittest.main()

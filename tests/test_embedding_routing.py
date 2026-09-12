import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from feedsummary_core.llm_client import create_llm_client
from feedsummary_core.llm_client.fallback_client import FallbackLLMClient
from feedsummary_core.llm_client.ollama_cloud import OllamaCloudClient
from feedsummary_core.llm_client.ollama_local import OllamaConfig, OllamaLocalClient


class FakeClient:
    def __init__(self, name):
        self.name = name
        self.chat_calls = []
        self.embed_calls = []

    async def chat(self, messages, *, temperature=0.2):
        self.chat_calls.append((messages, temperature))
        return self.name

    async def embed(self, text, **kwargs):
        self.embed_calls.append((text, kwargs))
        return [1.0, 2.0]


class EmbeddingRoutingTests(unittest.TestCase):
    def test_local_qwen_request_contains_instruction_and_dimensions(self):
        class Response:
            status = 200

            async def json(self):
                return {"embeddings": [[0.0] * 1024]}

        class RequestContext:
            async def __aenter__(self):
                return Response()

            async def __aexit__(self, *_args):
                return None

        class Session:
            def __init__(self):
                self.request = None

            def post(self, url, *, json):
                self.request = (url, json)
                return RequestContext()

        session = Session()
        client = OllamaLocalClient(
            OllamaConfig(
                embedding_model="qwen3-embedding:0.6b",
                embedding_dimensions=1024,
            )
        )
        client._rate_gate = AsyncMock()
        client._get_session = AsyncMock(return_value=session)

        vector = asyncio.run(
            client.embed(
                "Article text",
                instruction="Represent an event.",
                dimensions=1024,
            )
        )

        self.assertEqual(1024, len(vector))
        self.assertEqual(
            {
                "model": "qwen3-embedding:0.6b",
                "input": "Instruct: Represent an event.\nQuery:Article text",
                "dimensions": 1024,
            },
            session.request[1],
        )

    def test_fallback_uses_dedicated_embedding_client(self):
        cloud = FakeClient("cloud")
        local = FakeClient("local")
        client = FallbackLLMClient(
            clients=[cloud, local],
            embedding_client=local,
        )

        result = asyncio.run(client.embed("test text"))

        self.assertEqual([1.0, 2.0], result)
        self.assertEqual([], cloud.embed_calls)
        self.assertEqual(
            [("test text", {"instruction": "", "dimensions": None})],
            local.embed_calls,
        )

    def test_factory_selects_ollama_local_for_embeddings(self):
        created_clients = []

        def create_fake(config):
            client = FakeClient(config["provider"])
            created_clients.append(client)
            return client

        config = {
            "llm": [
                {"provider": "ollama_cloud", "api_key": "unused-in-test"},
                {
                    "provider": "ollama_local",
                    "base_url": "http://localhost:11434",
                    "embedding_model": "embeddinggemma:latest",
                },
            ]
        }

        with patch("feedsummary_core.llm_client._create_single_llm", side_effect=create_fake):
            client = create_llm_client(config)

        result = asyncio.run(client.embed("test text"))

        self.assertEqual([1.0, 2.0], result)
        self.assertEqual([], created_clients[0].embed_calls)
        self.assertEqual(
            [("test text", {"instruction": "", "dimensions": None})],
            created_clients[1].embed_calls,
        )

    def test_factory_passes_local_embedding_configuration(self):
        config = {
            "llm": [
                {"provider": "ollama_cloud", "api_key": "unused-in-test"},
                {
                    "provider": "ollama_local",
                    "base_url": "http://local-ollama.test:11434",
                    "model": "local-chat:test",
                    "embedding_model": "local-embedding:test",
                    "max_rps": 3,
                },
            ]
        }

        client = create_llm_client(config)

        self.assertEqual(
            "http://local-ollama.test:11434",
            client.embedding_client.cfg.base_url,
        )
        self.assertEqual("local-embedding:test", client.embedding_client.cfg.embedding_model)
        self.assertEqual(1024, client.embedding_client.cfg.embedding_dimensions)
        self.assertEqual(3.0, client.embedding_client.cfg.max_rps)
        asyncio.run(client.aclose())

    def test_fallback_without_local_embedding_config_fails_clearly(self):
        client = FallbackLLMClient(clients=[FakeClient("cloud")])

        with self.assertRaisesRegex(RuntimeError, "ollama_local"):
            asyncio.run(client.embed("test text"))

    def test_cloud_client_rejects_embeddings_without_network_call(self):
        cloud = object.__new__(OllamaCloudClient)

        with self.assertRaisesRegex(RuntimeError, "ollama_local"):
            asyncio.run(cloud.embed("test text"))


if __name__ == "__main__":
    unittest.main()

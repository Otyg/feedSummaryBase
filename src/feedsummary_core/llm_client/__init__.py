# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
from __future__ import annotations

from feedsummary_core.llm_client.fallback_client import FallbackLLMClient, FallbackPolicy
import logging

from typing import Any, Dict, List, Optional, Protocol

log = logging.getLogger(__name__)


class LLMRateLimitError(RuntimeError):
    """Signal that an LLM provider rejected a request due to quota or rate limiting."""

    def __init__(self, message: str, retry_after_seconds: Optional[int] = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMError(Exception):
    """Base exception for generic LLM client failures."""

    pass


class LLMClient(Protocol):
    """Protocol implemented by asynchronous chat-oriented LLM clients."""

    async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.2) -> str: ...


def _create_single_llm(llm_cfg: Dict[str, Any]):
    provider = (llm_cfg.get("provider") or "ollama").lower()

    if provider == "ollama_cloud":
        from feedsummary_core.llm_client.ollama_cloud import OllamaCloudClient

        return OllamaCloudClient(llm_cfg)

    if provider == "ollama_local":
        from feedsummary_core.llm_client.ollama_local import OllamaLocalClient, OllamaConfig

        cfg = OllamaConfig(
            base_url=str(llm_cfg.get("base_url", "http://localhost:11434")),
            model=str(llm_cfg.get("model", "llama3.1:latest")),
            max_rps=float(llm_cfg.get("max_rps", 1.0)),
            first_byte_timeout_s=int(llm_cfg.get("first_byte_timeout_s", 900)),
            sock_read_timeout_s=int(llm_cfg.get("sock_read_timeout_s", 300)),
            progress_log_every_s=float(llm_cfg.get("progress_log_every_s", 2.0)),
            max_retries=int(llm_cfg.get("max_retries", 3)),
        )
        return OllamaLocalClient(cfg)

    raise ValueError(f"Unsupported LLM provider: {provider}")


def get_primary_llm_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returnerar första LLM-konfigurationen (dict) oavsett om config['llm']
    är dict eller lista av dict.
    """
    llm_cfg = config.get("llm")
    if isinstance(llm_cfg, dict):
        return llm_cfg
    if isinstance(llm_cfg, list):
        for item in llm_cfg:
            if isinstance(item, dict):
                return item
        return {}
    return {}


def _collect_llm_chain_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    llm_raw = config.get("llm")
    chain_cfgs: List[Dict[str, Any]] = []

    if isinstance(llm_raw, dict):
        chain_cfgs.append(llm_raw)
    elif isinstance(llm_raw, list):
        for i, item in enumerate(llm_raw, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"config['llm'][{i}] måste vara ett objekt/dict.")
            chain_cfgs.append(item)
    elif llm_raw is None:
        chain_cfgs = []
    else:
        raise ValueError("config['llm'] måste vara dict eller lista av dict.")

    # Bakåtkompatibilitet: appenda legacy llm_fallback sist om den finns.
    fallback_cfg = config.get("llm_fallback")
    if isinstance(fallback_cfg, dict):
        chain_cfgs.append(fallback_cfg)
        if isinstance(llm_raw, list):
            log.warning(
                "Både config['llm'] (lista) och legacy config['llm_fallback'] finns. "
                "Appenderar llm_fallback sist i fallback-kedjan."
            )
    elif fallback_cfg is not None:
        raise ValueError("config['llm_fallback'] måste vara dict om den anges.")

    return chain_cfgs


def create_llm_client(config: Dict[str, Any]):
    """
    Bygger LLM-kedja från config['llm'].
    - dict: en primär provider (ev + legacy llm_fallback)
    - list: providers i prioriterad ordning (0..n), fallback till nästa vid trigger
    """
    chain_cfgs = _collect_llm_chain_configs(config)
    if not chain_cfgs:
        raise ValueError("Saknar LLM-konfiguration: config['llm'] är tom.")

    clients = [_create_single_llm(c) for c in chain_cfgs]

    # Policy tas från primär (första) config.
    quota_cfg = (chain_cfgs[0].get("quota") or {}) if chain_cfgs else {}
    policy = FallbackPolicy(
        max_quota_retries=int(quota_cfg.get("max_quota_retries", 1)),
        default_wait_s=int(quota_cfg.get("default_wait_s", 30)),
    )

    if len(clients) == 1:
        return clients[0]
    return FallbackLLMClient(clients=clients, policy=policy)

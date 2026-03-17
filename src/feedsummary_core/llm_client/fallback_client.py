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

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from feedsummary_core.llm_client.ollama_cloud import LLMRateLimitError, LLMUnavailableError

log = logging.getLogger(__name__)


class LLMClient(Protocol):
    """Minimal async chat contract used by the fallback wrapper."""

    async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.2) -> str: ...


@dataclass
class FallbackPolicy:
    """Retry and wait settings for quota-triggered provider fallback."""

    max_quota_retries: int = 1  # "vid retry" -> om det slår i igen efter 1 retry: fallback
    default_wait_s: int = 30  # om retry_after saknas
    jitter_s: float = 0.0  # håll 0 om du vill vara deterministisk


class FallbackLLMClient:
    """
    Wrapper som kör providers i prioritetsordning.

    Beteende:
      - Kör första provider i listan.
      - Vid LLMRateLimitError/LLMUnavailableError:
          * vänta + retry upp till max_quota_retries
          * vid LLMUnavailableError: prova nästa provider endast för aktuellt anrop
          * vid LLMRateLimitError: växla permanent till nästa provider i listan
      - Finns ingen nästa provider: bubbla felet.
    """

    def __init__(self, clients: List[LLMClient], policy: Optional[FallbackPolicy] = None):
        if not clients:
            raise ValueError("FallbackLLMClient kräver minst en LLM-klient.")
        self.clients = clients
        self.policy = policy or FallbackPolicy()
        self._active_idx = 0
        self._blocked_indices: set[int] = set()

    def _next_provider_idx(self, start_idx: int) -> Optional[int]:
        for idx in range(max(0, start_idx), len(self.clients)):
            if idx not in self._blocked_indices:
                return idx
        return None

    def _try_advance_provider_permanently(self, current_idx: int, reason: str) -> Optional[int]:
        self._blocked_indices.add(current_idx)
        next_idx = self._next_provider_idx(current_idx + 1)

        if next_idx is None:
            return None

        old_i = current_idx
        if self._active_idx == current_idx:
            self._active_idx = next_idx
        log.warning(
            "LLM %s efter %d retry(s). Växlar permanent provider %d/%d -> %d/%d.",
            reason,
            self.policy.max_quota_retries,
            old_i + 1,
            len(self.clients),
            next_idx + 1,
            len(self.clients),
        )
        return next_idx

    async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.2) -> str:
        provider_idx = self._next_provider_idx(self._active_idx)
        if provider_idx is None:
            raise RuntimeError("Ingen LLM-provider tillgänglig; samtliga providers är permanent blockerade.")

        while True:
            active = self.clients[provider_idx]

            attempt = 0
            while True:
                try:
                    return await active.chat(messages, temperature=temperature)
                except LLMUnavailableError as e:
                    attempt += 1
                    wait_s = int(self.policy.default_wait_s)

                    if attempt > self.policy.max_quota_retries:
                        next_idx = self._next_provider_idx(provider_idx + 1)
                        if next_idx is not None:
                            log.warning(
                                "LLM otillgänglig efter %d retry(s). "
                                "Provar nästa provider %d/%d -> %d/%d för aktuellt anrop.",
                                self.policy.max_quota_retries,
                                provider_idx + 1,
                                len(self.clients),
                                next_idx + 1,
                                len(self.clients),
                            )
                            provider_idx = next_idx
                            break
                        raise

                    log.warning(
                        "LLM provider %d/%d otillgänglig, %s.\n (attempt %d/%d) Väntar %ss och retry...",
                        provider_idx + 1,
                        len(self.clients),
                        e,
                        attempt,
                        self.policy.max_quota_retries,
                        wait_s,
                    )
                    await asyncio.sleep(wait_s)
                except LLMRateLimitError as e:
                    attempt += 1
                    wait_s = int(e.retry_after_seconds or self.policy.default_wait_s)

                    if attempt > self.policy.max_quota_retries:
                        next_idx = self._try_advance_provider_permanently(
                            provider_idx, "quota/rate-limit"
                        )
                        if next_idx is not None:
                            provider_idx = next_idx
                            break
                        raise

                    log.warning(
                        "LLM provider %d/%d rate-limited (attempt %d/%d). Väntar %ss och retry...",
                        provider_idx + 1,
                        len(self.clients),
                        attempt,
                        self.policy.max_quota_retries,
                        wait_s,
                    )
                    await asyncio.sleep(wait_s)

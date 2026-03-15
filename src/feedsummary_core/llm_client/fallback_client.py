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
          * om problemet kvarstår, växla permanent till nästa provider i listan
      - Finns ingen nästa provider: bubbla felet.
    """

    def __init__(self, clients: List[LLMClient], policy: Optional[FallbackPolicy] = None):
        if not clients:
            raise ValueError("FallbackLLMClient kräver minst en LLM-klient.")
        self.clients = clients
        self.policy = policy or FallbackPolicy()
        self._active_idx = 0

    def _try_advance_provider(self, reason: str) -> bool:
        if self._active_idx >= (len(self.clients) - 1):
            return False
        old_i = self._active_idx
        self._active_idx += 1
        log.warning(
            "LLM %s efter %d retry(s). Växlar provider %d/%d -> %d/%d.",
            reason,
            self.policy.max_quota_retries,
            old_i + 1,
            len(self.clients),
            self._active_idx + 1,
            len(self.clients),
        )
        return True

    async def chat(self, messages: List[Dict[str, str]], *, temperature: float = 0.2) -> str:
        while True:
            active_idx = min(self._active_idx, len(self.clients) - 1)
            active = self.clients[active_idx]

            attempt = 0
            while True:
                try:
                    return await active.chat(messages, temperature=temperature)
                except LLMUnavailableError as e:
                    attempt += 1
                    wait_s = int(self.policy.default_wait_s)

                    if attempt > self.policy.max_quota_retries:
                        # Växla endast om vi fortfarande är på samma provider.
                        if self._active_idx == active_idx and self._try_advance_provider("otillgänglig"):
                            break
                        if self._active_idx > active_idx:
                            break
                        raise

                    log.warning(
                        "LLM provider %d/%d otillgänglig, %s.\n (attempt %d/%d) Väntar %ss och retry...",
                        active_idx + 1,
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
                        # Växla endast om vi fortfarande är på samma provider.
                        if self._active_idx == active_idx and self._try_advance_provider(
                            "quota/rate-limit"
                        ):
                            break
                        if self._active_idx > active_idx:
                            break
                        raise

                    log.warning(
                        "LLM provider %d/%d rate-limited (attempt %d/%d). Väntar %ss och retry...",
                        active_idx + 1,
                        len(self.clients),
                        attempt,
                        self.policy.max_quota_retries,
                        wait_s,
                    )
                    await asyncio.sleep(wait_s)

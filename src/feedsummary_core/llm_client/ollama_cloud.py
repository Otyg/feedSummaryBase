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
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import weakref

from ollama import AsyncClient

logger = logging.getLogger(__name__)

# Delat tillstånd mellan klientinstanser för samma host/model/api-key.
_GLOBAL_LAST_CALL_TS: Dict[str, float] = {}
_GLOBAL_PREFLIGHT_OK_TS: Dict[str, float] = {}


@dataclass
class _LoopSyncPrimitives:
    throttle_lock: asyncio.Lock
    concurrency_sem: asyncio.Semaphore


# asyncio-primitiver är loop-bundna, så de måste vara per event loop.
_GLOBAL_LOOP_SYNCS: Dict[str, weakref.WeakKeyDictionary] = {}


class LLMRateLimitError(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: Optional[int] = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class LLMAuthError(RuntimeError):
    pass


class LLMUnavailableError(RuntimeError):
    pass


@dataclass
class OllamaCloudConfig:
    host: str = "https://ollama.com"
    model: str = "gemma3:270m"
    api_key: str = ""
    # quota/preflight
    preflight: bool = True
    min_interval_seconds: float = 1.0
    timeout_seconds: float = 180.0
    max_concurrency: int = 1


def _resolve_env(value: str) -> str:
    """
    Stöd för ${VAR} och $VAR i config.yaml.
    """
    if not isinstance(value, str):
        return str(value)
    return os.path.expandvars(value)


def _extract_retry_after_seconds(exc: Exception) -> Optional[int]:
    """
    Försök läsa Retry-After från exception/response om biblioteket exponerar det.
    """
    for attr in ("retry_after", "retry_after_seconds"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    return None


def _is_status(exc: Exception, code: int) -> bool:
    sc = getattr(exc, "status_code", None)
    if isinstance(sc, int) and sc == code:
        return True
    st = getattr(exc, "status", None)
    return isinstance(st, int) and st == code


def _client_key(cfg: OllamaCloudConfig) -> str:
    # Inkludera auth i nyckeln för att inte dela state mellan olika konton.
    # Vi loggar inte hela nyckeln, bara använder den internt.
    return f"{cfg.host.rstrip('/')}|{cfg.model}|{cfg.api_key}"


class OllamaCloudClient:
    """
    LLMClient-implementation för Ollama Cloud API.

    Viktiga egenskaper i denna version:
    - global throttle mellan klientinstanser
    - global preflight-cache mellan klientinstanser
    - global samtidighetsgräns mellan klientinstanser
    """

    def __init__(self, cfg: Dict[str, Any]):
        llm_cfg = cfg or {}
        quota_cfg = llm_cfg.get("quota") or {}

        self.cfg = OllamaCloudConfig(
            host=str(llm_cfg.get("host", "https://ollama.com")),
            model=str(llm_cfg.get("model", "gemma3:270m")),
            api_key=_resolve_env(str(llm_cfg.get("api_key", ""))),
            preflight=bool(quota_cfg.get("preflight", True)),
            min_interval_seconds=float(quota_cfg.get("min_interval_seconds", 1.0)),
            timeout_seconds=float(llm_cfg.get("timeout_seconds", 360.0)),
            max_concurrency=max(1, int(quota_cfg.get("max_concurrency", 1))),
        )

        if not self.cfg.api_key:
            raise LLMAuthError("Ollama Cloud kräver api_key i config.yaml (eller via env-var).")

        self._client = AsyncClient(
            host=self.cfg.host,
            headers={"Authorization": f"Bearer {self.cfg.api_key}"},
            timeout=self.cfg.timeout_seconds,
        )
        self.log = logging.getLogger(__name__)

        self._key = _client_key(self.cfg)

        if self._key not in _GLOBAL_LAST_CALL_TS:
            _GLOBAL_LAST_CALL_TS[self._key] = 0.0
        if self._key not in _GLOBAL_PREFLIGHT_OK_TS:
            _GLOBAL_PREFLIGHT_OK_TS[self._key] = 0.0

    def _get_loop_sync(self) -> _LoopSyncPrimitives:
        loop = asyncio.get_running_loop()

        by_loop = _GLOBAL_LOOP_SYNCS.get(self._key)
        if by_loop is None:
            by_loop = weakref.WeakKeyDictionary()
            _GLOBAL_LOOP_SYNCS[self._key] = by_loop

        sync = by_loop.get(loop)
        if sync is None:
            sync = _LoopSyncPrimitives(
                throttle_lock=asyncio.Lock(),
                concurrency_sem=asyncio.Semaphore(self.cfg.max_concurrency),
            )
            by_loop[loop] = sync
        return sync

    async def _throttle(self) -> None:
        sync = self._get_loop_sync()
        async with sync.throttle_lock:
            now = time.time()
            dt = now - _GLOBAL_LAST_CALL_TS[self._key]
            if dt < self.cfg.min_interval_seconds:
                await asyncio.sleep(self.cfg.min_interval_seconds - dt)
            _GLOBAL_LAST_CALL_TS[self._key] = time.time()

    async def _preflight_quota_check(self) -> None:
        if not self.cfg.preflight:
            return

        now = time.time()
        if (now - _GLOBAL_PREFLIGHT_OK_TS[self._key]) < 10.0:
            return

        try:
            await self._client.list()
            _GLOBAL_PREFLIGHT_OK_TS[self._key] = time.time()
        except Exception as e:
            if _is_status(e, 401) or _is_status(e, 403):
                raise LLMAuthError("Ollama Cloud auth misslyckades (api_key fel/nekad).") from e
            if _is_status(e, 429):
                ra = _extract_retry_after_seconds(e)
                raise LLMRateLimitError(
                    "Ollama Cloud: rate limit/quota uppnådd (preflight).",
                    ra,
                ) from e
            raise LLMUnavailableError(f"Ollama Cloud preflight misslyckades: {e}") from e

    async def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        """
        Returnerar en enda textsträng.
        Hanterar:
        - global samtidighetsgräns
        - global throttle
        - optional preflight
        - tydlig 429/auth/server-timeout-hantering
        """
        sync = self._get_loop_sync()
        async with sync.concurrency_sem:
            await self._throttle()
            await self._preflight_quota_check()

            self.log.info("LLM request start (model=%s)", self.cfg.model)
            payload_options = {"temperature": temperature}

            try:
                resp = await self._client.chat(
                    model=self.cfg.model,
                    messages=messages,
                    stream=False,
                    options=payload_options,
                )
                if isinstance(resp, dict):
                    return (resp.get("message") or {}).get("content", "") or ""
                return getattr(resp.message, "content", "") or ""

            except Exception as e:
                try:
                    body = getattr(e, "error", None) or getattr(e, "message", None) or str(e)
                    if isinstance(body, str) and body.strip():
                        self.log.error("Ollama Cloud error response/body:\n%s", body[:12000])
                except Exception:
                    pass

                if _is_status(e, 401) or _is_status(e, 403):
                    raise LLMAuthError(
                        "Ollama Cloud: auth misslyckades (api_key fel/nekad)."
                    ) from e

                if _is_status(e, 429):
                    ra = _extract_retry_after_seconds(e)
                    raise LLMRateLimitError(
                        "Ollama Cloud: rate limit/quota uppnådd.",
                        ra,
                    ) from e

                if _is_status(e, 500):
                    raise LLMUnavailableError(
                        f"Ollama Cloud: server error 500. Möjlig överbelastning.\n{e}"
                    ) from e

                name = e.__class__.__name__.lower()
                if "timeout" in name:
                    raise LLMUnavailableError(f"Ollama Cloud: timeout: {e}") from e

                raise LLMUnavailableError(f"Ollama Cloud: chat misslyckades: {e}") from e

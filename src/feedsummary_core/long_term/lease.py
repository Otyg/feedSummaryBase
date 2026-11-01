# LICENSE HEADER MANAGED BY add-license-header
#
# BSD 3-Clause License
#
# Copyright (c) 2026, Martin Vesterlund
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""Lease heartbeat for long-running asynchronous long-term analysis phases."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Protocol, Self


class LeaseRenewalStore(Protocol):
    def renew_long_term_lease(
        self,
        profile_id: str,
        owner_id: str,
        *,
        now_ts: int,
        lease_seconds: int,
    ) -> bool: ...


class LeaseGuard(Protocol):
    async def ensure_owned(self) -> None: ...


class LeaseLostError(RuntimeError):
    """The current worker can no longer prove ownership of the profile lease."""


class LongTermLeaseHeartbeat:
    """Renew a claimed lease in the background and verify it before writes."""

    def __init__(
        self,
        store: LeaseRenewalStore,
        *,
        profile_id: str,
        owner_id: str,
        lease_seconds: int,
        interval_seconds: float | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.profile_id = str(profile_id or "").strip()
        self.owner_id = str(owner_id or "").strip()
        self.lease_seconds = int(lease_seconds)
        if not self.profile_id or not self.owner_id or self.lease_seconds < 1:
            raise ValueError("profile, owner and positive lease_seconds are required")
        default_interval = self.lease_seconds / 3
        self.interval_seconds = float(
            interval_seconds if interval_seconds is not None else default_interval
        )
        if self.interval_seconds <= 0 or self.interval_seconds >= self.lease_seconds:
            raise ValueError("heartbeat interval must be positive and shorter than the lease")
        self._clock = clock
        self._task: asyncio.Task[None] | None = None
        self._renew_lock = asyncio.Lock()
        self._failure: LeaseLostError | None = None

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("lease heartbeat is already running")
        self._failure = None
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.stop()

    async def _renew_once(self) -> None:
        async with self._renew_lock:
            if self._failure is not None:
                raise self._failure
            now_ts = int(self._clock())
            try:
                renewed = await asyncio.to_thread(
                    self.store.renew_long_term_lease,
                    self.profile_id,
                    self.owner_id,
                    now_ts=now_ts,
                    lease_seconds=self.lease_seconds,
                )
            except Exception as error:
                self._failure = LeaseLostError(
                    f"lease renewal failed for profile {self.profile_id}"
                )
                raise self._failure from error
            if not renewed:
                self._failure = LeaseLostError(
                    f"profile lease was lost: {self.profile_id}"
                )
                raise self._failure

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.interval_seconds)
                await self._renew_once()
        except asyncio.CancelledError:
            raise
        except LeaseLostError:
            return

    async def ensure_owned(self) -> None:
        """Atomically renew now, failing before the caller performs a durable write."""
        if self._task is None:
            raise RuntimeError("lease heartbeat is not running")
        if self._failure is not None:
            raise self._failure
        await self._renew_once()

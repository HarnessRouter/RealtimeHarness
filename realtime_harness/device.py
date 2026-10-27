"""The device's side of a call, as one socket that outlives any one connection (design §9, item 2; check R-11).

Pipecat's WebSocket transport reads and writes a `DeviceLink` as it would the device's own socket. The call
(the pipeline and the provider's connection) lives as long as the link, not as long as one connection:

- A device that closes its socket with code 1000 hangs up.
- Any other drop pauses the call. The provider connection is held, so the conversation is intact. What the
  model says meanwhile is recorded and not heard. The device comes back with its resume token, within the
  grace window, to the same call. A device that does not come back ends the call with reason `device_lost`.
- A device may come back before the server has noticed the drop (a half-open connection): the new connection
  replaces the old one.

Each connection is read by its own task into one queue, so an old connection that never finishes closing
cannot hold up the next.
"""
from __future__ import annotations

import asyncio
import time
from typing import Callable

from starlette.websockets import WebSocketState

NORMAL_CLOSURE = 1000
_DISCONNECT = {"type": "websocket.disconnect", "code": NORMAL_CLOSURE}


class DeviceLink:
    def __init__(self, websocket, grace_s: float, on_lost: Callable[[], None], on_back: Callable[[], None]):
        self.grace_s = grace_s
        self.on_lost = on_lost
        self.on_back = on_back
        self.headers: dict = {}
        self.end_reason: str | None = None          # hangup | device_lost, once the device has gone for good
        self._q: asyncio.Queue = asyncio.Queue()
        self._ws = None
        self._lost_at: float | None = None
        self._closed = False
        self._readers: dict = {}
        self._done: dict = {}
        self.attach(websocket)

    # ── what Pipecat's transport uses ────────────────────────────────────────────────────────
    @property
    def client_state(self) -> WebSocketState:
        return WebSocketState.DISCONNECTED if self._closed else WebSocketState.CONNECTED

    @property
    def application_state(self) -> WebSocketState:
        return self.client_state

    @property
    def paused(self) -> bool:
        return self._ws is None and not self._closed

    async def receive(self) -> dict:
        while True:
            if self._closed:
                return _DISCONNECT
            if self._ws is None:
                left = (self._lost_at or 0) + self.grace_s - time.monotonic()
                if left <= 0:
                    self.end_reason = self.end_reason or "device_lost"
                    self._closed = True
                    return _DISCONNECT
                try:
                    ws, msg = await asyncio.wait_for(self._q.get(), left)
                except asyncio.TimeoutError:
                    continue
            else:
                ws, msg = await self._q.get()
            if ws is None or ws is not self._ws:
                continue                               # a replaced connection's late messages, or a wake-up
            if msg.get("type") == "websocket.disconnect":
                if msg.get("code") == NORMAL_CLOSURE:
                    self.end_reason = "hangup"
                    self._release(ws)
                    self._ws = None
                    self._closed = True
                    return _DISCONNECT
                self._lose(ws)
                continue
            return msg

    async def send_bytes(self, data: bytes) -> None:
        await self._send(data)

    async def send_text(self, data: str) -> None:
        await self._send(data)

    async def close(self, code: int = NORMAL_CLOSURE) -> None:
        self._closed = True
        ws, self._ws = self._ws, None
        self._q.put_nowait((None, None))
        if ws is not None:
            self._release(ws)
            await _quiet_close(ws, code)
        for done in self._done.values():
            done.set()

    # ── what the server uses ─────────────────────────────────────────────────────────────────
    def attach(self, websocket) -> asyncio.Event:
        """Make this connection the device's. Returns an event set when it is no longer the device's (it
        dropped, was replaced, or the call ended), for the route that accepted it to return on."""
        done = asyncio.Event()
        if self._closed:
            done.set()
            return done
        old, was_paused = self._ws, self._ws is None and self._lost_at is not None
        self._ws, self._lost_at = websocket, None
        self._done[websocket] = done
        self._readers[websocket] = asyncio.get_running_loop().create_task(self._read(websocket))
        if old is not None:
            self._release(old)
            asyncio.get_running_loop().create_task(_quiet_close(old, 4000))
        if was_paused:
            self.on_back()
        return done

    # ── inside ───────────────────────────────────────────────────────────────────────────────
    async def _read(self, ws) -> None:
        try:
            while True:
                msg = await ws.receive()
                await self._q.put((ws, msg))
                if msg.get("type") == "websocket.disconnect":
                    return
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001: a connection that fails to read has dropped
            await self._q.put((ws, {"type": "websocket.disconnect", "code": 1006}))

    async def _send(self, data) -> None:
        ws = self._ws
        if ws is None or self._closed:
            return                                     # paused: what the model says is recorded, not heard
        try:
            if isinstance(data, (bytes, bytearray)):
                await ws.send_bytes(bytes(data))
            else:
                await ws.send_text(data)
        except Exception:  # noqa: BLE001: a connection that fails to write has dropped
            self._lose(ws)

    def _lose(self, ws) -> None:
        if ws is not self._ws or self._closed:
            return
        self._ws, self._lost_at = None, time.monotonic()
        self._release(ws)
        self.on_lost()

    def _release(self, ws) -> None:
        reader = self._readers.pop(ws, None)
        if reader is not None and reader is not asyncio.current_task():
            reader.cancel()
        done = self._done.pop(ws, None)
        if done is not None:
            done.set()


async def _quiet_close(ws, code: int) -> None:
    try:
        await asyncio.wait_for(ws.close(code=code), 2)
    except Exception:  # noqa: BLE001: a dead connection may not close cleanly; it is gone either way
        pass

"""A device on WebRTC (design §6.2, item 3): the device POSTs its SDP offer to the connect URL with its token and
gets the answer back, the WHIP pattern. Audio is Opus, and a data channel the device opens (named `uhp`)
carries the same JSON events as the WebSocket protocol, both ways. The transport is Pipecat's SmallWebRTC, on
aiortc (`pip install "realtime-harness[webrtc]"`).

A closed or failed peer connection pauses the call, as a dropped WebSocket does (design §9, item 2): the
provider connection is held, and the device comes back by POSTing a new offer with its resume token, which
restarts the peer connection into the same call. Over WebRTC a hang-up is explicit: `realtime.session.end` on
the data channel, or the client's cancel. A device that does not come back within the grace ends the call
with reason `device_lost`.
"""
from __future__ import annotations

import asyncio
import os
from typing import Callable

from .device import SAMPLE_RATE


def available() -> bool:
    try:
        import aiortc  # noqa: F401
        from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection  # noqa: F401
    except ImportError:
        return False
    return True


def ice_servers() -> list[str]:
    """STUN servers for the server's side of the connection, from RTH_ICE_SERVERS (comma separated). None by
    default: a server with a public address needs none."""
    return [u.strip() for u in (os.environ.get("RTH_ICE_SERVERS") or "").split(",") if u.strip()]


class WebRTCDevice:
    def __init__(self, connection, grace_s: float, on_lost: Callable[[], None], on_back: Callable[[], None]):
        self.connection = connection
        self.grace_s = grace_s
        self.on_lost = on_lost
        self.on_back = on_back
        self.end_reason: str | None = None
        self.paused = False
        self._grace: asyncio.TimerHandle | None = None
        self._finish: Callable[[str], None] | None = None
        self._client = None
        self._out_track = None

    @classmethod
    async def answer(cls, sdp: str, grace_s: float, on_lost, on_back) -> tuple["WebRTCDevice", str]:
        from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
        conn = SmallWebRTCConnection(ice_servers=ice_servers() or None)
        await conn.initialize(sdp=sdp, type="offer")
        return cls(conn, grace_s, on_lost, on_back), conn.get_answer()["sdp"]

    async def resume(self, sdp: str) -> str:
        """The device is back with a new offer: a new peer connection carries the same call."""
        old = self._out_track
        await self.connection.renegotiate(sdp=sdp, type="offer", restart_pc=True)
        _settle(old)
        return self.connection.get_answer()["sdp"]

    def send(self, event: dict) -> None:
        """An event for the device only (queued until its data channel is open)."""
        self.connection.send_app_message(event)

    def transport(self, on_device_event: Callable[[dict], None], finish: Callable[[str], None]):
        from pipecat.transports.base_transport import TransportParams
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
        self._finish = finish
        t = SmallWebRTCTransport(self.connection, TransportParams(
            audio_in_enabled=True, audio_out_enabled=True, audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE))

        @t.event_handler("on_app_message")
        async def _message(_t, message, _sender):
            on_device_event(message if isinstance(message, dict) else {})

        @t.event_handler("on_client_disconnected")
        async def _gone(_t, _c):
            self._lost()

        @t.event_handler("on_client_connected")
        async def _here(_t, _c):
            self._out_track = getattr(self._client, "_audio_output_track", None)
            self._back()
        self._client = getattr(t, "_client", None)
        return t

    def _lost(self) -> None:
        _settle(self._out_track)
        self._out_track = None
        if self.end_reason or self.paused:
            return
        self.paused = True
        self.on_lost()
        self._grace = asyncio.get_running_loop().call_later(self.grace_s, self._expire)

    def _expire(self) -> None:
        if self.paused and not self.end_reason:
            self.end_reason = "device_lost"
            if self._finish is not None:
                self._finish("device_lost")

    def _back(self) -> None:
        if not self.paused:
            return
        self.paused = False
        if self._grace is not None:
            self._grace.cancel()
            self._grace = None
        self.on_back()

    async def close(self) -> None:
        _settle(self._out_track)
        self.end_reason = self.end_reason or "hangup"
        if self._grace is not None:
            self._grace.cancel()
        try:
            await self.connection.disconnect()
        except Exception:  # noqa: BLE001  # a peer connection that is already gone closes as it is
            pass


def _settle(track) -> None:
    """Pipecat's output track resolves a write when the peer connection has read it. A track whose peer
    connection closed is never read again, so a write in flight on it would wait out the transport's write
    timeout, and the transport would give up on the call. Settling its pending writes lets the call go on
    while the device is away, and on the new peer connection when it is back."""
    queue = getattr(track, "_chunk_queue", None)
    if not queue:
        return
    for _chunk, fut in list(queue):
        if fut is not None and not fut.done():
            fut.set_result(None)
    queue.clear()

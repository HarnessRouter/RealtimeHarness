"""A realtime provider for tests: the OpenAI realtime wire, enough of it for a call, with a script instead of a
model. It hears speech as loud audio followed by a pause, and takes the words from a list of utterances, in
order. It answers from what its conversation holds, so a test can tell whether the conversation reached it:

- a question about the weather, with a weather tool offered: it calls the tool, then answers from the result
- "which city": it names the last city its conversation mentions, or says it does not know
- anything else: "Hello."

`drop_after` closes a connection, without a close frame, after that many answers on it: a provider that goes.
"""
from __future__ import annotations

import asyncio
import base64
import json
import math
import re
import secrets

import websockets

CITIES = ("Paris", "Lisbon", "Tokyo")
SAMPLE_RATE = 24000


def _id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def _tone(seconds: float) -> bytes:
    n = int(SAMPLE_RATE * seconds)
    return b"".join(int(3000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE)).to_bytes(2, "little", signed=True)
                    for i in range(n))


class FakeRealtime:
    def __init__(self, utterances: list[str], drop_after: int | None = None):
        self.utterances = list(utterances)
        self.drop_after = drop_after
        self.connections = 0
        self.dropped = 0
        self.sessions: list[dict] = []          # per connection: what it was told and what it holds

    async def start(self) -> str:
        self._server = await websockets.serve(self._serve, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        return f"ws://127.0.0.1:{port}/v1/realtime"

    async def stop(self) -> None:
        self._server.close()
        await self._server.wait_closed()

    async def _serve(self, ws) -> None:
        self.connections += 1
        s = {"items": [], "tools": [], "instructions": "", "answers": 0, "loud": 0, "quiet": 0, "speaking": False}
        self.sessions.append(s)

        async def send(ev: dict) -> None:
            await ws.send(json.dumps({"event_id": _id("event"), **ev}))

        await send({"type": "session.created", "session": {"type": "realtime", "id": _id("sess")}})
        try:
            async for raw in ws:
                m = json.loads(raw)
                t = m.get("type")
                if t == "session.update":
                    sess = m.get("session") or {}
                    s["instructions"] = sess.get("instructions") or s["instructions"]
                    if sess.get("tools") is not None:
                        s["tools"] = [x.get("name") for x in sess["tools"]]
                    await send({"type": "session.updated", "session": {"type": "realtime"}})
                elif t == "conversation.item.create":
                    item = {**m["item"], "id": m["item"].get("id") or _id("item")}
                    s["items"].append(item)
                    await send({"type": "conversation.item.added", "item": item})
                    await send({"type": "conversation.item.done", "item": item})
                elif t == "input_audio_buffer.append":
                    await self._hear(s, send, base64.b64decode(m.get("audio") or ""))
                elif t == "response.create":
                    await self._respond(s, send)
                if self.drop_after is not None and s["answers"] >= self.drop_after and self.dropped < 1:
                    self.dropped += 1
                    await asyncio.sleep(0.3)            # what was sent reaches the harness first
                    ws.transport.abort()
                    return
        except websockets.ConnectionClosed:
            return

    async def _hear(self, s: dict, send, pcm: bytes) -> None:
        if not pcm:
            return
        peak = max(abs(int.from_bytes(pcm[i:i + 2], "little", signed=True)) for i in range(0, len(pcm) - 1, 2))
        ms = len(pcm) / 2 / SAMPLE_RATE * 1000
        if peak > 1000:
            s["loud"] += ms
            s["quiet"] = 0
            if not s["speaking"] and s["loud"] >= 200:
                s["speaking"] = True
                s["speech_item"] = _id("item")
                await send({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0, "item_id": s["speech_item"]})
        else:
            s["quiet"] += ms
            if s["speaking"] and s["quiet"] >= 500:
                s["speaking"], s["loud"] = False, 0
                words = self.utterances.pop(0) if self.utterances else ""
                iid = s["speech_item"]
                await send({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 0, "item_id": iid})
                await send({"type": "input_audio_buffer.committed", "item_id": iid})
                item = {"id": iid, "type": "message", "role": "user", "content": [{"type": "input_audio", "transcript": words}]}
                s["items"].append(item)
                await send({"type": "conversation.item.added", "item": item})
                await send({"type": "conversation.item.input_audio_transcription.completed", "item_id": iid,
                            "content_index": 0, "transcript": words})
                await self._respond(s, send)            # server turn detection answers on its own

    def _text(self, item: dict) -> str:
        return " ".join(str(p.get("text") or p.get("transcript") or "") for p in item.get("content") or [])

    async def _respond(self, s: dict, send) -> None:
        last = s["items"][-1] if s["items"] else {}
        if last.get("type") == "function_call_output":
            out = str(last.get("output") or "")
            city = next((c for c in CITIES if c in out), "the city")
            celsius = re.search(r'celsius\\?"?:\s*(\d+)', out)
            await self._say(s, send, f"{city} is clear and {celsius.group(1) if celsius else 'mild'} degrees.")
            return
        heard = self._text(last) if last.get("role") == "user" else ""
        if "weather" in heard.lower() and "weather" in s["tools"]:
            city = next((c for c in CITIES if c in heard), "Paris")
            await self._call(s, send, "weather", {"city": city})
            return
        if "which city" in heard.lower():
            said = [c for it in s["items"][:-1] for c in CITIES if c in self._text(it) or c in str(it.get("arguments") or "")]
            await self._say(s, send, f"You asked about {said[-1]}." if said else "I do not know which city.")
            return
        await self._say(s, send, "Hello.")

    async def _say(self, s: dict, send, text: str) -> None:
        rid, iid = _id("resp"), _id("item")
        await send({"type": "response.created", "response": {"id": rid, "status": "in_progress", "output": []}})
        item = {"id": iid, "type": "message", "role": "assistant", "status": "in_progress", "content": []}
        await send({"type": "response.output_item.added", "response_id": rid, "output_index": 0, "item": item})
        await send({"type": "conversation.item.added", "item": item})
        pcm = _tone(0.4)
        where = {"response_id": rid, "item_id": iid, "output_index": 0, "content_index": 0}
        for i in range(0, len(pcm), 4800):
            await send({"type": "response.output_audio.delta", **where, "delta": base64.b64encode(pcm[i:i + 4800]).decode()})
        await send({"type": "response.output_audio_transcript.delta", **where, "delta": text})
        await send({"type": "response.output_audio.done", **where})
        await send({"type": "response.output_audio_transcript.done", **where, "transcript": text})
        done = {**item, "status": "completed", "content": [{"type": "output_audio", "transcript": text}]}
        s["items"].append(done)
        await send({"type": "response.output_item.done", "response_id": rid, "output_index": 0, "item": done})
        await send({"type": "response.done", "response": {"id": rid, "status": "completed", "output": [done], "usage": {
            "total_tokens": 130, "input_tokens": 100, "output_tokens": 30,
            "input_token_details": {"text_tokens": 80, "audio_tokens": 20, "cached_tokens": 0},
            "output_token_details": {"text_tokens": 10, "audio_tokens": 20}}}})
        s["answers"] += 1

    async def _call(self, s: dict, send, name: str, args: dict) -> None:
        rid, iid, cid = _id("resp"), _id("item"), _id("call")
        await send({"type": "response.created", "response": {"id": rid, "status": "in_progress", "output": []}})
        item = {"id": iid, "type": "function_call", "status": "in_progress", "call_id": cid, "name": name, "arguments": ""}
        await send({"type": "response.output_item.added", "response_id": rid, "output_index": 0, "item": item})
        await send({"type": "conversation.item.added", "item": item})
        arguments = json.dumps(args)
        await send({"type": "response.function_call_arguments.done", "response_id": rid, "item_id": iid,
                    "output_index": 0, "call_id": cid, "arguments": arguments})
        done = {**item, "status": "completed", "arguments": arguments}
        s["items"].append(done)
        await send({"type": "response.output_item.done", "response_id": rid, "output_index": 0, "item": done})
        await send({"type": "response.done", "response": {"id": rid, "status": "completed", "output": [done], "usage": {
            "total_tokens": 110, "input_tokens": 100, "output_tokens": 10}}})

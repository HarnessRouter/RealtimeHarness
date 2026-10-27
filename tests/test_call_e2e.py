"""Whole calls with no provider key: the server, a WebSocket device, the fixture MCP server, and a scripted
provider on the OpenAI realtime wire (tests/fake_provider.py). What a call records (R-04, R-05, R-10), that a
device cannot change the configuration (R-07), a gapless stream (R-09), a device that drops and comes back
(R-11), and a provider that drops mid-call (design §9 item 2)."""
import asyncio
import contextlib
import json
import math

import httpx
import uvicorn
import websockets

from conftest import free_port
from fake_provider import FakeRealtime
from realtime_harness.config import HarnessConfig, McpServer
from realtime_harness.credentials import Connection, Credentials
from realtime_harness.uhp import Server, create_app

KEY = "e2e-key"
H = {"Authorization": f"Bearer {KEY}"}
SPEECH = b"".join(int(8000 * math.sin(2 * math.pi * 220 * i / 24000)).to_bytes(2, "little", signed=True) for i in range(24000))


class FakeCredentials(Credentials):
    """One credential per provider connection, as a server's mint gives; each is released once."""

    def __init__(self, url: str):
        self.url, self.minted, self.released = url, 0, []

    async def connect(self, model: str) -> Connection:
        self.minted += 1
        return Connection(url=f"{self.url}?model={model}", headers={"Authorization": f"Bearer secret-{self.minted}"},
                          model=model, ref={"n": self.minted})

    async def release(self, conn: Connection) -> None:
        self.released.append(conn.ref["n"])


@contextlib.asynccontextmanager
async def serving(mcp_url: str, fake: FakeRealtime):
    creds = FakeCredentials(await fake.start())
    server = Server([HarnessConfig(id="desk", name="Desk", instructions="Answer in English.",
                                   mcp_servers=[McpServer("fx", mcp_url)])], creds, {KEY})
    port = free_port()
    uv = uvicorn.Server(uvicorn.Config(create_app(server), host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(uv.serve())
    while not uv.started:
        await asyncio.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}", server, creds
    finally:
        uv.should_exit = True
        await task
        await fake.stop()


class Phone:
    """A device: a call, its event stream, and the device's socket."""

    def __init__(self, base: str):
        self.base, self.events, self.ws = base, [], None

    async def call(self, body: dict | None = None) -> "Phone":
        async with httpx.AsyncClient() as c:
            s = (await c.post(f"{self.base}/v1/realtime/sessions", headers=H, json=body or {})).json()
        self.id, self.url = s["id"], s["connect"]["url"]
        self._follow = asyncio.create_task(self._stream())
        return await self.connect(s["connect"]["token"])

    async def connect(self, token: str) -> "Phone":
        self.ws = await websockets.connect(f"{self.url}?token={token}")
        self.token = json.loads(await self.ws.recv())["token"]          # the first event: the way back in
        self._read = asyncio.create_task(self._drain(self.ws))         # a device reads its socket: the model's audio
        await self.until("realtime.session.live", "realtime.session.resumed")
        return self

    async def _drain(self, ws) -> None:
        try:
            async for _ in ws:
                pass
        except websockets.ConnectionClosed:
            pass

    async def _stream(self) -> None:
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("GET", f"{self.base}/v1/sessions/{self.id}/events", headers=H) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        self.events.append(json.loads(line[6:]))

    async def until(self, *types: str, n: int = 1, timeout: float = 10) -> None:
        async def wait():
            while sum(1 for e in self.events if e["type"] in types) < n:
                await asyncio.sleep(0.02)
        await asyncio.wait_for(wait(), timeout)

    async def say(self) -> None:
        """A second of speech and a pause: the fake provider hears its next utterance."""
        for i in range(0, len(SPEECH), 4800):
            await self.ws.send(SPEECH[i:i + 4800])
            await asyncio.sleep(0.02)
        for _ in range(8):
            await self.ws.send(b"\x00" * 4800)
            await asyncio.sleep(0.02)

    async def answers(self, n: int) -> None:
        async def wait():
            while sum(1 for e in self.events if e["type"] == "response.completed"
                      and e["response"]["output"][-1]["type"] == "message") < n:
                await asyncio.sleep(0.02)
        await asyncio.wait_for(wait(), 15)

    async def hang_up(self) -> dict:
        await self.ws.close(code=1000)
        await asyncio.wait_for(self._follow, 10)
        async with httpx.AsyncClient() as c:
            return (await c.get(f"{self.base}/v1/sessions/{self.id}", headers=H)).json()

    def said(self, turn: dict) -> str:
        return [i for i in turn["output"] if i["type"] == "message"][-1]["content"][0]["text"]


def test_a_call_records_the_question_the_tool_and_the_answer_and_the_device_cannot_widen_it(mcp_url):
    async def go():
        fake = FakeRealtime(["What is the weather in Paris?"])
        async with serving(mcp_url, fake) as (base, server, creds):
            p = await Phone(base).call()
            await p.ws.send(json.dumps({"type": "session.update", "session": {"instructions": "Answer in French.", "tools": []}}))
            await p.say()
            await p.answers(1)
            rec = await p.hang_up()
        t = rec["responses"][0]
        assert t["input"][0]["content"][0]["text"] == "What is the weather in Paris?"                   # R-04
        types = [i["type"] for i in t["output"]]
        assert types == ["function_call", "function_call_output", "message"]                             # R-05
        assert t["output"][0]["call_id"] == t["output"][1]["call_id"] and "18" in p.said(t)
        assert t["usage"]["output_tokens"] > 0 and t["usage"]["output_audio_tokens"] > 0                 # R-10
        assert fake.sessions[0]["instructions"] == "Answer in English." and fake.sessions[0]["tools"]    # R-07
        seqs = [e["sequence_number"] for e in p.events]
        assert seqs == list(range(len(seqs))) and p.events[-1]["type"] == "realtime.session.ended"       # R-09
        assert p.events[-1]["reason"] == "hangup" and p.events[-1]["ignored_device_events"] == 1
        assert sorted(creds.released) == list(range(1, creds.minted + 1))                                # every credential released
    asyncio.run(go())


def test_a_device_that_drops_comes_back_to_the_same_provider_conversation(mcp_url):
    async def go():
        fake = FakeRealtime(["What is the weather in Lisbon?", "Which city did I just ask about?"])
        async with serving(mcp_url, fake) as (base, server, creds):
            p = await Phone(base).call()
            await p.say()
            await p.answers(1)
            p.ws.transport.abort()                                       # the network goes
            await p.until("realtime.session.paused")
            await p.connect(p.token)
            await p.say()
            await p.answers(2)
            rec = await p.hang_up()
        assert fake.connections == 1 and creds.minted == 1               # one provider conversation throughout
        assert p.said(rec["responses"][-1]) == "You asked about Lisbon."
        kinds = [e["type"] for e in p.events if e["type"].startswith("realtime.session.")]
        assert kinds == ["realtime.session.live", "realtime.session.paused", "realtime.session.resumed", "realtime.session.ended"]
    asyncio.run(go())


def test_a_provider_that_drops_is_replaced_and_the_new_one_knows_what_was_said(mcp_url):
    async def go():
        fake = FakeRealtime(["What is the weather in Tokyo?", "Which city did I just ask about?"], drop_after=1)
        async with serving(mcp_url, fake) as (base, server, creds):
            p = await Phone(base).call()
            await p.say()
            await p.answers(1)
            await p.until("realtime.session.provider_reconnected")
            await p.say()
            await p.answers(2)
            rec = await p.hang_up()
        assert fake.connections == 2 and creds.minted == 2
        assert sorted(creds.released) == [1, 2] and creds.released[0] == 1     # the dropped one released first
        seam = next(e for e in p.events if e["type"] == "realtime.session.provider_reconnected")
        assert seam["seeded_messages"] >= 2
        assert p.said(rec["responses"][-1]) == "You asked about Tokyo."        # from the seeded transcript
        assert p.events[-1]["reason"] == "hangup"
    asyncio.run(go())

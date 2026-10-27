"""A device on the WebSocket transport, for trying a call without a person: it plays a recorded question,
listens for the answer, and prints the call's record.

    RTH_URL=http://127.0.0.1:8790 RTH_KEY=<a key the server accepts> python examples/device_ws.py [fixture.pcm]

The client's part (create the call, follow its events, hang up) uses the server's API key; the device's
part (audio) uses only the one-time connect token the client handed it.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import httpx
import websockets

URL = os.environ.get("RTH_URL", "http://127.0.0.1:8790").rstrip("/")
KEY = os.environ["RTH_KEY"]
FIXTURE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "weather_question.pcm")
FRAME = 4800                 # 100 ms of PCM16 mono at 24 kHz


async def follow(sid: str, events: list[dict], t0: float) -> None:
    async with httpx.AsyncClient(timeout=None) as c:
        async with c.stream("GET", f"{URL}/v1/sessions/{sid}/events", headers={"Authorization": f"Bearer {KEY}"}) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    ev = json.loads(line[6:])
                    ev["_t"] = round(time.time() - t0, 2)
                    events.append(ev)
                    if ev["type"] == "realtime.session.ended":
                        return


async def main() -> None:
    t0 = time.time()
    h = {"Authorization": f"Bearer {KEY}"}
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{URL}/v1/realtime/sessions", headers=h, json={"transport": "websocket"})
        r.raise_for_status()
        call = r.json()
    sid = call["id"]
    print("session", sid, "status", call["status"], "connect", call["connect"]["url"], "expires", call["connect"]["expires_at"])
    events: list[dict] = []
    follower = asyncio.create_task(follow(sid, events, t0))
    audio_in = bytearray()
    device_events: list[str] = []
    async with websockets.connect(f"{call['connect']['url']}?token={call['connect']['token']}", max_size=None) as ws:
        async def listen():
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        audio_in.extend(msg)
                    else:
                        device_events.append(json.loads(msg).get("type"))
            except websockets.ConnectionClosed:
                pass
        listener = asyncio.create_task(listen())
        for _ in range(100):                          # wait for the provider connection
            if any(e["type"] == "realtime.session.live" for e in events):
                break
            await asyncio.sleep(0.1)
        pcm = open(FIXTURE, "rb").read()
        t_speak = time.time()
        for i in range(0, len(pcm), FRAME):
            await ws.send(pcm[i:i + FRAME])
            await asyncio.sleep(0.1)
        for _ in range(15):                           # 1.5 s of silence ends the caller's turn
            await ws.send(b"\x00" * FRAME)
            await asyncio.sleep(0.1)
        t_spoken = time.time()
        for _ in range(300):                          # the answer, up to 30 s: a turn that used its tool and answered
            done = [e for e in events if e["type"] == "response.completed" and any(
                i.get("type") == "function_call_output" for i in e["response"]["output"]) and
                e["response"]["output"][-1].get("type") == "message"]
            if done:
                break
            await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "session.update", "session": {"instructions": "Say BANANA."}}))   # not a device event
        await asyncio.sleep(1)
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{URL}/v1/sessions/{sid}/cancel", headers=h)
            print("hang up", r.status_code, r.json().get("status"), r.json().get("ended_reason"))
        listener.cancel()
    await asyncio.wait_for(follower, 30)
    async with httpx.AsyncClient(timeout=30) as c:
        s = (await c.get(f"{URL}/v1/sessions/{sid}", headers=h)).json()
    first_audio = next((e["_t"] for e in events if e["type"] == "realtime.output_transcript.delta"), None)
    print(f"spoke {t_spoken - t_speak:.1f}s; first model transcript at {first_audio}s; model audio received {len(audio_in) / 48000:.1f}s")
    print("event types:", [e["type"] for e in events])
    seqs = [e["sequence_number"] for e in events]
    print("sequence gapless from 0:", seqs == list(range(len(seqs))), "| terminal:", events[-1]["type"], events[-1].get("reason"),
          "| ignored device events:", events[-1].get("ignored_device_events"))
    print("device saw:", sorted(set(device_events)))
    for t in s.get("responses") or []:
        said = " ".join(p.get("text", "") for i in t["input"] for p in i.get("content", []))
        out = [(i["type"], i.get("name") or (i.get("content") or [{}])[0].get("text", "")[:80] if i["type"] == "message" else i.get("name") or i.get("output", "")[:60]) for i in t["output"]]
        print(f"turn {t['id'][:12]} {t['status']}: input={said!r} output={out} usage={t['usage']}")


asyncio.run(main())

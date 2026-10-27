"""Live checks of a call (needs a running server with a provider; each check is one short call):

    resume     a device drops (no close frame), comes back with its resume token within the grace, and asks
               about what it said before the drop (R-11)
    continue   a call that continues a text turn starts from what was said in it, and does not speak first
    slow-tool  the caller speaks while a slow tool runs: the tool is not cancelled, and its result is
               recorded where the model takes it (design §5.4)

    RTH_URL=http://127.0.0.1:8790 RTH_KEY=<key> python tests/live/call_checks.py [resume|continue|slow-tool ...]
"""
import asyncio
import json
import os
import sys
import time

import httpx
import websockets

URL = os.environ.get("RTH_URL", "http://127.0.0.1:8790").rstrip("/")
H = {"Authorization": f"Bearer {os.environ['RTH_KEY']}"}
FIX = os.path.join(os.path.dirname(__file__), "..", "fixtures")
FRAME = 4800                                       # 100 ms of PCM16 mono at 24 kHz


class Call:
    def __init__(self, body: dict | None = None):
        self.body = body or {}
        self.events: list[dict] = []

    async def start(self) -> "Call":
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(f"{URL}/v1/realtime/sessions", headers=H, json=self.body)
        r.raise_for_status()
        s = r.json()
        self.id, self.url, self.token = s["id"], s["connect"]["url"], s["connect"]["token"]
        self._follow = asyncio.create_task(self._events())
        return self

    async def _events(self) -> None:
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("GET", f"{URL}/v1/sessions/{self.id}/events", headers=H) as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        self.events.append(json.loads(line[6:]))
                        if self.events[-1]["type"] == "realtime.session.ended":
                            return

    def count(self, pred) -> int:
        return sum(1 for e in self.events if pred(e))

    async def until(self, pred, timeout: float = 45) -> bool:
        t = time.time()
        while time.time() - t < timeout:
            if any(pred(e) for e in self.events):
                return True
            await asyncio.sleep(0.1)
        return False

    async def answers(self, n: int, timeout: float = 45) -> bool:
        """n turns completed with the model's spoken answer as their last item."""
        def done(e):
            return e["type"] == "response.completed" and (e["response"]["output"] or [{}])[-1].get("type") == "message"
        t = time.time()
        while time.time() - t < timeout:
            if self.count(done) >= n:
                return True
            await asyncio.sleep(0.1)
        return False

    async def record(self) -> dict:
        await asyncio.wait_for(self._follow, 30)
        async with httpx.AsyncClient(timeout=30) as c:
            return (await c.get(f"{URL}/v1/sessions/{self.id}", headers=H)).json()


class Device:
    def __init__(self):
        self.messages: list[dict] = []
        self.audio = 0

    async def connect(self, url: str, token: str) -> "Device":
        self.ws = await websockets.connect(f"{url}?token={token}", max_size=None)
        self._listen = asyncio.create_task(self._read())
        return self

    async def _read(self) -> None:
        try:
            async for m in self.ws:
                if isinstance(m, str):
                    self.messages.append(json.loads(m))
                else:
                    self.audio += len(m)
        except websockets.ConnectionClosed:
            pass

    def resume_token(self) -> str | None:
        return next((m["token"] for m in self.messages if m.get("type") == "realtime.session.resume_token"), None)

    async def speak(self, name: str) -> None:
        pcm = open(os.path.join(FIX, name), "rb").read()
        for i in range(0, len(pcm), FRAME):
            await self.ws.send(pcm[i:i + FRAME])
            await asyncio.sleep(0.1)
        for _ in range(12):                         # a pause, so the provider hears the end of the question
            await self.ws.send(b"\x00" * FRAME)
            await asyncio.sleep(0.1)

    async def keep_quiet(self, seconds: float) -> None:
        for _ in range(int(seconds * 10)):
            await self.ws.send(b"\x00" * FRAME)
            await asyncio.sleep(0.1)

    def drop(self) -> None:
        self.ws.transport.abort()                   # the network goes: no close frame
        self._listen.cancel()

    async def hang_up(self) -> None:
        await self.ws.close(code=1000)


def said(turn: dict) -> tuple[str, str]:
    heard = " ".join(p.get("text", "") for i in turn["input"] for p in i.get("content", []))
    ans = [(i.get("content") or [{}])[0].get("text", "") for i in turn["output"] if i["type"] == "message"]
    return heard, (ans[-1] if ans else "")


def report(name: str, call: Call, rec: dict, checks: dict) -> bool:
    seqs = [e["sequence_number"] for e in call.events]
    checks = {"gapless": seqs == list(range(len(seqs))), **checks}
    print(f"== {name}: {'PASS' if all(checks.values()) else 'FAIL'}  {checks}")
    print("   events:", [e["type"].replace("realtime.session.", "") for e in call.events if e["type"].startswith("realtime.session.")],
          "| ended:", call.events[-1].get("reason") if call.events else None)
    for t in rec["responses"]:
        heard, answer = said(t)
        print(f"   turn {t['status']}: heard={heard!r} items={[i['type'] for i in t['output']]} answer={answer!r}")
    return all(checks.values())


async def resume() -> bool:
    call = await Call().start()
    a = await Device().connect(call.url, call.token)
    await call.until(lambda e: e["type"] == "realtime.session.live")
    await a.speak("weather_question.pcm")
    first = await call.answers(1)
    token = a.resume_token()
    a.drop()
    paused = await call.until(lambda e: e["type"] == "realtime.session.paused", 15)
    await asyncio.sleep(3)
    b = await Device().connect(call.url, token)
    resumed = await call.until(lambda e: e["type"] == "realtime.session.resumed", 10)
    await b.keep_quiet(1)
    await b.speak("which_city.pcm")
    second = await call.answers(2)
    await b.keep_quiet(1)
    await b.hang_up()
    rec = await call.record()
    last = said(rec["responses"][-1])[1] if rec["responses"] else ""
    return report("resume", call, rec, {
        "first answer": first, "paused": paused, "resumed": resumed, "second answer": second,
        "new token on return": bool(b.resume_token()) and b.resume_token() != token,
        "one live event": call.count(lambda e: e["type"] == "realtime.session.live") == 1,
        "remembers Paris": "paris" in last.lower(),
        "ended hangup": bool(call.events) and call.events[-1].get("reason") == "hangup"})


async def continue_() -> bool:
    async with httpx.AsyncClient(timeout=120) as c:
        r = (await c.post(f"{URL}/v1/responses", headers=H, json={
            "input": "My name is Ada. What is the weather in Lisbon? Use the weather tool."})).json()
    call = await Call({"previous_response_id": r["id"]}).start()
    a = await Device().connect(call.url, call.token)
    await call.until(lambda e: e["type"] == "realtime.session.live")
    await a.keep_quiet(3)
    spoke_first = call.count(lambda e: e["type"] == "response.created") > 0
    await a.speak("which_city.pcm")
    answered = await call.answers(1)
    await a.keep_quiet(1)
    await a.hang_up()
    rec = await call.record()
    last = said(rec["responses"][-1])[1] if rec["responses"] else ""
    return report("continue", call, rec, {
        "text turn": r.get("status") == "completed", "same session": call.id == r["metadata"]["session_id"],
        "did not speak first": not spoke_first, "answered": answered, "remembers Lisbon": "lisbon" in last.lower()})


async def slow_tool() -> bool:
    call = await Call().start()
    a = await Device().connect(call.url, call.token)
    await call.until(lambda e: e["type"] == "realtime.session.live")
    await a.speak("order_question.pcm")

    def order_call(e):
        it = e.get("item") or {}
        return e["type"] == "response.output_item.added" and it.get("type") == "function_call" and it.get("name") == "order_status"
    started = await call.until(order_call, 30)
    await a.speak("weather_question.pcm")                 # spoken while the order lookup runs
    call_id = next((e["item"]["call_id"] for e in call.events if order_call(e)), None)

    def order_result(e):
        it = e.get("item") or {}
        return e["type"] == "response.output_item.done" and it.get("type") == "function_call_output" and it.get("call_id") == call_id
    resulted = await call.until(order_result, 30)
    await asyncio.sleep(1)
    await call.answers(call.count(lambda e: e["type"] == "response.completed") + 1, 15)
    await a.keep_quiet(1)
    await a.hang_up()
    rec = await call.record()
    out = next((e["item"]["output"] for e in call.events if order_result(e)), "")
    weather = any(i.get("type") == "function_call" and i.get("name") == "weather" for t in rec["responses"] for i in t["output"])
    return report("slow-tool", call, rec, {
        "order call made": started, "order result recorded": resulted, "not cancelled": "shipped" in out and "cancel" not in out,
        "weather answered meanwhile": weather})


async def main(names: list[str]) -> int:
    checks = {"resume": resume, "continue": continue_, "slow-tool": slow_tool}
    ok = True
    for n in names or list(checks):
        ok = await checks[n]() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1:])))

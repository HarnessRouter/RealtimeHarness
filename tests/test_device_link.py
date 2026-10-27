"""The device's link (design §9 item 2, check R-11): a drop pauses the call and the device comes back to it; a
close with code 1000 hangs up; a device that does not come back within the grace ends the call; a device that
comes back before the server noticed the drop replaces its old connection."""
import asyncio

from realtime_harness.device import DeviceLink


class FakeSocket:
    def __init__(self):
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent: list = []
        self.closed_with: int | None = None

    async def receive(self) -> dict:
        return await self.incoming.get()

    async def send_bytes(self, data: bytes) -> None:
        if self.closed_with is not None:
            raise RuntimeError("closed")
        self.sent.append(data)

    async def send_text(self, data: str) -> None:
        if self.closed_with is not None:
            raise RuntimeError("closed")
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        self.closed_with = code

    def says(self, data: bytes) -> None:
        self.incoming.put_nowait({"type": "websocket.receive", "bytes": data})

    def drops(self, code: int = 1006) -> None:
        self.incoming.put_nowait({"type": "websocket.disconnect", "code": code})


def _link(ws, grace=5.0):
    seen = []
    link = DeviceLink(ws, grace, on_lost=lambda: seen.append("lost"), on_back=lambda: seen.append("back"))
    return link, seen


def test_a_drop_pauses_and_the_device_comes_back_to_the_same_call():
    async def go():
        a, b = FakeSocket(), FakeSocket()
        link, seen = _link(a)
        a.says(b"one")
        assert (await link.receive())["bytes"] == b"one"
        await link.send_bytes(b"to a")
        a.drops()
        reading = asyncio.create_task(link.receive())          # Pipecat's read waits through the pause
        await asyncio.sleep(0.05)
        assert link.paused and seen == ["lost"] and not reading.done()
        await link.send_bytes(b"while away")                   # recorded upstream, not sent anywhere
        done_b = link.attach(b)
        b.says(b"two")
        assert (await asyncio.wait_for(reading, 1))["bytes"] == b"two"
        await link.send_bytes(b"to b")
        assert seen == ["lost", "back"] and a.sent == [b"to a"] and b.sent == [b"to b"]
        assert not done_b.is_set() and link.end_reason is None
    asyncio.run(go())


def test_closing_with_1000_hangs_up():
    async def go():
        a = FakeSocket()
        link, seen = _link(a)
        a.drops(1000)
        msg = await asyncio.wait_for(link.receive(), 1)
        assert msg["type"] == "websocket.disconnect" and link.end_reason == "hangup" and seen == []
    asyncio.run(go())


def test_a_device_that_does_not_come_back_ends_the_call():
    async def go():
        a = FakeSocket()
        link, seen = _link(a, grace=0.3)
        a.drops()
        msg = await asyncio.wait_for(link.receive(), 2)
        assert msg["type"] == "websocket.disconnect" and link.end_reason == "device_lost" and seen == ["lost"]
        late = FakeSocket()
        assert link.attach(late).is_set()                      # too late: the call is over
    asyncio.run(go())


def test_a_device_back_before_the_drop_was_noticed_replaces_its_old_connection():
    async def go():
        a, b = FakeSocket(), FakeSocket()                      # a is half-open: it never says anything again
        link, seen = _link(a)
        reading = asyncio.create_task(link.receive())
        await asyncio.sleep(0.05)
        done_b = link.attach(b)
        b.says(b"from b")
        assert (await asyncio.wait_for(reading, 1))["bytes"] == b"from b"
        await asyncio.sleep(0.05)
        assert a.closed_with == 4000 and seen == []            # replaced, never paused
        assert not done_b.is_set()                              # b carries the call now
        a.drops()                                               # a's late goodbye changes nothing
        b.says(b"still b")
        assert (await asyncio.wait_for(link.receive(), 1))["bytes"] == b"still b"
        assert seen == []
    asyncio.run(go())


def test_ending_the_call_releases_the_device_connection():
    async def go():
        a = FakeSocket()
        link, _ = _link(a)
        b = FakeSocket()
        done_b = link.attach(b)
        await link.close()
        assert done_b.is_set() and b.closed_with == 1000
        assert (await asyncio.wait_for(link.receive(), 1))["type"] == "websocket.disconnect"
    asyncio.run(go())

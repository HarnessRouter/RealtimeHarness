"""The tool executor against a real MCP server (tests/fixture_mcp.py over Streamable HTTP): what is offered,
what is never contacted, and that every call is answered (design §5, UHP Harnesses §4.1 and §4.3)."""
import asyncio
import pathlib
import socket
import subprocess
import sys
import time

import pytest

from realtime_harness.config import HarnessConfig, McpServer
from realtime_harness.tools import ToolExecutor

FIXTURE = pathlib.Path(__file__).with_name("fixture_mcp.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def mcp_url():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, str(FIXTURE), str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 15
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}/mcp"
    proc.terminate()
    proc.wait(timeout=5)


def _run(coro):
    return asyncio.run(coro)


def test_tools_are_offered_from_enabled_servers_and_a_disabled_tool_is_not(mcp_url):
    async def go():
        ex = ToolExecutor(HarnessConfig(id="h", name="h", mcp_servers=[McpServer("fx", mcp_url, disabled_tools=["delete_account"])]))
        await ex.open()
        try:
            names = sorted(f["name"] for f in ex.functions())
            assert names == ["broken", "order_status", "weather"]
            weather = next(f for f in ex.functions() if f["name"] == "weather")
            assert weather["type"] == "function" and weather["parameters"]["properties"]["city"]["type"] == "string"
            assert "$schema" not in weather["parameters"]
            # a hard block: refused here, never sent to the server
            assert (await ex.call("delete_account", {"user": "x"})) == {"error": "refused", "detail": "the tool 'delete_account' is disabled"}
        finally:
            await ex.close()
    _run(go())


def test_a_disabled_server_is_never_contacted_and_an_unreachable_one_fails_nothing(mcp_url):
    async def go():
        with socket.socket() as s:                       # a port nobody listens on
            s.bind(("127.0.0.1", 0))
            dead = s.getsockname()[1]
        ex = ToolExecutor(HarnessConfig(id="h", name="h", mcp_servers=[
            McpServer("off", f"http://127.0.0.1:{dead}/mcp", enabled=False),
            McpServer("gone", f"http://127.0.0.1:{dead}/mcp"),
            McpServer("fx", mcp_url)]))
        await ex.open()
        try:
            assert ex.servers["off"].error == "disabled" and ex.servers["off"].task is None     # no connection made
            assert ex.servers["gone"].error.startswith("unreachable")
            assert "weather" in ex.tools and ex.left_out["server:gone"].startswith("unreachable")
        finally:
            await ex.close()
    _run(go())


def test_every_call_is_answered(mcp_url):
    async def go():
        ex = ToolExecutor(HarnessConfig(id="h", name="h", mcp_servers=[McpServer("fx", mcp_url)], tool_timeout_seconds=1))
        await ex.open()
        try:
            done = await ex.call("weather", '{"city": "Paris"}')
            assert done["status"] == "done" and '"celsius": 18' in done["output"]
            assert await ex.call("order_status", {"order": "1", "delay_s": 3}) == {
                "error": "timeout", "detail": "the tool did not answer within 1 second"}
            failed = await ex.call("broken", {})
            assert failed["error"] == "tool_error"
            assert (await ex.call("nope", {}))["error"] == "refused"
            assert (await ex.call("weather", "not json"))["detail"] == "the arguments are not JSON"
            assert (await ex.call("weather", "[1]"))["detail"] == "the arguments are not a JSON object"
        finally:
            await ex.close()
    _run(go())

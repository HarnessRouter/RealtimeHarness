"""What the tests share: a free port, and the fixture MCP server (tests/fixture_mcp.py) over Streamable HTTP."""
import pathlib
import socket
import subprocess
import sys
import time

import pytest

FIXTURE = pathlib.Path(__file__).with_name("fixture_mcp.py")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def mcp_url():
    port = free_port()
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

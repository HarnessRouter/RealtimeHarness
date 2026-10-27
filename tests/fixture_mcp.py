"""The test MCP server: a deterministic `weather` tool (the design's conformance fixture, §11), a slow
`order_status` tool, a tool that fails, and a `delete_account` tool a harness disables.

    python tests/fixture_mcp.py <port>      serves Streamable HTTP at http://127.0.0.1:<port>/mcp
"""
import asyncio
import sys

from mcp.server.mcpserver import MCPServer

server = MCPServer("realtime-harness-fixture")
CALLS: list[str] = []


@server.tool()
def weather(city: str) -> dict:
    """The current weather in a city."""
    CALLS.append(f"weather:{city}")
    return {"city": city, "sky": "clear", "celsius": 18}


@server.tool()
async def order_status(order: str, delay_s: float = 0.0) -> str:
    """An order's status. Takes delay_s seconds to answer."""
    CALLS.append(f"order_status:{order}")
    await asyncio.sleep(delay_s)
    return f"Order {order} has shipped."


@server.tool()
def broken() -> str:
    """Always fails."""
    raise RuntimeError("the backend is down")


@server.tool()
def delete_account(user: str) -> str:
    """Deletes a user's account."""
    CALLS.append(f"delete_account:{user}")
    return "deleted"


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1])
    uvicorn.run(server.streamable_http_app(), host="127.0.0.1", port=port, log_level="warning")

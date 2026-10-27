"""The tool executor: a call's MCP servers, their tools offered to the model, and every call run here.

Every realtime model hands a function call back to its caller (design §1.2). The harness is that
caller: it runs the call on the MCP server that owns the tool and returns the result. The person's
device never runs a tool and never sees a tool's credential.

The rules (UHP Harnesses §4.1 and §4.3, design §5):

- A disabled server is never contacted. A server that cannot be reached does not fail the call: its
  tools are left out and the reason is kept.
- A disabled tool is not offered, and a call to it is refused here (a hard block, not a hint).
- Every call is answered: a result, a refusal, a timeout or an error, each as the call's output. A call
  left unanswered stalls some providers outright (design §5.3).
- A call is bounded by the harness's tool_timeout_seconds.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
from dataclasses import dataclass, field

from .config import HarnessConfig, McpServer

# the realtime APIs' function name rule
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DESCRIPTION_MAX = 1024          # characters of a tool description handed to a speech model
OUTPUT_MAX = 16000              # characters of a tool result handed back for the model to speak from
CONNECT_TIMEOUT_S = 10


@dataclass
class Tool:
    name: str                   # the name the model sees
    server: str                 # the MCP server that owns it
    remote_name: str            # the name on that server
    description: str
    parameters: dict

    def function(self) -> dict:
        """The tool as a realtime session's function definition."""
        return {"type": "function", "name": self.name, "description": self.description, "parameters": self.parameters}


@dataclass
class ServerState:
    config: McpServer
    session: object | None = None
    error: str = ""             # why it is not in use, when it is not
    task: asyncio.Task | None = None
    stop: asyncio.Event = field(default_factory=asyncio.Event)


def output_text(result) -> str:
    """An MCP tool result as the text the model reads: structured content when the tool gave it, else
    its text parts. A part the model cannot hear (an image, audio) is named, not dropped in silence."""
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        text = json.dumps(structured, ensure_ascii=False)
    else:
        parts = []
        for c in getattr(result, "content", None) or []:
            kind = getattr(c, "type", "")
            if kind == "text":
                parts.append(getattr(c, "text", ""))
            else:
                parts.append(f"[{kind or 'content'} not shown in a spoken call]")
        text = "\n".join(p for p in parts if p)
    return text if len(text) <= OUTPUT_MAX else text[:OUTPUT_MAX] + " [cut]"


@dataclass
class ToolExecutor:
    harness: HarnessConfig
    servers: dict[str, ServerState] = field(default_factory=dict)
    tools: dict[str, Tool] = field(default_factory=dict)
    left_out: dict[str, str] = field(default_factory=dict)      # tool or server -> why it is not offered

    async def open(self) -> None:
        """Connect to the harness's enabled servers, at once, and list their tools. Each server's
        connection lives in a task of its own, which opens it and closes it (the MCP client's transports
        must be left in the task that entered them), so a server that fails takes nothing else with it.
        Calls may come from any task."""
        blocked = set(self.harness.disabled_tools)
        ready: dict[str, asyncio.Future] = {}
        for cfg in self.harness.mcp_servers:
            state = ServerState(cfg)
            self.servers[cfg.name] = state
            if not cfg.enabled:
                state.error = "disabled"                    # never contacted
                continue
            ready[cfg.name] = asyncio.get_running_loop().create_future()
            state.task = asyncio.create_task(self._hold(state, ready[cfg.name]))
        for cfg in self.harness.mcp_servers:
            if cfg.name not in ready:
                continue
            state = self.servers[cfg.name]
            try:
                session, listed = await ready[cfg.name]
            except Exception as e:  # noqa: BLE001  # an unreachable server does not fail the call
                state.error = f"unreachable: {type(e).__name__}"
                self.left_out[f"server:{cfg.name}"] = state.error
                continue
            state.session = session
            for t in listed.tools:
                remote = t.name
                if remote in blocked or remote in cfg.disabled_tools:
                    self.left_out[remote] = "disabled"
                    continue
                name = remote if remote not in self.tools else f"{cfg.name}__{remote}"
                if not _NAME.match(name):
                    self.left_out[remote] = "name the realtime APIs cannot take"
                    continue
                params = {k: v for k, v in (t.input_schema or {}).items() if k != "$schema"} or {"type": "object", "properties": {}}
                params.setdefault("type", "object")
                self.tools[name] = Tool(name=name, server=cfg.name, remote_name=remote,
                                        description=str(t.description or t.title or remote)[:DESCRIPTION_MAX],
                                        parameters=params)

    async def _hold(self, state: ServerState, ready: asyncio.Future) -> None:
        """One server's connection, from open to close, in this task."""
        try:
            async with contextlib.AsyncExitStack() as stack:
                async with asyncio.timeout(CONNECT_TIMEOUT_S):
                    session = await self._connect(stack, state.config)
                    listed = await session.list_tools()
                ready.set_result((session, listed))
                await state.stop.wait()
        except BaseException as e:  # noqa: BLE001  # reported through `ready`; a closing call ends here too
            if not ready.done():
                ready.set_exception(e if isinstance(e, Exception) else RuntimeError(type(e).__name__))

    @staticmethod
    async def _connect(stack: contextlib.AsyncExitStack, cfg: McpServer):
        from mcp import ClientSession
        if cfg.transport == "sse":
            from mcp.client.sse import sse_client
            streams = await stack.enter_async_context(sse_client(cfg.url, headers=cfg.headers or None))
        else:
            import httpx2
            from mcp.client.streamable_http import streamable_http_client
            client = await stack.enter_async_context(httpx2.AsyncClient(headers=cfg.headers or None, timeout=30))
            streams = await stack.enter_async_context(streamable_http_client(cfg.url, http_client=client))
        read, write = streams[0], streams[1]
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        return session

    def functions(self) -> list[dict]:
        return [t.function() for t in self.tools.values()]

    async def call(self, name: str, arguments: str | dict) -> dict:
        """Run one call and return its output object, which is also what the record keeps."""
        tool = self.tools.get(name)
        if tool is None:
            why = self.left_out.get(name)
            return {"error": "refused", "detail": f"the tool {name!r} is {why}" if why else f"no tool named {name!r} in this call"}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except ValueError:
                return {"error": "refused", "detail": "the arguments are not JSON"}
        if not isinstance(arguments, dict):
            return {"error": "refused", "detail": "the arguments are not a JSON object"}
        session = self.servers[tool.server].session
        timeout = self.harness.tool_timeout_seconds
        try:
            result = await asyncio.wait_for(session.call_tool(tool.remote_name, arguments), timeout)
        except asyncio.TimeoutError:
            return {"error": "timeout", "detail": f"the tool did not answer within {timeout:g} second{'' if timeout == 1 else 's'}"}
        except Exception as e:  # noqa: BLE001  # the model is told, and the record keeps it
            return {"error": "tool_error", "detail": f"{type(e).__name__}: {str(e)[:300]}"}
        text = output_text(result)
        if getattr(result, "is_error", False):
            return {"error": "tool_error", "detail": text[:2000]}
        return {"status": "done", "output": text}

    async def close(self) -> None:
        tasks = []
        for state in self.servers.values():
            state.stop.set()
            state.session = None
            if state.task is not None:
                tasks.append(state.task)
        if tasks:
            await asyncio.wait(tasks, timeout=5)

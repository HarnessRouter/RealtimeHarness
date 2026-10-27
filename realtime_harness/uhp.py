"""The UHP server: Core on text turns (design §6.4), and the Realtime chapter as proposed (design §6).

Version 2026-10-04 of the Unified Harness Protocol (unifiedharnessprotocol.org). Core here means:
discovery, harness and model discovery, tasks in blocking, streaming and background modes, sessions chained
by `previous_response_id`, cancellation and the error model. A realtime call is a session too: its turns
are created by speech instead of by `POST /v1/responses`, and read through the same endpoints.

The objects live in this process. A harness keeps its own record of every call it serves; a server in
front of it (HarnessRouter) keeps the durable one.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .call import CallRecord, run_call
from .config import HarnessConfig
from .credentials import CredentialError, Credentials
from .textturn import Listener, TurnResult, run_text_turn
from .tools import ToolExecutor

VERSIONS = ["2026-10-04"]
BASE = "realtime"
BASE_LABEL = "Realtime"
DEFAULT_MODEL = "gpt-realtime-2.1"
TERMINAL = {"completed", "failed", "incomplete", "cancelled"}
KEEPALIVE_S = 15
# Request fields this server acts on; anything else a request carries is ignored and named in
# metadata.ignored_fields (Tasks §1.1). `tools` and `include` are reserved and always ignored.
ACTED_ON = {"input", "model", "metadata", "stream", "previous_response_id", "instructions", "store",
            "background", "max_step", "timeout_seconds"}
STATUS_HTTP = {"invalid_request_error": 400, "authentication_error": 401, "permission_error": 403,
               "rate_limit_error": 429, "harness_error": 500, "server_error": 500}


class UhpError(Exception):
    def __init__(self, status: int, code: str, message: str, etype: str = "invalid_request_error",
                 param: str | None = None, detail: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.etype, self.param, self.detail = status, code, message, etype, param, detail

    def envelope(self) -> dict:
        return {"error": {"type": self.etype, "code": self.code, "message": self.message, "param": self.param,
                          "detail": self.detail}}


def _rid(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(16)}"


def harness_id(h: HarnessConfig) -> str:
    return h.id if h.id.startswith("chrn_") else f"chrn_{h.id}"


@dataclass
class Session:
    id: str
    harness: HarnessConfig
    items: list[dict] = field(default_factory=list)      # the conversation so far, as UHP items
    running: str | None = None                           # the response in flight
    last_response_id: str | None = None
    kind: str = "text"                                   # text | realtime
    created_at: int = field(default_factory=lambda: int(time.time()))


@dataclass
class Call:
    """A realtime session's call: the device's one-time connect token, the record and the event stream."""
    session: Session
    model: str
    token: str
    token_expires: float
    record: CallRecord | None = None
    status: str = "connecting"                    # connecting | live | ended
    reason: str | None = None
    token_used: bool = False
    events: list[dict] = field(default_factory=list)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    ended: asyncio.Event = field(default_factory=asyncio.Event)       # asks a live call to end
    closed: asyncio.Event = field(default_factory=asyncio.Event)      # the call is over
    end_reason: list[str] = field(default_factory=list)

    def publish(self, ev: dict) -> None:
        ev = {**ev, "sequence_number": len(self.events)}     # one gapless sequence across the whole call
        self.events.append(ev)
        for q in list(self.subscribers):
            q.put_nowait(ev)


CONNECT_TOKEN_TTL_S = 60


@dataclass
class Task:
    id: str
    session: Session
    model: str
    created_at: int
    previous_response_id: str | None
    store: bool
    metadata: dict
    status: str = "in_progress"
    output: list[dict] = field(default_factory=list)
    usage: dict | None = None
    error: dict | None = None
    incomplete_details: dict | None = None
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    runner: asyncio.Task | None = None

    def to_dict(self) -> dict:
        return {"id": self.id, "object": "response", "created_at": self.created_at, "status": self.status,
                "error": self.error if self.status == "failed" else None, "incomplete_details": self.incomplete_details,
                "previous_response_id": self.previous_response_id, "model": self.model,
                "output": [dict(i) for i in self.output], "store": self.store, "usage": self.usage,
                "metadata": dict(self.metadata)}

    def publish(self, ev: dict) -> None:
        for q in list(self.subscribers):
            q.put_nowait(ev)


class StreamListener(Listener):
    """A text turn's progress as UHP stream events (Streaming §2): per item, added, then its text, then done."""

    def __init__(self, task: Task):
        self.task = task

    def _index(self, item: dict) -> int:
        return next(i for i, it in enumerate(self.task.output) if it is item)

    async def item_added(self, item: dict) -> None:
        self.task.output.append(item)              # a read while the turn runs sees what it has made so far
        shell = dict(item)
        if item["type"] == "message":
            shell["content"] = []
        self.task.publish({"type": "response.output_item.added", "output_index": self._index(item), "item": shell})
        if item["type"] == "message":
            self.task.publish({"type": "response.content_part.added", "output_index": self._index(item), "item_id": item["id"],
                               "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}})

    async def text_delta(self, item: dict, delta: str) -> None:
        self.task.publish({"type": "response.output_text.delta", "output_index": self._index(item), "item_id": item["id"],
                           "content_index": 0, "delta": delta})

    async def item_done(self, item: dict) -> None:
        idx = self._index(item)
        if item["type"] == "message":
            text = item["content"][0]["text"]
            self.task.publish({"type": "response.output_text.done", "output_index": idx, "item_id": item["id"],
                               "content_index": 0, "text": text})
            self.task.publish({"type": "response.content_part.done", "output_index": idx, "item_id": item["id"],
                               "content_index": 0, "part": dict(item["content"][0])})
        elif item["type"] == "function_call":
            self.task.publish({"type": "response.function_call_arguments.done", "output_index": idx, "item_id": item["id"],
                               "arguments": item.get("arguments") or "{}"})
        self.task.publish({"type": "response.output_item.done", "output_index": idx, "item": dict(item)})


def _input_text(inp) -> str:
    """A request's input as the user's text: a string, or input items whose input_text parts are joined."""
    if isinstance(inp, str):
        return inp
    if not isinstance(inp, list):
        raise UhpError(400, "invalid_input", "input is a string or a list of input items.", param="input")
    parts: list[str] = []
    for it in inp:
        if not isinstance(it, dict):
            continue
        content = it.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts += [str(p.get("text") or "") for p in content if isinstance(p, dict) and p.get("type") in ("input_text", "text")]
        elif it.get("type") == "input_text":
            parts.append(str(it.get("text") or ""))
    return "\n".join(p for p in parts if p)


class Server:
    def __init__(self, harnesses: list[HarnessConfig], credentials: Credentials | None, api_keys: set[str],
                 models: list[str] | None = None):
        if not harnesses:
            raise ValueError("a server needs at least one harness")
        self.harnesses = {harness_id(h): h for h in harnesses}
        self.default_harness = harness_id(harnesses[0])
        self.credentials = credentials
        self.api_keys = api_keys
        self.models = models or [DEFAULT_MODEL]
        self.tasks: dict[str, Task] = {}
        self.sessions: dict[str, Session] = {}
        self.calls: dict[str, Call] = {}                     # by session id
        self.tokens: dict[str, str] = {}                     # connect token -> session id

    # ── discovery ────────────────────────────────────────────────────────────────────────────
    def discovery(self) -> dict:
        return {"object": "uhp.discovery", "protocol": "uhp", "versions": VERSIONS, "default_version": VERSIONS[0],
                "conformance_class": "core",
                "capabilities": {"streaming": True, "sessions": True, "cancellation": True, "files_input": False,
                                 "files_output": False, "session_listing": False, "harness_management": False,
                                 "session_sharing": False, "idempotency": False, "plugins": False, "environments": False,
                                 "memories": False,
                                 # the Realtime chapter, proposed for the next version (design §6.2): additive
                                 "realtime": True},
                "realtime_transports": ["websocket"],
                "implementation": {"name": "realtime-harness", "version": __version__}}

    def harness(self, hid: str) -> HarnessConfig:
        h = self.harnesses.get(hid)
        if h is None:
            raise UhpError(404, "harness_not_found", "No harness with that id.", param="harness_id")
        return h

    def harness_dict(self, h: HarnessConfig) -> dict:
        return {"id": harness_id(h), "object": "harness", "name": h.name, "base": BASE, "baseLabel": BASE_LABEL,
                "defaultModel": h.default_model or self.models[0], "systemPrompt": h.instructions,
                "mcpServers": [{"name": s.name, "url": s.url, "transport": s.transport, "enabled": s.enabled}
                               for s in h.mcp_servers],
                "skills": [], "plugins": [], "disabledTools": list(h.disabled_tools), "maxStep": None,
                "timeoutSeconds": None}

    def model_list(self) -> list[dict]:
        ok = self.credentials is not None
        return [{"id": m, "available": ok, "backend": BASE} for m in self.models]

    # ── tasks ────────────────────────────────────────────────────────────────────────────────
    async def create(self, body: dict, subscribe: bool = False) -> tuple[Task, asyncio.Queue | None]:
        if "input" not in body:
            raise UhpError(400, "invalid_input", "input is required.", param="input")
        text = _input_text(body.get("input"))
        if not text.strip():
            raise UhpError(400, "invalid_input", "input has no text.", param="input")
        meta = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
        prev = body.get("previous_response_id")
        if prev:
            prior = self.tasks.get(str(prev))
            if prior is None:
                raise UhpError(404, "response_not_found", "No response with that id.", param="previous_response_id")
            session = prior.session
            if meta.get("harness_id") and str(meta["harness_id"]) != harness_id(session.harness):
                raise UhpError(409, "harness_mismatch", "That response belongs to another harness.", param="metadata.harness_id")
        else:
            h = self.harness(str(meta.get("harness_id") or self.default_harness))
            session = Session(id=_rid("hsess"), harness=h)
            self.sessions[session.id] = session
        if session.running is not None:
            raise UhpError(409, "session_busy", "This session is already running a task.")
        model = str(body.get("model") or session.harness.default_model or self.models[0])
        if model not in self.models:
            raise UhpError(422, "model_unavailable", f"This server does not run {model!r}.", param="model",
                           detail={"available": list(self.models)})
        if self.credentials is None:
            raise UhpError(503, "server_unavailable", "This server has no model connection configured.", etype="server_error")
        ignored = sorted(k for k in body if k not in ACTED_ON)
        metadata = {**{k: v for k, v in meta.items() if k != "harness_id"}, "session_id": session.id,
                    "harness_id": harness_id(session.harness)}
        if ignored:
            metadata["ignored_fields"] = ignored
        task = Task(id=_rid("resp"), session=session, model=model, created_at=int(time.time()),
                    previous_response_id=str(prev) if prev else None, store=bool(body.get("store", True)), metadata=metadata)
        self.tasks[task.id] = task
        session.running = task.id
        queue: asyncio.Queue | None = None
        if subscribe:                                  # before the turn starts, so no event is missed
            queue = asyncio.Queue()
            task.subscribers.append(queue)
        max_step = body.get("max_step") if isinstance(body.get("max_step"), int) and body["max_step"] > 0 else None
        timeout = body.get("timeout_seconds") if isinstance(body.get("timeout_seconds"), (int, float)) and body["timeout_seconds"] > 0 else None
        instructions = session.harness.instructions
        if isinstance(body.get("instructions"), str) and body["instructions"].strip():
            instructions = f"{instructions}\n\n{body['instructions']}".strip()
        task.runner = asyncio.create_task(self._run(task, text, instructions, max_step, timeout))
        return task, queue

    async def _run(self, task: Task, text: str, instructions: str, max_step: int | None, timeout: float | None) -> None:
        session = task.session
        executor = ToolExecutor(session.harness)
        conn = None
        user_item = {"id": _rid("msg"), "type": "message", "role": "user", "status": "completed",
                     "content": [{"type": "input_text", "text": text}]}
        result = TurnResult(status="failed", error={"code": "harness_error", "message": "The turn did not run."})
        try:
            await executor.open()
            conn = await self.credentials.connect(task.model)
            turn = run_text_turn(conn, instructions=instructions, executor=executor, history=list(session.items),
                                 text=text, listener=StreamListener(task), cancelled=task.cancelled, max_step=max_step)
            if timeout:
                try:
                    async with asyncio.timeout(timeout):
                        result = await turn
                except TimeoutError:
                    result = TurnResult(status="incomplete", output=task.output, usage=task.usage)
                    task.incomplete_details = {"reason": "timeout_seconds"}
            else:
                result = await turn
        except CredentialError as e:
            result = TurnResult(status="failed", error={"code": "provider_unavailable", "message": str(e)})
        except Exception as e:  # noqa: BLE001 — a failed task is a 200 with status failed, never a crash
            result = TurnResult(status="failed", error={"code": "harness_error", "message": f"The turn failed ({type(e).__name__})."})
        finally:
            if conn is not None:
                await self.credentials.release(conn)
            await executor.close()
        if task.cancelled.is_set() and result.status != "failed":
            result.status = "cancelled"
        if result.output and len(result.output) >= len(task.output):
            task.output = result.output
        task.usage = result.usage
        task.status = result.status
        if result.status == "failed":
            err = result.error or {}
            task.error = {"type": "harness_error", "code": err.get("code") or "harness_error",
                          "message": err.get("message") or "The turn failed."}
        if result.status == "incomplete" and task.incomplete_details is None:
            task.incomplete_details = {"reason": "max_step"}
        if result.status in ("completed", "incomplete"):
            session.items += [user_item] + [dict(i) for i in task.output]
        session.last_response_id = task.id
        session.running = None
        terminal = {"completed": "response.completed", "incomplete": "response.incomplete"}.get(task.status, "response.failed")
        task.publish({"type": terminal, "response": task.to_dict()})
        task.done.set()
        for q in list(task.subscribers):
            q.put_nowait(None)

    # ── realtime calls (design §6.2) ─────────────────────────────────────────────────────────
    def create_call(self, body: dict, ws_base: str) -> dict:
        meta = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
        h = self.harness(str(meta.get("harness_id") or self.default_harness))
        model = str(body.get("model") or h.default_model or self.models[0])
        if model not in self.models:
            raise UhpError(422, "model_unavailable", f"This server does not run {model!r}.", param="model",
                           detail={"available": list(self.models)})
        transport = str(body.get("transport") or "websocket")
        if transport != "websocket":
            raise UhpError(400, "invalid_input", "This server connects calls over websocket.", param="transport")
        if self.credentials is None:
            raise UhpError(503, "server_unavailable", "This server has no model connection configured.", etype="server_error")
        session = Session(id=_rid("hsess"), harness=h, kind="realtime")
        self.sessions[session.id] = session
        token = secrets.token_urlsafe(32)
        call = Call(session=session, model=model, token=token, token_expires=time.time() + CONNECT_TOKEN_TTL_S)
        call.record = CallRecord(session.id, harness_id(h), model, call.publish)
        self.calls[session.id] = call
        self.tokens[token] = session.id
        ignored = sorted(k for k in body if k not in {"model", "metadata", "transport", "voice"})
        return {**self.session_dict(session), "connect": {
            "transport": "websocket", "url": f"{ws_base}/v1/realtime/connect", "token": token,
            "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(call.token_expires))},
            **({"metadata": {"ignored_fields": ignored}} if ignored else {})}

    def claim(self, token: str) -> Call:
        """A connect token is good for one connection, within its minute (design §6.2)."""
        sid = self.tokens.pop(token, None)
        call = self.calls.get(sid or "")
        if call is None or call.token_used or time.time() > call.token_expires:
            raise UhpError(401, "invalid_credential", "The connect token is not valid.", etype="authentication_error")
        call.token_used = True
        return call

    async def serve_call(self, websocket, call: Call) -> None:
        call.status = "live"
        await run_call(websocket, harness=call.session.harness, credentials=self.credentials, model=call.model,
                       record=call.record, ended=call.ended, end_reason=call.end_reason)
        call.reason = call.end_reason[0] if call.end_reason else "hangup"
        call.status = "ended"
        call.record.event({"type": "realtime.session.ended", "reason": call.reason,
                           "ignored_device_events": call.record.ignored_device_events})
        call.closed.set()
        for q in list(call.subscribers):
            q.put_nowait(None)

    def session_dict(self, s: Session) -> dict:
        call = self.calls.get(s.id)
        d = {"id": s.id, "object": "session", "created_at": s.created_at, "harness_id": harness_id(s.harness),
             "kind": s.kind, "status": (call.status if call else ("running" if s.running else "idle")),
             "last_response_id": s.last_response_id}
        if call is not None:
            d["model"] = call.model
            d["responses"] = [call.record.response_dict(t) for t in call.record.turns]
            if call.reason:
                d["ended_reason"] = call.reason
            d["last_response_id"] = call.record.turns[-1].id if call.record.turns else None
        return d

    def session(self, sid: str) -> Session:
        s = self.sessions.get(sid)
        if s is None:
            raise UhpError(404, "session_not_found", "No session with that id.", param="session_id")
        return s

    async def call_events(self, call: Call) -> AsyncIterator[bytes]:
        """The call's events from the first, with their call-wide sequence numbers, until it ends."""
        q: asyncio.Queue = asyncio.Queue()
        backlog = list(call.events)
        call.subscribers.append(q)
        try:
            for ev in backlog:
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode()
            if call.closed.is_set():
                return
            seen = len(backlog)
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), KEEPALIVE_S)
                except asyncio.TimeoutError:
                    yield b": keep-alive\n\n"
                    continue
                if ev is None:
                    return
                if ev["sequence_number"] < seen:
                    continue
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode()
        finally:
            if q in call.subscribers:
                call.subscribers.remove(q)

    def response_of_call(self, rid: str) -> dict | None:
        for call in self.calls.values():
            for t in call.record.turns:
                if t.id == rid:
                    return call.record.response_dict(t)
        return None

    def get(self, rid: str) -> Task:
        t = self.tasks.get(rid)
        if t is None:
            raise UhpError(404, "response_not_found", "No response with that id.", param="response_id")
        return t

    async def cancel(self, rid: str) -> Task:
        t = self.get(rid)
        if t.status in TERMINAL:
            return t                    # cancelling a finished task succeeds and changes nothing
        t.cancelled.set()
        try:
            await asyncio.wait_for(t.done.wait(), 30)
        except asyncio.TimeoutError:
            pass
        return t

    async def cancel_session(self, sid: str) -> dict:
        s = self.session(sid)
        call = self.calls.get(sid)
        if call is not None:                         # hangs up; the record stays readable
            if call.status == "live":
                if not call.end_reason:
                    call.end_reason.append("hangup")
                call.ended.set()
                try:
                    await asyncio.wait_for(call.closed.wait(), 30)
                except asyncio.TimeoutError:
                    pass
            elif call.status == "connecting":
                call.status, call.reason = "ended", "hangup"
                self.tokens.pop(call.token, None)
                call.record.event({"type": "realtime.session.ended", "reason": "hangup", "ignored_device_events": 0})
                call.closed.set()
            return self.session_dict(s)
        if s.running:
            await self.cancel(s.running)
        return {"id": s.id, "object": "session", "status": "idle"}

    def delete(self, rid: str) -> dict:
        t = self.get(rid)
        if t.status not in TERMINAL:
            raise UhpError(409, "session_busy", "A running task is cancelled, not deleted.")
        self.tasks.pop(rid, None)
        return {"id": rid, "object": "response", "deleted": True}

    async def stream(self, task: Task, q: asyncio.Queue) -> AsyncIterator[bytes]:
        """The task's events from `response.created` to its terminal event, numbered from 0 with no gap. A
        client that goes away does not stop the task (Streaming §5)."""
        seq = 0

        def line(ev: dict) -> bytes:
            nonlocal seq
            ev = {**ev, "sequence_number": seq}
            seq += 1
            return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode()

        try:
            created = task.to_dict()
            created["status"] = "in_progress"
            yield line({"type": "response.created", "response": created})
            yield line({"type": "response.in_progress", "response": created})
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), KEEPALIVE_S)
                except asyncio.TimeoutError:
                    yield b": keep-alive\n\n"
                    continue
                if ev is None:
                    return
                yield line(ev)
        finally:
            if q in task.subscribers:
                task.subscribers.remove(q)


def create_app(server: Server) -> FastAPI:
    app = FastAPI(title="Realtime Harness", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.uhp = server

    def version_of(request: Request) -> str:
        asked = request.headers.get("uhp-version")
        if asked and asked not in VERSIONS:
            raise UhpError(400, "unsupported_protocol_version", f"This server does not speak UHP {asked}.",
                           param="UHP-Version", detail={"supported": VERSIONS})
        return asked or VERSIONS[0]

    def answer(body, status: int = 200, version: str = VERSIONS[0]) -> JSONResponse:
        return JSONResponse(body, status_code=status, headers={"UHP-Version": version})

    def fail(e: UhpError, version: str = VERSIONS[0]) -> JSONResponse:
        return answer(e.envelope(), e.status, version)

    def auth(request: Request) -> None:
        header = request.headers.get("authorization") or ""
        if not header:
            raise UhpError(401, "missing_credential", "A bearer token is required.", etype="authentication_error")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token or token.strip() not in server.api_keys:
            raise UhpError(401, "invalid_credential", "The bearer token is not valid.", etype="authentication_error")

    @app.exception_handler(UhpError)
    async def _uhp_error(request: Request, e: UhpError):
        return fail(e)

    @app.exception_handler(Exception)
    async def _crash(request: Request, e: Exception):      # never a stack trace, a path or a host in a message
        return fail(UhpError(500, "server_error", "The server could not answer this request.", etype="server_error"))

    async def guarded(request: Request, handler, *, public: bool = False):
        try:
            version = version_of(request)
        except UhpError as e:
            return fail(e)
        try:
            if not public:
                auth(request)
            out = await handler()
            if isinstance(out, (JSONResponse, StreamingResponse)):
                out.headers["UHP-Version"] = version
                return out
            return answer(out, 200, version)
        except UhpError as e:
            return fail(e, version)

    async def json_body(request: Request) -> dict:
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            raise UhpError(400, "invalid_input", "The body is not JSON.")
        if not isinstance(body, dict):
            raise UhpError(400, "invalid_input", "The body is a JSON object.")
        return body

    @app.get("/v1/uhp")
    async def discovery(request: Request):
        async def h():
            return server.discovery()
        return await guarded(request, h, public=True)

    @app.get("/v1/harnesses")
    async def harnesses(request: Request):
        async def h():
            return {"object": "list", "harnesses": [server.harness_dict(x) for x in server.harnesses.values()]}
        return await guarded(request, h)

    @app.get("/v1/harnesses/{hid}")
    async def harness(request: Request, hid: str):
        async def h():
            return server.harness_dict(server.harness(hid))
        return await guarded(request, h)

    @app.get("/v1/harnesses/{hid}/models")
    async def harness_models(request: Request, hid: str):
        async def h():
            x = server.harness(hid)
            return {"harness_id": harness_id(x), "backend": BASE, "default": x.default_model or server.models[0],
                    "models": server.model_list()}
        return await guarded(request, h)

    @app.get("/v1/models")
    async def models(request: Request):
        async def h():
            return {"backends": {BASE: {"default": server.models[0], "models": server.model_list()}}}
        return await guarded(request, h)

    @app.post("/v1/responses")
    async def responses(request: Request):
        async def h():
            body = await json_body(request)
            task, queue = await server.create(body, subscribe=bool(body.get("stream")))
            if queue is not None:
                return StreamingResponse(server.stream(task, queue), media_type="text/event-stream",
                                         headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
            if body.get("background"):
                return task.to_dict()
            await task.done.wait()
            return task.to_dict()
        return await guarded(request, h)

    @app.get("/v1/responses/{rid}")
    async def get_response(request: Request, rid: str):
        async def h():
            in_call = server.response_of_call(rid)
            return in_call if in_call is not None else server.get(rid).to_dict()
        return await guarded(request, h)

    @app.post("/v1/realtime/sessions")
    async def create_call(request: Request):
        async def h():
            body = await json_body(request)
            scheme = "wss" if (request.headers.get("x-forwarded-proto") or request.url.scheme) in ("https", "wss") else "ws"
            host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
            return JSONResponse(server.create_call(body, f"{scheme}://{host}"), status_code=201)
        return await guarded(request, h)

    @app.websocket("/v1/realtime/connect")
    async def connect(websocket: WebSocket):
        token = websocket.query_params.get("token") or ""
        try:
            call = server.claim(token)
        except UhpError:
            await websocket.close(code=4401)
            return
        await websocket.accept()
        await server.serve_call(websocket, call)

    @app.get("/v1/sessions/{sid}")
    async def get_session(request: Request, sid: str):
        async def h():
            return server.session_dict(server.session(sid))
        return await guarded(request, h)

    @app.get("/v1/sessions/{sid}/events")
    async def session_events(request: Request, sid: str):
        async def h():
            server.session(sid)
            call = server.calls.get(sid)
            if call is None:
                raise UhpError(404, "session_not_found", "That session is not a realtime call.", param="session_id")
            return StreamingResponse(server.call_events(call), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
        return await guarded(request, h)

    @app.post("/v1/responses/{rid}/cancel")
    async def cancel_response(request: Request, rid: str):
        async def h():
            return (await server.cancel(rid)).to_dict()
        return await guarded(request, h)

    @app.delete("/v1/responses/{rid}")
    async def delete_response(request: Request, rid: str):
        async def h():
            return server.delete(rid)
        return await guarded(request, h)

    @app.post("/v1/sessions/{sid}/cancel")
    async def cancel_session(request: Request, sid: str):
        async def h():
            return await server.cancel_session(sid)
        return await guarded(request, h)

    return app


def api_keys_from_env() -> set[str]:
    """RTH_API_KEYS (comma separated) or RTH_API_KEYS_FILE (one key per line)."""
    keys = {k.strip() for k in (os.environ.get("RTH_API_KEYS") or "").split(",") if k.strip()}
    path = os.environ.get("RTH_API_KEYS_FILE")
    if path:
        with open(path, encoding="utf-8") as f:
            keys |= {ln.strip() for ln in f if ln.strip()}
    return keys

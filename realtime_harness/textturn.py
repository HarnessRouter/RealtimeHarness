"""A text turn on a realtime model, over the OpenAI realtime wire (OpenAI, Azure OpenAI): design §6.4.

A realtime model also takes a text turn, so the `realtime` base answers `POST /v1/responses` like any
other base. One turn is one WebSocket connection: the harness's configuration first (`session.update`:
instructions, text output, the tools), then the session's earlier items, then the ask. Every function call
the model makes is run by the tool executor and answered before the model continues, as on a call. The
turn ends when the model answers without calling a tool.
"""
from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import websockets

from .credentials import Connection
from .tools import ToolExecutor

OPEN_TIMEOUT_S = 15
EVENT_TIMEOUT_S = 120           # no event from the provider for this long ends the turn as failed


def item_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


@dataclass
class TurnResult:
    status: str = "completed"               # completed | incomplete | failed | cancelled
    output: list[dict] = field(default_factory=list)
    usage: dict | None = None
    error: dict | None = None


class Listener:
    """What a turn reports as it happens; the UHP layer turns these into stream events and the record."""
    async def item_added(self, item: dict) -> None: ...
    async def text_delta(self, item: dict, delta: str) -> None: ...
    async def item_done(self, item: dict) -> None: ...


def _add_usage(total: dict | None, u: dict | None) -> dict | None:
    """The provider's usage, summed over the turn's model responses, in UHP's names. None when the provider
    reported none: a fabricated zero would read as a free turn."""
    if not isinstance(u, dict):
        return total
    total = dict(total or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cache_read_tokens": 0})
    total["input_tokens"] += int(u.get("input_tokens") or 0)
    total["output_tokens"] += int(u.get("output_tokens") or 0)
    total["total_tokens"] += int(u.get("total_tokens") or 0)
    total["cache_read_tokens"] += int(((u.get("input_token_details") or {}).get("cached_tokens")) or 0)
    return total


def history_events(items: list[dict]) -> list[dict]:
    """A session's earlier items as the conversation items that restore it on a new connection."""
    out = []
    for it in items:
        t = it.get("type")
        if t == "message":
            role = it.get("role") or "user"
            text = "".join(str(p.get("text") or "") for p in it.get("content") or [] if isinstance(p, dict))
            if not text:
                continue
            kind = "output_text" if role == "assistant" else "input_text"
            out.append({"type": "conversation.item.create",
                        "item": {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}})
        elif t == "function_call":
            out.append({"type": "conversation.item.create", "item": {
                "type": "function_call", "call_id": it.get("call_id"), "name": it.get("name"), "arguments": it.get("arguments") or "{}"}})
        elif t == "function_call_output":
            out.append({"type": "conversation.item.create", "item": {
                "type": "function_call_output", "call_id": it.get("call_id"), "output": it.get("output") or ""}})
    return out


async def run_text_turn(conn: Connection, *, instructions: str, executor: ToolExecutor, history: list[dict],
                        text: str, listener: Listener, cancelled: asyncio.Event, max_step: int | None = None) -> TurnResult:
    result = TurnResult()
    try:
        ws = await websockets.connect(conn.url, additional_headers=conn.headers, max_size=None, open_timeout=OPEN_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001  # the provider could not be reached; the message names no secret
        return TurnResult(status="failed", error={"code": "provider_unavailable",
                                                  "message": f"The model could not be reached ({type(e).__name__})."})
    try:
        async def send(ev: dict) -> None:
            await ws.send(json.dumps(ev))

        async def next_event() -> dict:
            raw = await asyncio.wait_for(ws.recv(), EVENT_TIMEOUT_S)
            return json.loads(raw)

        async def until(*types: str) -> dict:
            while True:
                ev = await next_event()
                if ev.get("type") in types:
                    return ev
                if ev.get("type") == "error":
                    raise _ProviderError(ev.get("error") or {})

        await until("session.created")
        await send({"type": "session.update", "session": {
            "type": "realtime", "instructions": instructions, "output_modalities": ["text"],
            "tools": executor.functions(), "tool_choice": "auto"}})
        await until("session.updated")
        for ev in history_events(history):
            await send(ev)
        await send({"type": "conversation.item.create", "item": {"type": "message", "role": "user",
                                                                  "content": [{"type": "input_text", "text": text}]}})
        steps = 0
        while True:
            await send({"type": "response.create"})
            outcome, calls = await _one_response(next_event, send, listener, result, cancelled)
            if outcome != "completed":
                result.status = outcome
                return result
            if not calls:
                return result
            steps += 1
            if max_step is not None and steps >= max_step:
                result.status = "incomplete"          # the step budget ended the turn (Tasks §1.4)
                return result
            outputs = await asyncio.gather(*(executor.call(c["name"], c["arguments"]) for c in calls))
            for call, out in zip(calls, outputs):
                item = {"id": item_id("fco"), "type": "function_call_output", "call_id": call["call_id"],
                        "output": json.dumps(out, ensure_ascii=False)}
                result.output.append(item)
                await listener.item_added(item)
                await listener.item_done(item)
                await send({"type": "conversation.item.create", "item": {
                    "type": "function_call_output", "call_id": call["call_id"], "output": item["output"]}})
            if cancelled.is_set():
                result.status = "cancelled"
                return result
    except _ProviderError as e:
        return TurnResult(status="failed", output=result.output, usage=result.usage,
                          error={"code": "provider_error", "message": str(e)})
    except (asyncio.TimeoutError, websockets.ConnectionClosed) as e:
        return TurnResult(status="failed", output=result.output, usage=result.usage,
                          error={"code": "provider_error", "message": f"The model's connection ended ({type(e).__name__})."})
    finally:
        await ws.close()


class _ProviderError(Exception):
    def __init__(self, err: dict):
        # the provider's own code and message: they describe the request, never our credential
        super().__init__(f"{err.get('code') or err.get('type') or 'error'}: {str(err.get('message') or '')[:300]}")


async def _one_response(next_event: Callable[[], Awaitable[dict]], send: Callable[[dict], Awaitable[None]],
                        listener: Listener, result: TurnResult, cancelled: asyncio.Event) -> tuple[str, list[dict]]:
    """Follow one model response to its end; returns (status, the function calls it made)."""
    by_provider_id: dict[str, dict] = {}         # the provider's item id -> our item
    calls: list[dict] = []
    cancel_sent = False
    while True:
        if cancelled.is_set() and not cancel_sent:
            await send({"type": "response.cancel"})
            cancel_sent = True
        try:
            ev = await asyncio.wait_for(next_event(), 0.25 if not cancel_sent else EVENT_TIMEOUT_S)
        except asyncio.TimeoutError:
            continue
        t = ev.get("type")
        if t == "error":
            err = ev.get("error") or {}
            if cancel_sent and "cancel" in str(err.get("code") or "").lower():
                continue        # nothing left to cancel: the response.done that follows says how it ended
            raise _ProviderError(err)
        if t == "response.output_item.added":
            pit = ev.get("item") or {}
            if pit.get("type") == "message":
                item = {"id": item_id("msg"), "type": "message", "role": "assistant", "status": "in_progress",
                        "content": [{"type": "output_text", "text": "", "annotations": []}]}
            elif pit.get("type") == "function_call":
                item = {"id": item_id("fc"), "type": "function_call", "status": "in_progress",
                        "call_id": pit.get("call_id") or "", "name": pit.get("name") or "", "arguments": ""}
            else:
                continue
            by_provider_id[pit.get("id") or ""] = item
            result.output.append(item)
            await listener.item_added(item)
        elif t in ("response.output_text.delta", "response.text.delta"):
            item = by_provider_id.get(ev.get("item_id") or "")
            if item is not None and item["type"] == "message":
                item["content"][0]["text"] += ev.get("delta") or ""
                await listener.text_delta(item, ev.get("delta") or "")
        elif t == "response.function_call_arguments.done":
            item = by_provider_id.get(ev.get("item_id") or "")
            if item is not None:
                item["arguments"] = ev.get("arguments") or "{}"
                item["name"] = item["name"] or ev.get("name") or ""
                item["call_id"] = item["call_id"] or ev.get("call_id") or ""
        elif t == "response.output_item.done":
            pit = ev.get("item") or {}
            item = by_provider_id.get(pit.get("id") or "")
            if item is None:
                continue
            item["status"] = "completed"
            if item["type"] == "function_call":
                item["arguments"] = pit.get("arguments") or item["arguments"] or "{}"
                calls.append({"call_id": item["call_id"], "name": item["name"], "arguments": item["arguments"]})
            await listener.item_done(item)
        elif t == "response.done":
            resp = ev.get("response") or {}
            result.usage = _add_usage(result.usage, resp.get("usage"))
            status = str(resp.get("status") or "completed")
            for item in by_provider_id.values():
                if item.get("status") == "in_progress":
                    item["status"] = "incomplete"
                    await listener.item_done(item)
            if status == "completed":
                return "completed", calls
            if status == "cancelled" or cancel_sent:
                return "cancelled", []
            if status == "incomplete":
                return "incomplete", []
            details = resp.get("status_details") or {}
            err = details.get("error") or {}
            raise _ProviderError({"code": err.get("code") or status, "message": err.get("message") or details.get("reason") or status})

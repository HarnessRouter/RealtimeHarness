"""One realtime call: the device's audio, the provider's model, the harness's tools, and the record (design §4).

The pipeline is Pipecat's: its FastAPI WebSocket transport carries the device's audio, its realtime service
speaks the provider's wire, and its context aggregators send the follow-up response after a tool result.
What this module adds is what a harness owns (design §2):

- The configuration is the harness's. Instructions, voice and tools are set when the provider connection
  opens; the device can narrow a call (mute, cancel the model's answer, hang up) and nothing else. Any
  other event it sends is ignored and counted in the record.
- Every tool call is run here, by the tool executor, through one catch-all handler: the model's call, the
  harness's MCP server, the result back to the model.
- The record is written as the call happens: each model turn is a UHP response in the call's session, with
  what the person said (the provider's transcript) as its input, and the turn's function_call,
  function_call_output and message items as its output.
- The budget: a call ends at the harness's max_call_minutes, with reason `budget`.

The device protocol (design §6.2): one WebSocket. Binary messages are PCM16 mono audio at 24 kHz, both
ways. Text messages are JSON events: from the device, `realtime.input.mute`, `realtime.input.unmute`,
`realtime.response.cancel`, `realtime.session.end`; to the device, the session's events, the same ones
`GET /v1/sessions/{id}/events` streams.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Callable

from .credentials import Connection, Credentials
from .tools import ToolExecutor

SAMPLE_RATE = 24000
# The provider says a response is done before Pipecat has started the tool calls it made, so the end of a
# turn waits this long for a call that belongs to it (one turn: the model's words, its calls, and its
# answer to their results).
TURN_END_GRACE_S = 0.6
# A turn whose last item is a tool's result waits for the model's answer to it, which comes in the provider's
# next response; this bounds that wait for a model that never answers.
ANSWER_WAIT_S = 20.0
DEVICE_EVENTS = {"realtime.input.mute", "realtime.input.unmute", "realtime.response.cancel", "realtime.session.end"}


def _id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


@dataclass
class Turn:
    """One model turn of a call, as a UHP response."""
    id: str
    created_at: int
    previous_response_id: str | None
    input: list[dict] = field(default_factory=list)
    output: list[dict] = field(default_factory=list)
    status: str = "in_progress"
    usage: dict | None = None
    message: dict | None = None          # the item carrying what the model said
    calls: dict[str, dict] = field(default_factory=dict)

    def calls_pending(self) -> bool:
        """A turn whose calls are not all answered goes on: the model's answer to the results is the same turn."""
        answered = {i.get("call_id") for i in self.output if i.get("type") == "function_call_output"}
        return any(cid not in answered for cid in self.calls)


class CallRecord:
    """A realtime session's record and its event stream: one `sequence_number` across the whole call."""

    def __init__(self, session_id: str, harness_id: str, model: str, publish: Callable[[dict], None]):
        self.session_id = session_id
        self.harness_id = harness_id
        self.model = model
        self.turns: list[Turn] = []
        self.current: Turn | None = None
        self.pending_input: list[dict] = []
        self.ignored_device_events = 0
        self._publish = publish
        self._ending: asyncio.TimerHandle | None = None

    def event(self, ev: dict) -> None:
        self._publish(ev)

    def response_dict(self, t: Turn) -> dict:
        return {"id": t.id, "object": "response", "created_at": t.created_at, "status": t.status, "error": None,
                "incomplete_details": None, "previous_response_id": t.previous_response_id, "model": self.model,
                "input": t.input, "output": t.output, "store": True, "usage": t.usage,
                "metadata": {"session_id": self.session_id, "harness_id": self.harness_id, "realtime": True}}

    def open_turn(self) -> Turn:
        if self.current is None:
            prev = self.turns[-1].id if self.turns else None
            t = Turn(id=_id("resp"), created_at=int(time.time()), previous_response_id=prev, input=self.pending_input)
            self.pending_input = []
            self.turns.append(t)
            self.current = t
            self.event({"type": "response.created", "response": self.response_dict(t)})
        return self.current

    def user_said(self, text: str) -> None:
        item = {"id": _id("msg"), "type": "message", "role": "user", "status": "completed",
                "content": [{"type": "input_text", "text": text}]}
        # the provider transcribes the caller's speech after the model has started answering it, often:
        # the turn that answered it takes it as its input. Words after a barge-in are the interruption, and
        # belong to the turn that answers them, not to the one they cut off.
        if self.current is not None and not self.current.input:
            self.current.input.append(item)
        elif self.current is None and self.turns and not self.turns[-1].input and self.turns[-1].status == "completed":
            self.turns[-1].input.append(item)
        else:
            self.pending_input.append(item)
        self.event({"type": "realtime.input_transcript.done", "item_id": item["id"], "transcript": text})

    def model_said(self, delta: str) -> None:
        self._cancel_ending()
        t = self.open_turn()
        if t.message is None:
            t.message = {"id": _id("msg"), "type": "message", "role": "assistant", "status": "in_progress",
                         "content": [{"type": "output_text", "text": "", "annotations": []}]}
            t.output.append(t.message)
            self.event({"type": "response.output_item.added", "response_id": t.id,
                        "output_index": len(t.output) - 1, "item": {**t.message, "content": []}})
        t.message["content"][0]["text"] += delta
        self.event({"type": "realtime.output_transcript.delta", "response_id": t.id, "item_id": t.message["id"], "delta": delta})

    def _close_message(self, t: Turn, status: str = "completed") -> None:
        """What the model said so far in this turn is one message item; it closes when the model calls a tool
        or the turn ends, and anything it says after a tool's result is a new item."""
        if t.message is None:
            return
        m, t.message = t.message, None
        m["status"] = status
        self.event({"type": "realtime.output_transcript.done", "response_id": t.id, "item_id": m["id"],
                    "transcript": m["content"][0]["text"]})
        self.event({"type": "response.output_item.done", "response_id": t.id, "output_index": t.output.index(m), "item": m})

    def call_started(self, call_id: str, name: str, arguments) -> None:
        self._cancel_ending()
        t = self.open_turn()
        if call_id in t.calls:
            return                                   # a broadcast frame is seen once per direction
        self._close_message(t)
        args = arguments if isinstance(arguments, str) else json.dumps(arguments or {}, ensure_ascii=False)
        item = {"id": _id("fc"), "type": "function_call", "status": "completed", "call_id": call_id, "name": name,
                "arguments": args}
        t.calls[call_id] = item
        t.output.append(item)
        self.event({"type": "response.output_item.added", "response_id": t.id, "output_index": len(t.output) - 1, "item": item})
        self.event({"type": "response.output_item.done", "response_id": t.id, "output_index": len(t.output) - 1, "item": item})

    def call_finished(self, call_id: str, result) -> None:
        t = next((x for x in reversed(self.turns) if call_id in x.calls), None)
        if t is None or any(i.get("type") == "function_call_output" and i.get("call_id") == call_id for i in t.output):
            return
        item = {"id": _id("fco"), "type": "function_call_output", "call_id": call_id,
                "output": result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)}
        t.output.append(item)
        self.event({"type": "response.output_item.added", "response_id": t.id, "output_index": len(t.output) - 1, "item": item})
        self.event({"type": "response.output_item.done", "response_id": t.id, "output_index": len(t.output) - 1, "item": item})

    def usage(self, u) -> None:
        t = self.current or (self.turns[-1] if self.turns else None)
        if t is None:
            return
        total = dict(t.usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cache_read_tokens": 0})
        total["input_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
        total["output_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
        total["total_tokens"] += int(getattr(u, "total_tokens", 0) or 0)
        total["cache_read_tokens"] += int(getattr(u, "cache_read_input_tokens", 0) or 0)
        for k in ("input_audio_tokens", "output_audio_tokens"):
            v = getattr(u, k, None)
            if v:
                total[k] = total.get(k, 0) + int(v)
        t.usage = total

    def speech_started(self) -> None:
        self.event({"type": "realtime.speech_started"})

    def _cancel_ending(self) -> None:
        if self._ending is not None:
            self._ending.cancel()
            self._ending = None

    def response_ended(self) -> None:
        """The provider finished one response: the turn ends unless a call of this turn follows."""
        if self.current is None:
            return
        self._cancel_ending()
        self._ending = asyncio.get_running_loop().call_later(TURN_END_GRACE_S, self._end_if_settled)

    def _end_if_settled(self, waited: bool = False) -> None:
        self._ending = None
        t = self.current
        if t is None or t.calls_pending():
            return
        if not waited and t.output and t.output[-1].get("type") == "function_call_output":
            # the results are in, and the model's answer to them is the rest of this turn
            self._ending = asyncio.get_running_loop().call_later(ANSWER_WAIT_S, self._end_if_settled, True)
            return
        self.end_turn("completed")

    def end_turn(self, status: str = "completed") -> None:
        self._cancel_ending()
        t = self.current
        if t is None:
            return
        self.current = None
        self._close_message(t, "completed" if status == "completed" else "incomplete")
        t.status = status
        terminal = "response.completed" if status == "completed" else ("response.incomplete" if status == "incomplete" else "response.failed")
        self.event({"type": terminal, "response": self.response_dict(t)})


class DeviceSerializer:
    """The device protocol over Pipecat's FastAPI WebSocket transport: PCM16 audio as binary, events as JSON."""

    def __init__(self, on_device_event: Callable[[dict], None]):
        from pipecat.serializers.base_serializer import FrameSerializer  # noqa: F401 (the interface implemented)
        self.on_device_event = on_device_event
        self.muted = False

    async def setup(self, setup) -> None:
        return None

    def should_ignore_frame(self, frame) -> bool:
        return False

    async def serialize(self, frame):
        from pipecat.frames.frames import (AudioRawFrame, InterruptionFrame, OutputTransportMessageFrame,
                                           OutputTransportMessageUrgentFrame)
        if isinstance(frame, AudioRawFrame):
            return frame.audio
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return json.dumps(frame.message, ensure_ascii=False)
        if isinstance(frame, InterruptionFrame):
            # the model was cut off: the device drops the audio it has queued and not yet played
            return json.dumps({"type": "realtime.output_audio.cleared"})
        return None

    async def deserialize(self, data):
        from pipecat.frames.frames import InputAudioRawFrame
        if isinstance(data, (bytes, bytearray)):
            if self.muted or not data:
                return None
            return InputAudioRawFrame(audio=bytes(data), sample_rate=SAMPLE_RATE, num_channels=1)
        try:
            ev = json.loads(data)
        except (ValueError, TypeError):
            ev = {}
        self.on_device_event(ev if isinstance(ev, dict) else {})
        return None


def _serializer_class():
    """DeviceSerializer as a Pipecat FrameSerializer subclass (built lazily, so importing this module does not
    import Pipecat)."""
    from pipecat.serializers.base_serializer import FrameSerializer

    class _DeviceFrameSerializer(DeviceSerializer, FrameSerializer):
        def __init__(self, on_device_event):
            FrameSerializer.__init__(self)
            DeviceSerializer.__init__(self, on_device_event)
    return _DeviceFrameSerializer


async def provider_service(credentials: Credentials, model: str, settings, tool_timeout_s: float,
                           opened: list[Connection]):
    """The provider's realtime service for one call. With a server's mint, each connect (the first, and any
    reconnect Pipecat makes) uses a fresh short-lived credential; every one is released when the call ends."""
    from pipecat.services.azure.realtime.llm import AzureRealtimeLLMService
    from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
    first = await credentials.connect(model)
    opened.append(first)
    base = first.url.split("?", 1)[0]
    settings.model = first.model
    timeout = tool_timeout_s + 5          # the executor's own limit answers first; this is only a backstop
    if "Authorization" in first.headers:
        tokens = [first.headers["Authorization"].split(" ", 1)[1]]

        async def token_provider() -> str:
            if tokens:
                return tokens.pop()
            nxt = await credentials.connect(model)
            opened.append(nxt)
            return nxt.headers["Authorization"].split(" ", 1)[1]
        if ".azure.com" in base:
            return AzureRealtimeLLMService(base_url=base, token_provider=token_provider, settings=settings,
                                           function_call_timeout_secs=timeout)
        return OpenAIRealtimeLLMService(api_key=tokens[0], base_url=base, settings=settings, function_call_timeout_secs=timeout)
    return AzureRealtimeLLMService(base_url=base, api_key=first.headers.get("api-key"), settings=settings,
                                   function_call_timeout_secs=timeout)


def _recorder_class():
    from pipecat.frames.frames import (FunctionCallInProgressFrame, FunctionCallResultFrame, InterruptionFrame,
                                       LLMFullResponseEndFrame, MetricsFrame, TranscriptionFrame, TTSTextFrame)
    from pipecat.metrics.metrics import LLMUsageMetricsData
    from pipecat.observers.base_observer import BaseObserver

    class Recorder(BaseObserver):
        """Watches the call's frames and writes the record (an observer sees the upstream transcript too)."""

        def __init__(self, record: CallRecord, interrupted: Callable[[], None]):
            super().__init__(observe_every_push=False)
            self.record = record
            self.interrupted = interrupted
            self._seen: set = set()

        async def on_push_frame(self, data) -> None:
            f = data.frame
            key = getattr(f, "id", None)          # Pipecat's own frame id: a Python id() is reused after a frame is freed
            if key in self._seen:
                return
            self._seen.add(key)
            if len(self._seen) > 200000:
                self._seen.clear()
            if isinstance(f, TranscriptionFrame):
                if f.text and f.text.strip():
                    self.record.user_said(f.text.strip())
            elif isinstance(f, TTSTextFrame):
                if f.text:
                    self.record.model_said(f.text)
            elif isinstance(f, FunctionCallInProgressFrame):
                self.record.call_started(f.tool_call_id, f.function_name, f.arguments)
            elif isinstance(f, FunctionCallResultFrame):
                self.record.call_finished(f.tool_call_id, f.result)
            elif isinstance(f, MetricsFrame):
                for d in f.data or []:
                    if isinstance(d, LLMUsageMetricsData):
                        self.record.usage(d.value)
            elif isinstance(f, InterruptionFrame):
                self.record.speech_started()
                self.interrupted()
            elif isinstance(f, LLMFullResponseEndFrame):
                self.record.response_ended()
    return Recorder


async def run_call(websocket, *, harness, credentials: Credentials, model: str, record: CallRecord,
                   ended: asyncio.Event, end_reason: list[str]) -> None:
    """Serve one call on an accepted WebSocket until the device hangs up, the client cancels, the budget ends
    it, or the provider closes. `end_reason[0]` says which."""
    from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
    from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
    from pipecat.frames.frames import InterruptionFrame, OutputTransportMessageUrgentFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
    from pipecat.services.openai.realtime import events as ev
    from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMSettings
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
    from pipecat.workers.runner import WorkerRunner

    executor = ToolExecutor(harness)
    opened: list[Connection] = []
    worker: PipelineWorker | None = None
    loop = asyncio.get_running_loop()

    def finish(reason: str) -> None:
        if not end_reason:
            end_reason.append(reason)
        if worker is not None:
            loop.create_task(worker.cancel(reason=reason))

    serializer = None

    def on_device_event(e: dict) -> None:
        t = str(e.get("type") or "")
        if t not in DEVICE_EVENTS:
            record.ignored_device_events += 1           # a device narrows a call and never widens it
            return
        if t == "realtime.input.mute":
            serializer.muted = True
        elif t == "realtime.input.unmute":
            serializer.muted = False
        elif t == "realtime.response.cancel" and worker is not None:
            loop.create_task(worker.queue_frame(InterruptionFrame()))
        elif t == "realtime.session.end":
            finish("hangup")

    try:
        await executor.open()
        functions = executor.functions()
        tools = ToolsSchema(standard_tools=[FunctionSchema(
            name=f["name"], description=f["description"], properties=dict(f["parameters"].get("properties") or {}),
            required=list(f["parameters"].get("required") or [])) for f in functions]) if functions else None
        settings = OpenAIRealtimeLLMSettings(
            system_instruction=harness.instructions or None,
            session_properties=ev.SessionProperties(
                instructions=(harness.instructions or None),
                audio=ev.AudioConfiguration(
                    input=ev.AudioInput(transcription=ev.InputAudioTranscription(model=harness.transcription_model))
                    if harness.transcription_model else None,
                    output=ev.AudioOutput(voice=harness.voice)),
                tools=functions or None, tool_choice="auto" if functions else None))
        llm = await provider_service(credentials, model, settings, harness.tool_timeout_seconds, opened)

        async def run_tool(params) -> None:
            out = await executor.call(params.function_name, dict(params.arguments or {}))
            await params.result_callback(out)
        llm.register_function(None, run_tool)

        serializer = _serializer_class()(on_device_event)
        transport = FastAPIWebsocketTransport(websocket, FastAPIWebsocketParams(
            audio_in_enabled=True, audio_out_enabled=True, audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE, add_wav_header=False, serializer=serializer))
        context = LLMContext([], tools) if tools else LLMContext([])
        user_agg, assistant_agg = LLMContextAggregatorPair(context)
        pipeline = Pipeline([transport.input(), user_agg, llm, transport.output(), assistant_agg])
        recorder = _recorder_class()(record, interrupted=lambda: record.end_turn("cancelled") if record.current else None)
        worker = PipelineWorker(pipeline, params=PipelineParams(audio_in_sample_rate=SAMPLE_RATE, audio_out_sample_rate=SAMPLE_RATE,
                                                                enable_metrics=True, enable_usage_metrics=True),
                                observers=[recorder], enable_rtvi=False, idle_timeout_secs=None)

        def publish_to_device(e: dict) -> None:
            if worker is not None:
                loop.create_task(worker.queue_frame(OutputTransportMessageUrgentFrame(message=e)))
        record_publish = record._publish

        def both(e: dict) -> None:
            record_publish(e)
            publish_to_device(e)
        record._publish = both

        @transport.event_handler("on_client_disconnected")
        async def _gone(_t, _c):
            finish("hangup")

        budget = loop.call_later(harness.max_call_minutes * 60, finish, "budget")
        stop_watch = loop.create_task(_wait_and(ended, lambda: finish(end_reason[0] if end_reason else "hangup")))
        record.event({"type": "realtime.session.live", "session_id": record.session_id, "model": record.model,
                      "tools": [f["name"] for f in functions], "max_call_minutes": harness.max_call_minutes})
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        try:
            await runner.run()
        finally:
            budget.cancel()
            stop_watch.cancel()
        if not end_reason:
            end_reason.append("provider_closed")
    except Exception as e:  # noqa: BLE001 — the call ends, and the record says why
        if not end_reason:
            end_reason.append("error")
        record.event({"type": "error", "error": {"type": "harness_error", "code": "call_failed",
                                                 "message": f"The call could not continue ({type(e).__name__})."}})
    finally:
        if record.current is not None:
            record.end_turn("cancelled")
        for c in opened:
            await credentials.release(c)
        await executor.close()


async def _wait_and(event: asyncio.Event, then: Callable[[], None]) -> None:
    await event.wait()
    then()

# Realtime Harness: design

A harness for realtime models: the speech-to-speech models that hold a live conversation over one
long-lived connection (OpenAI's gpt-realtime and GPT-Live, the same models on Azure OpenAI, Gemini
Live, Amazon Nova Sonic, xAI's Grok Voice, and the others that follow). The harness is open source
(Apache-2.0) and will plug into HarnessRouter as the base `realtime`.

This document is the design of record, agreed in review on 2026-10-27. Code starts once section 1
is measured. Section 1 is what a realtime model is, from the providers' documentation and the source of
the framework we build on. Its live measurements are marked **to measure**, and they replace the
documentation's word before the first release. Section 2 derives the harness from section 1.
Sections 3 to 11 cover leverage, the architecture, tool calls, the UHP extension, credentials,
metering, operations, providers and conformance. Section 12 is the first release, and section 13
records the decisions made in review, and the one still open.

## 1. What a realtime model is

Everything below comes from the providers' documentation as read on 2026-10-27 and from the source
of Pipecat v1.14.0 (§3). Where the two disagree, or a document is silent, the row says so. Nothing
is inferred. The live calls that confirm or correct each row are listed in §1.6.

### 1.1 The connection

A text model answers a request. A realtime model holds a conversation: one bidirectional stream per
call, with audio in, and audio plus text out. The provider runs the turn-taking itself: voice
activity detection decides when the person has finished, the model answers, and when the person
speaks over the model, the model's answer is cut off ("barge-in"). One model turn is one
*response* inside the conversation. A conversation lasts minutes, and holds many responses.

Three wire families cover the providers we target:

| Family | Providers | Shape |
|---|---|---|
| **OpenAI Realtime** | OpenAI (gpt-realtime), Azure OpenAI (gpt-realtime deployments), xAI (Grok Voice: "compatible", its first event is `conversation.created`, not `session.created`) | JSON events over WebSocket, or WebRTC with a data channel, or SIP. Client: `session.update`, `input_audio_buffer.append`, `conversation.item.create`, `response.create`, `response.cancel`. Server: `response.output_item.added`/`.done`, `response.function_call_arguments.delta`/`.done`, transcript deltas, `response.done` carrying `usage`. |
| **Gemini Live** | Google (Gemini API; Vertex AI) | `BidiGenerateContent` over WebSocket: `setup`, `realtimeInput`, `serverContent`, `toolCall` (`functionCalls[]` with `id`, `name`, `args`), `toolResponse` (`functionResponses[]` matched by `id`), `toolCallCancellation`, `usageMetadata`. |
| **Nova Sonic** | AWS Bedrock | `InvokeModelWithBidirectionalStream`, HTTP/2 signed with SigV4: `sessionStart`, `promptStart` (where tools are declared), `contentStart`/`contentEnd` blocks typed AUDIO, TEXT or TOOL, `audioInput`, `toolUse` (`toolName`, `toolUseId`, `content`), `toolResult`, usage events. |

OpenAI's **GPT-Live** (`gpt-live-1`) belongs to the first family's lineage but changes the shape: a
voice *frontend* that delegates reasoning and tools to a backend. The backend is either a Responses
model OpenAI runs (`delegation.type: "responses"`) or the application itself
(`delegation.type: "client"`: "your application runs the backend, which can be any model, agent, or
service"). The second mode matters to this design (§5.5): it is a provider-native place for a UHP
harness to be the backend.

### 1.2 Function calls

Every target can call functions, and in every one of them **the caller executes the function**: the
model emits a call, something on the caller's side runs it and returns the result, and the model
continues. None of them runs a custom function by itself. Hosted tools such as web search are the
exception, run by the provider, and they are out of scope here.

| | Declared in | Call arrives as | Result goes back as | Async results | Cancel on barge-in |
|---|---|---|---|---|---|
| OpenAI Realtime | `session.update` `tools` | `response.function_call_arguments.done` / `output_item.done` (`call_id`) | `conversation.item.create` `function_call_output`, then `response.create` | The output can be sent later; the model can keep talking | `response.cancel`; documentation silent on pending calls. **To measure** |
| GPT-Live, responses delegation | `delegation.responses.tools` | nested `response.event` → `response.output_item.done` | `response.item.create` + `response.create`; every pending call needs a result (`function_call_outputs_required`) | Not documented | **To measure** |
| GPT-Live, client delegation | the application's own backend | `session.delegation.created` | `session.commentary.append` (spoken) or `session.thinking.append` (unspoken), 500 tokens each | Yes, by construction | **To measure** |
| Azure OpenAI | as OpenAI Realtime | as OpenAI Realtime | as OpenAI Realtime | Yes: a result 8 s late was taken, and the model answered other speech meanwhile (§1.7) | The response is cancelled; a pending call is not, and its late result is taken (§1.7) |
| xAI Grok Voice | `session.update` `tools` (per integrators) | `response.function_call_arguments.done` | `function_call_output` item + one `response.create` after all results | Not documented | **To measure** |
| Gemini Live | `setup.tools[].function_declarations` | `toolCall.functionCalls[]` | `toolResponse.functionResponses[]` by `id` | `behavior: NON_BLOCKING`, `scheduling` `INTERRUPT` / `WHEN_IDLE` / `SILENT`. Google's table: 2.5 Flash Live sync and async, 3.1 Flash Live **sync only** ("the model will not start responding until you've sent the tool response") | `toolCallCancellation` |
| Nova Sonic | `promptStart` tool configuration | `toolUse` (`toolUseId`) | TOOL content block with `toolResult` | Documentation silent. **To measure** | Documentation silent. **To measure** |

One practitioner report says Nova Sonic gathers a tool's parameters and then, about a quarter of the
time, does not emit the call. That is a reliability question the column measures (§11), not a design
input.

### 1.3 Credentials

A browser cannot be given a provider key. Every provider but AWS has a short-lived credential for
exactly this case, and they differ in what the credential can pin:

| Provider | Short-lived credential | Can it lock the session's instructions and tools? |
|---|---|---|
| OpenAI | `POST /v1/realtime/client_secrets` (session config set at mint) | The mint sets the config. The documentation does not say a client is prevented from sending `session.update` later. **To measure** |
| Azure OpenAI | `POST /openai/v1/realtime/client_secrets` on the resource (the GA endpoint; the preview `sessions` endpoint is deprecated) | **No** (measured, §1.7). A client holding the secret replaced the tools and the instructions with `session.update`. |
| xAI | `POST /v1/realtime/client_secrets` with `expires_after` only; the mint "does not support" a `session` field | **No.** Nothing can be pinned at mint. |
| Gemini | `auth_tokens` (`expireTime` default 30 min, `newSessionExpireTime` default 1 min, `uses`), `liveConnectConstraints` and `lockAdditionalFields`; Gemini API `v1beta` only | Model and config can be locked. Whether the system instruction and tools are lockable is not stated. **To measure** |
| AWS Bedrock | STS session credentials scoped to `InvokeModelWithBidirectionalStream` on one model | Nothing is pinned. The caller signs every request, and there is no browser path. |

### 1.4 Who is on the connection

The person's device needs audio to and from the model. The harness needs to run tools, keep the
record and enforce the budget. Two topologies satisfy both:

- **Relay.** Device ↔ harness ↔ provider. The harness holds the only provider connection, and the
  device talks only to the harness. Every provider works this way. It costs one network hop on the
  audio path.
- **Direct with a sideband.** Device ↔ provider for audio; harness ↔ provider on a second
  connection to the same call, for tools and the record. OpenAI documents the sideband
  (`wss://api.openai.com/v1/realtime?call_id=…`, the call id from the `Location` header of the
  WebRTC answer, and `/v1/live/sessions/{id}/attach` for GPT-Live). Azure's documentation does not
  say. xAI, Gemini and Nova Sonic have no sideband: the one connection is the whole session.

Developer reports on the OpenAI sideband include connections dropping after a period of silence and a
`call_id` that answered 404. Both go into §1.6.

### 1.5 Usage and limits

The providers report usage per response, in tokens split by modality: OpenAI and Azure in
`response.done.usage`, Gemini in `usageMetadata`, Nova Sonic in usage events. Some voice offerings
are priced per minute rather than per token. The harness records what the provider reported and
fabricates nothing (§8).

Session length is capped by every provider, and the caps differ, per provider in §10. Azure OpenAI
holds a session for 60 minutes: its `session.expires_at` is the session's start plus 3,600 s (§1.7).

### 1.6 The measurements this section waits for

Each one is run on the provider's live service, from a server in the same region as the provider's
endpoint, with a recorded audio fixture (§11) so a run is repeatable:

1. **Tool round trip:** speech end → call emitted → result sent → first audio of the answer, in
   milliseconds, per provider, with one call, two parallel calls, and a call whose result arrives
   after 8 s.
2. **Barge-in during a call:** what the provider does with the pending call, and what the model says
   when the late result arrives.
3. **The relay hop:** first-audio latency through the relay versus direct, on the same provider, and
   the same region. The extra hop's cost is a number in this document, not an adjective.
4. **Sideband:** whether Azure accepts it, and whether OpenAI's still drops when the call goes silent.
5. **Locking:** whether a client holding a short-lived credential can change the session's
   instructions or tools after the mint, per provider.
6. **Usage:** the fields each provider reports, checked against the provider's own billing for the
   same session.
7. **Session caps:** the longest session each provider holds, and what it sends when the cap is
   reached.

### 1.7 Measured on Azure OpenAI (2026-10-27)

The first provider measured. The calls ran on `gpt-realtime-2.1` (a US data zone deployment), opened
with a short-lived client secret minted by a gateway that holds the resource key, from a test server
in the US. Each session was ended through the gateway's meter. Turns were sent as text input; the
barge-in clip is the model's own speech of one sentence, captured from an earlier answer and sent
back as the caller's audio (PCM16, 24 kHz, in 100 ms frames). The tools are two function tools the
client runs itself. Small samples: these numbers place the design, they are not a benchmark.

| What | Measured |
|---|---|
| Mint, then WebSocket open | 2.1 s to mint (the gateway's whole request), 0.7 s to open the socket. The client secret opens a **WebSocket** as a bearer token (`wss://<resource>/openai/v1/realtime?model=<deployment>`), not only WebRTC, so the relay (§1.4) can carry a per-session credential. |
| Ask to call emitted | 1.2 to 1.4 s from the question to `response.function_call_arguments.done` (3 calls). |
| Result to first audio | 330 to 470 ms from the `function_call_output` to the first audio delta; the answer finished 1.1 to 1.2 s after the result. |
| Two calls at once | Asked for two cities, the model emitted both calls in one response and answered both in one sentence, 1.9 s after the results. |
| A result 8 s late | Nothing happened while it waited: no event, no audio, no timeout. With the result, the model answered at once (0.8 s to the end of its answer). |
| Barge-in while the model speaks | Speech detected 0.1 to 0.65 s into the caller's audio. The model's answer was cut off in 2 of 3 trials, about 0.2 s in (`response.done` with status `cancelled`); in the third the answer was not cancelled. The caller's speech was transcribed each time. |
| The caller speaks while a call is pending | The server took the speech as a turn and the model answered it while the call was still open. The late `function_call_output` was then accepted without an error, and the next response completed. A pending call blocks nothing. |
| Through a relay hop | First audio after an ask: 1.0 to 1.4 s direct, 0.8 to 1.2 s through a one-hop relay on the same server (3 each, medians 1.33 s and 1.07 s). The hop's own cost is below this noise; what a relay adds is the network path to it, which placement (§9) decides. |
| Changing the call after the mint | `session.update` from the client replaced the tools (with none) and the instructions, and the server confirmed both (`session.updated`). A model change was confirmed and did not take effect: the session kept its model. So Azure, like OpenAI's mint, cannot lock a call against the person holding its secret; the relay topology (§1.4, §13 decision 1) is what keeps the configuration the owner's. |
| The secret's lifetime | Minted for 120 s, the secret still opened a session at 135 s. The expiry bounds starting a call only loosely; it is not a revocation. |
| Usage | `response.done.usage` per response: `input_tokens` and `output_tokens`, split into text, audio and image tokens, cached input (with its own split), and `reasoning_tokens` when the model reasoned. Checking these against Azure's own billing needs a voice-input session and is still to do. |
| Session cap | 60 minutes (`session.expires_at` = start + 3,600 s). What the server sends at the cap is still to measure. |
| Sideband | Not measurable from the client: attaching to a call takes the resource key. Azure's documentation still says nothing; this is measured from the upstream service (§7) when it exists. |

Every other provider in §10 is measured the same way before its row is shipped.

## 2. First principles: what a harness for a realtime model is

A realtime model already has the loop a coding harness provides. It decides when to speak, what to
say, and when to call a tool. A harness that re-implements that loop around the model would add
latency and remove nothing. So the harness owns what the model cannot own:

1. **The configuration, against the caller.** Instructions, voice, tools and budgets belong to
   the configured harness: its owner decides them once, and they hold for every session. The person
   on the call can narrow a session (mute, end it, cut the model off) but never widen it. This is
   UHP's rule ("narrowing is safe, widening is escalation", Tasks §1.4) applied to a connection that
   lasts twenty minutes instead of one request.
2. **Tool execution.** Every provider hands the call back to the caller (§1.2). The harness is that
   caller: it runs the call against the harness's own tools and returns the result. The person's
   device never runs a tool, and never sees a tool's credential.
3. **The record.** A voice conversation leaves nothing behind unless something writes it down. The
   harness turns it into a UHP session: one response per model turn, with the transcript and every
   tool call and result as output items, readable after the call through the same endpoints as any
   other session.
4. **The credential and the meter.** The provider key never reaches the device, or the harness's
   tools. Usage is measured where the provider reports it, and charged to the organization that
   started the session.
5. **Budgets.** A call can run for an hour. The harness enforces maximum duration, idle time and
   cost, and ends the call itself when one is reached: the model says goodbye, then the line closes.
   No open-ended spend.
6. **The provider, as a setting.** A product moves from one provider to another by changing the
   harness's model, not its client.

What it is not: a speech-to-text, text model, text-to-speech cascade (§13, decision 4); a telephony
carrier; or a media server.

**Latency is the product.** A person notices a pause of 800 ms. Every hop, every tool call and every
ten milliseconds of buffering has to justify its place. That is why §1.6 measures the relay hop
before the design commits to relay as the default, and why the tool executor runs next to the
provider connection, not behind another queue.

## 3. Leverage

The industry already has frameworks for this. The brief was to build on one if it is widely used
and fits, and to lead only where nobody does.

| Project | Stars (2026-10-27) | License | Realtime providers | Fit |
|---|---|---|---|---|
| pipecat-ai/pipecat | 16.8k | BSD-2-Clause | OpenAI Realtime and GPT-Live, Azure Realtime and Voice Live, Gemini Live (Gemini API and Vertex), Nova Sonic, Grok Voice, Inworld, Ultravox | Every target provider, in one framework that runs in our process, with no media server required. **Chosen.** |
| livekit/agents | 14.9k | Apache-2.0 | OpenAI Realtime first; Azure realtime and Gemini Live not first-class in its README | Production needs a LiveKit server (an SFU) in the path. |
| openai/openai-agents-python | 30k | MIT | OpenAI only | One provider. |
| TEN framework | 11.2k | unclear | several | License unclear. Not usable until it is clear. |

### 3.1 Pipecat, verified in its source

The brief's open question was whether Pipecat really runs function calls on every realtime service,
and whether it has MCP. Both were read in the v1.14.0 source (commit `82c4fdb`), service by service.
Nothing was run. Paths are relative to `src/pipecat/`.

**Every speech-to-speech service runs tools through one dispatch point.** Each service parses its
provider's call event into `FunctionCallFromLLM` and calls `LLMService.run_function_calls`
(`services/llm_service.py`). That builds `FunctionCallParams` (`function_name`, `tool_call_id`,
`arguments`, `result_callback`) and awaits the registered handler. The handler can answer later, so
async results are possible.

| Service | Tools sent | Call parsed from | Result returned as | Parallel | Cancel | Usage | Short-lived credential |
|---|---|---|---|---|---|---|---|
| OpenAI Realtime (`openai/realtime/llm.py`) | `session.update` | `response.function_call_arguments.done` | `conversation.item.create` `function_call_output`, then `response.create` | concurrent, not grouped | yes (`"CANCELLED"` returned) | full: audio in/out, cached, cached audio | no: an API key as Bearer |
| Azure Realtime (`azure/realtime/llm.py`, a subclass) | as OpenAI | as OpenAI | as OpenAI | as OpenAI | as OpenAI | full | **Entra token provider**, fetched on each connect |
| OpenAI GPT-Live (`openai/live/llm.py`) | responses delegation only, in `session.start` | `response.output_item.done` in `response.event` | `response.item.create`, then `response.create` once all are answered | waits for all | an interruption never cancels a call | seconds from the live model (logged only), tokens from the backend | no |
| Azure Voice Live (`azure/voice_live/llm.py`) | `session.update` | `…arguments.done` | only on a later context update | one at a time | `response.cancel` | totals only | Entra |
| Gemini Live (`google/gemini_live/llm.py`) | at connect only | `tool_call.function_calls` | `send_tool_response` | **one batch** | no provider-side cancel; a cancelled call gets no response | full, with audio and cached | no: an API key |
| Gemini Live on Vertex (`google/gemini_live/vertex/llm.py`) | as Gemini | as Gemini, call ids made locally | as Gemini, async switched off | yes | as Gemini | full | service account, refreshed once at startup |
| Nova Sonic (`aws/nova_sonic/llm.py`) | `promptStart` `toolConfiguration` | `toolUse` | TOOL `contentStart`, then `toolResult` | one at a time | nothing sent to the provider | totals only, no audio split | STS session token accepted, never refreshed |
| Grok Voice (`xai/realtime/llm.py`) | `session.update` (plus xAI-hosted MCP) | `…arguments.done` | `function_call_output`, then `response.create` | one at a time | yes | totals only | no |
| Inworld (`inworld/realtime/llm.py`) | `session.update` | `…arguments.done` | final results only | one at a time | yes | totals only | JWT bearer |
| Ultravox (`ultravox/llm.py`) | fixed at call creation | `client_tool_invocation` | `client_tool_result` | one at a time | partial | **none** | a `join_url` keeps the key outside |

What this means for the harness:

- **The hook is ours to own.** Pipecat has no approval hook: `on_function_calls_started` runs as a
  background task and cannot block. The harness subclasses each service it uses and overrides
  `run_function_calls`. Every call then passes through the harness's tool executor (§5.2) before
  any handler runs. The executor asks the server's approval gate where there is one, and it
  refuses, times out, records and answers. One override covers every provider.
- **An unknown tool name must never reach a service unanswered.** Nova Sonic raises
  `AWSNovaSonicUnhandledFunctionException` for a name with no handler, and its receive loop then
  resets the conversation. The harness registers a catch-all handler (`register_function(None, …)`;
  `has_function` is then true for every name), and that handler refuses the call (§5.3).
- **Provider-hosted tools are switched off.** Grok's xAI-hosted MCP calls never pass through the
  dispatch point, so the harness could neither record nor refuse them. The harness never configures
  them.
- **Usage must be switched on.** Pipecat emits usage only with `enable_usage_metrics=True`
  (`pipeline/worker.py`), and four services report totals without the audio split. That is one
  more reason the server's upstream is the authoritative meter (§8).
- **Pipecat never connects a device to a provider.** No service creates a client secret or an auth
  token, and every provider connection opens from the Pipecat process. That is the relay topology
  (§1.4), which the first release ships. Direct mode would be ours to build.
- **Gaps carried as measurements, not assumptions:** Gemini sends nothing back for a cancelled call;
  GPT-Live never cancels on interruption; OpenAI Realtime has a TODO on cancelled responses
  (`openai/realtime/llm.py:1092`); Azure Voice Live returns results only on a later context update.
  Each one is a row in §1.6 item 2.

**MCP.** `MCPClient` (`services/mcp_service.py`) speaks stdio, SSE and Streamable HTTP (with a bearer
header), and `await mcp.tools()` returns a provider-neutral tool schema that every service converts.
It keeps only the text parts of a result, and its filters (an allowlist, a post-call filter, fixed
arguments) cannot approve or refuse a call before it runs. So the harness keeps Pipecat's schema
conversion, but runs calls through its own executor on the `mcp` SDK. That executor is where
Harnesses §4.1 (disabled servers never contacted, unreachable servers don't fail the session), the
timeout and the record live.

**Packaging.** Python 3.11 or newer, and 3.12 for the Nova Sonic extra. Two core dependencies are
LGPL (`soxr`, `num2words`). They are used as unmodified libraries, which LGPL permits in an Apache-2.0
project, and the NOTICE file names them. The extras a realtime build needs are permissive:
google-genai and the Bedrock SDK are Apache-2.0, `mcp` and FastAPI are MIT, and aiortc is BSD-3.

### 3.2 What this harness builds

What none of them have:

- **Tool calls conformant to UHP:** tools come from the configured harness, never the client, and
  are recorded as UHP items (§5).
- **The UHP surface for a live session:** create, connect, follow, end and read back (§6).
- **A credential model where the key never leaves the server**, and a meter the harness cannot
  under-report (§7, §8).
- **One provider matrix, measured**, with a conformance report per provider (§10, §11).

## 4. Architecture

```
                    media path (audio)                          control path
┌────────┐  WebRTC / WebSocket / SIP  ┌──────────────────────────────────────┐   UHP (HTTP + SSE)   ┌────────┐
│ device │ ◀────────────────────────▶ │ Realtime Harness                     │ ◀──────────────────▶ │ client │
└────────┘                            │  transport ─ session ─ provider      │                      └────────┘
                                      │               │  │      adapter ─────┼──▶ provider (via the
                                      │   tool executor  recorder · meter    │     server's upstream)
                                      └──────┬───────────────────────────────┘
                                             │ MCP (the harness's servers)
                                             ▼
                                          tools
```

- **Transport.** WebRTC for browsers and apps, WebSocket for servers, telephony through a carrier's
  media stream. All three come from Pipecat: SmallWebRTC (aiortc, no media server), the FastAPI
  WebSocket transport, and the serializers for Twilio, Telnyx, Plivo, Exotel, Vonage and Genesys
  AudioHook. Daily and LiveKit transports exist too, for products that already run one.
- **Session.** One per call: the configured harness, the budgets, the state machine
  (`connecting` → `live` → `ending` → `ended`), and the clock.
- **Provider adapter.** One per wire family (§1.1), each a speech-to-speech service in Pipecat.
- **Tool executor.** Connects to the harness's enabled MCP servers when the session starts, offers
  their tools to the model, and runs each call (§5).
- **Recorder.** Writes the UHP session as the call happens, not at the end, so a dropped harness loses
  at most the turn in flight.
- **Meter.** Forwards the provider's usage reports. In a hosted server the authoritative meter is the
  server's upstream (§8), and the harness's own report is a cross-check.

The **client** (the product's backend) creates the session and follows it over UHP. The **device**
(the person's phone or browser) carries only audio and a few control events, and connects with a
short-lived token the client hands it. They are often different parties, which is why only the
client's credential can configure anything.

## 5. Tool calls, conformant to UHP

### 5.1 Where tools come from

From the configured harness, and nowhere else: its `mcpServers` (Harnesses §4.1), its plugins'
servers (Plugins §4), and minus its `disabledTools` (Harnesses §4.3). The rules carry over unchanged:

- A disabled server is never contacted.
- An unreachable server does not fail the session. The session starts without those tools and the
  record says which server was unreachable.
- A disabled tool is a hard block. The tool is not offered to the model, and a call naming it is
  refused (§5.3).

Request-level `tools` stay reserved and ignored, as in Tasks §1.4. In relay mode, the device's
control channel has no message that adds a tool or changes the instructions. In direct mode
(§1.4), the harness re-applies the configuration whenever it sees a `session.updated` it did not
send, and refuses any call to a tool it did not offer. Direct mode is allowed only for a provider
where §1.6 item 5 shows the mint can lock the configuration.

### 5.2 Running a call

1. The provider adapter turns the provider's call event into one shape: `call_id`, `name`,
   `arguments`.
2. The recorder writes a `function_call` item to the current response, and streams
   `response.output_item.added`, then `response.function_call_arguments.done`.
3. The tool executor runs the call on the MCP server that owns the tool, under the harness's
   per-call timeout (`toolTimeoutSeconds`, default 10 s, §13 decision 7: the provider waits on a
   call in complete silence, so the limit is how long the caller hears nothing). The harness adds
   one line to the instructions, to say so before using a tool, so the wait starts after the
   caller has heard why. Behind a server with an approval gate, the call goes through that gate exactly as
   a text turn's call does, and a held call answers at once (§5.3).
4. The result goes back to the provider in its own shape (§1.2), and the recorder writes the
   `function_call_output` item with the same `call_id`.

Parallel calls run in parallel. Each result is returned as it completes, where the provider takes it
that way; otherwise all results go back together, followed by one `response.create` (xAI, OpenAI).

### 5.3 Refusals and failures

The model always gets an answer to its call: a refusal, a timeout or a server error each come back
as the call's result, a short JSON object such as
`{"error": "timeout", "detail": "the tool did not answer within 10 seconds"}`. The model can then say
so. A call left unanswered stalls Gemini 3.1 Flash Live and GPT-Live's responses delegation
outright, and the others in practice. The recorded `function_call_output` carries the same object.

**A call that needs a person's approval is held, never waited on.** A server that gates some tool
calls behind a person's approval (sending mail outside an allowed list, a write to a connected app, a
paid run over a set amount) applies the same gate to a realtime session's calls as to a text turn's.
The gate answers at once with its own held result, such as
`{"status": "held", "approval_id": "…", "says": "…"}`. That is the call's result and the recorded
`function_call_output`, byte for byte what a text turn records. It is neither a refusal nor a wait.
The model tells the person the action is waiting for approval, and the conversation goes on. The
person approves later, wherever they approve today, and the gate then runs the action as it does for
a text turn. If the call is still live when the approval resolves, the harness may pass the outcome
to the model as unspoken context, the async path of §5.4. Otherwise, the approval surface and the
record carry it. Nothing on the call waits synchronously for a person, and voice keeps parity with
text: whatever the harness can do in a text turn, it can do on a call, under the same gate.

### 5.4 Barge-in, cancellation, slow tools

When the person speaks over the model while a call is running:

- Where the provider cancels pending calls (Gemini's `toolCallCancellation`), the harness cancels
  the MCP request, and records the output as `{"error": "cancelled"}`.
- Where it does not, the call finishes, and its result is returned if the provider still takes it.
  It is always recorded. Azure OpenAI does not cancel: it answers the caller while the call is open,
  and takes the late result afterwards without an error (§1.7).

A tool expected to be slow can be marked `async` on the harness (§13, decision 3). The model is then
told at once that the work has started, and the result is injected when it arrives (Gemini
`NON_BLOCKING` with `WHEN_IDLE`, an OpenAI-family conversation item, a GPT-Live
`session.thinking.append`).

### 5.5 Delegating to another harness

GPT-Live's client delegation (§1.1) names the pattern most products want from a voice agent: a fast
voice in front, and real work behind. HarnessRouter already has the "real work" side: every other
base. A Realtime Harness can name a **delegate**, another configured harness on the same server.
The model then gets one tool, `delegate(task)`. Calling it starts a UHP task on the delegate as a
continuation of one session, streams the delegate's progress back as unspoken context, and returns
its final message as the call's result. On GPT-Live this maps onto client delegation natively. On
every other provider it is an ordinary function call. It is the same rule as §5.1: the delegate is
part of the configuration, never chosen by the caller.

This is second-release work (§12). It is in the design now so the first release's record and tool
executor leave room for it.

## 6. UHP: a Realtime sub-protocol

### 6.1 The fit

UHP's object model already holds a conversation: a **session** is a chain of **responses**. A
realtime call is a session whose responses are created by speech instead of by `POST /v1/responses`.
So the extension adds a way to *start* a session that is live, a way to *connect* the audio, and a
few event types. It reuses everything else:

| UHP today | In a realtime session |
|---|---|
| Session (`hsess…`) | The call. Created by `POST /v1/realtime/sessions`, and listed, inspected, shared and deleted as any other session. |
| Response (`resp_…`) | One model turn. `input` holds what the person said (transcribed), and `output` holds the turn's `function_call` and `function_call_output` items and a `message` with the transcript of what the model said. Turns are chained by `previous_response_id` as usual. |
| `status` | `completed`, `incomplete` (a budget ended the turn), `failed`, and `cancelled` (barged in, or hung up mid-turn). |
| `usage` | The provider's per-turn usage, with optional `input_tokens_details` / `output_tokens_details` (`audio_tokens`, `text_tokens`, `cached_tokens`). `null` when the provider reported none. |
| `POST /v1/sessions/{id}/cancel` | Hangs up. The session stays readable and continuable (as a new call, or as text turns). |
| Streaming vocabulary | Unchanged per turn: `response.created` … `response.completed`, with output item and function-call events. |

### 6.2 What is new

1. **Discovery.** A capability `realtime` (a boolean, `false` when absent), and
   `realtime_transports` (`["webrtc", "websocket"]`).
2. **`POST /v1/realtime/sessions`.** Body: `model`, `metadata.harness_id`, optional `voice` (when the
   harness allows a choice), `transport`, `previous_response_id` (start the call with an earlier
   session's context), and `metadata`. Returns the session with
   `connect: { transport, url, token, expires_at }`. The `token` is good for one connection, within
   about a minute. The client passes it to the device.
3. **Connecting.** WebRTC: the device POSTs its SDP offer to `connect.url` with the token, and gets
   the answer back (the WHIP pattern). Audio is Opus, and a data channel named `uhp` carries events.
   WebSocket: one socket at `connect.url`, PCM16 frames, with JSON events in the same vocabulary.
4. **The session's event stream.** `GET /v1/sessions/{id}/events` (SSE, one `sequence_number`
   across the whole call), and the same events on the data channel. Per turn: the existing
   vocabulary. Per call, new and optional: `realtime.session.live`, `realtime.input_transcript.delta`
   / `.done`, `realtime.output_transcript.delta` / `.done`, `realtime.speech_started` (barge-in),
   `realtime.session.ended` (the terminal event, with `reason`: `hangup`, `budget`, `idle`,
   `provider_closed`, `error`).
5. **Device events.** These narrow only: `realtime.input.mute`, `realtime.input.unmute`,
   `realtime.response.cancel`, `realtime.session.end`. The server ignores any other event from the
   device, and counts it in the record.

### 6.3 Checked against VERSIONING

Everything above is additive. It adds new endpoints, a capability that is `false` for every
existing server, new event types, which every conformant client already ignores (Client rules 2),
and optional fields on `usage`. Nothing is renamed, retyped, removed or made required, and a client
that has never heard of realtime reads a realtime session as a session of text turns. A new chapter
is still a new version, as Plugins was in `2026-10-04`, so this is proposed as the optional
**Realtime** chapter of the next published version. That version keeps serving `2026-10-04` from
the same code path.

The event names avoid the provider's names on purpose (`realtime.session.live`, not
`session.created`), so a client that also speaks a provider's wire never confuses the two.

### 6.4 Text turns on the same base

A realtime model can also take a text turn. So the `realtime` base answers `POST /v1/responses` as
any other base does, and a realtime session can be continued in text. That makes the base
conformant at Core through the existing suite, with no realtime knowledge in the suite. The
realtime checks are added beside it (§11).

## 7. Credentials and the provider connection

- **Self-hosted, standalone.** The harness reads the provider key from its environment. "Your keys,
  your infrastructure", as for every HarnessRouter base.
- **Behind a HarnessRouter server.** The harness never holds the key. It opens the provider
  connection through the server's **upstream**: a WebSocket endpoint on the server (the same pattern
  as the loopback relay other bases use for HTTP). The harness authenticates to it with a credential
  scoped to this one session and revocable alone (Security §1.1). The server attaches the provider
  credential, or mints the provider's short-lived one, and forwards frames both ways. Pipecat's
  OpenAI, Azure and xAI services take a `base_url`, and its Gemini service takes `http_options`, so
  they reach the upstream unmodified. Its Nova Sonic service has no endpoint setting: it signs to
  Bedrock through the AWS SDK. There, the upstream speaks SigV4 HTTP/2 to Bedrock, and the harness
  needs a small adapter that sends Nova Sonic's events over the upstream's WebSocket. That adapter
  is measured before Nova Sonic is offered behind a server.
- **Direct mode** (where allowed, §5.1): the server mints the provider's short-lived credential with
  the configuration locked, the device connects to the provider, and the harness attaches the
  sideband through the upstream.

The upstream validates every target against the server's configured provider list. A caller never
names an upstream URL (Security §1.1).

**The upstream is its own deployment.** A server's request-serving replicas roll often, and a roll
waits only for the few turns in flight. Calls held open on those replicas would either block every
roll for up to an hour, or be cut by it. So the upstream is a small service of its own, built from
the server's code so it shares the server's credential broker, provider list and meter, and it stays
inside the server's trust boundary so the meter remains authoritative. It has its own release cycle
and its own drain (§9).

## 8. Metering and billing

- **Who pays:** the organization of the principal that created the session, exactly as for a task.
  The device's token carries no billing identity of its own.
- **What is counted:** the provider's usage reports, read by the server's upstream as they pass
  through it. The harness forwards the same reports in the record, and the two are compared at the
  end of the call: a difference is an alert, never a silent correction. Per-minute pricing, where a
  provider uses it, is counted from the upstream's own connection time.
- **Before the call:** the server's credit gate runs at `POST /v1/realtime/sessions`, as it does
  before a task.
- **During the call:** the server tracks the running cost. When the organization's balance, or the
  harness's `maxCostPerSession`, would be crossed, the session ends gracefully with
  `reason: "budget"`. A call that cannot be paid for does not continue.
- **Never fabricated:** a turn the provider reported no usage for records `usage: null`, never zero
  (Tasks §3).

## 9. Operations

A realtime call is a socket held for minutes. A server that runs tasks of seconds needs four things
it did not:

1. **Drain, don't drop.** A replica of the harness or the upstream (§7) that is being replaced stops
   taking new sessions, and keeps its live ones until they end or reach their maximum duration. A
   deployment counts a live call as a live turn, and waits for it. The platform's termination grace
   period caps how long a drain can hold a call. So the harness's `maxCallDuration` is set below that
   grace period, or, where a call must outlast it, the harness records a planned resume seam at the
   cap (item 2) rather than an unplanned cut.
2. **Resume.** When the device's connection drops, it reconnects to the same session within a grace
   window (default 20 s), with a resume token from the original `connect`. When the provider
   connection drops, the harness resumes where the provider supports it (Gemini's
   `sessionResumption` handle). Otherwise it opens a new provider session seeded with the transcript
   so far, and records the seam.
3. **Fail fast at connect.** A device connect that gets no answer within 5 s retries once, then
   reports the failure. An ingress with no ready backend can hold a connection for 30 s, and on a
   voice call that reads as a dead line.
4. **Placement.** The harness runs in the same region as the provider endpoint, and the upstream on
   the same network as the harness. The relay hop's measured cost (§1.6 item 3) decides whether this
   is enough.

## 10. Providers and the model matrix

Model ids are confirmed from each provider's model list at measurement, not copied from examples.

| Provider | Wire | Transports to the harness | Tools | Async tools | Short-lived credential | Sideband | Status |
|---|---|---|---|---|---|---|---|
| OpenAI (gpt-realtime family) | OpenAI Realtime | WebSocket | yes | output can follow | yes | documented | docs read; **to measure** |
| OpenAI GPT-Live | GPT-Live | WebSocket | delegation | yes | yes | documented | docs read; **to measure** |
| Azure OpenAI (gpt-realtime deployment) | OpenAI Realtime | WebSocket | yes | as OpenAI | yes | not documented | first to measure |
| Gemini Live (Gemini API; Vertex AI) | Gemini Live | WebSocket | yes | 2.5 Flash Live only | Gemini API only | no | docs read; **to measure** |
| Nova Sonic (Bedrock) | Nova Sonic | HTTP/2 SigV4 | yes | undocumented | STS | no | docs read; **to measure** |
| xAI Grok Voice | OpenAI Realtime (compatible) | WebSocket | yes | not documented | yes, cannot lock | no | docs read; **to measure** |
| Azure Voice Live | Azure Voice Live | WebSocket | results only on a later context update (§3.1) | errors on intermediate results | Entra | no | after the first release |
| Inworld | OpenAI Realtime (variant) | WebSocket | yes | errors on async | JWT | no | after the first release |
| Ultravox | Ultravox | its own call (`join_url`) | fixed at call creation | not documented | `join_url` | no | after the first release; reports no usage |

## 11. Conformance

**Core**, with the existing suite, on text turns (§6.4). It runs unchanged, and the report is
published in the README as System One Harness publishes its own.

**Realtime**, as new checks in the suite, under a `realtime` class that requires Core. They run on
the WebSocket transport, with recorded audio fixtures: short WAV files of a synthetic voice asking a
fixed question, so a run needs no person and gives the same input every time. For example, "What is
the weather in Paris? Use the weather tool." against the suite's MCP fixture, which gains a
deterministic `weather` tool.

| Id | Check |
|---|---|
| R-01 | Discovery advertises `realtime` and its transports, consistently with the class claimed. |
| R-02 | `POST /v1/realtime/sessions` returns a session with a `connect` block, and an unknown harness gets `404 harness_not_found`. |
| R-03 | The token connects once, then is refused, and is refused after `expires_at`. |
| R-04 | The fixture's question produces a turn whose record has the transcript in `input`, and a `message` in `output`. |
| R-05 | The weather question produces a `function_call` and a `function_call_output` with the same `call_id`, in the record and in the stream, and the answer uses the tool's value. |
| R-06 | A tool on a disabled MCP server is never offered, and that server receives no connection. |
| R-07 | A device event that tries to add a tool or change the instructions has no effect. |
| R-08 | `POST /v1/sessions/{id}/cancel` ends the call with `realtime.session.ended`, and the record stays readable. |
| R-09 | `sequence_number` is gapless from first event to last, and exactly one terminal event ends the stream. |
| R-10 | `usage` on each turn is either the provider's numbers or `null`, never zero for a turn that produced audio. |
| R-11 | A device connection dropped and resumed within the grace window continues the same session. |

The column (§10) runs these checks per provider and model, and the README carries the matrix with
the suite's own output, never an adjective.

## 12. The first release

**In:**

- The Python package `realtime_harness` (Python 3.11+, 3.12 with Nova Sonic), on a pinned Pipecat
  release for transports and provider services, with the session, the tool executor (on the `mcp`
  SDK, for Streamable HTTP and SSE), the recorder, the meter and the budget guard.
- Providers: Azure OpenAI and OpenAI (gpt-realtime), and Gemini Live. Then Nova Sonic and Grok Voice,
  each added when its row in §10 is measured.
- Transports: WebSocket and WebRTC (the device side), relay topology only.
- The UHP server (`rth serve`): Core on text turns, and the Realtime chapter (§6) as proposed.
- A browser example (one page: connect, talk, see the transcript and the tool calls live) and a
  WebSocket example that plays the fixtures.
- The conformance report at Core and the realtime checks, per provider.
- Unit tests on recorded provider event streams, so the tool path, the record and every terminal
  state are tested without a key.

**Out, and why:** direct mode (needs §1.6 items 4 and 5 measured), delegation to another harness
(§5.5, needs the record in production first), telephony (a carrier's media stream is a transport,
added once the WebSocket transport is measured), cascades (§13, decision 4), and GPT-Live (its
delegation model is the second release's subject).

**Naming:** the repository `RealtimeHarness`, the package `realtime_harness`, the CLI `rth`, and the
HarnessRouter base `realtime`.

## 13. Decisions

Decided in review on 2026-10-27:

1. **Relay only in the first release.** It works for every provider and keeps one code path, and
   §1.6 will say what direct mode would save.
2. **Placement behind a hosted server: a realtime plane**, not the per-task sandbox runner. It is a
   service of harness processes in the same region as the provider endpoint, with many sessions per
   process and no sandbox. A call is long-lived and runs no shell, and a sandbox per call would add
   startup time to every call. A harness whose tools need a sandbox (a local MCP server in a
   plugin) is refused for this base in the first release. The upstream is a separate service (§7).
3. **Async tools are marked by the owner** (`asyncTools` on the harness), never inferred. The owner
   knows which tool is slow; a guess would make the model announce work that finishes instantly.
4. **Cascades** (speech-to-text, any HarnessRouter text model, text-to-speech) are a separate base,
   after this one. Pipecat does them well, and they would put every text model behind a voice, but
   they are a different product with different latency.
5. **Approvals: the same gate as in a text turn, answered at once** (§5.3). An approval-gated tool
   is offered in a realtime session exactly as in a text turn. Leaving it out would make the voice
   agent say it cannot do what the same harness does in text. Spoken confirmation before a call
   ("Shall I send it?") remains the model's job, prompted by the instructions.
6. **Providers.** The first release ships Azure OpenAI, OpenAI and Gemini Live. Nova Sonic and Grok
   Voice follow, each when its row in §10 is measured. GPT-Live and delegation to another harness
   (§5.5) are the second release.

7. **The default `toolTimeoutSeconds` is 10 s** (decided 2026-10-27 from §1.7). The provider waits
   on a call in silence, without a limit of its own, and a pending call blocks nothing: the caller
   can speak and the late result is still taken. So the limit bounds only the dead air before the
   model can say what happened, and a shorter one costs nothing in correctness. With the one-line
   instruction to say so before a tool (§5.2), the caller hears why before the wait starts. A
   harness's owner raises it per tool, or marks a slow tool `async` (decision 3).

## 14. Sources

- OpenAI: Realtime server controls (sideband), GPT-Live delegation, Realtime with WebRTC:
  https://developers.openai.com/api/docs/guides/realtime-server-controls,
  https://developers.openai.com/api/docs/guides/live-delegation,
  https://developers.openai.com/api/docs/guides/realtime-webrtc
- Microsoft: Realtime audio over WebRTC on Azure OpenAI:
  https://learn.microsoft.com/azure/foundry/openai/how-to/realtime-audio-webrtc
- Google: Live API tools, ephemeral tokens: https://ai.google.dev/gemini-api/docs/live-tools,
  https://ai.google.dev/gemini-api/docs/ephemeral-tokens
- xAI: ephemeral tokens: https://docs.x.ai/developers/model-capabilities/audio/ephemeral-tokens
- AWS: Nova Sonic input and output events, tool use:
  https://docs.aws.amazon.com/nova/latest/userguide/output-events.html,
  https://docs.aws.amazon.com/nova/latest/userguide/speech-tools-use.html, and the Nova 2 Sonic
  tool configuration page
- Pipecat v1.14.0 source (§3), read file by file.
- Unified Harness Protocol `2026-10-04`: Architecture, Lifecycle, Tasks, Streaming, Sessions,
  Harnesses, Plugins, Security, VERSIONING.

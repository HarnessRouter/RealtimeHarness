<h1 align="center">
  Realtime Harness
  <br>
  <sub>The harness for realtime speech models.</sub>
</h1>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-285AFF?style=flat&amp;labelColor=444c56" alt="License: Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Version-0.1.0.dev0-285AFF?style=flat&amp;labelColor=444c56" alt="Version 0.1.0.dev0"></a>
  <a href="docs/reports/uhp-conformance-core-2026-10-27.json"><img src="https://img.shields.io/badge/UHP%202026--10--04-Core-16824B?style=flat&amp;labelColor=444c56" alt="UHP 2026-10-04 conformance: Core"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-285AFF?style=flat&amp;labelColor=444c56" alt="Python 3.11 or newer"></a>
</p>

**Put a realtime speech model behind your product, with your tools and a full record of every call.**
A realtime model already runs its own conversation: it listens, decides when to speak, and calls functions.
What it cannot do is decide who configures it, run your tools, keep a record, or hold the credential and the
meter. Realtime Harness does those four things, and serves the result over the
[Unified Harness Protocol](https://unifiedharnessprotocol.org) (UHP), the same API as every other harness.

- **The configuration is yours, not the caller's.** Instructions, voice and tools are set by the harness's
  owner when the call opens. The person on the call can mute, cut the model off and hang up. Nothing they
  send can add a tool or change the instructions.
- **Every tool call runs here.** The model's function calls are run on the harness's MCP servers and answered,
  each within the harness's timeout. A disabled server is never contacted and a disabled tool is never run.
- **Every call is a record.** Each model turn is a UHP response, written while the call happens: what the person
  said (transcribed), what the model said, each tool call and its result, and the provider's usage.
- **The key stays on the server.** Standalone, the harness reads a provider key from its environment. Behind a
  HarnessRouter server, it holds no provider key: each connection uses a short-lived credential the server mints
  for one call and meters.

The design, with the measurements it rests on, is [docs/design.md](docs/design.md).

## Status

This is the first slice, not a release.

| | |
|---|---|
| Provider | Azure OpenAI (`gpt-realtime-2.1`), on the OpenAI realtime wire. OpenAI direct uses the same path and has not been measured yet. Gemini Live, Nova Sonic and Grok Voice follow, each when it is measured (design §10). |
| Transport | WebSocket (PCM16 audio and JSON events). WebRTC is next. |
| UHP | **Core, 40/40** on text turns ([report](docs/reports/uhp-conformance-core-2026-10-27.json)). The Realtime chapter is implemented as proposed in design §6 and is not yet part of a published UHP version. |
| Tools | MCP servers over Streamable HTTP or SSE. |

## Quickstart

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[test]"
```

A harness file says what every call runs with:

```yaml
# harness.yaml
id: weather-desk
name: Weather desk
instructions: You are a helpful assistant. Use the weather tool for weather questions. Keep your answers short.
voice: marin
tool_timeout_seconds: 10
max_call_minutes: 20
mcp_servers:
  - name: weather
    url: http://127.0.0.1:18933/mcp
```

Serve it, with a provider key from the environment (or a HarnessRouter server's mint, below):

```bash
export RTH_API_KEYS=$(python3 -c "import secrets; print(secrets.token_hex(16))")   # the keys your clients use
export AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com AZURE_OPENAI_API_KEY=...
python tests/fixture_mcp.py 18933 &               # a test MCP server with a weather tool
rth serve --config harness.yaml --port 8790
```

Then make a call with the example device, which plays a recorded question and prints the call's record:

```bash
RTH_URL=http://127.0.0.1:8790 RTH_KEY=$RTH_API_KEYS python examples/device_ws.py
```

```
turn resp_… completed: input='What is the weather in Paris? Use the weather tool.'
  output=[message 'Let me check the current conditions for Paris.', function_call weather,
          function_call_output {"city": "Paris", "sky": "clear", "celsius": 18},
          message 'Paris is clear and about 18 degrees Celsius.']
```

### Behind a HarnessRouter server

The harness holds no provider key. For each connection it asks the server's realtime mint for a short-lived
credential, scoped to one call and billed to the caller's organization:

```bash
export RTH_SERVER_URL=https://<your-server>/api/harness RTH_SERVER_HARNESS=<the server's harness id>
export RTH_SERVER_TOKEN_FILE=<a file holding the server's API token>
```

## The API

Everything a UHP Core server serves (`/v1/uhp`, `/v1/harnesses`, `/v1/models`, `/v1/responses` in blocking,
streaming and background modes, sessions by `previous_response_id`, cancellation). A text turn runs on the same
realtime model, with the same tools.

A call adds four endpoints (design §6.2):

| | |
|---|---|
| `POST /v1/realtime/sessions` | Makes a call. Returns the session and `connect: {transport, url, token, expires_at}`. The token connects once, within a minute. With `previous_response_id`, the call continues that session and starts from what was said in it. |
| `WS /v1/realtime/connect?token=…` | The device's connection. Binary messages are PCM16 mono audio at 24 kHz, both ways. Text messages are JSON events; the first is `realtime.session.resume_token`. |
| `GET /v1/sessions/{id}/events` | The call's events (SSE), with one `sequence_number` from 0 across the whole call, ending with `realtime.session.ended`. |
| `POST /v1/sessions/{id}/cancel` | Hangs up. The session and its turns stay readable (`GET /v1/sessions/{id}`, `GET /v1/responses/{id}`). |

Events, to the client and to the device: `realtime.session.live`, `realtime.session.paused` and `.resumed`, `realtime.speech_started`,
`realtime.input_transcript.done`, `realtime.output_transcript.delta` and `.done`, per turn `response.created`,
`response.output_item.added` and `.done`, `response.completed` (or `response.failed` with status `cancelled`
after a barge-in), and `realtime.session.ended` with its `reason`: `hangup`, `budget`, `device_lost`,
`provider_closed` or `error`. The device additionally gets `realtime.output_audio.cleared` when the model is cut off, so it drops
the audio it has not played.

From the device, four events are honoured: `realtime.input.mute`, `realtime.input.unmute`,
`realtime.response.cancel` and `realtime.session.end`. Any other event is ignored and counted in the record.

**A dropped device comes back to the same call.** Closing the socket with code 1000 hangs up. Any other drop
pauses the call for 20 seconds: the provider connection is held, so the conversation is intact. The device
reconnects at the same URL with its resume token in place of the connect token, and gets a new one. What the
model said meanwhile is in the record, between `realtime.session.paused` and `realtime.session.resumed`. A
device that does not come back ends the call with reason `device_lost`.

**A call and the text around it are one session.** A text turn can continue from a call's turn, and a call
can start from any turn, with what each side said as the model's context.

**Speaking over the model does not cancel a tool.** The model answers the caller while the tool runs, and the
tool's result is recorded in the turn where the model takes it.

## Measured, not implied

[Design §1.7](docs/design.md) holds the measurements this slice is built on, taken on a live Azure OpenAI
realtime deployment through a server's short-lived credential: the tool round trip (330 to 470 ms from a tool's
result to the model's first audio), barge-in, what a client can change after the credential is minted (the
tools and the instructions, which is why the harness keeps the provider connection itself), the credential's
lifetime, the session cap (60 minutes) and the usage the provider reports. The default tool timeout, 10 s,
was set from them.

## Test

```bash
.venv/bin/python -m pytest -q
```

The unit tests need no provider key: the tool executor runs against a real MCP server
([tests/fixture_mcp.py](tests/fixture_mcp.py)), and the call's record is tested on the events a call produces.

[tests/live/call_checks.py](tests/live/call_checks.py) checks a running server with a provider, one short call
each: a device that drops and comes back, a call that continues a text turn, and a caller who speaks while a
slow tool runs. The questions are recorded speech in [tests/fixtures](tests/fixtures).

## Built on

[Pipecat](https://github.com/pipecat-ai/pipecat) for the transport and the provider's realtime service, and the
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) for tools.

## License

Apache-2.0. See [LICENSE](LICENSE).

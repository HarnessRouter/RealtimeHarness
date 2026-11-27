"""A call placed by the server this harness works for (design §7): with an operator key, the request that
creates a call carries the call's configuration and its own way to the provider. From any other caller the
same fields are ignored and named, so a call's configuration stays its owner's (design §2.1)."""
import asyncio
import contextlib
import json

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from conftest import free_port
from fake_provider import FakeRealtime
from realtime_harness import credentials as creds_mod
from realtime_harness import uhp
from realtime_harness.config import HarnessConfig
from realtime_harness.credentials import Connection, CredentialError, Credentials, ServerCall
from realtime_harness.uhp import Server, create_app
from test_call_e2e import Phone

KEY, OP = "caller-key", "operator-key"
H, HOP = {"Authorization": f"Bearer {KEY}"}, {"Authorization": f"Bearer {OP}"}
GIVEN = {"id": "chrn_neo", "name": "Neo", "instructions": "You are Neo.", "voice": "cedar", "max_call_minutes": 7,
         "mcp_servers": [{"name": "plugs", "url": "https://gw.example/v1/mcp/plugs",
                          "headers": {"Authorization": "Bearer hrs_call"}}]}
UPSTREAM = {"url": "https://gw.example/v1/realtime/sessions/hrrt_1/upstream", "token": "call-token"}


class OwnProvider(Credentials):
    async def connect(self, model: str) -> Connection:
        raise AssertionError("the server's own credentials are not this call's")


def _app(harnesses=None, credentials=None, keys=frozenset({KEY})):
    server = Server([HarnessConfig(id="desk", name="Desk", instructions="You are the desk.")] if harnesses is None else harnesses,
                    credentials, set(keys), operator_keys={OP})
    return server, TestClient(create_app(server))


# ── who may hand a call its configuration ───────────────────────────────────────────────────────
def test_an_operators_call_runs_on_the_harness_and_the_upstream_it_was_handed():
    server, c = _app(credentials=OwnProvider())
    r = c.post("/v1/realtime/sessions", headers=HOP, json={"harness": GIVEN, "upstream": UPSTREAM})
    assert r.status_code == 201, r.text
    s = r.json()
    assert s["harness_id"] == "chrn_neo" and "metadata" not in s          # nothing was ignored
    call = server.calls[s["id"]]
    h = call.session.harness
    assert (h.instructions, h.voice, h.max_call_minutes) == ("You are Neo.", "cedar", 7.0)
    assert h.mcp_servers[0].url == GIVEN["mcp_servers"][0]["url"]
    assert h.mcp_servers[0].headers == {"Authorization": "Bearer hrs_call"}
    assert isinstance(call.credentials, ServerCall) and call.credentials.url == UPSTREAM["url"]
    assert "chrn_neo" not in server.harnesses                            # this call's, not the server's from now on
    assert "call-token" not in json.dumps(s) and "hrs_call" not in json.dumps(s)


def test_any_other_callers_harness_and_upstream_are_ignored_and_named():
    own = OwnProvider()
    server, c = _app(credentials=own)
    r = c.post("/v1/realtime/sessions", headers=H, json={"harness": GIVEN, "upstream": UPSTREAM})
    assert r.status_code == 201
    s = r.json()
    assert s["metadata"]["ignored_fields"] == ["harness", "upstream"]
    call = server.calls[s["id"]]
    assert call.session.harness.id == "desk" and call.session.harness.instructions == "You are the desk."
    assert call.credentials is own


def test_an_operator_key_is_an_api_key_too_and_a_wrong_key_is_no_operator():
    _, c = _app(credentials=OwnProvider())
    assert c.get("/v1/harnesses", headers=HOP).status_code == 200
    assert c.post("/v1/realtime/sessions", headers={"Authorization": "Bearer nope"},
                  json={"harness": GIVEN, "upstream": UPSTREAM}).status_code == 401


def test_a_handed_header_is_taken_as_written_never_read_from_this_process(monkeypatch):
    monkeypatch.setenv("RTH_TEST_SECRET", "from-the-environment")
    server, c = _app(credentials=OwnProvider())
    given = {**GIVEN, "mcp_servers": [{"name": "x", "url": "https://elsewhere.example/mcp",
                                       "headers": {"Authorization": "Bearer ${RTH_TEST_SECRET}"}}]}
    s = c.post("/v1/realtime/sessions", headers=HOP, json={"harness": given, "upstream": UPSTREAM}).json()
    assert server.calls[s["id"]].session.harness.mcp_servers[0].headers == {"Authorization": "Bearer ${RTH_TEST_SECRET}"}
    # ...while a harness FILE's author may name the environment, as before
    assert HarnessConfig.from_dict(given).mcp_servers[0].headers == {"Authorization": "Bearer from-the-environment"}


@pytest.mark.parametrize("body, param", [
    ({"harness": "neo", "upstream": UPSTREAM}, "harness"),
    ({"harness": {"name": "no id"}, "upstream": UPSTREAM}, "harness"),
    ({"harness": {**GIVEN, "max_call_minutes": -1}, "upstream": UPSTREAM}, "harness"),
    ({"harness": {**GIVEN, "mcp_servers": [{"name": "x"}]}, "upstream": UPSTREAM}, "harness"),
    ({"harness": GIVEN, "upstream": "https://gw.example"}, "upstream"),
    ({"harness": GIVEN, "upstream": {"url": "ftp://gw.example/x", "token": "t"}}, "upstream"),
    ({"harness": GIVEN, "upstream": {"url": UPSTREAM["url"]}}, "upstream"),
], ids=["not-an-object", "no-id", "negative-minutes", "server-without-url", "upstream-not-an-object", "not-http", "no-token"])
def test_a_handed_configuration_that_cannot_be_used_is_refused_by_name(body, param):
    server, c = _app(credentials=OwnProvider())
    r = c.post("/v1/realtime/sessions", headers=HOP, json=body)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_input" and r.json()["error"]["param"] == param
    assert server.calls == {}


def test_a_call_that_continues_a_session_keeps_that_sessions_harness():
    _, c = _app(credentials=OwnProvider())
    r = c.post("/v1/realtime/sessions", headers=HOP, json={"harness": GIVEN, "upstream": UPSTREAM,
                                                           "previous_response_id": "resp_x"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "harness"


def test_a_server_with_an_operator_key_needs_no_harness_and_no_provider_of_its_own():
    server, c = _app(harnesses=[], credentials=None, keys=frozenset())
    assert c.post("/v1/realtime/sessions", headers=HOP, json={"harness": GIVEN, "upstream": UPSTREAM}).status_code == 201
    r = c.post("/v1/realtime/sessions", headers=HOP, json={"upstream": UPSTREAM})           # no harness to fall back on
    assert r.status_code == 404 and r.json()["error"]["code"] == "harness_not_found"
    r = c.post("/v1/realtime/sessions", headers=HOP, json={"harness": GIVEN})               # no way to the provider
    assert r.status_code == 503 and r.json()["error"]["code"] == "server_unavailable"
    with pytest.raises(ValueError):
        Server([], None, {KEY})                                                             # as before, without one


# ── the call's own way to the provider ──────────────────────────────────────────────────────────
def _upstream(monkeypatch, handler):
    seen: list[httpx.Request] = []

    def wrapped(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)
    real = httpx.AsyncClient
    monkeypatch.setattr(creds_mod.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(wrapped), **kw))
    return seen


def test_each_connection_asks_the_server_with_the_calls_token_and_opens_what_it_answers(monkeypatch):
    seen = _upstream(monkeypatch, lambda r: httpx.Response(200, json={
        "client_secret": "ek_once", "webrtc_url": "https://res.openai.azure.com/openai/v1/realtime/calls",
        "deployment": "gpt-realtime-2-1"}))
    up = ServerCall(UPSTREAM["url"], " call-token ")
    conn = asyncio.run(up.connect("gpt-realtime-2.1"))
    (req,) = seen
    assert str(req.url) == UPSTREAM["url"] and req.method == "POST"
    assert req.headers["authorization"] == "Bearer call-token" and json.loads(req.content) == {"model": "gpt-realtime-2.1"}
    assert conn.url == "wss://res.openai.azure.com/openai/v1/realtime?model=gpt-realtime-2-1"
    assert conn.headers == {"Authorization": "Bearer ek_once"} and conn.model == "gpt-realtime-2-1"
    assert "ek_once" not in json.dumps(conn.ref)
    asyncio.run(up.release(conn))                    # ending the call is the server's: nothing is asked
    assert len(seen) == 1


@pytest.mark.parametrize("answer, says", [
    (httpx.Response(403, json={"error": {"code": "call_ended", "message": "This call has ended."}}), "HTTP 403, call_ended This call has ended."),
    (httpx.Response(502, text="bad gateway"), "HTTP 502"),
    (httpx.Response(200, json={"webrtc_url": "https://res.openai.azure.com/x", "deployment": "d"}), "without a usable credential"),
    (httpx.Response(200, text="not json"), "without a usable credential"),
], ids=["refused", "not-json-error", "no-secret", "not-json"])
def test_an_upstream_that_does_not_mint_is_an_error_in_words_without_the_token(monkeypatch, answer, says):
    _upstream(monkeypatch, lambda r: answer)
    with pytest.raises(CredentialError) as e:
        asyncio.run(ServerCall(UPSTREAM["url"], "call-token").connect("m"))
    assert says in str(e.value) and "call-token" not in str(e.value)


def test_an_upstream_that_cannot_be_reached_is_an_error_in_words(monkeypatch):
    def down(request):
        raise httpx.ConnectError("no route")
    _upstream(monkeypatch, down)
    with pytest.raises(CredentialError) as e:
        asyncio.run(ServerCall(UPSTREAM["url"], "call-token").connect("m"))
    assert "could not be reached (ConnectError)" in str(e.value) and "call-token" not in str(e.value)


# ── a whole call on a handed configuration ──────────────────────────────────────────────────────
class HandedConnection(Credentials):
    """What ServerCall is to a real server, pointed at the scripted provider: it records what it was built from."""
    made: list["HandedConnection"] = []

    def __init__(self, url: str, token: str):
        self.url, self.token, self.minted = url, token, 0
        HandedConnection.made.append(self)

    async def connect(self, model: str) -> Connection:
        self.minted += 1
        return Connection(url=f"{HandedConnection.provider}?model={model}",
                          headers={"Authorization": f"Bearer ek_{self.minted}"}, model=model, ref={})


@contextlib.asynccontextmanager
async def serving(fake: FakeRealtime):
    HandedConnection.provider, HandedConnection.made = await fake.start(), []
    server = Server([], None, set(), operator_keys={OP})
    port = free_port()
    uv = uvicorn.Server(uvicorn.Config(create_app(server), host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(uv.serve())
    while not uv.started:
        await asyncio.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}", server
    finally:
        uv.should_exit = True
        await task
        await fake.stop()


def test_a_whole_call_runs_on_the_instructions_tools_and_connection_the_server_handed_it(mcp_url, monkeypatch):
    monkeypatch.setattr(uhp, "ServerCall", HandedConnection)
    monkeypatch.setattr("test_call_e2e.H", HOP)                  # the phone's own requests carry the operator's key

    async def go():
        fake = FakeRealtime(["What is the weather in Paris?"])
        async with serving(fake) as (base, server):
            p = await Phone(base).call({"harness": {"id": "chrn_neo", "name": "Neo", "instructions": "Answer in German.",
                                                    "mcp_servers": [{"name": "fx", "url": mcp_url}]},
                                        "upstream": UPSTREAM})
            await p.say()
            await p.answers(1)
            rec = await p.hang_up()
        t = rec["responses"][0]
        assert rec["harness_id"] == "chrn_neo"
        assert [i["type"] for i in t["output"]] == ["function_call", "function_call_output", "message"]
        assert fake.sessions[0]["instructions"].startswith("Answer in German.") and fake.sessions[0]["tools"]
        (handed,) = HandedConnection.made
        assert (handed.url, handed.token) == (UPSTREAM["url"], UPSTREAM["token"]) and handed.minted == 1
    asyncio.run(go())


# ── the command line ────────────────────────────────────────────────────────────────────────────
def _serve(monkeypatch, argv, env):
    from realtime_harness import cli
    for name in ("RTH_API_KEYS", "RTH_API_KEYS_FILE", "RTH_OPERATOR_KEYS", "RTH_OPERATOR_KEYS_FILE", "RTH_SERVER_URL",
                 "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    ran = []
    import uvicorn as uv
    monkeypatch.setattr(uv, "run", lambda app, **kw: ran.append(app))
    return cli.main(["serve", *argv]), ran


def test_a_server_behind_an_operator_starts_with_no_harness_file_and_no_provider_key(monkeypatch):
    code, ran = _serve(monkeypatch, [], {"RTH_OPERATOR_KEYS": OP})
    assert code == 0 and len(ran) == 1
    server = ran[0].state.uhp
    assert server.operator_keys == {OP} and server.api_keys == {OP} and server.harnesses == {} and server.credentials is None


def test_without_an_operator_key_the_harness_file_and_the_provider_are_still_required(monkeypatch, tmp_path, capsys):
    code, ran = _serve(monkeypatch, [], {"RTH_API_KEYS": KEY})
    assert code == 2 and ran == [] and "--config" in capsys.readouterr().err
    f = tmp_path / "h.yaml"
    f.write_text("id: desk\nname: Desk\n", encoding="utf-8")
    code, ran = _serve(monkeypatch, ["--config", str(f)], {"RTH_API_KEYS": KEY})
    assert code == 2 and ran == [] and "no provider credentials" in capsys.readouterr().err

"""The Realtime chapter's API without a provider (design §6.2, checks R-01, R-02, R-03, R-08): discovery says
realtime, a call is created with a one-time connect token, the token is refused a second time and after it
expires, and hanging up a call that never connected ends it with a readable record."""
import time

from fastapi.testclient import TestClient

from realtime_harness.config import HarnessConfig
from realtime_harness.credentials import Connection, Credentials
from realtime_harness.uhp import Server, UhpError, create_app

KEY = "test-key"
H = {"Authorization": f"Bearer {KEY}"}


class NoProvider(Credentials):
    async def connect(self, model: str) -> Connection:
        raise AssertionError("no provider in these tests")


def _app():
    server = Server([HarnessConfig(id="desk", name="Desk")], NoProvider(), {KEY})
    return server, TestClient(create_app(server))


def test_discovery_advertises_realtime_and_its_transport():
    _, c = _app()
    d = c.get("/v1/uhp").json()
    assert d["capabilities"]["realtime"] is True and d["realtime_transports"] == ["websocket"]
    assert d["capabilities"]["environments"] is False and d["capabilities"]["memories"] is False   # false, not omitted


def test_a_call_gets_a_connect_block_and_an_unknown_harness_does_not():
    _, c = _app()
    r = c.post("/v1/realtime/sessions", headers=H, json={"transport": "websocket"})
    assert r.status_code == 201
    s = r.json()
    assert s["id"].startswith("hsess_") and s["kind"] == "realtime" and s["status"] == "connecting"
    assert s["connect"]["transport"] == "websocket" and s["connect"]["url"].endswith("/v1/realtime/connect")
    assert s["connect"]["token"] and s["connect"]["expires_at"].endswith("Z")
    r = c.post("/v1/realtime/sessions", headers=H, json={"metadata": {"harness_id": "chrn_nope"}})
    assert r.status_code == 404 and r.json()["error"]["code"] == "harness_not_found"
    assert c.post("/v1/realtime/sessions", json={}).status_code == 401


def test_a_connect_token_works_once_and_not_after_it_expires():
    server, c = _app()
    s = c.post("/v1/realtime/sessions", headers=H, json={}).json()
    tok = s["connect"]["token"]
    call = server.claim(tok)
    assert call.session.id == s["id"]
    try:
        server.claim(tok)
        raise AssertionError("a token was accepted twice")
    except UhpError as e:
        assert e.status == 401
    s2 = c.post("/v1/realtime/sessions", headers=H, json={}).json()
    server.calls[s2["id"]].token_expires = time.time() - 1
    try:
        server.claim(s2["connect"]["token"])
        raise AssertionError("an expired token was accepted")
    except UhpError as e:
        assert e.status == 401


def test_hanging_up_ends_the_call_and_the_record_stays_readable():
    _, c = _app()
    s = c.post("/v1/realtime/sessions", headers=H, json={}).json()
    r = c.post(f"/v1/sessions/{s['id']}/cancel", headers=H)
    assert r.status_code == 200 and r.json()["status"] == "ended" and r.json()["ended_reason"] == "hangup"
    got = c.get(f"/v1/sessions/{s['id']}", headers=H).json()
    assert got["status"] == "ended" and got["responses"] == []
    with c.stream("GET", f"/v1/sessions/{s['id']}/events", headers=H) as ev:
        lines = [ln for ln in ev.iter_lines() if ln.startswith("data: ")]
    assert len(lines) == 1 and '"realtime.session.ended"' in lines[0] and '"sequence_number": 0' in lines[0]


def test_a_call_and_the_text_around_it_are_one_session():
    server, c = _app()
    s = c.post("/v1/realtime/sessions", headers=H, json={}).json()
    call = server.calls[s["id"]]
    call.record.user_said("What is the weather in Paris?")
    call.record.model_said("It is clear.")
    call.record.end_turn("completed")
    turn_id = call.record.turns[0].id
    # while the call is open the session is busy for a text turn
    r = c.post("/v1/responses", headers=H, json={"input": "And tomorrow?", "previous_response_id": turn_id})
    assert r.status_code == 409 and r.json()["error"]["code"] == "session_busy"
    # hung up: a text turn continues the call's session, and the call's turn is readable as a response
    c.post(f"/v1/sessions/{s['id']}/cancel", headers=H)
    assert server.continued(turn_id, {}) is call.session
    assert [i["type"] for i in call.session.items] == ["message", "message"]          # the call joined the history
    assert c.get(f"/v1/responses/{turn_id}", headers=H).json()["output"][0]["content"][0]["text"] == "It is clear."
    # a new call can start from the same turn, in the same session
    s2 = c.post("/v1/realtime/sessions", headers=H, json={"previous_response_id": turn_id}).json()
    assert s2["id"] == s["id"] and "ignored_fields" not in (s2.get("metadata") or {})


def test_a_call_starts_with_what_each_side_said_before():
    from realtime_harness.call import context_messages
    history = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
               {"type": "function_call", "call_id": "c", "name": "weather", "arguments": "{}"},
               {"type": "function_call_output", "call_id": "c", "output": "{}"},
               {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "It is clear."}]}]
    assert context_messages(history) == [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "It is clear."}]

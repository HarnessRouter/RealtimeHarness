"""A call's record and its event stream, without a provider (design §6): one turn holds what the person said,
what the model said, every tool call and its result; the stream is gapless; a barge-in cancels the turn."""
import asyncio
import json

from realtime_harness.call import TURN_END_GRACE_S, CallRecord
from realtime_harness.device import DeviceSerializer


def _record():
    events = []
    rec = CallRecord("hsess_x", "chrn_h", "gpt-realtime-2.1", lambda e: events.append({**e, "sequence_number": len(events)}))
    return rec, events


def test_one_turn_holds_the_words_the_call_its_result_and_the_answer():
    async def go():
        rec, events = _record()
        for d in ("Let me ", "check."):
            rec.model_said(d)
        rec.response_ended()                       # the provider's first response ends before the call starts
        rec.call_started("call_1", "weather", {"city": "Paris"})
        rec.call_finished("call_1", {"status": "done", "output": "clear, 18"})
        rec.call_started("call_1", "weather", {"city": "Paris"})          # a broadcast frame seen twice
        rec.user_said("What is the weather in Paris?")                   # transcribed after the answer began
        rec.model_said("It is clear.")
        rec.usage(type("U", (), {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cache_read_input_tokens": 2,
                                 "input_audio_tokens": 3, "output_audio_tokens": 4})())
        rec.response_ended()
        await asyncio.sleep(TURN_END_GRACE_S + 0.2)
        assert len(rec.turns) == 1
        t = rec.turns[0]
        assert t.status == "completed"
        assert [i["type"] for i in t.output] == ["message", "function_call", "function_call_output", "message"]
        assert t.output[0]["content"][0]["text"] == "Let me check." and t.output[3]["content"][0]["text"] == "It is clear."
        assert t.output[1]["call_id"] == t.output[2]["call_id"] == "call_1"
        assert json.loads(t.output[1]["arguments"]) == {"city": "Paris"}
        assert t.input[0]["content"][0]["text"] == "What is the weather in Paris?"
        assert t.usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "cache_read_tokens": 2,
                           "input_audio_tokens": 3, "output_audio_tokens": 4}
        types = [e["type"] for e in events]
        assert types[0] == "response.created" and types[-1] == "response.completed"
        assert [e["sequence_number"] for e in events] == list(range(len(events)))
        assert types.count("realtime.output_transcript.done") == 2
        fc = types.index("response.function_call_arguments.done")              # between the call's added and done
        assert types[fc - 1] == "response.output_item.added" and types[fc + 1] == "response.output_item.done"
    asyncio.run(go())


def test_a_turn_waits_for_its_pending_call():
    async def go():
        rec, _ = _record()
        rec.model_said("One moment.")
        rec.call_started("call_9", "order_status", "{}")
        rec.response_ended()
        await asyncio.sleep(TURN_END_GRACE_S + 0.2)
        assert rec.current is not None and rec.turns[0].status == "in_progress"      # the call is not answered yet
        rec.call_finished("call_9", "shipped")
        rec.model_said("It shipped.")
        rec.response_ended()
        await asyncio.sleep(TURN_END_GRACE_S + 0.2)
        assert rec.turns[0].status == "completed" and rec.current is None
    asyncio.run(go())


def test_a_barge_in_cancels_the_turn_and_the_next_words_open_another():
    async def go():
        rec, events = _record()
        rec.model_said("A long story begins")
        rec.speech_started()
        rec.end_turn("cancelled")
        rec.user_said("Stop, please.")
        rec.model_said("Of course.")
        rec.response_ended()
        await asyncio.sleep(TURN_END_GRACE_S + 0.2)
        assert [t.status for t in rec.turns] == ["cancelled", "completed"]
        assert rec.turns[0].output[0]["status"] == "incomplete"
        assert rec.turns[1].input[0]["content"][0]["text"] == "Stop, please."
        assert rec.turns[1].previous_response_id == rec.turns[0].id
        failed = [e for e in events if e["type"] == "response.failed"]
        assert failed and failed[0]["response"]["status"] == "cancelled"      # never reported as failed
    asyncio.run(go())


def test_the_device_narrows_a_call_and_never_widens_it():
    async def go():
        from realtime_harness.call import _gate_class
        seen = []
        s = DeviceSerializer(seen.append)
        frame = await s.deserialize(b"\x01\x00" * 2400)
        assert frame is not None and frame.sample_rate == 24000 and frame.num_channels == 1
        await s.deserialize(json.dumps({"type": "realtime.input.mute"}))
        await s.deserialize(json.dumps({"type": "session.update", "session": {"instructions": "x"}}))
        await s.deserialize("not json")
        assert [e.get("type") for e in seen] == ["realtime.input.mute", "session.update", None]
        gate = _gate_class()()
        assert gate.passes(frame)
        gate.muted = True
        assert not gate.passes(frame)                                     # muted: no audio reaches the model
    asyncio.run(go())


def test_a_turn_with_results_in_waits_for_the_answer_to_them():
    async def go():
        rec, _ = _record()
        rec.model_said("Checking.")
        rec.call_started("call_2", "weather", "{}")
        rec.call_finished("call_2", "clear")
        rec.response_ended()
        await asyncio.sleep(TURN_END_GRACE_S + 0.4)            # the model is still thinking about the result
        assert rec.current is not None and len(rec.turns) == 1
        rec.model_said("It is clear.")
        rec.response_ended()
        await asyncio.sleep(TURN_END_GRACE_S + 0.2)
        assert len(rec.turns) == 1 and rec.turns[0].status == "completed"
        assert rec.turns[0].output[-1]["content"][0]["text"] == "It is clear."
    asyncio.run(go())


def test_a_result_that_outlives_its_turn_is_recorded_where_the_model_takes_it():
    async def go():
        rec, events = _record()
        rec.model_said("Checking the order.")
        rec.call_started("call_o", "order_status", {"order": "4417"})
        rec.speech_started()
        rec.end_turn("cancelled")                    # the caller spoke while the call ran
        rec.user_said("And the weather in Paris?")
        rec.model_said("It is clear in Paris.")
        rec.response_ended()
        await asyncio.sleep(TURN_END_GRACE_S + 0.2)
        rec.call_finished("call_o", "Order 4417 has shipped.")
        rec.model_said("Your order has shipped.")
        rec.response_ended()
        await asyncio.sleep(TURN_END_GRACE_S + 0.2)
        first, second, third = rec.turns
        assert first.status == "cancelled" and [i["type"] for i in first.output] == ["message", "function_call"]
        assert second.status == "completed"
        assert [i["type"] for i in third.output] == ["function_call_output", "message"] and third.status == "completed"
        assert third.output[0]["call_id"] == "call_o"
        # nothing about a response after its terminal event
        ended = set()
        for e in events:
            rid = e.get("response_id") or (e.get("response") or {}).get("id")
            assert rid not in ended, e["type"]
            if e["type"] in ("response.completed", "response.failed", "response.incomplete"):
                ended.add(rid)
    asyncio.run(go())


def test_a_provider_drop_fails_a_turn_it_cut_off_and_keeps_one_it_had_answered():
    async def go():
        rec, _ = _record()
        rec.model_said("It is clear.")
        rec.response_ended()                       # answered; only the end-of-turn grace is left
        rec.provider_lost()
        rec.model_said("Let me check")             # the next turn, cut off mid-sentence
        rec.provider_lost()
        assert [t.status for t in rec.turns] == ["completed", "failed"]
        assert rec.turns[1].output[0]["status"] == "incomplete"
    asyncio.run(go())


def test_a_call_with_tools_is_told_to_say_so_before_using_one():
    from realtime_harness.call import SAY_BEFORE_A_TOOL, call_instructions
    assert call_instructions("Be brief.", True) == "Be brief.\n\n" + SAY_BEFORE_A_TOOL
    assert call_instructions("Be brief.", False) == "Be brief."
    assert call_instructions("", True) == SAY_BEFORE_A_TOOL and call_instructions("", False) is None


def test_each_provider_response_is_its_own_message():
    async def go():
        rec, _ = _record()
        rec.model_said("Checking both.")
        rec.call_started("c1", "weather", "{}")
        rec.call_finished("c1", "clear")
        rec.model_said("Paris is clear.")
        rec.response_ended()
        rec.model_said("Your order has shipped.")             # the provider's next response, same turn
        rec.response_ended()
        await asyncio.sleep(TURN_END_GRACE_S + 0.2)
        texts = [i["content"][0]["text"] for i in rec.turns[0].output if i["type"] == "message"]
        assert texts == ["Checking both.", "Paris is clear.", "Your order has shipped."]
    asyncio.run(go())

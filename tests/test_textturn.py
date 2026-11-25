"""A text turn says which model ran it.

On Azure a request names a deployment, and the provider's own `session.created` names the model
behind it. A server that runs this harness one turn per process has no other place to read the
served model from, so the turn's result carries it: what the provider said, never what was asked
for, and empty when the provider named none.
"""
import asyncio

from realtime_harness.config import HarnessConfig
from realtime_harness.credentials import Connection
from realtime_harness.textturn import Listener, run_text_turn
from realtime_harness.tools import ToolExecutor

from fake_provider import FakeRealtime

ASKED = "my-deployment"


async def _turn(fake: FakeRealtime, text: str = "Hello"):
    url = await fake.start()
    executor = ToolExecutor(HarnessConfig(id="h", name="h"))
    await executor.open()
    try:
        return await run_text_turn(Connection(url=url, headers={}, model=ASKED, ref={}), instructions="Be brief.",
                                   executor=executor, history=[], text=text, listener=Listener(),
                                   cancelled=asyncio.Event())
    finally:
        await executor.close()
        await fake.stop()


def test_the_result_names_the_model_the_provider_said_it_is_running():
    result = asyncio.run(_turn(FakeRealtime([], model="gpt-realtime-2.1-2026-10-01")))
    assert result.status == "completed"
    assert result.model == "gpt-realtime-2.1-2026-10-01"
    assert result.model != ASKED


def test_a_provider_that_names_no_model_leaves_it_empty_rather_than_the_one_asked_for():
    result = asyncio.run(_turn(FakeRealtime([])))
    assert result.status == "completed"
    assert result.model == ""


def test_a_turn_the_provider_fails_still_names_the_model():
    result = asyncio.run(_turn(FakeRealtime([], refuse_after=0, model="gpt-realtime-2.1-2026-10-01")))
    assert result.status == "failed" and result.error["code"] == "provider_error"
    assert result.model == "gpt-realtime-2.1-2026-10-01"

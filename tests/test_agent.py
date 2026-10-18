import asyncio
from types import SimpleNamespace

from forecasting_tools import BinaryQuestion

import agent as agent_module
from agent import Agent

QUESTION = BinaryQuestion(
    question_text="Will the BLS report more than 100,000 jobs added in October 2026?",
    page_url="https://www.metaculus.com/questions/9/",
    resolution_criteria="Resolves per https://www.bls.gov/news.release/empsit.nr0.htm",
)


def response(content=None, calls=(), cost=0.001):
    tool_calls = [
        SimpleNamespace(id=f"call-{i}", function=SimpleNamespace(name=name, arguments=args))
        for i, (name, args) in enumerate(calls)
    ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=SimpleNamespace(cost=cost))


def test_agent_reads_the_source_then_writes_its_brief(monkeypatch):
    script = [
        response(calls=[("read_page", '{"url": "https://www.bls.gov/news.release/empsit.nr0.htm"}')]),
        response(content="Brief: September added 29,000 jobs; October is due 6 Nov."),
    ]
    seen = []

    async def fake_completion(**kwargs):
        seen.append(kwargs)
        return script.pop(0)

    monkeypatch.setattr(agent_module.litellm, "acompletion", fake_completion)
    monkeypatch.setattr(agent_module.resolution_sources, "read_page", lambda url: "Total nonfarm payroll +29,000")
    run = Agent(QUESTION)
    brief = asyncio.run(run.research())
    assert brief.startswith("Brief: September added 29,000")
    assert run.tool_calls == 1
    assert run.cost == 0.002
    assert "https://www.bls.gov/news.release/empsit.nr0.htm" in seen[0]["messages"][0]["content"]
    assert seen[1]["messages"][-1] == {"role": "tool", "tool_call_id": "call-0", "content": "Total nonfarm payroll +29,000"}


def test_a_failing_tool_is_reported_to_the_agent_not_raised(monkeypatch):
    def broken(url):
        raise TimeoutError("slow site")

    monkeypatch.setattr(agent_module.resolution_sources, "read_page", broken)
    result = asyncio.run(Agent(QUESTION).run_tool("read_page", '{"url": "https://example.org"}'))
    assert result == "The tool failed: TimeoutError."


def test_the_agent_stops_after_its_turn_budget(monkeypatch):
    calls = {"n": 0}

    async def endless(**kwargs):
        calls["n"] += 1
        if kwargs.get("tool_choice") == "none":
            return response(content="Brief written at the limit.")
        return response(calls=[("similar_questions", '{"query": "jobs report"}')])

    monkeypatch.setattr(agent_module.litellm, "acompletion", endless)
    monkeypatch.setattr(agent_module.history, "load", lambda: [])
    run = Agent(QUESTION)
    assert asyncio.run(run.research()) == "Brief written at the limit."
    assert calls["n"] == agent_module.MAX_TURNS + 1

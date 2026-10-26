"""Provider-neutral LLM layer: OpenAI-compatible adapter, presets, factory, number grounding."""
import json

import pytest

from optlib_app.agent import TOOLS, ToolRecord, ungrounded_numbers
from optlib_app.llm import (
    AnthropicLLM,
    LLMError,
    LLMNotConfigured,
    OpenAICompatibleLLM,
    PRESETS,
    ToolCall,
    make_llm,
)


class Resp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = json.dumps(body) if not isinstance(body, str) else body

    def json(self):
        if isinstance(self._body, str):
            raise ValueError("not json")
        return self._body


class Session:
    def __init__(self, responses):
        self.responses, self.posts = list(responses), []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "json": json, "headers": headers})
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def completion(content=None, tool_calls=None, finish="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
                         "finish_reason": finish}]}


def test_openai_compatible_request_and_tool_call_parsing():
    s = Session([Resp(200, completion(None, [{"id": "c1", "type": "function", "function": {
        "name": "price_option", "arguments": json.dumps({"expiry": "2026-10-27", "strike": 22500,
                                                         "option_type": "call"})}}], "tool_calls"))])
    llm = OpenAICompatibleLLM("groq", PRESETS["groq"], "openai/gpt-oss-120b", "gsk_test", session=s)
    msgs = [{"role": "user", "content": "price it"},
            {"role": "assistant", "content": "", "tool_calls": [ToolCall("c0", "market_overview", {"underlying": "NIFTY"})]},
            {"role": "tool", "tool_call_id": "c0", "name": "market_overview", "content": "{}", "is_error": False}]
    reply = llm.chat("SYSTEM", msgs, TOOLS)
    post = s.posts[0]
    assert post["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert post["headers"]["Authorization"] == "Bearer gsk_test"
    body = post["json"]
    assert body["model"] == "openai/gpt-oss-120b" and body["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert body["messages"][2]["tool_calls"][0]["function"] == {"name": "market_overview",
                                                                 "arguments": '{"underlying": "NIFTY"}'}
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "c0", "name": "market_overview", "content": "{}"}
    assert body["tools"][0]["type"] == "function" and body["tools"][0]["function"]["parameters"]["type"] == "object"
    assert reply.tool_calls[0].name == "price_option" and reply.tool_calls[0].arguments["strike"] == 22500
    assert reply.stop == "tool_calls"


def test_openai_compatible_bad_arguments_and_plain_text():
    s = Session([Resp(200, completion(None, [{"id": "c1", "function": {"name": "price_option",
                                                                      "arguments": "{strike: 22500"}}])),
                 Resp(200, completion("Hello"))])
    llm = OpenAICompatibleLLM("ollama", PRESETS["ollama"], "qwen3", session=s)
    r1 = llm.chat("S", [{"role": "user", "content": "x"}], TOOLS)
    assert r1.tool_calls[0].parse_error and r1.tool_calls[0].arguments == {}
    r2 = llm.chat("S", [{"role": "user", "content": "x"}])
    assert r2.text == "Hello" and r2.tool_calls == [] and "tools" not in s.posts[1]["json"]
    assert "Authorization" not in s.posts[0]["headers"]  # no key configured: none sent


def test_openai_compatible_retries_then_reports_errors():
    sleeps = []
    s = Session([Resp(429, {"error": {"message": "slow down"}}, {"Retry-After": "3"}),
                 Resp(200, completion("ok"))])
    llm = OpenAICompatibleLLM("gemini", PRESETS["gemini"], "gemini-3.8-flash", "k", session=s, sleep=sleeps.append)
    assert llm.chat("S", [{"role": "user", "content": "x"}]).text == "ok" and sleeps == [3.0]

    s = Session([Resp(401, {"error": {"message": "invalid api key"}})])
    with pytest.raises(LLMError, match="401.*invalid api key"):
        OpenAICompatibleLLM("openrouter", PRESETS["openrouter"], "m", "k", session=s).chat("S", [{"role": "user",
                                                                                                 "content": "x"}])
    s = Session([Resp(429, {"error": {"message": "quota"}})] * 4)
    with pytest.raises(LLMError, match="rate limit"):
        OpenAICompatibleLLM("groq", PRESETS["groq"], "m", "k", session=s, sleep=lambda x: None).chat(
            "S", [{"role": "user", "content": "x"}])
    s = Session([ConnectionError("refused")] * 4)
    with pytest.raises(LLMError, match="cannot reach"):
        OpenAICompatibleLLM("ollama", PRESETS["ollama"], "m", session=s, sleep=lambda x: None).chat(
            "S", [{"role": "user", "content": "x"}])


def test_anthropic_message_conversion_groups_tool_results():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "thinking", "tool_calls": [ToolCall("a", "t1", {"x": 1}),
                                                                       ToolCall("b", "t2", {})]},
            {"role": "tool", "tool_call_id": "a", "name": "t1", "content": "1", "is_error": False},
            {"role": "tool", "tool_call_id": "b", "name": "t2", "content": "oops", "is_error": True}]
    out = AnthropicLLM.to_provider(msgs)
    assert len(out) == 3 and out[2]["role"] == "user" and len(out[2]["content"]) == 2
    assert out[2]["content"][1] == {"type": "tool_result", "tool_use_id": "b", "content": "oops", "is_error": True}
    assert out[1]["content"][0] == {"type": "text", "text": "thinking"}


def settings(**kv):
    return lambda name, default="": kv.get(name, default)


def test_factory_presets_and_errors():
    llm = make_llm(settings(LLM_PROVIDER="groq", LLM_API_KEY="k"))
    assert llm.provider == "groq" and llm.model == "openai/gpt-oss-120b" and "groq.com" in llm.base_url
    llm = make_llm(settings(LLM_PROVIDER="gemini", LLM_API_KEY="k"))
    assert llm.model == "gemini-3.8-flash" and "generativelanguage" in llm.base_url
    llm = make_llm(settings(LLM_PROVIDER="ollama", LLM_MODEL="qwen3"))  # local: no key needed
    assert llm.base_url == "http://localhost:11434/v1"
    llm = make_llm(settings(LLM_PROVIDER="custom", LLM_API_KEY="k", LLM_MODEL="m",
                            LLM_BASE_URL="https://llm.internal.dpa/v1/"))
    assert llm.base_url == "https://llm.internal.dpa/v1"
    v = make_llm(settings(LLM_PROVIDER="groq", LLM_API_KEY="k", VALIDATOR_MODEL="qwen/qwen3.8-27b"), "validator")
    assert v.model == "qwen/qwen3.8-27b"
    assert make_llm(settings(ANTHROPIC_API_KEY="sk-ant")).provider == "anthropic"  # first-setup compatibility
    for bad, msg in ((settings(), "No LLM configured"), (settings(LLM_PROVIDER="nope"), "not one of"),
                     (settings(LLM_PROVIDER="openrouter", LLM_API_KEY="k"), "LLM_MODEL"),
                     (settings(LLM_PROVIDER="groq"), "LLM_API_KEY"),
                     (settings(LLM_PROVIDER="custom", LLM_API_KEY="k", LLM_MODEL="m"), "LLM_BASE_URL")):
        with pytest.raises(LLMNotConfigured, match=msg):
            make_llm(bad)


def test_key_not_in_repr():
    llm = OpenAICompatibleLLM("groq", PRESETS["groq"], "m", "gsk_secret_value", session=Session([]))
    assert "gsk_secret_value" not in repr(llm)


def test_number_grounding():
    rec = [ToolRecord("price_option", {"strike": 22500, "lots": 2},
                      {"price_per_unit": 163.441777, "value_rs": 21247.43, "vol": 0.162788, "rate": 0.05797,
                       "validation": {"checks": [{"detail": "gap -0.021% (warn > 0.05%)"}]}})]
    ok = ("The 22,500 call costs 163.44 (Rs 21,247 for 2 lots) at 16.28% vol, rate 5.797%, "
          "expiring 27 Oct 2026; forward gap -0.021%.")
    assert ungrounded_numbers(ok, "price 2 lots of the 22500 call", rec) == []
    bad = "Breakeven is 22,663.44 and the delta-adjusted value is 9,999.5."
    assert ungrounded_numbers(bad, "q", rec) == ["22,663.44", "9,999.5"]
    assert ungrounded_numbers("Expiry 2026-10-27, 3 checks passed.", "q", rec) == []

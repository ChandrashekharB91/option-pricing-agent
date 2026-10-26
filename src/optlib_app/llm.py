"""Provider-neutral LLM layer: the agents run on whichever API is configured in .env.

Two adapters cover the field:
  - AnthropicLLM            Claude via the Anthropic SDK
  - OpenAICompatibleLLM     any /chat/completions API with tool calling: OpenAI, Groq, Google Gemini,
                            OpenRouter, GitHub Models, Cerebras, Mistral, a local Ollama server, or a custom URL

Configuration (.env):
    LLM_PROVIDER      anthropic | openai | groq | gemini | openrouter | github | cerebras | mistral | ollama | custom
    LLM_API_KEY       the provider's key (ollama needs none; anthropic also reads ANTHROPIC_API_KEY)
    LLM_MODEL         model id at that provider (defaults exist for anthropic, groq and gemini)
    LLM_BASE_URL      overrides the preset URL (required for custom)
    VALIDATOR_MODEL   optional: a different model for the validator agent

Messages use one neutral format, converted per provider:
    {"role": "user", "content": str}
    {"role": "assistant", "content": str, "tool_calls": [ToolCall, ...]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str, "is_error": bool}
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

PRESETS = {
    "anthropic": None,
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openrouter": "https://openrouter.ai/api/v1",
    "github": "https://models.github.ai/inference",
    "cerebras": "https://api.cerebras.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "ollama": "http://localhost:11434/v1",
    "custom": None,
}
DEFAULT_MODELS = {  # only where the provider's own docs name a tool-capable model
    "anthropic": "claude-sonnet-5-5",
    "groq": "openai/gpt-oss-120b",
    "gemini": "gemini-3.8-flash",
}
KEY_NOT_NEEDED = {"ollama"}


class LLMError(Exception):
    """The LLM API failed (network, auth, rate limit, bad response)."""


class LLMNotConfigured(Exception):
    """No usable LLM settings; the message says what to put in .env."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    parse_error: Optional[str] = None  # the model sent arguments that are not valid JSON


@dataclass
class LLMReply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop: str = ""


class LLMClient(Protocol):
    provider: str
    model: str

    def chat(self, system: str, messages: list[dict], tools: Optional[list[dict]] = None,
             max_tokens: int = 2048) -> LLMReply: ...


# ------------------------------------------------------------------ Anthropic

class AnthropicLLM:
    provider = "anthropic"

    def __init__(self, model: str, api_key: str = "", client=None):
        self.model = model
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        self.client = client

    @staticmethod
    def to_provider(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m.get("content") else []
                blocks += [{"type": "tool_use", "id": t.id, "name": t.name, "input": t.arguments}
                           for t in m.get("tool_calls") or []]
                out.append({"role": "assistant", "content": blocks or m.get("content", "")})
            elif m["role"] == "tool":
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"],
                         "is_error": bool(m.get("is_error"))}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)  # results of one turn go back in one message
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def chat(self, system, messages, tools=None, max_tokens=2048) -> LLMReply:
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system, messages=self.to_provider(messages))
        if tools:
            kwargs["tools"] = tools
        try:
            resp = self.client.messages.create(**kwargs)
        except Exception as exc:  # SDK errors carry status and message
            raise LLMError(f"anthropic: {exc}") from exc
        text = "\n".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if getattr(b, "type", "") == "tool_use"]
        return LLMReply(text, calls, resp.stop_reason or "")


# ------------------------------------------------------------------ OpenAI-compatible

class OpenAICompatibleLLM:
    def __init__(self, provider: str, base_url: str, model: str, api_key: str = "", session=None,
                 timeout: float = 120.0, max_retries: int = 3, sleep: Callable[[float], None] = time.sleep):
        if session is None:
            import requests

            session = requests.Session()
        self.provider, self.model = provider, model
        self.base_url = base_url.rstrip("/")
        self._key = api_key
        self.session, self.timeout, self.max_retries, self._sleep = session, timeout, max_retries, sleep

    def __repr__(self) -> str:
        return f"OpenAICompatibleLLM(provider={self.provider!r}, model={self.model!r}, base_url={self.base_url!r})"

    @staticmethod
    def to_provider(system: str, messages: list[dict]) -> list[dict]:
        out = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"id": t.id, "type": "function",
                                          "function": {"name": t.name, "arguments": json.dumps(t.arguments)}}
                                         for t in m["tool_calls"]]
                out.append(msg)
            elif m["role"] == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "name": m.get("name", ""),
                            "content": m["content"]})
        return out

    @staticmethod
    def tools_to_provider(tools: list[dict]) -> list[dict]:
        return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["input_schema"]}} for t in tools]

    def chat(self, system, messages, tools=None, max_tokens=2048) -> LLMReply:
        body = {"model": self.model, "messages": self.to_provider(system, messages), "max_tokens": max_tokens}
        if tools:
            body["tools"] = self.tools_to_provider(tools)
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        url = f"{self.base_url}/chat/completions"
        for attempt in range(1, self.max_retries + 2):
            try:
                resp = self.session.post(url, json=body, headers=headers, timeout=self.timeout)
            except Exception as exc:
                if attempt > self.max_retries:
                    raise LLMError(f"{self.provider}: cannot reach {self.base_url} ({type(exc).__name__})") from exc
                self._sleep(2 ** (attempt - 1))
                continue
            if resp.status_code in (429, 500, 502, 503, 504) and attempt <= self.max_retries:
                retry_after = resp.headers.get("Retry-After") if hasattr(resp, "headers") else None
                try:
                    delay = float(retry_after)
                except (TypeError, ValueError):
                    delay = 2 ** (attempt - 1)
                self._sleep(min(delay, 30.0))
                continue
            break
        if resp.status_code >= 400:
            try:
                err = resp.json().get("error", {})
                detail = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            except ValueError:
                detail = getattr(resp, "text", "")[:300]
            hint = " (rate limit: free tiers cap requests per minute/day)" if resp.status_code == 429 else ""
            raise LLMError(f"{self.provider} HTTP {resp.status_code}: {detail}{hint}")
        try:
            choice = resp.json()["choices"][0]
        except (ValueError, KeyError, IndexError) as exc:
            raise LLMError(f"{self.provider}: unexpected response shape") from exc
        msg = choice.get("message") or {}
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                args, err = (raw if isinstance(raw, dict) else json.loads(raw)), None
            except json.JSONDecodeError as exc:
                args, err = {}, f"tool arguments are not valid JSON: {exc}"
            calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args, err))
        return LLMReply((msg.get("content") or "").strip(), calls, choice.get("finish_reason") or "")


# ------------------------------------------------------------------ factory

def make_llm(get: Callable[[str, str], str], role: str = "orchestrator") -> LLMClient:
    """Build the configured client. ``get(name, default)`` reads settings (env / .env)."""
    provider = get("LLM_PROVIDER", "").lower()
    if not provider and get("ANTHROPIC_API_KEY", ""):
        provider = "anthropic"  # backwards compatible with the first setup
    if not provider or provider == "none":
        raise LLMNotConfigured(
            "No LLM configured. In .env set LLM_PROVIDER (e.g. groq, gemini, openrouter, ollama, anthropic), "
            "LLM_API_KEY and LLM_MODEL, then restart the API.")
    if provider not in PRESETS:
        raise LLMNotConfigured(f"LLM_PROVIDER={provider!r} is not one of: {', '.join(PRESETS)}")

    model = get("LLM_MODEL", "") or DEFAULT_MODELS.get(provider, "")
    if role == "validator":
        model = get("VALIDATOR_MODEL", "") or model
    if not model:
        raise LLMNotConfigured(f"Set LLM_MODEL in .env to a tool-capable model id from {provider}.")
    key = get("LLM_API_KEY", "") or (get("ANTHROPIC_API_KEY", "") if provider == "anthropic" else "")
    if not key and provider not in KEY_NOT_NEEDED:
        raise LLMNotConfigured(f"Set LLM_API_KEY in .env (your {provider} API key).")

    if provider == "anthropic":
        return AnthropicLLM(model, key)
    base_url = get("LLM_BASE_URL", "") or PRESETS[provider]
    if not base_url:
        raise LLMNotConfigured("LLM_PROVIDER=custom needs LLM_BASE_URL (an OpenAI-compatible /v1 URL).")
    return OpenAICompatibleLLM(provider, base_url, model, key or "ollama")

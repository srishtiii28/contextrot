"""Minimal JSON-mode chat client for the judge. Separate from the tutor adapter so the judge can use
a different provider or model (JUDGE_PROVIDER / JUDGE_MODEL), which limits self-preference bias."""
import asyncio
import json
import time
from contextvars import ContextVar

import httpx

from app.core.config import get_settings


# Callers that need cost accounting (the evaluation harness) set a dict here; calls add token usage to it.
USAGE: ContextVar[dict | None] = ContextVar("judge_usage", default=None)


def _record_usage(prompt_tokens, completion_tokens):
    holder = USAGE.get()
    if holder is not None:
        holder["calls"] = holder.get("calls", 0) + 1
        holder["prompt_tokens"] = holder.get("prompt_tokens", 0) + int(prompt_tokens or 0)
        holder["completion_tokens"] = holder.get("completion_tokens", 0) + int(completion_tokens or 0)


class JudgeConfigurationError(RuntimeError):
    """The configured judge provider cannot be used."""


def judge_target() -> tuple[str, str]:
    s = get_settings()
    provider = (s.judge_provider or s.llm_provider or "").lower()
    if provider == "groq":
        if not s.groq_api_key:
            raise JudgeConfigurationError("Judge provider groq needs GROQ_API_KEY")
        return provider, s.judge_model or s.groq_model
    if provider == "gemini":
        if not s.google_api_key or not (s.judge_model or s.gemini_model):
            raise JudgeConfigurationError("Judge provider gemini needs GOOGLE_API_KEY and JUDGE_MODEL or GEMINI_MODEL")
        return provider, s.judge_model or s.gemini_model
    raise JudgeConfigurationError(
        f"RELIABILITY_DETECTOR=judge needs JUDGE_PROVIDER groq or gemini (got {provider or 'none'}); "
        "use RELIABILITY_DETECTOR=rules for the offline demo")


def _parse(text: str) -> dict:
    """First JSON object in the reply; tolerates code fences and text around it."""
    text = (text or "").strip()
    decoder = json.JSONDecoder()
    index = text.find("{")
    while index >= 0:
        try:
            value, _ = decoder.raw_decode(text, index)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        index = text.find("{", index + 1)
    raise ValueError("Judge returned no JSON object")


async def complete_json(system: str, user: str) -> dict:
    s = get_settings()
    provider, model = judge_target()
    # Stay inside the reliability budget: a judge that cannot answer in time fails visibly, it is not retried past it.
    deadline = time.monotonic() + s.reliability_timeout_seconds
    timeout = min(s.model_timeout_seconds, s.reliability_timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if provider == "groq":
            body = {"model": model, "temperature": 0, "max_tokens": 1200, "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            while True:
                response = await client.post(f"{s.groq_base_url.rstrip('/')}/chat/completions",
                                             headers={"Authorization": f"Bearer {s.groq_api_key}"}, json=body)
                retry_after = response.headers.get("retry-after", "")
                wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 1.0
                if response.status_code == 429 and time.monotonic() + wait < deadline - 1:
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                data = response.json()
                usage = data.get("usage") or {}
                _record_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))
                return _parse(data["choices"][0]["message"]["content"])
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": s.google_api_key},
            json={"systemInstruction": {"parts": [{"text": system}]},
                  "contents": [{"role": "user", "parts": [{"text": user}]}],
                  "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}})
        response.raise_for_status()
        data = response.json()
        usage = data.get("usageMetadata") or {}
        _record_usage(usage.get("promptTokenCount"), usage.get("candidatesTokenCount"))
        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        return _parse("".join(p.get("text", "") for p in parts))

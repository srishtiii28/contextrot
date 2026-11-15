"""Minimal JSON-mode chat client for the judge. Separate from the tutor adapter so the judge can use
a different provider or model (JUDGE_PROVIDER / JUDGE_MODEL), which limits self-preference bias."""
import asyncio
import json
import re
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


class RateLimited(RuntimeError):
    """The provider's rate limit was hit and the wait does not fit in the check budget. The worker re-queues
    the job after `retry_after` seconds instead of marking the answer unverified."""

    def __init__(self, retry_after: float):
        super().__init__(f"rate limited; retry after {retry_after:.1f}s")
        self.retry_after = retry_after


def _duration(value: str | None) -> float | None:
    """Parses '7.5', '23.174s', '577ms', '1m26.4s'."""
    if not value:
        return None
    value = value.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", value):
        return float(value)
    total, found = 0.0, False
    for number, unit in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", value):
        found = True
        total += float(number) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total if found else None


def rate_limit_wait(response) -> float:
    """Seconds to wait after a 429, from retry-after or Groq's token/request reset headers."""
    headers = response.headers
    retry_after = _duration(headers.get("retry-after"))  # authoritative when the provider sends it
    if retry_after is not None:
        return max(1.0, retry_after)
    waits = [w for w in (_duration(headers.get(k)) for k in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests")) if w is not None]
    return max(1.0, min(waits)) if waits else 5.0


def judge_target() -> tuple[str, str]:
    s = get_settings()
    provider = (s.judge_provider or s.llm_provider or "").lower()
    if provider == "groq":
        if not (s.judge_api_key or s.groq_api_key):
            raise JudgeConfigurationError("Judge provider groq needs JUDGE_API_KEY or GROQ_API_KEY")
        return provider, s.judge_model or s.groq_model
    if provider == "gemini":
        if not (s.judge_api_key or s.google_api_key) or not (s.judge_model or s.gemini_model):
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
            body = {"model": model, "temperature": 0, "max_tokens": 700, "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            enlarged = False
            while True:
                response = await client.post(f"{s.groq_base_url.rstrip('/')}/chat/completions",
                                             headers={"Authorization": f"Bearer {s.judge_api_key or s.groq_api_key}"}, json=body)
                if response.status_code == 429:
                    wait = rate_limit_wait(response)
                    # Only short waits are absorbed here; longer ones re-queue the job so the worker is not blocked.
                    if wait <= 2 and time.monotonic() + wait < deadline - 2:
                        await asyncio.sleep(wait)
                        continue
                    raise RateLimited(wait)
                if response.status_code == 400 and "json_validate_failed" in response.text and not enlarged:
                    # The verdict ran past the output cap before the JSON closed: retry once with more room.
                    enlarged = True
                    body["max_tokens"] = 1500
                    continue
                response.raise_for_status()
                data = response.json()
                usage = data.get("usage") or {}
                _record_usage(usage.get("prompt_tokens"), usage.get("completion_tokens"))
                return _parse(data["choices"][0]["message"]["content"])
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": s.judge_api_key or s.google_api_key},
            json={"systemInstruction": {"parts": [{"text": system}]},
                  "contents": [{"role": "user", "parts": [{"text": user}]}],
                  "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}})
        if response.status_code == 429:
            raise RateLimited(rate_limit_wait(response))
        response.raise_for_status()
        data = response.json()
        usage = data.get("usageMetadata") or {}
        _record_usage(usage.get("promptTokenCount"), usage.get("candidatesTokenCount"))
        candidates = data.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        return _parse("".join(p.get("text", "") for p in parts))

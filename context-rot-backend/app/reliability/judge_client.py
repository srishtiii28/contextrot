"""Minimal JSON-mode chat client for the judge. Separate from the tutor adapter so the judge can use
a different provider or model (JUDGE_PROVIDER / JUDGE_MODEL), which limits self-preference bias."""
import asyncio
import json

import httpx

from app.core.config import get_settings


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
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Judge returned no JSON object")
    return json.loads(text[start:end + 1])


async def complete_json(system: str, user: str) -> dict:
    s = get_settings()
    provider, model = judge_target()
    timeout = s.model_timeout_seconds
    async with httpx.AsyncClient(timeout=timeout) as client:
        if provider == "groq":
            body = {"model": model, "temperature": 0, "max_tokens": 1200, "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            for attempt in range(3):
                response = await client.post(f"{s.groq_base_url.rstrip('/')}/chat/completions",
                                             headers={"Authorization": f"Bearer {s.groq_api_key}"}, json=body)
                if response.status_code == 429 and attempt < 2:
                    retry_after = response.headers.get("retry-after", "")
                    wait = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else 2.0
                    await asyncio.sleep(min(max(wait, 1.0), 6.0))
                    continue
                response.raise_for_status()
                return _parse(response.json()["choices"][0]["message"]["content"])
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": s.google_api_key},
            json={"systemInstruction": {"parts": [{"text": system}]},
                  "contents": [{"role": "user", "parts": [{"text": user}]}],
                  "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}})
        response.raise_for_status()
        candidates = response.json().get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        return _parse("".join(p.get("text", "") for p in parts))

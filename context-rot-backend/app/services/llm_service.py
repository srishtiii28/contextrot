from app.core.config import get_settings
from app.integrations.rag_client import RetrievedChunk


async def generate_response(*, user_message: str, conversation_history: list[dict[str, str]], retrieved_context: list[RetrievedChunk]) -> str:
    """Single LLM boundary. Add Gemini/OpenAI providers here, never in routes."""
    provider = get_settings().llm_provider.lower()
    if provider != "mock":
        raise NotImplementedError(f"LLM provider '{provider}' is not configured yet. Set LLM_PROVIDER=mock to run locally.")
    context_note = f" I found {len(retrieved_context)} relevant study-note chunk(s)." if retrieved_context else ""
    return f"[Mock tutor reply] You asked: {user_message}.{context_note} Configure Gemini or OpenAI in app/services/llm_service.py when the team API key is ready."


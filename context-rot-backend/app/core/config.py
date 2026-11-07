from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Context Rot Detection API"
    database_url: str = "sqlite:///./context_rot.db"
    jwt_secret_key: str = "development-only-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    llm_provider: str = "mock"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    openai_api_key: str | None = None
    google_api_key: str | None = None
    chroma_db_dir: str = "./chroma_store"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

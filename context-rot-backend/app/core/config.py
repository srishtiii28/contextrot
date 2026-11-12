from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Context Rot Detection API"
    database_url: str = "sqlite:///./context_rot.db"
    jwt_secret_key: str = "development-only-change-me"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    llm_provider: str = "mock"
    openai_api_key: str | None = None
    google_api_key: str | None = None
    gemini_model: str = ""
    model_timeout_seconds: float = 45
    embedded_worker: bool = True
    worker_poll_seconds: float = 0.25
    job_lease_seconds: int = 300
    recovery_threshold: float = 65
    enable_demo_faults: bool = True
    rag_service_url: str | None = None
    rot_service_url: str | None = None
    integration_api_key: str | None = None
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5500,http://127.0.0.1:5500,http://localhost:5173,http://127.0.0.1:5173"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

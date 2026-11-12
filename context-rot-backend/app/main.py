from contextlib import asynccontextmanager
import asyncio
from pathlib import Path
from fastapi.staticfiles import StaticFiles

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.database.database import create_tables
from app.routes import analysis, auth, chat, integrations, sessions, users, runtime

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    create_tables()
    from app.services.course_service import seed_course
    from app.services.worker import run_worker
    seed_course()
    task=asyncio.create_task(run_worker()) if settings.embedded_worker else None
    try:yield
    finally:
        if task:
            task.cancel()
            try:await task
            except asyncio.CancelledError:pass


app = FastAPI(title=settings.app_name, version="0.7.0", description="Arijit's semester 7 platform backend. Offline defaults are demonstration baselines, not validated rot detection.", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(sessions.router)
app.include_router(chat.router)
app.include_router(analysis.router)
app.include_router(integrations.router)
app.include_router(runtime.router)
demo_dir=Path(__file__).resolve().parents[2]/"context-rot-frontend"
if demo_dir.exists():app.mount("/demo",StaticFiles(directory=demo_dir,html=True),name="demo")


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"message": "Context Rot Detection API is running"}


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {"status": "healthy"}

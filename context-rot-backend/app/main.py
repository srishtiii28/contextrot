from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.database.database import create_tables, get_db
from app.database.models import DocumentImage
from app.routes import analysis, auth, chat, integrations, knowledge_bases, sessions, users

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    create_tables()
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_private_network_headers(request, call_next):
    if request.method == "OPTIONS" and request.headers.get("Access-Control-Request-Private-Network"):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response
    response = await call_next(request)
    response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(sessions.router)
app.include_router(chat.router)
app.include_router(analysis.router)
app.include_router(integrations.router)
app.include_router(knowledge_bases.router)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"message": "Context Rot Detection API is running"}


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/images/{image_id}", tags=["images"])
def get_image(image_id: int, db=Depends(get_db)):
    image = db.get(DocumentImage, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return Response(content=image.image_data, media_type="image/png")

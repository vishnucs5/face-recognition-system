from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.comfyui import router as comfyui_router
from app.api.controlnet import router as controlnet_router
from app.api.copilot import router as copilot_router
from app.api.diagnostics import router as diagnostics_router
from app.api.generations import router as generations_router
from app.api.health import router as health_router
from app.api.loras import router as loras_router
from app.api.models import router as models_router
from app.api.presets import router as presets_router
from app.api.queue import router as queue_router
from app.api.status import router as status_router
from app.api.training import router as training_router
from app.api.websocket import router as ws_router
from app.config import settings
from app.database import async_session_maker, init_db
from app.services.queue_scheduler import queue_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure tables exist and directories are created
    await init_db()
    async with async_session_maker() as session:
        await queue_scheduler.reconcile_stale_jobs(session)
    settings.generated_path
    settings.uploads_path
    yield


app = FastAPI(
    title="Forge Studio Backend",
    version="0.1.0",
    description="Local AI Image Generation Workstation API for SDXL and ComfyUI",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Local workstation development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount local data directory for static image serving
settings.data_path.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=str(settings.data_path)), name="data")

# Include API Routers under /api
app.include_router(health_router, prefix="/api")
app.include_router(status_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(loras_router, prefix="/api")
app.include_router(training_router, prefix="/api")
app.include_router(comfyui_router, prefix="/api")
app.include_router(controlnet_router, prefix="/api")
app.include_router(generations_router, prefix="/api")
app.include_router(presets_router, prefix="/api")
app.include_router(queue_router, prefix="/api")
app.include_router(copilot_router, prefix="/api")
app.include_router(diagnostics_router, prefix="/api")
app.include_router(ws_router, prefix="/api")


@app.get("/")
async def root():
    return {
        "app": "Forge Studio API",
        "status": "online",
        "docs_url": "/docs",
        "health_url": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)

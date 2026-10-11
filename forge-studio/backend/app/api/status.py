from fastapi import APIRouter
from app.config import settings
from app.schemas.system import SystemStatusResponse, TelemetryResponse
from app.services.comfyui_client import ComfyUIClient
from app.services.gpu_monitor import GpuMonitor
from app.services.gpu_safety import gpu_safety_manager
from app.services.ollama_client import OllamaClient

router = APIRouter(tags=["Status"])


@router.get("/status", response_model=SystemStatusResponse)
async def system_status():
    comfy_client = ComfyUIClient()
    ollama_client = OllamaClient()

    comfy_online = await comfy_client.is_available()
    ollama_online = await ollama_client.is_available()
    gpu_stats = GpuMonitor.get_status()

    return SystemStatusResponse(
        comfyui_online=comfy_online,
        comfyui_url=settings.COMFYUI_URL,
        ollama_online=ollama_online,
        ollama_url=settings.OLLAMA_URL,
        ollama_model=settings.OLLAMA_MODEL,
        demo_mode=settings.DEMO_MODE or not comfy_online,
        gpu=gpu_stats
    )


@router.get("/system/telemetry", response_model=TelemetryResponse)
@router.get("/telemetry", response_model=TelemetryResponse, include_in_schema=False)
async def system_telemetry():
    return await gpu_safety_manager.get_telemetry()


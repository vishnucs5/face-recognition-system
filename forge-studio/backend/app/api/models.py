from fastapi import APIRouter
from app.schemas.system import ModelInfo
from app.services.comfyui_client import ComfyUIClient

router = APIRouter(tags=["Models"])


@router.get("/models", response_model=list[ModelInfo])
async def list_models():
    client = ComfyUIClient()
    return await client.get_checkpoints()

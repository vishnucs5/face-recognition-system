from fastapi import APIRouter
from app.schemas.generation import ControlNetModelInfo
from app.services.comfyui_client import ComfyUIClient

router = APIRouter(prefix="/controlnet", tags=["ControlNet"])


@router.get("/models", response_model=list[ControlNetModelInfo])
async def list_controlnet_models():
    """Returns available ControlNet and T2I-Adapter models from ComfyUI and local storage."""
    client = ComfyUIClient()
    return await client.get_controlnet_models()


@router.get("/types")
async def list_controlnet_types():
    """Returns supported ControlNet types and preprocessors."""
    return [
        {
            "id": "canny",
            "name": "Canny Edge",
            "description": "High-fidelity edge detection preserving contours and structure",
            "preprocessor": "canny",
            "default_strength": 1.0,
        },
        {
            "id": "depth",
            "name": "Depth Map",
            "description": "3D spatial depth estimation preserving perspective and volume",
            "preprocessor": "depth",
            "default_strength": 0.9,
        },
        {
            "id": "openpose",
            "name": "OpenPose",
            "description": "Human skeletal pose tracking for figure positioning and anatomy",
            "preprocessor": "openpose",
            "default_strength": 1.0,
        },
        {
            "id": "lineart",
            "name": "Line Art",
            "description": "Crisp illustration contours and comic ink drawings",
            "preprocessor": "lineart",
            "default_strength": 1.0,
        },
        {
            "id": "soft_edge",
            "name": "Soft Edge (HED)",
            "description": "Smooth organic boundary transitions and painterly curves",
            "preprocessor": "soft_edge",
            "default_strength": 0.85,
        },
    ]

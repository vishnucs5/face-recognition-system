from pathlib import Path
from typing import Any
import httpx
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status
from pydantic import BaseModel

from app.config import settings
from app.services.comfyui_client import ComfyUIClient

router = APIRouter(prefix="/comfyui", tags=["ComfyUI Proxy"])
client = ComfyUIClient()


class PromptSubmission(BaseModel):
    prompt: dict[str, Any]
    client_id: str | None = None


@router.get("/status")
async def comfyui_status():
    """Check ComfyUI server connectivity and telemetry."""
    try:
        async with httpx.AsyncClient(timeout=2.0) as http_client:
            res = await http_client.get(f"{client.base_url}/system_stats")
            if res.status_code == 200:
                return {
                    "online": True,
                    "status": "connected",
                    "url": client.base_url,
                    "stats": res.json()
                }
    except Exception as e:
        return {
            "online": False,
            "status": "offline",
            "url": client.base_url,
            "error": str(e)
        }
    return {"online": False, "status": "offline", "url": client.base_url}


@router.get("/checkpoints")
async def comfyui_checkpoints():
    """Fetch all available checkpoints directly from the connected ComfyUI engine."""
    checkpoints = await client.get_checkpoints()
    return checkpoints


@router.get("/loras")
async def comfyui_loras():
    """Fetch all available LoRA models from the connected ComfyUI engine."""
    loras = await client.get_loras()
    return loras


@router.post("/prompt")
async def comfyui_prompt(submission: PromptSubmission):
    """Proxy prompt graph submission directly to ComfyUI."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            payload = {
                "prompt": submission.prompt,
                "client_id": submission.client_id or client.client_id
            }
            res = await http_client.post(f"{client.base_url}/prompt", json=payload)
            if res.status_code == 200:
                return res.json()
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"ComfyUI is offline at {client.base_url}."
        )


@router.get("/history/{prompt_id}")
async def comfyui_history(prompt_id: str):
    """Proxy execution history retrieval for a prompt ID."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as http_client:
            res = await http_client.get(f"{client.base_url}/history/{prompt_id}")
            if res.status_code == 200:
                return res.json()
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except httpx.ConnectError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ComfyUI offline.")


@router.get("/queue")
async def comfyui_queue():
    """Proxy ComfyUI queue inspection."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as http_client:
            res = await http_client.get(f"{client.base_url}/queue")
            if res.status_code == 200:
                return res.json()
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except httpx.ConnectError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ComfyUI offline.")


@router.post("/interrupt")
async def comfyui_interrupt():
    """Proxy interrupt command to halt active ComfyUI sampling."""
    success = await client.interrupt()
    return {"interrupted": success}


@router.post("/upload")
async def comfyui_upload(
    image: UploadFile = File(...),
    overwrite: bool = Form(True)
):
    """Upload a source image to ComfyUI input directory."""
    content = await image.read()
    filename = image.filename or "upload.png"

    try:
        async with httpx.AsyncClient(timeout=15.0) as http_client:
            files = {"image": (filename, content, image.content_type or "image/png")}
            data = {"overwrite": str(overwrite).lower()}
            res = await http_client.post(f"{client.base_url}/upload/image", files=files, data=data)
            if res.status_code == 200:
                return res.json()
            raise HTTPException(status_code=res.status_code, detail=res.text)
    except httpx.ConnectError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ComfyUI offline.")


@router.get("/view")
async def comfyui_view(filename: str, subfolder: str = "", type: str = "output"):
    """Proxy image viewing directly from ComfyUI."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client:
            view_url = f"{client.base_url}/view?filename={filename}&subfolder={subfolder}&type={type}"
            res = await http_client.get(view_url)
            if res.status_code == 200:
                media_type = res.headers.get("content-type", "image/png")
                return Response(content=res.content, media_type=media_type)
            raise HTTPException(status_code=res.status_code, detail="Image not found in ComfyUI.")
    except httpx.ConnectError:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ComfyUI offline.")

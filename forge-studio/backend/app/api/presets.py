import json
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.preset import (
    PresetCreateRequest,
    PresetDiffReview,
    PresetModel,
    PresetUpdateRequest,
)
from app.services.preset_engine import preset_engine

router = APIRouter(prefix="/presets", tags=["Presets"])


class DiffRequest(BaseModel):
    preset_id: str
    current_form: dict[str, Any] = Field(default_factory=dict)


class ImportRequest(BaseModel):
    json_data: str | None = None


@router.get("", response_model=list[PresetModel])
async def list_presets(db: AsyncSession = Depends(get_db)):
    """List all presets (merges built-in and user presets)."""
    return await preset_engine.get_all_presets(db)


@router.post("", response_model=PresetModel, status_code=status.HTTP_201_CREATED)
async def create_preset(req: PresetCreateRequest, db: AsyncSession = Depends(get_db)):
    """Create a new user workflow preset."""
    return await preset_engine.create_user_preset(db, req)


@router.get("/export")
async def export_presets(db: AsyncSession = Depends(get_db)):
    """Export all user presets to JSON format."""
    json_str = await preset_engine.export_presets_json(db)
    return Response(content=json_str, media_type="application/json")


@router.post("/diff", response_model=PresetDiffReview)
async def diff_preset(req: DiffRequest, db: AsyncSession = Depends(get_db)):
    """Side-by-side comparison between preset configuration and current form state."""
    preset = await preset_engine.get_preset_by_id(db, req.preset_id)
    if not preset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset with id '{req.preset_id}' not found.",
        )
    return preset_engine.calculate_diff(preset, req.current_form)


@router.post("/import")
async def import_presets(request: Request, db: AsyncSession = Depends(get_db)):
    """Import user presets from JSON payload with strict validation and sanitization."""
    # Support both raw JSON body and JSON object with 'json_data'
    raw_body = await request.body()
    try:
        body_text = raw_body.decode("utf-8")
        parsed = json.loads(body_text)
        if isinstance(parsed, dict) and "json_data" in parsed and isinstance(parsed["json_data"], str):
            json_str = parsed["json_data"]
        else:
            json_str = body_text
    except Exception as e:
        return {"imported_count": 0, "errors": [f"Malformed JSON body: {str(e)}"]}

    imported_count, errors = await preset_engine.import_presets_json(db, json_str)
    return {"imported_count": imported_count, "errors": errors}


@router.get("/{id}", response_model=PresetModel)
async def get_preset(id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve a preset by id, including dependency status verification."""
    preset = await preset_engine.get_preset_by_id(db, id)
    if not preset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Preset with id '{id}' not found.",
        )
    preset.dependency_status = await preset_engine.inspect_dependencies(preset)
    return preset


@router.put("/{id}", response_model=PresetModel)
async def update_preset(id: str, req: PresetUpdateRequest, db: AsyncSession = Depends(get_db)):
    """Update an existing user preset (immutable for built-in presets)."""
    return await preset_engine.update_user_preset(db, id, req)


@router.delete("/{id}")
async def delete_preset(id: str, db: AsyncSession = Depends(get_db)):
    """Delete a user preset (rejected for built-in presets)."""
    await preset_engine.delete_user_preset(db, id)
    return {"status": "deleted"}


@router.post("/{id}/duplicate", response_model=PresetModel)
async def duplicate_preset(id: str, db: AsyncSession = Depends(get_db)):
    """Duplicate an existing preset (creates a new user preset copy)."""
    return await preset_engine.duplicate_user_preset(db, id)

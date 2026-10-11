from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.lora import LoraImportRequest, LoraUpdateRequest, LoraVaultItem
from app.services.lora_service import LoraService

router = APIRouter(prefix="/loras", tags=["LoRAs"])


@router.get("", response_model=list[LoraVaultItem])
async def list_loras(db: AsyncSession = Depends(get_db)):
    return await LoraService.list_loras(db)


@router.post("/import", response_model=LoraVaultItem)
async def import_lora(req: LoraImportRequest, db: AsyncSession = Depends(get_db)):
    try:
        return await LoraService.import_lora(db, req)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import LoRA: {e}")


@router.get("/{lora_id}", response_model=LoraVaultItem)
async def get_lora(lora_id: str, db: AsyncSession = Depends(get_db)):
    item = await LoraService.get_lora(db, lora_id)
    if not item:
        raise HTTPException(status_code=404, detail="LoRA not found in index.")
    return item


@router.patch("/{lora_id}", response_model=LoraVaultItem)
async def update_lora(lora_id: str, req: LoraUpdateRequest, db: AsyncSession = Depends(get_db)):
    try:
        return await LoraService.update_lora(db, lora_id, req)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update LoRA: {e}")


@router.delete("/{lora_id}")
async def remove_lora(lora_id: str, db: AsyncSession = Depends(get_db)):
    success = await LoraService.remove_lora(db, lora_id)
    if not success:
        raise HTTPException(status_code=404, detail="LoRA not found in index.")
    return {"status": "removed", "id": lora_id}

import copy
import json
import uuid
from datetime import datetime, timezone
from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.preset import PresetRecord
from app.schemas.generation import LoraInput
from app.schemas.preset import (
    DISALLOWED_PATTERNS,
    PresetCreateRequest,
    PresetDiffReview,
    PresetModel,
    PresetUpdateRequest,
    sanitize_text,
)
from app.services.comfyui_client import ComfyUIClient
from app.services.queue_scheduler import queue_scheduler


FACTORY_BUILTIN_PRESETS: list[dict] = [
    {
        "id": "builtin-portrait",
        "name": "Portrait",
        "description": "Optimized for detailed facial features, realistic skin textures, and portrait studio lighting",
        "category": "built-in",
        "version": 1,
        "is_built_in": True,
        "mode": "txt2img",
        "base_model_family": "sd15",
        "checkpoint_name": "v1-5-pruned-emaonly.safetensors",
        "prompt_template": "Stunning studio portrait photo of {subject}, high detail, 85mm lens, soft studio lighting, sharp focus, 8k",
        "negative_prompt": "blurry, low quality, deformed, bad anatomy, bad eyes, extra limbs, amputee, missing fingers",
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "cfg_scale": 6.5,
        "sampler": "euler",
        "scheduler": "karras",
        "seed_policy": "random",
        "fixed_seed": None,
        "denoise": 0.75,
        "loras": [],
        "controlnet_settings": {},
        "vram_profile": "balanced",
    },
    {
        "id": "builtin-cinematic-landscape",
        "name": "Cinematic landscape",
        "description": "Ultra-wide panoramic scenic vistas with atmospheric cinematic grading and volumetric depth",
        "category": "built-in",
        "version": 1,
        "is_built_in": True,
        "mode": "txt2img",
        "base_model_family": "sdxl",
        "checkpoint_name": "sd_xl_base_1.0.safetensors",
        "prompt_template": "Breathtaking panoramic cinematic landscape of {subject}, golden hour, dramatic volumetric clouds, epic vista, photorealistic, 8k",
        "negative_prompt": "blurry, low quality, oversaturated, watermark, signature, hazy, noisy, artifacts",
        "width": 1344,
        "height": 768,
        "steps": 30,
        "cfg_scale": 6.5,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "seed_policy": "random",
        "fixed_seed": None,
        "denoise": 0.75,
        "loras": [],
        "controlnet_settings": {},
        "vram_profile": "balanced",
    },
    {
        "id": "builtin-product-photography",
        "name": "Product photography",
        "description": "Clean studio commercial product presentation with neutral balanced studio lighting",
        "category": "built-in",
        "version": 1,
        "is_built_in": True,
        "mode": "txt2img",
        "base_model_family": "sdxl",
        "checkpoint_name": "sd_xl_base_1.0.safetensors",
        "prompt_template": "Commercial product photography of {subject}, clean neutral lighting, studio cyclorama background, crisp sharp details, advertising aesthetic, 8k",
        "negative_prompt": "clean neutral lighting studio negative: messy background, shadows, harsh reflections, blurry, low resolution, dust, scratches, watermark",
        "width": 1152,
        "height": 896,
        "steps": 32,
        "cfg_scale": 6.5,
        "sampler": "euler",
        "scheduler": "karras",
        "seed_policy": "random",
        "fixed_seed": None,
        "denoise": 0.75,
        "loras": [],
        "controlnet_settings": {},
        "vram_profile": "balanced",
    },
    {
        "id": "builtin-fast-draft",
        "name": "Fast draft",
        "description": "High-speed iteration preset (~10s generation) for rapid concept exploration and composition vetting",
        "category": "built-in",
        "version": 1,
        "is_built_in": True,
        "mode": "txt2img",
        "base_model_family": "sd15",
        "checkpoint_name": "v1-5-pruned-emaonly.safetensors",
        "prompt_template": "Concept sketch of {subject}, vibrant colors, expressive composition, digital art",
        "negative_prompt": "blurry, worst quality, low quality",
        "width": 768,
        "height": 768,
        "steps": 16,
        "cfg_scale": 6.0,
        "sampler": "euler",
        "scheduler": "karras",
        "seed_policy": "random",
        "fixed_seed": None,
        "denoise": 0.75,
        "loras": [],
        "controlnet_settings": {},
        "vram_profile": "speed",
    },
    {
        "id": "builtin-inpainting-repair",
        "name": "Inpainting repair",
        "description": "Conservative inpainting repair for targeted seamless defect corrections with mask padding",
        "category": "built-in",
        "version": 1,
        "is_built_in": True,
        "mode": "inpaint",
        "base_model_family": "sd15",
        "checkpoint_name": "v1-5-pruned-emaonly.safetensors",
        "prompt_template": "Seamless inpaint repair of {subject}, matching surroundings, flawless texture, photorealistic",
        "negative_prompt": "blurry, seam, visible edges, artifacts, mismatching style",
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "cfg_scale": 6.5,
        "sampler": "euler",
        "scheduler": "karras",
        "seed_policy": "random",
        "fixed_seed": None,
        "denoise": 0.55,
        "loras": [],
        "controlnet_settings": {"mask_padding": 32},
        "vram_profile": "conservative",
    },
    {
        "id": "builtin-controlnet-composition",
        "name": "ControlNet composition",
        "description": "Structural edge guidance using Canny preprocessor for precise pose and composition",
        "category": "built-in",
        "version": 1,
        "is_built_in": True,
        "mode": "controlnet",
        "base_model_family": "sd15",
        "checkpoint_name": "v1-5-pruned-emaonly.safetensors",
        "prompt_template": "Detailed rendering of {subject}, coherent composition, perfect outlines, masterpiece",
        "negative_prompt": "blurry, deformed, distorted, chaotic background",
        "width": 1024,
        "height": 1024,
        "steps": 28,
        "cfg_scale": 6.5,
        "sampler": "euler",
        "scheduler": "karras",
        "seed_policy": "random",
        "fixed_seed": None,
        "denoise": 0.75,
        "loras": [],
        "controlnet_settings": {
            "controlnet_type": "canny",
            "controlnet_preprocessor": "canny",
            "controlnet_strength": 0.85,
            "controlnet_start_percent": 0.0,
            "controlnet_end_percent": 1.0,
        },
        "vram_profile": "balanced",
    },
    {
        "id": "builtin-lora-style-test",
        "name": "LoRA style test",
        "description": "Evaluates single LoRA adaptation with controlled baseline strength (0.75)",
        "category": "built-in",
        "version": 1,
        "is_built_in": True,
        "mode": "txt2img",
        "base_model_family": "sd15",
        "checkpoint_name": "v1-5-pruned-emaonly.safetensors",
        "prompt_template": "Stylized rendering of {subject}, high fidelity, sharp details, artistic flair",
        "negative_prompt": "blurry, oversaturated, low quality, artifact",
        "width": 1024,
        "height": 1024,
        "steps": 26,
        "cfg_scale": 6.5,
        "sampler": "euler",
        "scheduler": "karras",
        "seed_policy": "random",
        "fixed_seed": None,
        "denoise": 0.75,
        "loras": [{"name": "style_adaptation.safetensors", "strength": 0.75}],
        "controlnet_settings": {},
        "vram_profile": "balanced",
    },
]


class PresetEngine:
    @staticmethod
    def _record_to_model(rec: PresetRecord) -> PresetModel:
        loras = []
        for l in rec.loras:
            if isinstance(l, dict):
                loras.append(LoraInput(**l))
            elif isinstance(l, LoraInput):
                loras.append(l)

        return PresetModel(
            schema_version=1,
            id=rec.id,
            name=rec.name,
            description=rec.description or "",
            category=rec.category,
            version=rec.version,
            is_built_in=rec.is_built_in,
            mode=rec.mode,
            base_model_family=rec.base_model_family,
            checkpoint_name=rec.checkpoint_name,
            prompt_template=rec.prompt_template,
            negative_prompt=rec.negative_prompt or "",
            width=rec.width,
            height=rec.height,
            steps=rec.steps,
            cfg_scale=rec.cfg_scale,
            sampler=rec.sampler,
            scheduler=rec.scheduler,
            seed_policy=rec.seed_policy,
            fixed_seed=rec.fixed_seed,
            denoise=rec.denoise,
            loras=loras,
            controlnet_settings=rec.controlnet_settings,
            vram_profile=rec.vram_profile,
            created_at=rec.created_at.isoformat() if rec.created_at else None,
            updated_at=rec.updated_at.isoformat() if rec.updated_at else None,
        )

    def get_built_in_presets(self) -> list[PresetModel]:
        presets = []
        for p in FACTORY_BUILTIN_PRESETS:
            item = copy.deepcopy(p)
            loras = [LoraInput(**l) for l in item.get("loras", [])]
            item["loras"] = loras
            presets.append(PresetModel(**item))
        return presets

    async def get_all_presets(self, db: AsyncSession) -> list[PresetModel]:
        built_ins = self.get_built_in_presets()
        stmt = select(PresetRecord).order_by(PresetRecord.created_at.desc())
        res = await db.execute(stmt)
        user_records = res.scalars().all()
        user_presets = [self._record_to_model(r) for r in user_records]
        return built_ins + user_presets

    async def get_preset_by_id(self, db: AsyncSession, preset_id: str) -> PresetModel | None:
        for b in self.get_built_in_presets():
            if b.id == preset_id:
                return b

        stmt = select(PresetRecord).where(PresetRecord.id == preset_id)
        res = await db.execute(stmt)
        rec = res.scalar_one_or_none()
        if rec:
            return self._record_to_model(rec)
        return None

    async def create_user_preset(self, db: AsyncSession, req: PresetCreateRequest) -> PresetModel:
        new_id = str(uuid.uuid4())
        record = PresetRecord(
            id=new_id,
            name=req.name,
            description=req.description,
            category="user",
            version=1,
            is_built_in=False,
            mode=req.mode,
            base_model_family=req.base_model_family,
            checkpoint_name=req.checkpoint_name,
            prompt_template=req.prompt_template,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            steps=req.steps,
            cfg_scale=req.cfg_scale,
            sampler=req.sampler,
            scheduler=req.scheduler,
            seed_policy=req.seed_policy,
            fixed_seed=req.fixed_seed,
            denoise=req.denoise,
            loras_json=json.dumps([l.model_dump() for l in req.loras]),
            controlnet_settings_json=json.dumps(req.controlnet_settings),
            vram_profile=req.vram_profile,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(record)
        await db.commit()
        await db.refresh(record)
        return self._record_to_model(record)

    async def update_user_preset(
        self, db: AsyncSession, preset_id: str, req: PresetUpdateRequest
    ) -> PresetModel:
        builtin_ids = {b.id for b in self.get_built_in_presets()}
        if preset_id in builtin_ids or preset_id.startswith("builtin-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Built-in presets are immutable and cannot be updated.",
            )

        stmt = select(PresetRecord).where(PresetRecord.id == preset_id)
        res = await db.execute(stmt)
        rec = res.scalar_one_or_none()
        if not rec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset with id '{preset_id}' not found.",
            )

        if req.name is not None:
            rec.name = req.name
        if req.description is not None:
            rec.description = req.description
        if req.mode is not None:
            rec.mode = req.mode
        if req.base_model_family is not None:
            rec.base_model_family = req.base_model_family
        if req.checkpoint_name is not None:
            rec.checkpoint_name = req.checkpoint_name
        if req.prompt_template is not None:
            rec.prompt_template = req.prompt_template
        if req.negative_prompt is not None:
            rec.negative_prompt = req.negative_prompt
        if req.width is not None:
            rec.width = req.width
        if req.height is not None:
            rec.height = req.height
        if req.steps is not None:
            rec.steps = req.steps
        if req.cfg_scale is not None:
            rec.cfg_scale = req.cfg_scale
        if req.sampler is not None:
            rec.sampler = req.sampler
        if req.scheduler is not None:
            rec.scheduler = req.scheduler
        if req.seed_policy is not None:
            rec.seed_policy = req.seed_policy
        if req.fixed_seed is not None:
            rec.fixed_seed = req.fixed_seed
        if req.denoise is not None:
            rec.denoise = req.denoise
        if req.loras is not None:
            rec.loras_json = json.dumps([l.model_dump() for l in req.loras])
        if req.controlnet_settings is not None:
            rec.controlnet_settings_json = json.dumps(req.controlnet_settings)
        if req.vram_profile is not None:
            rec.vram_profile = req.vram_profile

        rec.version = rec.version + 1
        rec.updated_at = datetime.now(timezone.utc)

        await db.commit()
        await db.refresh(rec)
        return self._record_to_model(rec)

    async def delete_user_preset(self, db: AsyncSession, preset_id: str) -> bool:
        builtin_ids = {b.id for b in self.get_built_in_presets()}
        if preset_id in builtin_ids or preset_id.startswith("builtin-"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Built-in presets are immutable and cannot be deleted.",
            )

        stmt = select(PresetRecord).where(PresetRecord.id == preset_id)
        res = await db.execute(stmt)
        rec = res.scalar_one_or_none()
        if not rec:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset with id '{preset_id}' not found.",
            )

        await db.delete(rec)
        await db.commit()
        return True

    async def duplicate_user_preset(self, db: AsyncSession, preset_id: str) -> PresetModel:
        source_preset = await self.get_preset_by_id(db, preset_id)
        if not source_preset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Preset with id '{preset_id}' not found.",
            )

        new_name = f"{source_preset.name} (Copy)"
        # ensure length <= 100
        if len(new_name) > 100:
            new_name = new_name[:93] + " (Copy)"

        req = PresetCreateRequest(
            name=new_name,
            description=source_preset.description,
            mode=source_preset.mode,
            base_model_family=source_preset.base_model_family,
            checkpoint_name=source_preset.checkpoint_name,
            prompt_template=source_preset.prompt_template,
            negative_prompt=source_preset.negative_prompt,
            width=source_preset.width,
            height=source_preset.height,
            steps=source_preset.steps,
            cfg_scale=source_preset.cfg_scale,
            sampler=source_preset.sampler,
            scheduler=source_preset.scheduler,
            seed_policy=source_preset.seed_policy,
            fixed_seed=source_preset.fixed_seed,
            denoise=source_preset.denoise,
            loras=source_preset.loras,
            controlnet_settings=source_preset.controlnet_settings,
            vram_profile=source_preset.vram_profile,
        )
        return await self.create_user_preset(db, req)

    async def export_presets_json(self, db: AsyncSession) -> str:
        stmt = select(PresetRecord).order_by(PresetRecord.created_at)
        res = await db.execute(stmt)
        records = res.scalars().all()
        user_presets = [self._record_to_model(r).model_dump() for r in records]

        export_data = {
            "schema_version": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "presets": user_presets,
        }
        return json.dumps(export_data, indent=2)

    async def import_presets_json(
        self, db: AsyncSession, json_str: str
    ) -> tuple[int, list[str]]:
        errors: list[str] = []
        try:
            data = json.loads(json_str)
        except Exception as e:
            return 0, [f"Invalid JSON payload: {str(e)}"]

        if not isinstance(data, dict):
            return 0, ["Top-level JSON structure must be an object."]

        schema_version = data.get("schema_version")
        if schema_version != 1:
            return 0, [f"Unsupported or missing schema_version (expected 1, got {schema_version})."]

        presets_raw = data.get("presets", [])
        if not isinstance(presets_raw, list):
            return 0, ["'presets' field must be an array."]

        imported_count = 0
        for idx, item in enumerate(presets_raw):
            if not isinstance(item, dict):
                errors.append(f"Preset #{idx + 1}: expected an object.")
                continue

            try:
                # Validate and sanitize all fields through PresetCreateRequest
                create_req = PresetCreateRequest(**item)
                await self.create_user_preset(db, create_req)
                imported_count += 1
            except Exception as e:
                errors.append(f"Preset #{idx + 1} ('{item.get('name', 'unnamed')}'): {str(e)}")

        return imported_count, errors

    async def inspect_dependencies(self, preset: PresetModel) -> dict[str, bool]:
        client = ComfyUIClient()
        results: dict[str, bool] = {}
        all_satisfied = True

        if preset.checkpoint_name:
            checkpoints = await client.get_checkpoints()
            names = {c.filename.lower() for c in checkpoints} | {c.name.lower() for c in checkpoints}
            exists = preset.checkpoint_name.lower() in names
            results[f"checkpoint:{preset.checkpoint_name}"] = exists
            if not exists:
                all_satisfied = False

        if preset.loras:
            loras = await client.get_loras()
            lora_names = {l.filename.lower() for l in loras} | {l.name.lower() for l in loras}
            for lora in preset.loras:
                exists = lora.name.lower() in lora_names
                results[f"lora:{lora.name}"] = exists
                if not exists:
                    all_satisfied = False

        if preset.controlnet_settings:
            cnet_model = preset.controlnet_settings.get("controlnet_model")
            if cnet_model:
                cnet_models = await client.get_controlnet_models()
                cnet_names = {m.filename.lower() for m in cnet_models} | {m.name.lower() for m in cnet_models}
                exists = cnet_model.lower() in cnet_names
                results[f"controlnet:{cnet_model}"] = exists
                if not exists:
                    all_satisfied = False

        results["all_satisfied"] = all_satisfied
        return results

    def calculate_diff(self, preset: PresetModel, current_form: dict) -> PresetDiffReview:
        differences: dict[str, dict[str, Any]] = {}

        fields_to_compare = [
            ("mode", preset.mode, current_form.get("mode")),
            ("checkpoint_name", preset.checkpoint_name, current_form.get("model") or current_form.get("checkpoint_name")),
            ("prompt", preset.prompt_template, current_form.get("prompt")),
            ("negative_prompt", preset.negative_prompt, current_form.get("negative_prompt")),
            ("width", preset.width, current_form.get("width")),
            ("height", preset.height, current_form.get("height")),
            ("steps", preset.steps, current_form.get("steps")),
            ("cfg_scale", preset.cfg_scale, current_form.get("cfg_scale")),
            ("sampler", preset.sampler, current_form.get("sampler")),
            ("scheduler", preset.scheduler, current_form.get("scheduler")),
            ("seed_policy", preset.seed_policy, current_form.get("seed_policy")),
            ("fixed_seed", preset.fixed_seed, current_form.get("seed") or current_form.get("fixed_seed")),
            ("denoise", preset.denoise, current_form.get("denoise")),
            ("vram_profile", preset.vram_profile, current_form.get("vram_profile")),
        ]

        for field_name, p_val, c_val in fields_to_compare:
            if c_val is not None and p_val != c_val:
                differences[field_name] = {"preset": p_val, "current": c_val}

        # Compare loras
        curr_loras = current_form.get("loras", [])
        if isinstance(curr_loras, list):
            norm_curr_loras = []
            for l in curr_loras:
                if isinstance(l, dict):
                    norm_curr_loras.append({"name": l.get("name"), "strength": l.get("strength", 0.8)})
                elif hasattr(l, "model_dump"):
                    norm_curr_loras.append(l.model_dump())
            p_loras = [l.model_dump() for l in preset.loras]
            if p_loras != norm_curr_loras:
                differences["loras"] = {"preset": p_loras, "current": norm_curr_loras}

        # Compare controlnet
        curr_cnet = current_form.get("controlnet_settings")
        if curr_cnet is not None and preset.controlnet_settings != curr_cnet:
            differences["controlnet_settings"] = {
                "preset": preset.controlnet_settings,
                "current": curr_cnet,
            }

        # Estimate VRAM
        preset_vram, _ = queue_scheduler.estimate_resources(
            model=preset.checkpoint_name or "sd_xl_base_1.0.safetensors",
            width=preset.width,
            height=preset.height,
            steps=preset.steps,
            loras_count=len(preset.loras),
            has_controlnet=bool(preset.controlnet_settings),
        )

        curr_model = current_form.get("model") or current_form.get("checkpoint_name") or preset.checkpoint_name or "sd_xl_base_1.0.safetensors"
        curr_w = current_form.get("width") or preset.width
        curr_h = current_form.get("height") or preset.height
        curr_steps = current_form.get("steps") or preset.steps
        curr_loras_count = len(curr_loras) if isinstance(curr_loras, list) else 0
        curr_has_cnet = bool(current_form.get("controlnet_type") or current_form.get("controlnet_model") or curr_cnet)

        curr_vram, _ = queue_scheduler.estimate_resources(
            model=curr_model,
            width=curr_w,
            height=curr_h,
            steps=curr_steps,
            loras_count=curr_loras_count,
            has_controlnet=curr_has_cnet,
        )

        vram_diff = curr_vram - preset_vram
        summary = (
            f"{len(differences)} field(s) differ from preset '{preset.name}'. "
            f"VRAM change: {'+' if vram_diff > 0 else ''}{vram_diff} MB."
        )

        return PresetDiffReview(
            preset_id=preset.id,
            preset_name=preset.name,
            differences=differences,
            preset_estimated_vram_mb=preset_vram,
            current_estimated_vram_mb=curr_vram,
            estimated_vram_diff_mb=vram_diff,
            summary=summary,
        )


preset_engine = PresetEngine()

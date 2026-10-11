import asyncio
import copy
import json
import random
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any
import httpx
import websockets

from app.config import settings
from app.schemas.generation import GenerationProgressUpdate, GenerationRequest, ControlNetModelInfo
from app.schemas.system import LoraInfo, ModelInfo


class ComfyUIClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.COMFYUI_URL).rstrip("/")
        self.client_id = str(uuid.uuid4())

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                res = await client.get(f"{self.base_url}/system_stats")
                return res.status_code == 200
        except Exception:
            return False

    async def interrupt(self) -> bool:
        """Sends an interrupt signal to stop any currently sampling ComfyUI workflow."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.post(f"{self.base_url}/interrupt")
                return res.status_code == 200
        except Exception:
            return False

    async def delete_queue_item(self, prompt_id: str) -> bool:
        """Removes a pending prompt from ComfyUI queue via /queue {"delete": [prompt_id]}."""
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.post(f"{self.base_url}/queue", json={"delete": [prompt_id]})
                return res.status_code == 200
        except Exception:
            return False

    async def upload_image(self, image_path: Path) -> str:
        """Uploads an image to ComfyUI's input directory via /upload/image."""
        if not image_path.exists():
            raise FileNotFoundError(f"Source image {image_path} does not exist.")

        async with httpx.AsyncClient(timeout=15.0) as client:
            with open(image_path, "rb") as f:
                files = {"image": (image_path.name, f, "image/png")}
                data = {"overwrite": "true"}
                res = await client.post(f"{self.base_url}/upload/image", files=files, data=data)
                if res.status_code == 200:
                    info = res.json()
                    return info.get("name", image_path.name)
                raise RuntimeError(f"Failed to upload image to ComfyUI: {res.text}")

    async def get_checkpoints(self) -> list[ModelInfo]:
        """Fetch models from ComfyUI API or fallback to scanning models/checkpoints."""
        models: list[ModelInfo] = []
        try:
            if await self.is_available():
                async with httpx.AsyncClient(timeout=2.0) as client:
                    res = await client.get(f"{self.base_url}/object_info/CheckpointLoaderSimple")
                    if res.status_code == 200:
                        data = res.json()
                        input_info = data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {})
                        ckpt_raw = input_info.get("ckpt_name", [[]])
                        ckpt_items = ckpt_raw[0] if ckpt_raw else []
                        if isinstance(ckpt_items, list):
                            for name in ckpt_items:
                                if name and isinstance(name, str):
                                    models.append(ModelInfo(name=Path(name).stem, filename=name))
                        elif isinstance(ckpt_items, str) and ckpt_items and not ckpt_items.startswith("@{"):
                            models.append(ModelInfo(name=Path(ckpt_items).stem, filename=ckpt_items))
                        if models:
                            return models
        except Exception:
            pass

        # Scan local directory & ComfyUI models directory
        search_dirs = [settings.checkpoints_path, Path("c:/vishnu/ComfyUI/models/checkpoints")]
        seen_filenames = set()
        for ckpt_dir in search_dirs:
            if ckpt_dir.exists():
                for f in ckpt_dir.glob("*.*"):
                    if f.suffix.lower() in [".safetensors", ".ckpt", ".pt", ".bin"]:
                        if f.name not in seen_filenames:
                            seen_filenames.add(f.name)
                            size_mb = round(f.stat().st_size / (1024 * 1024), 1)
                            models.append(ModelInfo(name=f.stem, filename=f.name, size_mb=size_mb))

        return models

    async def get_loras(self) -> list[LoraInfo]:
        """Fetch LoRAs from ComfyUI or fallback to scanning models/loras."""
        loras: list[LoraInfo] = []
        try:
            if await self.is_available():
                async with httpx.AsyncClient(timeout=2.0) as client:
                    res = await client.get(f"{self.base_url}/object_info/LoraLoader")
                    if res.status_code == 200:
                        data = res.json()
                        input_info = data.get("LoraLoader", {}).get("input", {}).get("required", {})
                        lora_names = input_info.get("lora_name", [[]])[0]
                        for name in lora_names:
                            loras.append(LoraInfo(name=Path(name).stem, filename=name))
                        if loras:
                            return loras
        except Exception:
            pass

        lora_dir = settings.loras_path
        if lora_dir.exists():
            for f in lora_dir.glob("*.*"):
                if f.suffix.lower() in [".safetensors", ".pt"]:
                    size_mb = round(f.stat().st_size / (1024 * 1024), 1)
                    loras.append(LoraInfo(name=f.stem, filename=f.name, size_mb=size_mb))

        return loras

    async def get_controlnet_models(self) -> list[ControlNetModelInfo]:
        """Fetch ControlNet models from ComfyUI or fallback to scanning models/controlnet."""
        models: list[ControlNetModelInfo] = []
        try:
            if await self.is_available():
                async with httpx.AsyncClient(timeout=2.0) as client:
                    res = await client.get(f"{self.base_url}/object_info/ControlNetLoader")
                    if res.status_code == 200:
                        data = res.json()
                        input_info = data.get("ControlNetLoader", {}).get("input", {}).get("required", {})
                        cnet_raw = input_info.get("control_net_name", [[]])
                        cnet_names = cnet_raw[0] if cnet_raw else []
                        if isinstance(cnet_names, list):
                            for name in cnet_names:
                                if name and isinstance(name, str) and not name.startswith("@{"):
                                    models.append(self._to_controlnet_info(name))
                            if models:
                                return models
        except Exception:
            pass

        search_dirs = [settings.base_dir / "models" / "controlnet", Path("c:/vishnu/ComfyUI/models/controlnet")]
        seen = set()
        for d in search_dirs:
            if d.exists():
                for f in d.glob("*.*"):
                    if f.suffix.lower() in [".safetensors", ".pth", ".pt", ".bin"] and f.name not in seen:
                        seen.add(f.name)
                        models.append(self._to_controlnet_info(f.name, f.stat().st_size))

        return models

    def _to_controlnet_info(self, filename: str, size_bytes: int = 0) -> ControlNetModelInfo:
        nl = filename.lower()
        c_type = "other"
        if "canny" in nl:
            c_type = "canny"
        elif "depth" in nl:
            c_type = "depth"
        elif "openpose" in nl or "pose" in nl:
            c_type = "openpose"
        elif "lineart" in nl:
            c_type = "lineart"
        elif "soft" in nl or "hed" in nl or "pidinet" in nl or "edge" in nl:
            c_type = "soft_edge"

        size_mb = round(size_bytes / (1024 * 1024), 1) if size_bytes > 0 else 0.0
        return ControlNetModelInfo(
            name=Path(filename).stem,
            filename=filename,
            type=c_type,
            size_mb=size_mb,
        )

    def load_workflow_template(self, mode: str) -> dict[str, Any]:
        if mode == "img2img":
            template_name = "img2img.json"
        elif mode == "inpaint":
            template_name = "inpaint.json"
        elif mode == "controlnet":
            template_name = "controlnet.json"
        else:
            template_name = "txt2img.json"

        workflow_path = settings.base_dir / "workflows" / template_name
        if not workflow_path.exists():
            raise FileNotFoundError(f"Workflow template {workflow_path} not found.")
        with open(workflow_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def prepare_workflow(
        self,
        request: GenerationRequest,
        job_id: str,
        uploaded_source_filename: str | None = None,
        uploaded_mask_filename: str | None = None,
    ) -> dict[str, Any]:
        prompt_graph = self.load_workflow_template(request.mode)
        resolved_seed = request.seed if (request.seed is not None and request.seed >= 0) else random.randint(1, 2**32 - 1)

        # Match model to available checkpoint
        checkpoint_name = request.model
        if not checkpoint_name.endswith((".safetensors", ".ckpt", ".pt", ".bin")):
            for ext in [".safetensors", ".ckpt"]:
                candidate = f"{checkpoint_name}{ext}"
                if (settings.checkpoints_path / candidate).exists() or Path(f"c:/vishnu/ComfyUI/models/checkpoints/{candidate}").exists():
                    checkpoint_name = candidate
                    break

        # Mutate nodes based on standard node class_types
        for node_id, node in prompt_graph.items():
            class_type = node.get("class_type")
            inputs = node.get("inputs", {})

            if class_type == "CheckpointLoaderSimple":
                inputs["ckpt_name"] = checkpoint_name
            elif class_type == "EmptyLatentImage":
                inputs["width"] = request.width
                inputs["height"] = request.height
                inputs["batch_size"] = 1
            elif class_type == "LoadImage":
                if request.mode in ["img2img", "inpaint", "controlnet"] and uploaded_source_filename:
                    inputs["image"] = uploaded_source_filename
            elif class_type == "LoadImageMask" and request.mode == "inpaint":
                if uploaded_mask_filename:
                    inputs["image"] = uploaded_mask_filename
            elif class_type == "ControlNetLoader" and request.mode == "controlnet":
                if request.controlnet_model:
                    inputs["control_net_name"] = request.controlnet_model
            elif class_type == "ControlNetApplyAdvanced" and request.mode == "controlnet":
                inputs["strength"] = request.controlnet_strength
                inputs["start_percent"] = request.controlnet_start_percent
                inputs["end_percent"] = request.controlnet_end_percent
            elif class_type == "CLIPTextEncode":
                # Check positive vs negative by node ID or text contents
                if node_id == "6" or "positive" in str(node):
                    inputs["text"] = request.prompt
                elif node_id == "7" or "negative" in str(node):
                    inputs["text"] = request.negative_prompt
            elif class_type == "KSampler":
                inputs["seed"] = resolved_seed
                inputs["steps"] = request.steps
                inputs["cfg"] = request.cfg_scale
                inputs["sampler_name"] = request.sampler
                if request.mode in ["img2img", "inpaint"]:
                    inputs["denoise"] = request.denoise
                else:
                    inputs["denoise"] = 1.0
            elif class_type == "SaveImage":
                inputs["filename_prefix"] = f"ForgeStudio_{job_id}"

        # Inject dynamic LoRA nodes (supports up to 2)
        if request.loras:
            last_model_ref = ["4", 0]
            last_clip_ref = ["4", 1]
            for idx, lora in enumerate(request.loras[:2]):
                lora_node_id = f"lora_{idx + 1}"
                prompt_graph[lora_node_id] = {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "lora_name": lora.name,
                        "strength_model": lora.strength,
                        "strength_clip": lora.strength,
                        "model": last_model_ref,
                        "clip": last_clip_ref,
                    }
                }
                last_model_ref = [lora_node_id, 0]
                last_clip_ref = [lora_node_id, 1]

            # Re-link KSampler model input and CLIP conditioning inputs
            if "3" in prompt_graph and "model" in prompt_graph["3"]["inputs"]:
                prompt_graph["3"]["inputs"]["model"] = last_model_ref
            if "6" in prompt_graph:
                prompt_graph["6"]["inputs"]["clip"] = last_clip_ref
            if "7" in prompt_graph:
                prompt_graph["7"]["inputs"]["clip"] = last_clip_ref

        return prompt_graph

    async def execute_generation(
        self,
        job_id: str,
        request: GenerationRequest,
        source_image_path: Path | None = None,
        mask_image_path: Path | None = None,
        progress_callback: Callable[[GenerationProgressUpdate], Any] | None = None,
        on_prompt_id: Callable[[str], Any] | None = None,
    ) -> Path:
        """Sends workflow to ComfyUI, streams progress, and returns saved image path."""
        uploaded_source_filename = None
        uploaded_mask_filename = None

        if request.mode in ["img2img", "inpaint", "controlnet"] and source_image_path:
            uploaded_source_filename = await self.upload_image(source_image_path)

        if request.mode == "inpaint" and mask_image_path:
            uploaded_mask_filename = await self.upload_image(mask_image_path)

        if request.mode == "controlnet":
            # Strict safety check: ensure model exists in available models
            available_cnet = await self.get_controlnet_models()
            avail_names = [m.filename for m in available_cnet]
            if not request.controlnet_model or request.controlnet_model not in avail_names:
                raise ValueError(
                    "ControlNet model unavailable — place a controlnet model (e.g. control_v11p_sd15_canny.safetensors) in ComfyUI/models/controlnet/ to enable ControlNet generation."
                )

        workflow = self.prepare_workflow(
            request,
            job_id,
            uploaded_source_filename,
            uploaded_mask_filename
        )

        # 1. Post prompt to ComfyUI
        async with httpx.AsyncClient(timeout=10.0) as client:
            payload = {
                "prompt": workflow,
                "client_id": self.client_id
            }
            res = await client.post(f"{self.base_url}/prompt", json=payload)
            if res.status_code != 200:
                err_text = res.text
                try:
                    err_json = res.json()
                    if "node_errors" in err_json and err_json["node_errors"]:
                        err_text = f"ComfyUI node validation error: {err_json['node_errors']}"
                    elif "error" in err_json:
                        err_text = f"ComfyUI error: {err_json['error']}"
                except Exception:
                    pass
                raise RuntimeError(err_text)
            data = res.json()
            prompt_id = data.get("prompt_id")

            if prompt_id and on_prompt_id:
                res_cb = on_prompt_id(prompt_id)
                if asyncio.iscoroutine(res_cb):
                    await res_cb

        # 2. Connect to ComfyUI WebSocket for progress tracking
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + f"/ws?clientId={self.client_id}"

        try:
            async with websockets.connect(ws_url, close_timeout=3) as ws:
                async for message in ws:
                    if isinstance(message, str):
                        msg = json.loads(message)
                        msg_type = msg.get("type")
                        msg_data = msg.get("data", {})

                        if msg_type == "progress":
                            curr_step = msg_data.get("value", 0)
                            max_step = msg_data.get("max", request.steps)
                            pct = round((curr_step / max_step) * 100, 1) if max_step > 0 else 0
                            if progress_callback:
                                update = GenerationProgressUpdate(
                                    job_id=job_id,
                                    status="running",
                                    step=curr_step,
                                    total_steps=max_step,
                                    progress_percentage=pct,
                                    message=f"Sampling step {curr_step}/{max_step}"
                                )
                                res = progress_callback(update)
                                if asyncio.iscoroutine(res):
                                    await res

                        elif msg_type == "executing" and msg_data.get("node") is None and msg_data.get("prompt_id") == prompt_id:
                            # Finished execution
                            break
        except Exception:
            # Fallback to polling /history if WebSocket closes early
            pass

        # 3. Retrieve generated image from ComfyUI history with retry loop
        for attempt in range(120):
            async with httpx.AsyncClient(timeout=10.0) as client:
                hist_res = await client.get(f"{self.base_url}/history/{prompt_id}")
                if hist_res.status_code == 200:
                    history = hist_res.json().get(prompt_id, {})
                    status_info = history.get("status", {})
                    if status_info.get("status_str") == "error":
                        messages = status_info.get("messages", [])
                        raise RuntimeError(f"ComfyUI node execution failed: {messages}")

                    outputs = history.get("outputs", {})
                    for _, node_output in outputs.items():
                        if "images" in node_output and node_output["images"]:
                            img_info = node_output["images"][0]
                            filename = img_info.get("filename")
                            subfolder = img_info.get("subfolder", "")
                            img_type = img_info.get("type", "output")

                            view_url = f"{self.base_url}/view?filename={filename}&subfolder={subfolder}&type={img_type}"
                            img_res = await client.get(view_url)
                            if img_res.status_code == 200:
                                saved_path = settings.generated_path / f"{job_id}.png"
                                with open(saved_path, "wb") as f:
                                    f.write(img_res.content)
                                return saved_path
            await asyncio.sleep(1.0)

        raise RuntimeError("ComfyUI finished execution but output image could not be retrieved.")

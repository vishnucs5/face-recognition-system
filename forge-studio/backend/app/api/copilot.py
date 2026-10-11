from typing import Any
from fastapi import APIRouter
from app.schemas.copilot import (
    CopilotActionRequest,
    CopilotActionResponse,
    CopilotRequest,
    CopilotResponse,
)
from app.services.ollama_client import OllamaClient
from app.services.gpu_monitor import GpuMonitor
from app.services.preset_engine import FACTORY_BUILTIN_PRESETS, preset_engine

router = APIRouter(prefix="/copilot", tags=["Copilot"])


@router.post("/prompt", response_model=CopilotResponse)
async def prompt_copilot(request: CopilotRequest):
    client = OllamaClient()
    return await client.process_copilot_request(request)


@router.post("/action", response_model=CopilotActionResponse)
async def copilot_action(request: CopilotActionRequest) -> CopilotActionResponse:
    """
    Structured Copilot advisory actions for Forge Studio.
    Strictly review-only! Never modifies database records or dispatches generations.
    Supports 5 structured actions with truthful offline heuristics when Ollama is unavailable.
    """
    client = OllamaClient()
    action_type = request.action_type
    current_form = request.current_form or {}

    # Check Ollama availability
    is_online = await client.is_available()

    if action_type == "fast_generation":
        suggested = {
            "steps": 16,
            "width": 768,
            "height": 768,
            "cfg_scale": 6.0,
            "sampler": "euler",
        }
        orig_prompt = str(current_form.get("prompt") or "").strip()
        explanation = (
            "Suggested fast iteration settings: steps reduced to 16, resolution set to 768x768, "
            "CFG 6.0, and standard Euler sampler. This halves latency while retaining clean composition."
        )

        metrics: dict[str, Any] = {"duration_ms": 0, "offline_heuristic": not is_online}

        if is_online and orig_prompt:
            system_prompt = (
                "You are Studio Copilot. The user wants fast generation settings. "
                "Output JSON with keys 'explanation' (string) and 'prompt' (concise, optimized for speed)."
            )
            res = await client.generate_structured(
                prompt=f"Optimize this prompt for fast rendering: {orig_prompt}",
                system=system_prompt,
                timeout=15.0,
            )
            if res.get("status") == "ok" and isinstance(res.get("data"), dict):
                data = res["data"]
                if data.get("prompt"):
                    suggested["prompt"] = client.truncate_prompt(str(data["prompt"]))
                if data.get("explanation"):
                    explanation = str(data["explanation"])
                metrics = res.get("metrics", metrics)
        elif orig_prompt:
            suggested["prompt"] = f"{orig_prompt}, crisp focus, speed optimized"

        return CopilotActionResponse(
            action_type=action_type,
            explanation=explanation,
            suggested_changes=suggested,
            metrics=metrics,
        )

    elif action_type == "quality":
        suggested = {
            "steps": 30,
            "width": 1024,
            "height": 1024,
            "cfg_scale": 6.5,
            "sampler": "dpmpp_2m",
        }
        orig_prompt = str(current_form.get("prompt") or "").strip()
        explanation = (
            "Configured for maximum fidelity: 30 denoising steps with DPM++ 2M sampler and native 1024x1024 resolution. "
            "Prompt enriched with photographic detail cues to maximize micro-texture on RTX 5070 Ti."
        )

        metrics = {"duration_ms": 0, "offline_heuristic": not is_online}

        if is_online and orig_prompt:
            system_prompt = (
                "You are Studio Copilot. Output JSON with keys 'explanation' (string) and "
                "'prompt' (photorealistic, highly detailed version of the user prompt)."
            )
            res = await client.generate_structured(
                prompt=f"Enhance this prompt for maximum photographic quality: {orig_prompt}",
                system=system_prompt,
                timeout=15.0,
            )
            if res.get("status") == "ok" and isinstance(res.get("data"), dict):
                data = res["data"]
                if data.get("prompt"):
                    suggested["prompt"] = client.truncate_prompt(str(data["prompt"]))
                if data.get("explanation"):
                    explanation = str(data["explanation"])
                metrics = res.get("metrics", metrics)
        elif orig_prompt:
            suggested["prompt"] = f"{orig_prompt}, 8k uhd, masterpiece, highly detailed, professional photography, studio lighting, sharp focus"

        return CopilotActionResponse(
            action_type=action_type,
            explanation=explanation,
            suggested_changes=suggested,
            metrics=metrics,
        )

    elif action_type == "vram_safe_resolution":
        model_str = str(current_form.get("model") or "").lower()
        family_str = str(current_form.get("base_model_family") or "").lower()
        is_sdxl = "sdxl" in model_str or "sdxl" in family_str or "xl" in model_str

        gpu_status = GpuMonitor.get_status()
        vram_mb = gpu_status.vram_total_mb if gpu_status.available else 16384

        if is_sdxl:
            if vram_mb >= 12000:
                width, height = 1024, 1024
                explanation = (
                    f"SDXL architecture detected with {vram_mb} MB VRAM. "
                    "Native 1024x1024 resolution is safe and optimal for the SDXL latent space without VRAM paging."
                )
            else:
                width, height = 896, 896
                explanation = (
                    f"SDXL architecture detected with constrained VRAM ({vram_mb} MB). "
                    "Recommending 896x896 to ensure headroom against out-of-memory errors."
                )
        else:
            if vram_mb >= 8000:
                width, height = 768, 768
                explanation = (
                    f"SD 1.5 architecture detected with {vram_mb} MB VRAM. "
                    "768x768 resolution delivers clean high-res output without latent duplication."
                )
            else:
                width, height = 512, 512
                explanation = (
                    f"SD 1.5 architecture detected. 512x512 is the native training baseline "
                    "and guaranteed safe against VRAM exhaustion."
                )

        return CopilotActionResponse(
            action_type=action_type,
            explanation=explanation,
            suggested_changes={"width": width, "height": height},
            metrics={"duration_ms": 0, "offline_heuristic": not is_online},
        )

    elif action_type == "explain_slowness":
        steps = int(current_form.get("steps") or 20)
        width = int(current_form.get("width") or 1024)
        height = int(current_form.get("height") or 1024)
        loras = current_form.get("loras") or []
        sampler = str(current_form.get("sampler") or "euler").lower()
        pixels = width * height

        reasons: list[str] = []
        fixes: dict[str, Any] = {}

        if steps > 30:
            reasons.append(
                f"High step count ({steps} steps): requires {steps} full denoising passes. "
                "Reducing to 24–28 steps yields equivalent perceptual quality in significantly less time."
            )
            fixes["steps"] = 28

        if pixels > 1024 * 1024:
            reasons.append(
                f"High resolution ({width}x{height} = {pixels:,} pixels): Latent diffusion compute scales quadratically with pixel area."
            )
            fixes["width"] = 1024
            fixes["height"] = 1024

        if len(loras) >= 2:
            reasons.append(
                f"Multiple LoRAs active ({len(loras)}): Dynamically patching model weights increases memory overhead and sampling latency."
            )

        if sampler in ["dpmpp_sde", "dpmpp_2m_sde", "heun"]:
            reasons.append(
                f"Sampler '{sampler}' evaluates two derivative steps per iteration (2x ODE calculations compared to Euler)."
            )
            fixes["sampler"] = "euler"

        if not reasons:
            reasons.append(
                "Current settings are well-balanced. Any generation slowness is likely due to initial checkpoint disk loading or background GPU utilization."
            )
            fixes["steps"] = min(steps, 24)

        explanation = "Generation performance evaluation:\n• " + "\n• ".join(reasons)

        return CopilotActionResponse(
            action_type=action_type,
            explanation=explanation,
            suggested_changes=fixes,
            metrics={"duration_ms": 0, "offline_heuristic": not is_online},
        )

    elif action_type == "suggest_preset":
        prompt_text = str(current_form.get("prompt") or "").lower()
        mode = str(current_form.get("mode") or "txt2img").lower()
        has_cnet = bool(current_form.get("controlnet_type") or current_form.get("controlnet_settings"))
        loras = current_form.get("loras") or []

        # Intent matching against the 7 factory built-in presets
        if mode == "inpaint" or any(w in prompt_text for w in ["inpaint", "mask", "repair", "erase", "blemish"]):
            preset_id = "builtin-inpainting-repair"
        elif has_cnet or any(w in prompt_text for w in ["controlnet", "canny", "depth", "pose", "openpose", "wireframe"]):
            preset_id = "builtin-controlnet-composition"
        elif len(loras) > 0 or any(w in prompt_text for w in ["lora", "style test", "stylization"]):
            preset_id = "builtin-lora-style-test"
        elif any(w in prompt_text for w in ["draft", "fast", "speed", "sketch", "rough", "concept"]):
            preset_id = "builtin-fast-draft"
        elif any(w in prompt_text for w in ["product", "bottle", "shoe", "packaging", "commercial", "studio shot", "packshot", "watch"]):
            preset_id = "builtin-product-photography"
        elif any(w in prompt_text for w in ["landscape", "mountain", "ocean", "vista", "forest", "sunset", "sky", "scenery", "scenic", "valley", "cinematic"]):
            preset_id = "builtin-cinematic-landscape"
        else:
            preset_id = "builtin-portrait"

        # Lookup preset specification
        builtins = preset_engine.get_built_in_presets()
        matched = next((p for p in builtins if p.id == preset_id), None)
        if not matched:
            raw = next((r for r in FACTORY_BUILTIN_PRESETS if r["id"] == preset_id), FACTORY_BUILTIN_PRESETS[0])
            name = raw["name"]
            desc = raw["description"]
            suggested_diff = {
                "width": raw["width"],
                "height": raw["height"],
                "steps": raw["steps"],
                "cfg_scale": raw["cfg_scale"],
                "sampler": raw["sampler"],
            }
        else:
            name = matched.name
            desc = matched.description
            suggested_diff = {
                "width": matched.width,
                "height": matched.height,
                "steps": matched.steps,
                "cfg_scale": matched.cfg_scale,
                "sampler": matched.sampler,
            }

        explanation = f"Matched built-in preset '{name}' based on prompt intent. {desc}."

        return CopilotActionResponse(
            action_type=action_type,
            explanation=explanation,
            suggested_changes=suggested_diff,
            preset_id=preset_id,
            preset_name=name,
            metrics={"duration_ms": 0, "offline_heuristic": not is_online},
        )

    # Fallback for unrecognized action
    return CopilotActionResponse(
        action_type=action_type,
        explanation="No action performed.",
        suggested_changes={},
        metrics={"duration_ms": 0},
    )

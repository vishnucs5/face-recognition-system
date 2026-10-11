import json
import re
import time
from typing import Any
import httpx

from app.config import settings
from app.schemas.copilot import CopilotRequest, CopilotResponse, CopilotSettingsSuggestion

DEFAULT_TIMEOUT: float = 120.0
MAX_CONTEXT_TOKENS: int = 8192
MAX_CONTEXT_CHARS: int = 32768

ALLOWED_SAMPLERS = {
    "euler",
    "euler_ancestral",
    "dpmpp_2m",
    "dpmpp_sde",
    "dpmpp_2m_sde",
    "heun",
    "ddim",
}

FORBIDDEN_PATTERNS = [
    r"\b(rm\s+-rf|del\s+/f|format\s+[a-z]:|powershell|cmd\.exe|/bin/sh|/bin/bash)\b",
    r"\b(curl\s+|wget\s+).+\|\s*(bash|sh)",
]


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.base_url = (base_url or settings.OLLAMA_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout

    def truncate_prompt(self, text: str | None, max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """
        Enforce maximum context protection ceiling (~8,192 tokens / 32,768 characters).
        Safely truncates oversized inputs to protect workstation VRAM and RAM.
        """
        if not text:
            return ""
        if len(text) > max_chars:
            return text[:max_chars]
        return text

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                return res.status_code == 200
        except Exception:
            return False

    async def get_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(f"{self.base_url}/api/tags")
                if res.status_code == 200:
                    data = res.json()
                    return [m.get("name") for m in data.get("models", []) if m.get("name")]
        except Exception:
            pass
        return []

    async def generate_structured(
        self,
        prompt: str,
        system: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> dict:
        """
        Sends a structured JSON generation request to Ollama with:
        - Maximum context ceiling enforcement (8,192 tokens / 32,768 chars)
        - Truthful offline state detection
        - Latency timing (duration_ms)
        - Generation metrics (eval_count, tokens_per_sec)
        """
        if not await self.is_available():
            return {
                "status": "offline",
                "error": f"Ollama is offline or unreachable at {self.base_url}",
                "data": None,
                "metrics": {
                    "duration_ms": 0,
                    "eval_count": 0,
                    "tokens_per_sec": 0.0,
                },
            }

        safe_prompt = self.truncate_prompt(prompt)
        safe_system = self.truncate_prompt(system) if system else None

        messages: list[dict[str, str]] = []
        if safe_system:
            messages.append({"role": "system", "content": safe_system})
        messages.append({"role": "user", "content": safe_prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "format": "json",
            "stream": False,
        }

        t0 = time.time()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                res = await client.post(f"{self.base_url}/api/chat", json=payload)
                duration_ms = int((time.time() - t0) * 1000)

                if res.status_code == 200:
                    data = res.json()
                    raw_content = data.get("message", {}).get("content", "").strip()
                    eval_count = data.get("eval_count", 0)
                    eval_duration_ns = data.get("eval_duration", 0)

                    if eval_duration_ns > 0:
                        tokens_per_sec = round(eval_count / (eval_duration_ns / 1_000_000_000), 2)
                    elif duration_ms > 0 and eval_count > 0:
                        tokens_per_sec = round(eval_count / (duration_ms / 1000.0), 2)
                    else:
                        tokens_per_sec = 0.0

                    parsed: dict | None = None
                    try:
                        parsed = json.loads(raw_content)
                    except Exception:
                        if "```json" in raw_content:
                            try:
                                parsed = json.loads(raw_content.split("```json")[1].split("```")[0].strip())
                            except Exception:
                                pass
                        if parsed is None and "{" in raw_content and "}" in raw_content:
                            try:
                                s = raw_content.index("{")
                                e = raw_content.rindex("}") + 1
                                parsed = json.loads(raw_content[s:e])
                            except Exception:
                                pass

                    if parsed is not None:
                        return {
                            "status": "ok",
                            "data": parsed,
                            "raw": raw_content,
                            "metrics": {
                                "duration_ms": duration_ms,
                                "eval_count": eval_count,
                                "tokens_per_sec": tokens_per_sec,
                            },
                        }
                    else:
                        return {
                            "status": "error",
                            "error": "Failed to parse structured JSON from model response",
                            "raw": raw_content,
                            "data": None,
                            "metrics": {
                                "duration_ms": duration_ms,
                                "eval_count": eval_count,
                                "tokens_per_sec": tokens_per_sec,
                            },
                        }
                else:
                    return {
                        "status": "error",
                        "error": f"Ollama HTTP {res.status_code}: {res.text}",
                        "data": None,
                        "metrics": {
                            "duration_ms": duration_ms,
                            "eval_count": 0,
                            "tokens_per_sec": 0.0,
                        },
                    }
        except Exception as exc:
            duration_ms = int((time.time() - t0) * 1000)
            return {
                "status": "error",
                "error": str(exc),
                "data": None,
                "metrics": {
                    "duration_ms": duration_ms,
                    "eval_count": 0,
                    "tokens_per_sec": 0.0,
                },
            }

    async def process_copilot_request(self, req: CopilotRequest) -> CopilotResponse:
        online = await self.is_available()

        if not online:
            return self._offline_fallback(req)

        system_instruction = (
            "You are Studio Copilot, an AI art director and SDXL generation specialist inside Forge Studio. "
            "You help users craft vivid, evocative prompts, configure optimal technical parameters (steps, CFG, sampler, resolution), "
            "and suggest structured generation recipes for their local RTX 5070 Ti workstation. "
            "Never execute shell commands or output system scripts. "
            "When providing settings suggestions, include a JSON block in format: "
            '```json\n{"prompt": "...", "negative_prompt": "...", "width": 1024, "height": 1024, "steps": 28, "cfg_scale": 6.5, "sampler": "euler"}\n```'
        )

        safe_input = self.truncate_prompt(req.message)
        user_content = (
            f"Task: {req.task}\n"
            f"User input: {safe_input}\n"
            f"Active studio context: {json.dumps(req.current_context)}"
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_content}
                    ],
                    "stream": False
                }
                res = await client.post(f"{self.base_url}/api/chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    content = data.get("message", {}).get("content", "")
                    return self._parse_llm_response(content, req)
        except Exception:
            return self._offline_fallback(req)

        return self._offline_fallback(req)


    def _sanitize_string(self, s: str | None) -> str | None:
        if not s:
            return None
        # Check against forbidden shell injection patterns
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, s, re.IGNORECASE):
                return "[Unsafe content filtered]"
        return s.strip()

    def _clamp_dimensions(self, val: int | None, default: int = 1024) -> int:
        if val is None:
            return default
        clamped = max(256, min(1536, val))
        return (clamped // 64) * 64

    def _parse_llm_response(self, text: str, req: CopilotRequest) -> CopilotResponse:
        suggested_settings: CopilotSettingsSuggestion | None = None
        variations: list[str] = []

        # Check for JSON block
        if "```json" in text:
            try:
                json_part = text.split("```json")[1].split("```")[0].strip()
                parsed = json.loads(json_part)

                sampler = parsed.get("sampler")
                if sampler and sampler not in ALLOWED_SAMPLERS:
                    sampler = "euler"

                steps = parsed.get("steps")
                if steps is not None:
                    steps = max(10, min(60, int(steps)))

                cfg = parsed.get("cfg_scale")
                if cfg is not None:
                    cfg = max(1.0, min(15.0, float(cfg)))

                denoise = parsed.get("denoise")
                if denoise is not None:
                    denoise = max(0.1, min(1.0, float(denoise)))

                cnet_type = parsed.get("controlnet_type")
                if cnet_type and cnet_type not in ["canny", "depth", "openpose", "lineart", "soft_edge"]:
                    cnet_type = None

                cnet_strength = parsed.get("controlnet_strength")
                if cnet_strength is not None:
                    cnet_strength = max(0.1, min(2.0, float(cnet_strength)))

                suggested_settings = CopilotSettingsSuggestion(
                    prompt=self._sanitize_string(parsed.get("prompt")),
                    negative_prompt=self._sanitize_string(parsed.get("negative_prompt")),
                    width=self._clamp_dimensions(parsed.get("width")),
                    height=self._clamp_dimensions(parsed.get("height")),
                    steps=steps,
                    cfg_scale=cfg,
                    sampler=sampler,
                    denoise=denoise,
                    controlnet_type=cnet_type,
                    controlnet_strength=cnet_strength,
                )
            except Exception:
                pass

        if req.task == "variations":
            lines = text.split("\n")
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.endswith(":") or stripped.startswith("`"):
                    continue
                if stripped.startswith(("-", "*")) or (stripped and stripped[0].isdigit() and ("." in stripped[:4] or ")" in stripped[:4])):
                    cleaned = stripped.lstrip("-*123456789.) ")
                    cleaned = self._sanitize_string(cleaned)
                    if cleaned and len(cleaned) > 10:
                        variations.append(cleaned)
            if not variations:
                variations = [
                    f"{req.message}, cinematic film still, 35mm lens, atmospheric haze",
                    f"{req.message}, dark fantasy concept art, dramatic rim lighting",
                    f"{req.message}, minimalist architectural photography, geometric framing",
                ]

        return CopilotResponse(
            reply=text,
            suggested_settings=suggested_settings,
            variations=variations[:5]
        )

    def _offline_fallback(self, req: CopilotRequest) -> CopilotResponse:
        p = req.message.strip()
        enhanced = f"{p}, cinematic composition, 8k uhd, studio lighting, highly detailed, photorealistic, octane render, sharp focus"
        negative = "blurry, deformed, lowres, distorted anatomy, watermark, oversaturated, amateur"

        if req.task == "improve_prompt":
            reply = (
                "*(Studio Copilot — Offline Heuristics)*\n\n"
                f"**Enhanced Prompt:**\n{enhanced}\n\n"
                f"**Negative Prompt:**\n{negative}\n\n"
                "*Advisory: Review the suggested parameters below before applying.*"
            )
            settings_sugg = CopilotSettingsSuggestion(
                prompt=enhanced,
                negative_prompt=negative,
                steps=30,
                cfg_scale=7.0,
                sampler="dpmpp_2m"
            )
            return CopilotResponse(reply=reply, suggested_settings=settings_sugg)

        elif req.task == "variations":
            vars_list = [
                f"{p}, cinematic film still, 35mm lens, atmospheric haze, Kodak Portra 400 tones",
                f"{p}, dark fantasy concept art, volumetric rim lighting, dramatic shadows, octane render",
                f"{p}, minimalist architectural photography, geometric framing, soft morning light",
                f"{p}, cyberpunk aesthetic, neon cyan and amber reflections, wet asphalt, ultra-detailed"
            ]
            reply = "*(Studio Copilot — Offline Heuristics)*\nHere are 4 stylistic prompt variations based on your subject:"
            return CopilotResponse(reply=reply, variations=vars_list)

        elif req.task == "negative_prompt":
            reply = (
                "*(Studio Copilot — Negative Prompt)*\n\n"
                f"**Negative Prompt:**\n{negative}\n\n"
                "Recommended baseline negative keywords to suppress common rendering artifacts."
            )
            settings_sugg = CopilotSettingsSuggestion(negative_prompt=negative)
            return CopilotResponse(reply=reply, suggested_settings=settings_sugg)

        elif req.task == "inpaint_advice":
            reply = (
                "*(Studio Copilot — Inpainting Advisory)*\n\n"
                "• **Prompting for Inpaint**: Describe only the object/details you want inside the masked region, "
                "or write the complete desired scene context. Avoid describing what you erased.\n"
                "• **Denoise Recommendations**:\n"
                "  - `0.40–0.60`: Subtle touchups, fixing skin, or altering textures.\n"
                "  - `0.75–0.90`: Complete object replacement or generating brand new elements.\n"
                "• **Masking Technique**: Paint slightly beyond the edges of the object to give the latent inpainter room to blend seams seamlessly."
            )
            settings_sugg = CopilotSettingsSuggestion(
                denoise=0.85,
                steps=30,
                cfg_scale=7.0,
            )
            return CopilotResponse(reply=reply, suggested_settings=settings_sugg)

        elif req.task == "controlnet_advice":
            reply = (
                "*(Studio Copilot — ControlNet Advisory)*\n\n"
                "• **Canny**: Best for hard edges, architecture, vehicles, and line-bound subjects. Strength: `0.9–1.0`.\n"
                "• **Depth**: Best for 3D staging, room geometry, landscape foreground/background layers. Strength: `0.85–1.0`.\n"
                "• **OpenPose**: Best for replicating human poses, limb placement, and gestures without locking clothing or background. Strength: `1.0`.\n"
                "• **Line Art & Soft Edge**: Best for turning line sketches or organic contours into finished renderings.\n"
                "• **Control Range**: End at `0.80` (80%) if you want the model to add creative micro-texture while keeping structural guidance."
            )
            settings_sugg = CopilotSettingsSuggestion(
                controlnet_type="canny",
                controlnet_strength=1.0,
                steps=30,
                cfg_scale=7.0,
            )
            return CopilotResponse(reply=reply, suggested_settings=settings_sugg)

        elif req.task == "explain":
            reply = (
                "*(Studio Copilot — Workstation Guide)*\n\n"
                "• **Width & Height**: Latent dimensions. SDXL native baseline is 1024×1024; SD 1.5 is 512×512.\n"
                "• **Steps**: Denoising iterations. 20–28 steps give optimal speed/fidelity on RTX 5070 Ti.\n"
                "• **CFG Scale**: Classifier-Free Guidance. Controls adherence to prompt (4.5 creative, 6.5 balanced, 8.0 strict).\n"
                "• **Sampler**: Math solver for ODEs. Euler is fast; DPM++ 2M SDE yields rich micro-details.\n"
                "• **Denoise**: In Image-to-Image / Inpaint, governs how much original latent data is replaced.\n"
                "• **ControlNet**: Uses a conditioning preprocessor (Canny, Depth, Pose) to preserve geometry and composition.\n"
                "• **Inpaint**: Masked region latent blending using VAEEncodeForInpaint and feathered edge blending."
            )
            return CopilotResponse(reply=reply)

        else:
            settings_sugg = CopilotSettingsSuggestion(
                prompt=enhanced,
                negative_prompt=negative,
                width=1024,
                height=1024,
                steps=28,
                cfg_scale=6.5,
                sampler="euler"
            )
            reply = f"*(Studio Copilot — Offline Heuristics)*\nRecommended balanced SDXL workstation configuration for '{p}'."
            return CopilotResponse(reply=reply, suggested_settings=settings_sugg)

    def _extract_fallback_settings(self, prompt: str) -> CopilotSettingsSuggestion:
        return CopilotSettingsSuggestion(
            prompt=f"{prompt}, high quality, crisp focus, studio lighting",
            negative_prompt="blurry, distorted, low quality",
            width=1024,
            height=1024,
            steps=28,
            cfg_scale=6.5,
            sampler="euler"
        )

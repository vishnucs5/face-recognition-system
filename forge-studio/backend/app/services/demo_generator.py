import asyncio
import hashlib
import math
import random
from collections.abc import Callable
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from app.config import settings
from app.schemas.generation import GenerationProgressUpdate, GenerationRequest


class DemoGenerator:
    @staticmethod
    def _get_palette(prompt: str) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
        p_lower = prompt.lower()
        if any(w in p_lower for w in ["dawn", "sunrise", "morning", "golden"]):
            return ((38, 22, 55), (170, 75, 95), (245, 175, 75))
        if any(w in p_lower for w in ["sunset", "dusk", "evening", "crimson"]):
            return ((40, 18, 32), (180, 50, 65), (240, 130, 60))
        if any(w in p_lower for w in ["cyber", "neon", "scifi", "future", "lab"]):
            return ((12, 16, 28), (28, 48, 76), (0, 220, 210))
        if any(w in p_lower for w in ["forest", "green", "emerald", "jungle"]):
            return ((16, 26, 20), (32, 60, 48), (95, 205, 145))
        if any(w in p_lower for w in ["space", "cosmic", "galaxy", "stars", "nebula"]):
            return ((10, 8, 22), (40, 25, 65), (190, 120, 240))
        # Default warm studio signature
        return ((24, 22, 19), (60, 48, 38), (215, 168, 92))

    @classmethod
    async def generate(
        cls,
        job_id: str,
        request: GenerationRequest,
        progress_callback: Callable[[GenerationProgressUpdate], asyncio.Future | None] | None = None
    ) -> Path:
        """Simulates generation steps with realistic progress updates, outputting an expressive procedural image."""
        steps = request.steps
        c_top, c_mid, accent = cls._get_palette(request.prompt)

        # Simulate progressive latent sampling
        for step in range(1, steps + 1):
            await asyncio.sleep(0.08)  # ~2 seconds total for 25 steps
            pct = round((step / steps) * 100, 1)
            if progress_callback:
                update = GenerationProgressUpdate(
                    job_id=job_id,
                    status="running",
                    step=step,
                    total_steps=steps,
                    progress_percentage=pct,
                    message=f"Sampling latent space (step {step}/{steps})"
                )
                res = progress_callback(update)
                if asyncio.iscoroutine(res):
                    await res

        # Generate styled high-resolution procedural image
        w, h = request.width, request.height
        seed = (request.seed or 123456) % 100000
        rng = random.Random(seed)

        img = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(img)

        horizon_y = int(h * 0.55)

        # 1. Sky Gradient (Multi-stop gradient from top to horizon)
        for y in range(horizon_y):
            t = y / max(1, horizon_y)
            if t < 0.5:
                st = t / 0.5
                r = int(c_top[0] + (c_mid[0] - c_top[0]) * st)
                g = int(c_top[1] + (c_mid[1] - c_top[1]) * st)
                b = int(c_top[2] + (c_mid[2] - c_top[2]) * st)
            else:
                st = (t - 0.5) / 0.5
                r = int(c_mid[0] + (accent[0] - c_mid[0]) * st)
                g = int(c_mid[1] + (accent[1] - c_mid[1]) * st)
                b = int(c_mid[2] + (accent[2] - c_mid[2]) * st)
            draw.line([(0, y), (w, y)], fill=(r, g, b))

        # 2. Glowing Celestial Sun / Moon on horizon
        sun_x = int(w * (0.35 + 0.3 * rng.random()))
        sun_y = horizon_y - 20
        sun_rad = min(w, h) // 7
        for rad in range(sun_rad, 0, -3):
            t = (sun_rad - rad) / sun_rad
            sr = int(255 * t + accent[0] * (1 - t))
            sg = int(240 * t + accent[1] * (1 - t))
            sb = int(180 * t + accent[2] * (1 - t))
            draw.ellipse([(sun_x - rad, sun_y - rad), (sun_x + rad, sun_y + rad)], fill=(sr, sg, sb))

        # 3. Distant mountain peaks (misty atmospheric haze)
        pts_distant = [(0, horizon_y)]
        for x in range(0, w + 40, 20):
            nh = math.sin(x * 0.008 + seed) * 120 + math.cos(x * 0.018) * 60 + math.sin(x * 0.035) * 25
            pts_distant.append((x, int(horizon_y - 130 + nh * 0.7)))
        pts_distant.extend([(w, horizon_y), (0, horizon_y)])
        draw.polygon(
            pts_distant,
            fill=(int(c_mid[0] * 0.75), int(c_mid[1] * 0.75), int(c_mid[2] * 0.75))
        )

        # 4. Midground sharp mountain peaks
        pts_mid = [(0, horizon_y)]
        for x in range(0, w + 30, 15):
            nh = math.sin(x * 0.012 + seed * 2) * 150 + math.cos(x * 0.024) * 80 + math.sin(x * 0.05) * 35
            pts_mid.append((x, int(horizon_y - 85 + nh * 0.65)))
        pts_mid.extend([(w, horizon_y), (0, horizon_y)])
        draw.polygon(
            pts_mid,
            fill=(int(c_top[0] * 1.5), int(c_top[1] * 1.5), int(c_top[2] * 1.5))
        )

        # 5. Foreground dark mountain ridges / silhouettes
        pts_fore = [(0, horizon_y)]
        for x in range(0, w + 25, 12):
            nh = math.sin(x * 0.015 + seed * 3) * 95 + math.cos(x * 0.03) * 45
            pts_fore.append((x, int(horizon_y - 25 + nh * 0.5)))
        pts_fore.extend([(w, horizon_y), (0, horizon_y)])
        draw.polygon(
            pts_fore,
            fill=(max(15, int(c_top[0] * 0.8)), max(15, int(c_top[1] * 0.8)), max(15, int(c_top[2] * 0.8)))
        )

        # 6. Serene reflective lake / water surface in bottom half
        for y in range(horizon_y, h):
            t = (y - horizon_y) / max(1, (h - horizon_y))
            # Water reflects sky colors at horizon and deepens downward
            r = int(accent[0] * 0.65 * (1 - t) + c_top[0] * 0.6 * t)
            g = int(accent[1] * 0.65 * (1 - t) + c_top[1] * 0.6 * t)
            b = int(accent[2] * 0.65 * (1 - t) + c_top[2] * 0.6 * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b))

        # 7. Water Specular Reflection of Sun
        for y in range(horizon_y, h):
            t = (y - horizon_y) / max(1, (h - horizon_y))
            spread = int(12 + 190 * t)
            wave_noise = math.sin(y * 0.45) * 7 + math.cos(y * 0.85) * 4
            center = sun_x + wave_noise
            ref_r = int(255 * (1 - t * 0.75))
            ref_g = int(225 * (1 - t * 0.8))
            ref_b = int(150 * (1 - t * 0.85))
            draw.line([(center - spread, y), (center + spread, y)], fill=(ref_r, ref_g, ref_b))

        # 8. Delicate horizontal water shimmer lines
        for y in range(horizon_y + 8, h, 7):
            t = (y - horizon_y) / max(1, (h - horizon_y))
            if rng.random() > 0.35:
                x_start = rng.randint(0, w // 2)
                length = rng.randint(50, 240)
                shimmer = (int(accent[0] * (1 - t * 0.6)), int(accent[1] * (1 - t * 0.6)), int(accent[2] * (1 - t * 0.6)))
                draw.line([(x_start, y), (x_start + length, y)], fill=shimmer, width=1)

        # 9. Decorative framing border
        pad = min(w, h) // 22
        draw.rectangle(
            [(pad, pad), (w - pad, h - pad)],
            outline=accent,
            width=2
        )

        font = ImageFont.load_default()

        # Header metadata watermark
        draw.text((pad + 20, pad + 20), "FORGE STUDIO  |  PROCEDURAL SYNTHESIS (DEMO MODE)", fill=accent, font=font)
        draw.text((pad + 20, pad + 40), f"Mode: {request.mode.upper()}  |  Model: {request.model}  |  Steps: {steps}  |  CFG: {request.cfg_scale}", fill=(225, 225, 225), font=font)
        draw.text((pad + 20, pad + 58), f"Resolution: {w}x{h}  |  Seed: {request.seed or 'Random'}", fill=(185, 185, 185), font=font)

        # Prompt text snippet at bottom
        prompt_snippet = request.prompt[:120] + ("..." if len(request.prompt) > 120 else "")
        draw.text((pad + 20, h - pad - 50), "PROMPT:", fill=accent, font=font)
        draw.text((pad + 20, h - pad - 32), f'"{prompt_snippet}"', fill=(245, 245, 245), font=font)

        # Footer instructions pill
        instructions = "Connect ComfyUI (:8188) & set DEMO_MODE=false in .env for RTX 5070 Ti neural SDXL weights"
        draw.text((pad + 20, h - pad - 16), instructions, fill=(180, 175, 170), font=font)

        # Save to generated directory
        output_filename = f"{job_id}.png"
        output_path = settings.generated_path / output_filename
        img.save(output_path, format="PNG")

        return output_path

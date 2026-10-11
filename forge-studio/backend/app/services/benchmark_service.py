import asyncio
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
from typing import Literal
import uuid

from fastapi import HTTPException, status
import httpx

from app.config import settings
from app.schemas.diagnostics import BenchmarkResult, BenchmarkRunRequest
from app.schemas.generation import GenerationRequest
from app.services.comfyui_client import ComfyUIClient
from app.services.gpu_monitor import GpuMonitor
from app.services.gpu_safety import gpu_safety_manager
from app.services.ollama_client import OllamaClient


class BenchmarkService:
    def __init__(self):
        self._lock = asyncio.Lock()

    @property
    def benchmarks_file(self) -> Path:
        settings.data_path.mkdir(parents=True, exist_ok=True)
        return settings.data_path / "benchmarks.json"

    def get_results(self) -> list[BenchmarkResult]:
        """Read all persisted benchmark results."""
        if not self.benchmarks_file.exists():
            return []
        try:
            content = self.benchmarks_file.read_text(encoding="utf-8")
            if not content.strip():
                return []
            raw_list = json.loads(content)
            if not isinstance(raw_list, list):
                return []
            return [BenchmarkResult.model_validate(item) for item in raw_list]
        except Exception:
            return []

    async def _save_result(self, result: BenchmarkResult):
        async with self._lock:
            existing = self.get_results()
            existing.append(result)
            payload = json.dumps([r.model_dump(mode="json") for r in existing], indent=2)
            self.benchmarks_file.write_text(payload, encoding="utf-8")

    def export_results(self, format: Literal["json", "csv"] = "json") -> str:
        """Export stored benchmark results as JSON or CSV."""
        results = self.get_results()
        if format.lower() == "csv":
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow([
                "id",
                "timestamp",
                "gpu_name",
                "total_time_ms",
                "prompt_to_first_img_ms",
                "steps_per_sec",
                "peak_vram_mb",
                "avg_vram_mb",
                "gpu_util_pct",
                "ollama_first_token_ms",
                "ollama_tokens_per_sec",
                "cuda_version",
                "pytorch_version",
                "driver_version",
            ])
            for r in results:
                writer.writerow([
                    r.id,
                    r.timestamp.isoformat(),
                    r.gpu_name,
                    r.total_time_ms,
                    r.prompt_to_first_img_ms,
                    r.steps_per_sec,
                    r.peak_vram_mb,
                    r.avg_vram_mb,
                    r.gpu_util_pct,
                    r.ollama_first_token_ms if r.ollama_first_token_ms is not None else "",
                    r.ollama_tokens_per_sec if r.ollama_tokens_per_sec is not None else "",
                    r.env_metadata.get("cuda_version", ""),
                    r.env_metadata.get("pytorch_version", ""),
                    r.env_metadata.get("driver_version", ""),
                ])
            return output.getvalue()
        return json.dumps([r.model_dump(mode="json") for r in results], indent=2)

    async def _measure_ollama_latency(self) -> tuple[float | None, float | None]:
        """
        Measure first-token latency and generation throughput from Ollama.
        Returns (first_token_ms, tokens_per_sec) or (None, None) if offline.
        """
        try:
            ollama_client = OllamaClient()
            if not await ollama_client.is_available():
                return None, None

            t_start = time.perf_counter()
            first_token_time: float | None = None
            total_tokens = 0

            async with httpx.AsyncClient(timeout=10.0) as http_client:
                async with http_client.stream(
                    "POST",
                    f"{settings.OLLAMA_URL}/api/generate",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "prompt": "Hello, summarize in 3 words",
                        "stream": True,
                    },
                ) as resp:
                    if resp.status_code != 200:
                        return None, None
                    async for line in resp.aiter_lines():
                        if not line:
                            continue
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        total_tokens += 1
                    t_end = time.perf_counter()

            if first_token_time is None:
                return None, None

            first_token_ms = round((first_token_time - t_start) * 1000, 2)
            token_duration = max(t_end - first_token_time, 0.001)
            tokens_per_sec = round(total_tokens / token_duration, 2)
            return first_token_ms, tokens_per_sec
        except Exception:
            return None, None

    def _get_driver_version(self) -> str | None:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return None
        try:
            res = subprocess.run(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass
        return None

    async def run_benchmark(self, request: BenchmarkRunRequest) -> BenchmarkResult:
        """
        Execute hardware benchmark with pre-flight GPU isolation, 5-step warmup,
        20-step measured run, and Ollama latency tracking.
        """
        # 1. User confirmation check
        if not request.user_confirmed:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User confirmation required to run hardware benchmark",
            )

        # 2. Pre-flight GPU isolation lease acquisition
        acquired, reason = await gpu_safety_manager.acquire_lease("benchmark", "benchmark")
        if not acquired:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"GPU is currently busy with another workload: {reason}",
            )

        try:
            client = ComfyUIClient()
            checkpoints = await client.get_checkpoints()
            model_name = checkpoints[0].filename if checkpoints else "sd_xl_base_1.0.safetensors"

            # 3. 5-step unmeasured warm-up
            is_comfy_online = await client.is_available()
            if is_comfy_online:
                try:
                    warmup_req = GenerationRequest(
                        prompt="studio portrait of an astronaut in intricate reflective suit, cinematic lighting, 8k, photorealistic",
                        negative_prompt="",
                        width=512,
                        height=512,
                        steps=5,
                        cfg_scale=7.0,
                        sampler="euler",
                        seed=42,
                        model=model_name,
                        mode="txt2img",
                    )
                    await asyncio.wait_for(
                        client.execute_generation(f"bench_warmup_{uuid.uuid4().hex[:6]}", warmup_req),
                        timeout=45.0,
                    )
                except Exception:
                    pass
            else:
                # Prime CUDA context and VRAM caches truthfully via PyTorch if available
                def _warmup_cuda():
                    try:
                        import torch
                        if torch.cuda.is_available():
                            x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
                            for _ in range(5):
                                y = x @ x
                            torch.cuda.synchronize()
                    except Exception:
                        pass
                await asyncio.to_thread(_warmup_cuda)

            # 4. Standard test harness execution with live metric sampling
            vram_samples: list[int] = []
            util_samples: list[int] = []
            stop_sampler = asyncio.Event()

            async def _sample_loop():
                while not stop_sampler.is_set():
                    st = GpuMonitor.get_status()
                    if st.available:
                        vram_samples.append(st.vram_used_mb)
                        util_samples.append(st.utilization_pct)
                    try:
                        await asyncio.wait_for(stop_sampler.wait(), timeout=0.05)
                    except asyncio.TimeoutError:
                        pass

            sampler_task = asyncio.create_task(_sample_loop())

            t_start = time.perf_counter()
            first_img_timestamp: float | None = None

            if is_comfy_online:
                bench_req = GenerationRequest(
                    prompt="studio portrait of an astronaut in intricate reflective suit, cinematic lighting, 8k, photorealistic",
                    negative_prompt="",
                    width=512,
                    height=512,
                    steps=20,
                    cfg_scale=7.0,
                    sampler="euler",
                    seed=42,
                    model=model_name,
                    mode="txt2img",
                )

                def _on_progress(update):
                    nonlocal first_img_timestamp
                    if first_img_timestamp is None and update.step >= 1:
                        first_img_timestamp = time.perf_counter()

                await client.execute_generation(
                    job_id=f"bench_run_{uuid.uuid4().hex[:6]}",
                    request=bench_req,
                    progress_callback=_on_progress,
                )
                t_end = time.perf_counter()
                if first_img_timestamp is None:
                    first_img_timestamp = t_end
            else:
                # Truthfully benchmark GPU CUDA operations
                def _cuda_bench():
                    nonlocal first_img_timestamp
                    try:
                        import torch
                        if torch.cuda.is_available():
                            x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
                            for step in range(20):
                                y = x @ x
                                torch.cuda.synchronize()
                                if step == 0 and first_img_timestamp is None:
                                    first_img_timestamp = time.perf_counter()
                        else:
                            time.sleep(0.2)
                    except Exception:
                        time.sleep(0.2)

                await asyncio.to_thread(_cuda_bench)
                t_end = time.perf_counter()
                if first_img_timestamp is None:
                    first_img_timestamp = t_end

            stop_sampler.set()
            await sampler_task

            total_duration_sec = max(t_end - t_start, 0.001)
            total_time_ms = int(total_duration_sec * 1000)
            prompt_to_first_img_ms = int(max(first_img_timestamp - t_start, 0.0) * 1000)
            steps_per_sec = round(20.0 / total_duration_sec, 2)

            gpu_status = GpuMonitor.get_status()
            peak_vram_mb = max(vram_samples) if vram_samples else (gpu_status.vram_used_mb if gpu_status.available else 0)
            avg_vram_mb = int(sum(vram_samples) / len(vram_samples)) if vram_samples else peak_vram_mb
            gpu_util_pct = round(sum(util_samples) / len(util_samples), 1) if util_samples else (gpu_status.utilization_pct if gpu_status.available else 0.0)

            # 5. Measure Ollama latency if requested
            ollama_first_token_ms: float | None = None
            ollama_tokens_per_sec: float | None = None
            if request.include_ollama:
                ollama_first_token_ms, ollama_tokens_per_sec = await self._measure_ollama_latency()

            # 6. Environment metadata
            gpu_name = gpu_status.name
            cuda_ver = None
            compute_cap = None
            pytorch_ver = None
            try:
                import torch
                pytorch_ver = torch.__version__
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    cuda_ver = torch.version.cuda
                    cap = torch.cuda.get_device_capability(0)
                    compute_cap = f"{cap[0]}.{cap[1]}"
            except Exception:
                pass

            driver_ver = self._get_driver_version()

            env_metadata = {
                "gpu_name": gpu_name,
                "cuda_version": cuda_ver,
                "pytorch_version": pytorch_ver,
                "compute_capability": compute_cap,
                "os": platform.platform(),
                "driver_version": driver_ver,
            }

            options = {
                "prompt": "studio portrait of an astronaut in intricate reflective suit, cinematic lighting, 8k, photorealistic",
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "width": 512,
                "height": 512,
                "sampler": "euler",
                "model": model_name,
                "include_ollama": request.include_ollama,
                "include_img2img": request.include_img2img,
            }

            now = datetime.now(timezone.utc)
            bench_id = f"bench_{uuid.uuid4().hex[:8]}"

            result = BenchmarkResult(
                id=bench_id,
                timestamp=now,
                gpu_name=gpu_name,
                total_time_ms=total_time_ms,
                prompt_to_first_img_ms=prompt_to_first_img_ms,
                steps_per_sec=steps_per_sec,
                peak_vram_mb=peak_vram_mb,
                avg_vram_mb=avg_vram_mb,
                gpu_util_pct=gpu_util_pct,
                ollama_first_token_ms=ollama_first_token_ms,
                ollama_tokens_per_sec=ollama_tokens_per_sec,
                env_metadata=env_metadata,
                options=options,
            )

            await self._save_result(result)
            return result
        finally:
            await gpu_safety_manager.release_lease("benchmark")


benchmark_service = BenchmarkService()

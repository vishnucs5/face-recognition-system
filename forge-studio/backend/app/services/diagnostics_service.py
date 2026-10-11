import asyncio
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid

from app.config import settings
from app.schemas.diagnostics import DiagnosticCheck, DiagnosticsReport
from app.services.comfyui_client import ComfyUIClient
from app.services.ollama_client import OllamaClient
from app.services.training_engine import TrainingEngineService


class DiagnosticsService:
    def __init__(self):
        self._last_report: DiagnosticsReport | None = None
        self._last_sampled_at: float = 0.0
        self._lock = asyncio.Lock()

    async def run_diagnostics(self, force_refresh: bool = False) -> DiagnosticsReport:
        """
        Execute all 9 system diagnostic checks with a 30-second cache TTL.
        """
        async with self._lock:
            now_ts = time.time()
            if not force_refresh and (now_ts - self._last_sampled_at < 30.0) and self._last_report is not None:
                return self._last_report

            now = datetime.now(timezone.utc)

            # Run all 9 checks concurrently with timeout protection
            check_tasks = [
                self._check_backend_api(now),
                self._check_comfyui(now),
                self._check_ollama(now),
                self._check_checkpoints(now),
                self._check_controlnet(now),
                self._check_cuda_execution(now),
                self._check_training_readiness(now),
                self._check_output_permissions(now),
                self._check_lora_vault(now),
            ]

            results = await asyncio.gather(*check_tasks, return_exceptions=True)
            checks: list[DiagnosticCheck] = []

            check_metadata = [
                ("backend_api", "Backend API Connectivity", "api", "Start-Process powershell -ArgumentList \"-NoExit\", \"-Command\", \"cd c:\\vishnu\\forge-studio\\backend; python -m uvicorn app.main:app --host 127.0.0.1 --port 8000\""),
                ("comfyui_endpoint", "ComfyUI Endpoint & WebSocket", "comfyui", "Start-Process powershell -ArgumentList \"-NoExit\", \"-Command\", \"cd c:\\vishnu\\ComfyUI; c:\\vishnu\\.venv\\Scripts\\python.exe main.py --listen 127.0.0.1 --port 8188\""),
                ("ollama_connectivity", "Ollama Connectivity & Model Check", "ollama", f"ollama run {settings.OLLAMA_MODEL}"),
                ("checkpoint_availability", "Discovered Checkpoint Availability", "checkpoints", "Copy-Item <path-to-model.safetensors> c:\\vishnu\\ComfyUI\\models\\checkpoints\\"),
                ("controlnet_presence", "ControlNet Model Presence", "controlnet", "New-Item -ItemType Directory -Force -Path c:\\vishnu\\ComfyUI\\models\\controlnet"),
                ("cuda_execution", "Live PyTorch / CUDA Tensor Execution", "cuda", "pip install torch --index-url https://download.pytorch.org/whl/cu128"),
                ("training_readiness", "Training Engine Readiness", "training", "Test-Path c:\\vishnu\\forge-studio\\backend\\app\\services\\training.py"),
                ("output_permissions", "Output Directory Permissions", "storage", "New-Item -ItemType Directory -Force -Path c:\\vishnu\\forge-studio\\backend\\storage\\generated"),
                ("lora_vault_sync", "LoRA Vault Sync State", "loras", "New-Item -ItemType Directory -Force -Path c:\\vishnu\\forge-studio\\backend\\storage\\loras"),
            ]

            for i, res in enumerate(results):
                cid, name, category, rem_cmd = check_metadata[i]
                if isinstance(res, Exception):
                    checks.append(
                        DiagnosticCheck(
                            id=cid,
                            name=name,
                            category=category,
                            status="error",
                            message=f"Check encountered unexpected exception: {str(res)}",
                            details=str(res),
                            remediation_cmd=rem_cmd,
                            checked_at=now,
                        )
                    )
                else:
                    checks.append(res)

            report = DiagnosticsReport(checks=checks, sampled_at=now, cache_ttl_sec=30)
            self._last_report = report
            self._last_sampled_at = time.time()
            return report

    async def _check_backend_api(self, now: datetime) -> DiagnosticCheck:
        rem_cmd = "Start-Process powershell -ArgumentList \"-NoExit\", \"-Command\", \"cd c:\\vishnu\\forge-studio\\backend; python -m uvicorn app.main:app --host 127.0.0.1 --port 8000\""
        return DiagnosticCheck(
            id="backend_api",
            name="Backend API Connectivity",
            category="api",
            status="ok",
            message="FastAPI application is online and responding",
            details=f"Host: {settings.HOST}, Port: {settings.PORT}",
            remediation_cmd=rem_cmd,
            checked_at=now,
        )

    async def _check_comfyui(self, now: datetime) -> DiagnosticCheck:
        rem_cmd = "Start-Process powershell -ArgumentList \"-NoExit\", \"-Command\", \"cd c:\\vishnu\\ComfyUI; c:\\vishnu\\.venv\\Scripts\\python.exe main.py --listen 127.0.0.1 --port 8188\""
        try:
            client = ComfyUIClient()
            online = await asyncio.wait_for(client.is_available(), timeout=2.5)
            if online:
                return DiagnosticCheck(
                    id="comfyui_endpoint",
                    name="ComfyUI Endpoint & WebSocket",
                    category="comfyui",
                    status="ok",
                    message="ComfyUI endpoint is online and responsive",
                    details=f"Target URL: {client.base_url}",
                    remediation_cmd=rem_cmd,
                    checked_at=now,
                )
            return DiagnosticCheck(
                id="comfyui_endpoint",
                name="ComfyUI Endpoint & WebSocket",
                category="comfyui",
                status="error",
                message="ComfyUI endpoint is offline or unreachable",
                details=f"Target URL: {client.base_url}",
                remediation_cmd=rem_cmd,
                checked_at=now,
            )
        except Exception as e:
            return DiagnosticCheck(
                id="comfyui_endpoint",
                name="ComfyUI Endpoint & WebSocket",
                category="comfyui",
                status="error",
                message=f"ComfyUI check failed: {str(e)}",
                details=str(e),
                remediation_cmd=rem_cmd,
                checked_at=now,
            )

    async def _check_ollama(self, now: datetime) -> DiagnosticCheck:
        rem_cmd = f"ollama run {settings.OLLAMA_MODEL}"
        try:
            client = OllamaClient()
            online = await asyncio.wait_for(client.is_available(), timeout=2.5)
            if not online:
                return DiagnosticCheck(
                    id="ollama_connectivity",
                    name="Ollama Connectivity & Model Check",
                    category="ollama",
                    status="error",
                    message="Ollama service is unreachable or not running",
                    details=f"Target URL: {client.base_url}/api/tags",
                    remediation_cmd=rem_cmd,
                    checked_at=now,
                )

            models = await asyncio.wait_for(client.get_models(), timeout=3.0)
            target_model = settings.OLLAMA_MODEL
            # Model name match check (e.g. qwen3-coder:30b or qwen3-coder:30b:latest)
            has_model = any(m == target_model or m.startswith(f"{target_model}:") for m in models)
            if has_model:
                return DiagnosticCheck(
                    id="ollama_connectivity",
                    name="Ollama Connectivity & Model Check",
                    category="ollama",
                    status="ok",
                    message=f"Ollama is online with model '{target_model}' loaded",
                    details=f"Installed models: {', '.join(models)}",
                    remediation_cmd=rem_cmd,
                    checked_at=now,
                )
            return DiagnosticCheck(
                id="ollama_connectivity",
                name="Ollama Connectivity & Model Check",
                category="ollama",
                status="warning",
                message=f"Ollama is running but model '{target_model}' is not installed",
                details=f"Installed models: {', '.join(models) if models else 'None'}",
                remediation_cmd=rem_cmd,
                checked_at=now,
            )
        except Exception as e:
            return DiagnosticCheck(
                id="ollama_connectivity",
                name="Ollama Connectivity & Model Check",
                category="ollama",
                status="error",
                message=f"Ollama check failed: {str(e)}",
                details=str(e),
                remediation_cmd=rem_cmd,
                checked_at=now,
            )

    async def _check_checkpoints(self, now: datetime) -> DiagnosticCheck:
        rem_cmd = "Copy-Item <path-to-model.safetensors> c:\\vishnu\\ComfyUI\\models\\checkpoints\\"
        try:
            client = ComfyUIClient()
            checkpoints = await asyncio.wait_for(client.get_checkpoints(), timeout=3.0)
            if len(checkpoints) >= 1:
                return DiagnosticCheck(
                    id="checkpoint_availability",
                    name="Discovered Checkpoint Availability",
                    category="checkpoints",
                    status="ok",
                    message=f"Discovered {len(checkpoints)} checkpoint model(s)",
                    details=f"Models: {', '.join(m.name for m in checkpoints[:5])}",
                    remediation_cmd=rem_cmd,
                    checked_at=now,
                )
            return DiagnosticCheck(
                id="checkpoint_availability",
                name="Discovered Checkpoint Availability",
                category="checkpoints",
                status="warning",
                message="No SDXL or SD1.5 checkpoints found",
                details=f"Search paths: {settings.checkpoints_path}, c:/vishnu/ComfyUI/models/checkpoints",
                remediation_cmd=rem_cmd,
                checked_at=now,
            )
        except Exception as e:
            return DiagnosticCheck(
                id="checkpoint_availability",
                name="Discovered Checkpoint Availability",
                category="checkpoints",
                status="warning",
                message=f"Checkpoint discovery query failed: {str(e)}",
                details=str(e),
                remediation_cmd=rem_cmd,
                checked_at=now,
            )

    async def _check_controlnet(self, now: datetime) -> DiagnosticCheck:
        rem_cmd = "New-Item -ItemType Directory -Force -Path c:\\vishnu\\ComfyUI\\models\\controlnet"
        try:
            client = ComfyUIClient()
            models = await asyncio.wait_for(client.get_controlnet_models(), timeout=3.0)
            if len(models) >= 1:
                return DiagnosticCheck(
                    id="controlnet_presence",
                    name="ControlNet Model Presence",
                    category="controlnet",
                    status="ok",
                    message=f"Discovered {len(models)} ControlNet model(s)",
                    details=f"Models: {', '.join(m.name for m in models[:5])}",
                    remediation_cmd=rem_cmd,
                    checked_at=now,
                )
            return DiagnosticCheck(
                id="controlnet_presence",
                name="ControlNet Model Presence",
                category="controlnet",
                status="warning",
                message="No ControlNet models found",
                details="Checked models/controlnet and c:/vishnu/ComfyUI/models/controlnet",
                remediation_cmd=rem_cmd,
                checked_at=now,
            )
        except Exception as e:
            return DiagnosticCheck(
                id="controlnet_presence",
                name="ControlNet Model Presence",
                category="controlnet",
                status="warning",
                message=f"ControlNet check query failed: {str(e)}",
                details=str(e),
                remediation_cmd=rem_cmd,
                checked_at=now,
            )

    async def _check_cuda_execution(self, now: datetime) -> DiagnosticCheck:
        rem_cmd = "pip install torch --index-url https://download.pytorch.org/whl/cu128"
        def _run_tensor_kernel():
            import torch
            if not torch.cuda.is_available():
                return False, f"PyTorch {torch.__version__} initialized without CUDA support", None
            x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
            y = x @ x
            torch.cuda.synchronize()
            dev_name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            details = f"Device: {dev_name}, Compute Capability: {cap[0]}.{cap[1]}, PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}"
            return True, dev_name, details

        try:
            success, dev_name, details = await asyncio.to_thread(_run_tensor_kernel)
            if success:
                return DiagnosticCheck(
                    id="cuda_execution",
                    name="Live PyTorch / CUDA Tensor Execution",
                    category="cuda",
                    status="ok",
                    message=f"Live CUDA kernel execution verified on {dev_name}",
                    details=details,
                    remediation_cmd=rem_cmd,
                    checked_at=now,
                )
            return DiagnosticCheck(
                id="cuda_execution",
                name="Live PyTorch / CUDA Tensor Execution",
                category="cuda",
                status="error",
                message="PyTorch installed but CUDA is not available",
                details=dev_name,
                remediation_cmd=rem_cmd,
                checked_at=now,
            )
        except Exception as e:
            return DiagnosticCheck(
                id="cuda_execution",
                name="Live PyTorch / CUDA Tensor Execution",
                category="cuda",
                status="error",
                message=f"Live CUDA tensor execution failed: {str(e)}",
                details=str(e),
                remediation_cmd=rem_cmd,
                checked_at=now,
            )

    async def _check_training_readiness(self, now: datetime) -> DiagnosticCheck:
        rem_cmd = "Test-Path c:\\vishnu\\forge-studio\\backend\\app\\services\\training.py"
        try:
            health = TrainingEngineService.get_health()
            if health.engine_available:
                return DiagnosticCheck(
                    id="training_readiness",
                    name="Training Engine Readiness",
                    category="training",
                    status="ok",
                    message=f"Training engine readiness confirmed ({health.engine_name})",
                    details=health.message,
                    remediation_cmd=rem_cmd,
                    checked_at=now,
                )
            return DiagnosticCheck(
                id="training_readiness",
                name="Training Engine Readiness",
                category="training",
                status="warning",
                message="No external training engine installed (kohya_ss / ai-toolkit)",
                details=health.message,
                remediation_cmd=rem_cmd,
                checked_at=now,
            )
        except Exception as e:
            return DiagnosticCheck(
                id="training_readiness",
                name="Training Engine Readiness",
                category="training",
                status="warning",
                message=f"Training engine readiness check failed: {str(e)}",
                details=str(e),
                remediation_cmd=rem_cmd,
                checked_at=now,
            )

    async def _check_output_permissions(self, now: datetime) -> DiagnosticCheck:
        rem_cmd = "New-Item -ItemType Directory -Force -Path c:\\vishnu\\forge-studio\\backend\\storage\\generated"
        def _test_file_io():
            gen_dir = settings.generated_path
            gen_dir.mkdir(parents=True, exist_ok=True)
            probe_file = gen_dir / f".probe_{uuid.uuid4().hex}.tmp"
            probe_file.write_text("forge_studio_diagnostic_probe", encoding="utf-8")
            content = probe_file.read_text(encoding="utf-8")
            if content != "forge_studio_diagnostic_probe":
                raise RuntimeError("Probe read verification mismatch")
            probe_file.unlink(missing_ok=True)
            return str(gen_dir)

        try:
            gen_path = await asyncio.to_thread(_test_file_io)
            return DiagnosticCheck(
                id="output_permissions",
                name="Output Directory Permissions",
                category="storage",
                status="ok",
                message="Output directory write and delete permissions verified",
                details=f"Path: {gen_path}",
                remediation_cmd=rem_cmd,
                checked_at=now,
            )
        except Exception as e:
            return DiagnosticCheck(
                id="output_permissions",
                name="Output Directory Permissions",
                category="storage",
                status="error",
                message=f"Output directory permission check failed: {str(e)}",
                details=str(e),
                remediation_cmd=rem_cmd,
                checked_at=now,
            )

    async def _check_lora_vault(self, now: datetime) -> DiagnosticCheck:
        rem_cmd = "New-Item -ItemType Directory -Force -Path c:\\vishnu\\forge-studio\\backend\\storage\\loras"
        try:
            loras_path = settings.loras_path
            if loras_path.exists():
                lora_files = [f for f in loras_path.glob("*.*") if f.suffix.lower() in [".safetensors", ".pt"]]
                return DiagnosticCheck(
                    id="lora_vault_sync",
                    name="LoRA Vault Sync State",
                    category="loras",
                    status="ok",
                    message=f"LoRA vault accessible with {len(lora_files)} model(s) discovered",
                    details=f"Path: {loras_path}",
                    remediation_cmd=rem_cmd,
                    checked_at=now,
                )
            return DiagnosticCheck(
                id="lora_vault_sync",
                name="LoRA Vault Sync State",
                category="loras",
                status="warning",
                message="LoRA directory does not exist",
                details=f"Expected: {loras_path}",
                remediation_cmd=rem_cmd,
                checked_at=now,
            )
        except Exception as e:
            return DiagnosticCheck(
                id="lora_vault_sync",
                name="LoRA Vault Sync State",
                category="loras",
                status="warning",
                message=f"LoRA vault check failed: {str(e)}",
                details=str(e),
                remediation_cmd=rem_cmd,
                checked_at=now,
            )


diagnostics_service = DiagnosticsService()

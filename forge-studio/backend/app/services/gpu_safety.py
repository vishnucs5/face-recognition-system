import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import subprocess
import time

from app.schemas.system import ActiveLeaseInfo, GpuStatus, ObservedProcess, TelemetryResponse
from app.services.gpu_monitor import GpuMonitor


class GpuSafetyManager:
    TTL_SECONDS: int = 300
    SAFETY_MARGIN: float = 1.20
    GRAPH_OVERHEAD_MB: float = 600.0
    CONTROLNET_OVERHEAD_MB: float = 1600.0

    SD15_BASELINE_MB: float = 2600.0
    SD15_LATENT_FACTOR: float = 1.8e-6
    SD15_LORA_MB: float = 200.0

    SDXL_BASELINE_MB: float = 6400.0
    SDXL_LATENT_FACTOR: float = 3.2e-6
    SDXL_LORA_MB: float = 500.0

    COPILOT_MIN_FREE_VRAM_MB: int = 4096

    POLICY_LIMITS: dict[str, float] = {
        "balanced": 0.85,
        "max_quality": 0.92,
        "max_speed": 0.78,
        "conservative": 0.65,
    }

    HEAVY_LEASE_TYPES: set[str] = {"generation", "training", "benchmark"}

    def __init__(self):
        self._lock = asyncio.Lock()
        self._active_lease: ActiveLeaseInfo | None = None
        self._waiting_jobs: set[str] = set()

    @staticmethod
    def estimate_vram(
        model_family: str,
        width: int,
        height: int,
        steps: int = 20,
        loras_count: int = 0,
        has_controlnet: bool = False,
    ) -> int:
        """
        Estimate peak VRAM required for a workload.
        Formula: int((baseline + latents + loras + cnet + 600) * 1.20)
        """
        family = (model_family or "").lower().replace("-", "").replace("_", "").replace(" ", "")
        is_sdxl = "sdxl" in family

        if is_sdxl:
            baseline = GpuSafetyManager.SDXL_BASELINE_MB
            latents = GpuSafetyManager.SDXL_LATENT_FACTOR * width * height
            loras = GpuSafetyManager.SDXL_LORA_MB * max(0, loras_count)
        else:
            baseline = GpuSafetyManager.SD15_BASELINE_MB
            latents = GpuSafetyManager.SD15_LATENT_FACTOR * width * height
            loras = GpuSafetyManager.SD15_LORA_MB * max(0, loras_count)

        cnet = GpuSafetyManager.CONTROLNET_OVERHEAD_MB if has_controlnet else 0.0
        overhead = GpuSafetyManager.GRAPH_OVERHEAD_MB

        total = (baseline + latents + loras + cnet + overhead) * GpuSafetyManager.SAFETY_MARGIN
        return int(total)

    @classmethod
    def get_policy_limit_mb(cls, policy: str = "balanced", total_vram_mb: int | None = None) -> int:
        """
        Calculate maximum VRAM allocated under a given user policy.
        Inspects physical VRAM dynamically via GpuMonitor if total_vram_mb is omitted.
        """
        if total_vram_mb is None:
            status = GpuMonitor.get_status()
            total_vram_mb = status.vram_total_mb if status.available else 0
        pct = cls.POLICY_LIMITS.get(policy.lower(), 0.85)
        return int(total_vram_mb * pct)

    @classmethod
    def can_run_copilot(
        cls,
        policy: str = "balanced",
        free_vram_mb: int | None = None,
        is_heavy_render_active: bool = False,
    ) -> tuple[bool, str]:
        """
        Evaluate if Copilot can run based on user policy and available free VRAM.
        """
        p = policy.lower()
        if p == "max_quality":
            if is_heavy_render_active:
                return False, "Copilot blocked during heavy render under max_quality policy"
            return True, "Copilot allowed"

        if free_vram_mb is None:
            status = GpuMonitor.get_status()
            free_vram_mb = status.vram_free_mb if status.available else 0

        if p == "balanced":
            if free_vram_mb >= cls.COPILOT_MIN_FREE_VRAM_MB:
                return True, "Copilot allowed"
            return False, f"Copilot requires >= 4.0 GB free VRAM (current: {free_vram_mb} MB) under balanced policy"

        if p == "conservative":
            if free_vram_mb >= cls.COPILOT_MIN_FREE_VRAM_MB:
                return True, "Copilot allowed"
            return False, f"Copilot blocked: insufficient free VRAM under conservative policy ({free_vram_mb} MB)"

        return True, "Copilot allowed"

    @classmethod
    def can_allocate(
        cls,
        model_family: str,
        width: int,
        height: int,
        steps: int = 20,
        loras_count: int = 0,
        has_controlnet: bool = False,
        policy: str = "balanced",
        total_vram_mb: int | None = None,
    ) -> tuple[bool, str]:
        estimated = cls.estimate_vram(model_family, width, height, steps, loras_count, has_controlnet)
        limit = cls.get_policy_limit_mb(policy, total_vram_mb)
        if limit > 0 and estimated > limit:
            return False, f"Estimated VRAM ({estimated} MB) exceeds {policy} policy limit ({limit} MB)"
        return True, f"Estimated VRAM ({estimated} MB) within {policy} policy limit ({limit} MB)"

    async def acquire_lease(self, job_id: str, lease_type: str, holder: str = "forge") -> tuple[bool, str]:
        """
        Atomically acquire exclusive heavy GPU lease.
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            if self._active_lease is not None:
                if self._active_lease.expires_at <= now:
                    self._active_lease = None
                elif self._active_lease.job_id == job_id:
                    self._active_lease.expires_at = now + timedelta(seconds=self.TTL_SECONDS)
                    return True, f"Lease renewed for job {job_id}"
                else:
                    return False, f"GPU busy: active {self._active_lease.lease_type} lease held by job {self._active_lease.job_id}"

            expires_at = now + timedelta(seconds=self.TTL_SECONDS)
            self._active_lease = ActiveLeaseInfo(
                lease_type=lease_type,
                job_id=job_id,
                holder=holder,
                acquired_at=now,
                expires_at=expires_at,
            )
            self._waiting_jobs.discard(job_id)
            return True, "Lease acquired"

    async def release_lease(self, job_id: str) -> bool:
        """
        Release held GPU lease.
        """
        async with self._lock:
            if self._active_lease is not None and self._active_lease.job_id == job_id:
                self._active_lease = None
                return True
            return False

    async def heartbeat(self, job_id: str) -> bool:
        """
        Renew lease TTL by 300s.
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            if self._active_lease is not None and self._active_lease.job_id == job_id:
                if self._active_lease.expires_at > now:
                    self._active_lease.expires_at = now + timedelta(seconds=self.TTL_SECONDS)
                    return True
                else:
                    self._active_lease = None
                    return False
            return False

    async def get_active_lease(self) -> ActiveLeaseInfo | None:
        async with self._lock:
            if self._active_lease is not None:
                now = datetime.now(timezone.utc)
                if self._active_lease.expires_at <= now:
                    self._active_lease = None
            return self._active_lease

    def register_waiting(self, job_id: str):
        self._waiting_jobs.add(job_id)

    def unregister_waiting(self, job_id: str):
        self._waiting_jobs.discard(job_id)

    def reset_leases(self):
        self._active_lease = None
        self._waiting_jobs.clear()

    @staticmethod
    def get_observed_processes() -> list[ObservedProcess]:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return []
        try:
            cmd = [
                nvidia_smi,
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            if result.returncode != 0 or not result.stdout.strip():
                return []

            observed = []
            target_names = ("llama-server", "python", "ollama")
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    try:
                        pid = int(parts[0])
                    except ValueError:
                        continue
                    raw_path = parts[1]
                    proc_name = Path(raw_path).name.lower()
                    if any(t in proc_name for t in target_names):
                        vram_mb = 0
                        if len(parts) >= 3 and parts[2] and parts[2] != "[N/A]":
                            try:
                                vram_mb = int(float(parts[2]))
                            except ValueError:
                                vram_mb = 0
                        observed.append(ObservedProcess(pid=pid, name=proc_name, vram_used_mb=vram_mb))
            return observed
        except Exception:
            return []

    def determine_state(
        self,
        gpu_status: GpuStatus,
        active_lease: ActiveLeaseInfo | None,
        observed_processes: list[ObservedProcess],
    ) -> str:
        if not gpu_status.available:
            name_lower = (gpu_status.name or "").lower()
            if "cuda" in name_lower:
                return "cuda_unavailable"
            return "gpu_unavailable"

        if active_lease is not None:
            lease_type = active_lease.lease_type.lower()
            if lease_type == "training":
                return "training_active"
            elif lease_type in ("generation", "comfyui"):
                return "comfyui_active"
            elif lease_type == "waiting":
                return "waiting_for_gpu"
            else:
                return "comfyui_active"

        if len(self._waiting_jobs) > 0:
            return "waiting_for_gpu"

        proc_names = [p.name.lower() for p in observed_processes]
        if any("llama-server" in name or "ollama" in name for name in proc_names):
            return "ollama_active"

        # Memory pressure check
        if gpu_status.vram_total_mb > 0:
            used_pct = (gpu_status.vram_used_mb / gpu_status.vram_total_mb) * 100
            if used_pct >= 85 or gpu_status.utilization_pct >= 85:
                return "gpu_memory_pressure"

        if any("python" in name for name in proc_names) and (gpu_status.vram_used_mb > 2000):
            return "comfyui_active"

        return "gpu_ready"

    async def get_telemetry(self) -> TelemetryResponse:
        start_time = time.perf_counter()
        now = datetime.now(timezone.utc)

        gpu_status = GpuMonitor.get_status()
        observed_processes = self.get_observed_processes()
        active_lease = await self.get_active_lease()

        cpu_ram_total_mb = 0
        cpu_ram_used_mb = 0
        try:
            import psutil
            vm = psutil.virtual_memory()
            cpu_ram_total_mb = int(vm.total / (1024 * 1024))
            cpu_ram_used_mb = int(vm.used / (1024 * 1024))
        except Exception:
            pass

        state = self.determine_state(gpu_status, active_lease, observed_processes)
        backend_latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return TelemetryResponse(
            gpu_name=gpu_status.name if gpu_status.available else "GPU Unavailable",
            total_vram_mb=gpu_status.vram_total_mb if gpu_status.available else 0,
            used_vram_mb=gpu_status.vram_used_mb if gpu_status.available else 0,
            free_vram_mb=gpu_status.vram_free_mb if gpu_status.available else 0,
            utilization_pct=gpu_status.utilization_pct if gpu_status.available else 0,
            temperature_c=gpu_status.temperature_c if gpu_status.available else None,
            cpu_ram_used_mb=cpu_ram_used_mb,
            cpu_ram_total_mb=cpu_ram_total_mb,
            observed_processes=observed_processes,
            active_lease=active_lease,
            state=state,
            pressure_threshold_pct=85,
            critical_threshold_pct=95,
            freshness_ms=0,
            is_stale=False,
            backend_latency_ms=backend_latency_ms,
            sampled_at=now,
        )


gpu_safety_manager = GpuSafetyManager()

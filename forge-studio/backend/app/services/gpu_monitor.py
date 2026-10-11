import shutil
import subprocess
from app.schemas.system import GpuStatus


class GpuMonitor:
    @staticmethod
    def get_status() -> GpuStatus:
        nvidia_smi = shutil.which("nvidia-smi")
        if not nvidia_smi:
            return GpuStatus(
                available=False,
                name="NVIDIA GPU Not Detected",
                vram_used_mb=0,
                vram_total_mb=0,
                vram_free_mb=0,
                utilization_pct=0,
                temperature_c=None
            )

        try:
            cmd = [
                nvidia_smi,
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 6:
                    name = parts[0]
                    total_mb = int(float(parts[1]))
                    used_mb = int(float(parts[2]))
                    free_mb = int(float(parts[3]))
                    util_pct = int(float(parts[4]))
                    temp_c = int(float(parts[5])) if parts[5].isdigit() else None

                    return GpuStatus(
                        available=True,
                        name=name,
                        vram_used_mb=used_mb,
                        vram_total_mb=total_mb,
                        vram_free_mb=free_mb,
                        utilization_pct=util_pct,
                        temperature_c=temp_c
                    )
        except Exception:
            pass

        return GpuStatus(
            available=False,
            name="GPU Query Inactive",
            vram_used_mb=0,
            vram_total_mb=0,
            vram_free_mb=0,
            utilization_pct=0
        )

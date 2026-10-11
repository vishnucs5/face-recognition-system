import re
from pydantic import BaseModel, Field


class ErrorClassification(BaseModel):
    subsystem: str  # "gpu", "comfyui", "ollama", "filesystem", "network", "validation"
    category: str   # "transient" or "permanent"
    can_retry: bool
    sanitized_message: str
    remediation_guidance: str | None = None
    error_code: str | None = None


# Patterns for redaction
WINDOWS_USER_PATH_PATTERN = re.compile(
    r"(?i)([a-z]:[\\/]users[\\/])[^\\/\s\"']+",
)

API_KEY_PATTERN = re.compile(
    r"sk-[a-zA-Z0-9_\-]{16,}",
)

BEARER_TOKEN_PATTERN = re.compile(
    r"(?i)(bearer\s+)[a-zA-Z0-9_\-\.]{16,}",
)

CREDENTIAL_PARAMS_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|access[_-]?token|auth[_-]?token|token|password|passwd|secret)\s*[:=]\s*)[^\s&\"',;]+",
)


def sanitize_error_message(raw_error: str) -> str:
    """
    Sanitizes raw error messages by redacting:
    - Windows user profile paths (e.g. C:\\Users\\<username>\\... -> C:\\Users\\[REDACTED]\\...)
    - API keys (e.g. sk-...)
    - Bearer authorization tokens
    - Embedded passwords/secrets/credentials in URLs or query strings
    """
    if not raw_error:
        return ""

    sanitized = str(raw_error)

    # 1. Redact Windows user profile paths
    sanitized = WINDOWS_USER_PATH_PATTERN.sub(r"\g<1>[REDACTED]", sanitized)

    # 2. Redact API keys (sk-...)
    sanitized = API_KEY_PATTERN.sub("sk-[REDACTED]", sanitized)

    # 3. Redact Bearer tokens
    sanitized = BEARER_TOKEN_PATTERN.sub(r"\g<1>[REDACTED]", sanitized)

    # 4. Redact password/token/secret parameters
    sanitized = CREDENTIAL_PARAMS_PATTERN.sub(r"\g<1>[REDACTED]", sanitized)

    return sanitized.strip()


def classify_error(error_msg: str, http_status: int | None = None) -> ErrorClassification:
    """
    Classifies an error message into a subsystem, category (transient vs permanent),
    determines retryability, and provides copyable remediation guidance.
    """
    sanitized = sanitize_error_message(error_msg)
    lower = sanitized.lower()

    # 1. GPU Subsystem
    if any(k in lower for k in [
        "cuda out of memory",
        "out of memory",
        "cublas",
        "cuda_error",
        "cuda error",
        "vram",
        "torch.cuda",
        "device-side assert",
    ]):
        if any(k in lower for k in ["out of memory", "vram", "allocation"]):
            return ErrorClassification(
                subsystem="gpu",
                category="transient",
                can_retry=True,
                sanitized_message=sanitized,
                remediation_guidance=(
                    "Reduce generation resolution (e.g. to 768x768 or 1024x1024), "
                    "reduce batch size, or release unused GPU leases."
                ),
                error_code="GPU_OOM",
            )
        return ErrorClassification(
            subsystem="gpu",
            category="permanent",
            can_retry=False,
            sanitized_message=sanitized,
            remediation_guidance="Check NVIDIA driver health via 'nvidia-smi' and verify CUDA installation.",
            error_code="GPU_HARDWARE_ERROR",
        )

    # 2. ComfyUI Subsystem
    if any(k in lower for k in ["comfyui", "8188", "comfy client", "prompt_id"]):
        if any(k in lower for k in [
            "offline",
            "connection refused",
            "connecterror",
            "connection reset",
            "cannot connect",
        ]):
            return ErrorClassification(
                subsystem="comfyui",
                category="transient",
                can_retry=True,
                sanitized_message=sanitized,
                remediation_guidance=(
                    "Start ComfyUI via PowerShell: "
                    "& 'c:\\vishnu\\ComfyUI\\venv\\Scripts\\python.exe' 'c:\\vishnu\\ComfyUI\\main.py' --port 8188"
                ),
                error_code="COMFYUI_OFFLINE",
            )
        if any(k in lower for k in [
            "no model checkpoints",
            "checkpoint not found",
            "missing checkpoint",
            "model checkpoint",
        ]):
            return ErrorClassification(
                subsystem="comfyui",
                category="permanent",
                can_retry=False,
                sanitized_message=sanitized,
                remediation_guidance="Place model checkpoint (.safetensors) into ComfyUI/models/checkpoints directory.",
                error_code="COMFYUI_MISSING_CHECKPOINT",
            )
        if any(k in lower for k in ["invalid sampler", "unknown node", "keyerror", "sampler"]):
            return ErrorClassification(
                subsystem="comfyui",
                category="permanent",
                can_retry=False,
                sanitized_message=sanitized,
                remediation_guidance="Select a supported sampler from the dropdown or verify ComfyUI node definitions.",
                error_code="COMFYUI_INVALID_CONFIG",
            )
        is_transient = "timeout" in lower or "timed out" in lower
        return ErrorClassification(
            subsystem="comfyui",
            category="transient" if is_transient else "permanent",
            can_retry=is_transient,
            sanitized_message=sanitized,
            remediation_guidance="Inspect ComfyUI console output for workflow execution traces.",
            error_code="COMFYUI_EXECUTION_ERROR",
        )

    # 3. Ollama Subsystem
    if any(k in lower for k in ["ollama", "11434", "qwen"]):
        if any(k in lower for k in [
            "offline",
            "unreachable",
            "connection refused",
            "connecterror",
            "cannot connect",
        ]):
            return ErrorClassification(
                subsystem="ollama",
                category="transient",
                can_retry=True,
                sanitized_message=sanitized,
                remediation_guidance="Start Ollama service via PowerShell: 'ollama serve'",
                error_code="OLLAMA_OFFLINE",
            )
        is_transient = "timeout" in lower or "timed out" in lower
        return ErrorClassification(
            subsystem="ollama",
            category="transient" if is_transient else "permanent",
            can_retry=is_transient,
            sanitized_message=sanitized,
            remediation_guidance="Verify Ollama model weights are pulled with 'ollama list'.",
            error_code="OLLAMA_ERROR",
        )

    # 4. Filesystem Subsystem
    if any(k in lower for k in [
        "filenotfound",
        "no such file",
        "permission denied",
        "oserror",
        "path does not exist",
    ]):
        return ErrorClassification(
            subsystem="filesystem",
            category="permanent",
            can_retry=False,
            sanitized_message=sanitized,
            remediation_guidance="Verify target file or directory path exists and Studio has read/write permissions.",
            error_code="FS_ERROR",
        )

    # 5. Network Subsystem
    if any(k in lower for k in ["timeout", "timed out", "connecttimeout", "readtimeout"]):
        return ErrorClassification(
            subsystem="network",
            category="transient",
            can_retry=True,
            sanitized_message=sanitized,
            remediation_guidance="Request timed out waiting for backend service. Check system load and try again.",
            error_code="NETWORK_TIMEOUT",
        )
    if any(k in lower for k in [
        "connection refused",
        "connecterror",
        "connection reset",
        "failed to establish a new connection",
    ]):
        return ErrorClassification(
            subsystem="network",
            category="transient",
            can_retry=True,
            sanitized_message=sanitized,
            remediation_guidance="Target network endpoint refused connection. Verify the required server is running.",
            error_code="NETWORK_REFUSED",
        )

    # 6. Validation Subsystem
    if (http_status and http_status in (400, 422)) or any(k in lower for k in [
        "validation",
        "valueerror",
        "invalid parameter",
        "bad request",
        "unsupported",
    ]):
        return ErrorClassification(
            subsystem="validation",
            category="permanent",
            can_retry=False,
            sanitized_message=sanitized,
            remediation_guidance="Review input parameters: ensure resolutions are multiples of 64 and within model limits.",
            error_code="VALIDATION_ERROR",
        )

    # 7. Generic Fallback
    is_transient = (http_status in (502, 503, 504)) or ("retry" in lower)
    return ErrorClassification(
        subsystem="network" if is_transient else "validation",
        category="transient" if is_transient else "permanent",
        can_retry=is_transient,
        sanitized_message=sanitized,
        remediation_guidance="Check application logs or retry the operation.",
        error_code="GENERIC_ERROR",
    )

"""Centralized application configuration management."""

import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SystemConfig(BaseSettings):
    """System-wide configuration settings with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project paths
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    enrolled_dir: Path = PROJECT_ROOT / "data" / "enrolled"
    test_dir: Path = PROJECT_ROOT / "data" / "test"
    database_path: Path = PROJECT_ROOT / "data" / "database" / "faces.db"
    models_dir: Path = PROJECT_ROOT / "models"
    output_dir: Path = PROJECT_ROOT / "outputs"
    eval_output_dir: Path = PROJECT_ROOT / "outputs" / "evaluation"

    # Model specifications
    detector_model_name: str = "YuNet"
    detector_model_file: str = "face_detection_yunet_2023mar.onnx"
    detector_url: str = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )
    detector_mirror_url: str = (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )

    embedder_model_name: str = "SFace"
    embedder_model_file: str = "face_recognition_sface_2021dec.onnx"
    embedder_url: str = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    )
    embedder_mirror_url: str = (
        "https://huggingface.co/opencv/opencv_zoo/resolve/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
    )

    embedding_dimension: int = 128

    # Thresholds
    detection_threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for YuNet face detection",
    )
    match_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for accepting a known identity",
    )

    # Strategy for multi-embedding aggregation per identity: 'max' or 'mean'
    matching_strategy: Literal["max", "mean"] = "max"

    # Inference device
    device: Literal["auto", "cpu", "cuda"] = "auto"

    # Quality control during enrollment
    min_face_size: int = 60  # Minimum face width/height in pixels
    min_laplacian_variance: float = 25.0  # Blur detection threshold
    max_image_dimension: int = 1920  # Auto-rescale large images to this max dimension

    # Top-K candidates to return during identification
    top_k_candidates: int = 3

    def get_yunet_path(self) -> Path:
        """Return absolute path to YuNet ONNX weights."""
        return self.models_dir / self.detector_model_file

    def get_sface_path(self) -> Path:
        """Return absolute path to SFace ONNX weights."""
        return self.models_dir / self.embedder_model_file

    def ensure_directories(self) -> None:
        """Create necessary directories if they do not exist."""
        for path in [
            self.data_dir,
            self.enrolled_dir,
            self.test_dir,
            self.database_path.parent,
            self.models_dir,
            self.output_dir,
            self.eval_output_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = SystemConfig()
config.ensure_directories()

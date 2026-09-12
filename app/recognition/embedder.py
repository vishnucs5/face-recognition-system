"""Face embedding extraction using OpenCV SFace deep CNN."""

import logging
import urllib.request
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np

from app.config import config
from app.detection.detector import FaceDetection
from app.utils.image_utils import load_image

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """Extracts 128-dimensional L2-normalized biometric embeddings via SFace."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else config.get_sface_path()
        self._recognizer: Optional[cv2.FaceRecognizerSF] = None
        self._ensure_model_downloaded()
        self._load_model()

    def _ensure_model_downloaded(self) -> None:
        """Download SFace ONNX weights if not available locally."""
        if self.model_path.exists() and self.model_path.stat().st_size > 30000000:
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        urls = [config.embedder_url, config.embedder_mirror_url]
        downloaded = False

        for url in urls:
            try:
                logger.info("Downloading SFace model from %s to %s", url, self.model_path)
                urllib.request.urlretrieve(url, str(self.model_path))
                if self.model_path.exists() and self.model_path.stat().st_size > 30000000:
                    downloaded = True
                    logger.info("SFace model downloaded successfully.")
                    break
            except Exception as e:
                logger.warning("Failed downloading SFace from %s: %s", url, e)

        if not downloaded:
            raise RuntimeError(
                f"Could not download SFace model to {self.model_path}. Please check internet connection."
            )

    def _load_model(self) -> None:
        """Initialize the cv2.FaceRecognizerSF instance."""
        try:
            self._recognizer = cv2.FaceRecognizerSF.create(
                model=str(self.model_path),
                config="",
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )
            logger.info("SFace embedder initialized successfully (128-d output)")
        except Exception as e:
            logger.error("Failed to load SFace model: %s", e)
            raise RuntimeError(f"Face embedder initialization failed: {e}")

    def align_face(
        self,
        img_bgr: np.ndarray,
        detection: Union[FaceDetection, np.ndarray],
    ) -> np.ndarray:
        """Align and crop face to 112x112 using facial landmarks.

        Args:
            img_bgr: Source image in BGR format.
            detection: FaceDetection object or raw 15-element array.

        Returns:
            np.ndarray: Aligned face of shape (112, 112, 3).
        """
        raw = detection.raw_array if isinstance(detection, FaceDetection) else detection
        aligned = self._recognizer.alignCrop(img_bgr, raw)
        return aligned

    def generate(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        detection: Optional[Union[FaceDetection, np.ndarray]] = None,
    ) -> np.ndarray:
        """Generate a 128-dimensional L2-normalized face embedding.

        Args:
            image: Image containing the face.
            detection: Optional detection information with landmarks for alignment.

        Returns:
            np.ndarray: 1D float32 vector of shape (128,) with unit L2 norm.

        Raises:
            ValueError: If input is invalid or embedding contains NaN/Inf values.
        """
        img_bgr = load_image(image)

        if detection is not None:
            aligned = self.align_face(img_bgr, detection)
        else:
            # If no detection landmarks provided, resize image directly to 112x112
            aligned = cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_AREA)

        # Extract raw features from SFace (shape: 1, 128)
        raw_feat = self._recognizer.feature(aligned)

        vec = np.asarray(raw_feat, dtype=np.float32).flatten()

        # Validate vector dimension
        if vec.shape != (config.embedding_dimension,):
            raise ValueError(
                f"Unexpected embedding dimension: got {vec.shape}, expected ({config.embedding_dimension},)"
            )

        # Numerical integrity check
        if not np.all(np.isfinite(vec)):
            raise ValueError("Face embedding contains non-finite values (NaN or Inf).")

        # L2 Normalization: ensures dot product equals cosine similarity
        norm = float(np.linalg.norm(vec))
        if norm < 1e-7:
            raise ValueError("Zero-norm face embedding generated from input.")

        normalized_vec = (vec / norm).astype(np.float32)
        return normalized_vec

"""Face detection module using OpenCV YuNet DNN."""

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from app.config import config
from app.utils.image_utils import load_image, resize_with_aspect_ratio

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Structured representation of a single detected face."""

    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: np.ndarray  # (5, 2) coords: right eye, left eye, nose, right mouth, left mouth
    raw_array: np.ndarray  # 15-element array for direct OpenCV SFace alignment


class FaceDetector:
    """YuNet deep learning face detector integrated with OpenCV DNN."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        confidence_threshold: Optional[float] = None,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        self.model_path = Path(model_path) if model_path else config.get_yunet_path()
        self.confidence_threshold = (
            confidence_threshold if confidence_threshold is not None else config.detection_threshold
        )
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._detector: Optional[cv2.FaceDetectorYN] = None
        self._ensure_model_downloaded()
        self._load_model()

    def _ensure_model_downloaded(self) -> None:
        """Download YuNet ONNX model from repository if not present locally."""
        if self.model_path.exists() and self.model_path.stat().st_size > 50000:
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        urls = [config.detector_url, config.detector_mirror_url]
        downloaded = False

        for url in urls:
            try:
                logger.info("Downloading YuNet model from %s to %s", url, self.model_path)
                urllib.request.urlretrieve(url, str(self.model_path))
                if self.model_path.exists() and self.model_path.stat().st_size > 50000:
                    downloaded = True
                    logger.info("YuNet model downloaded successfully.")
                    break
            except Exception as e:
                logger.warning("Failed downloading YuNet from %s: %e", url, e)

        if not downloaded:
            raise RuntimeError(
                f"Could not download YuNet model to {self.model_path}. Please check internet connection."
            )

    def _load_model(self) -> None:
        """Instantiate the cv2.FaceDetectorYN object."""
        try:
            self._detector = cv2.FaceDetectorYN.create(
                model=str(self.model_path),
                config="",
                input_size=(320, 320),
                score_threshold=float(self.confidence_threshold),
                nms_threshold=float(self.nms_threshold),
                top_k=int(self.top_k),
                backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
                target_id=cv2.dnn.DNN_TARGET_CPU,
            )
            logger.info("YuNet face detector initialized (threshold=%.2f)", self.confidence_threshold)
        except Exception as e:
            logger.error("Failed to load YuNet face detector: %s", e)
            raise RuntimeError(f"Face detector model initialization failed: {e}")

    def detect(
        self,
        image: Union[str, Path, bytes, np.ndarray],
    ) -> List[FaceDetection]:
        """Detect all faces in an input image.

        Args:
            image: Image source (file path, bytes, or BGR numpy array).

        Returns:
            List[FaceDetection]: List of detected faces sorted by bounding box area descending.
        """
        img_bgr = load_image(image)
        h, w = img_bgr.shape[:2]

        if h < 10 or w < 10:
            logger.warning("Image dimensions too small for face detection: %sx%s", w, h)
            return []

        # YuNet requires explicitly matching its input size to the image dimensions
        self._detector.setInputSize((w, h))

        retval, raw_faces = self._detector.detect(img_bgr)

        if raw_faces is None or len(raw_faces) == 0:
            logger.debug("No faces detected in image of size %sx%s", w, h)
            return []

        detections: List[FaceDetection] = []
        for face in raw_faces:
            # face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            x, y, fw, fh = face[0:4]
            conf = float(face[14])

            # Convert (x, y, w, h) to (x1, y1, x2, y2)
            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(w, int(round(x + fw)))
            y2 = min(h, int(round(y + fh)))

            # 5 landmarks
            landmarks = np.array(
                [
                    [face[4], face[5]],   # Right eye
                    [face[6], face[7]],   # Left eye
                    [face[8], face[9]],   # Nose tip
                    [face[10], face[11]], # Right mouth corner
                    [face[12], face[13]], # Left mouth corner
                ],
                dtype=np.float32,
            )

            detections.append(
                FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    landmarks=landmarks,
                    raw_array=face.astype(np.float32),
                )
            )

        # Sort by bounding box area (largest face first)
        detections.sort(
            key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
            reverse=True,
        )

        logger.info("Detected %d face(s) in image", len(detections))
        return detections

"""Face enrollment workflow with validation, quality checks, and multi-embedding support."""

import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import cv2
import numpy as np

from app.config import config
from app.database import DatabaseManager
from app.detection.detector import FaceDetection, FaceDetector
from app.recognition.embedder import FaceEmbedder
from app.utils.image_utils import crop_face, load_image
from app.utils.validation import (
    QualityAssessment,
    check_image_quality,
    validate_person_name,
)

logger = logging.getLogger(__name__)


@dataclass
class EnrollmentResult:
    """Result summary of face enrollment operation."""

    success: bool
    person_id: Optional[int]
    person_name: str
    embedding_id: Optional[int]
    is_new_person: bool
    total_embeddings_for_person: int
    quality: Optional[QualityAssessment]
    detection: Optional[FaceDetection]
    saved_image_path: Optional[str]
    message: str


class EnrollmentService:
    """Orchestrates identity validation, single-face verification, quality checks, and embedding persistence."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
    ):
        self.db = db_manager if db_manager is not None else DatabaseManager()
        self.detector = detector if detector is not None else FaceDetector()
        self.embedder = embedder if embedder is not None else FaceEmbedder()

    def enroll(
        self,
        name: str,
        image: Union[str, Path, bytes, np.ndarray],
        allow_save_image: bool = True,
        skip_quality_check: bool = False,
    ) -> EnrollmentResult:
        """Enroll a person's face into the local biometric database.

        Args:
            name: Intended identity name.
            image: Image containing exactly one face.
            allow_save_image: Whether to store reference image in data/enrolled/<name>/.
            skip_quality_check: Bypass blur/size quality checks if explicitly requested.

        Returns:
            EnrollmentResult: Status, IDs, quality score, and informative user message.
        """
        # 1. Validate identity name
        try:
            clean_name = validate_person_name(name)
        except ValueError as e:
            return EnrollmentResult(
                success=False,
                person_id=None,
                person_name=name,
                embedding_id=None,
                is_new_person=False,
                total_embeddings_for_person=0,
                quality=None,
                detection=None,
                saved_image_path=None,
                message=f"Invalid name: {e}",
            )

        # 2. Load and decode image
        try:
            img_bgr = load_image(image)
        except Exception as e:
            return EnrollmentResult(
                success=False,
                person_id=None,
                person_name=clean_name,
                embedding_id=None,
                is_new_person=False,
                total_embeddings_for_person=0,
                quality=None,
                detection=None,
                saved_image_path=None,
                message=f"Unable to read image: {e}",
            )

        # 3. Detect faces and enforce Single-Face Policy
        detections = self.detector.detect(img_bgr)
        if len(detections) == 0:
            logger.warning("Enrollment rejected for '%s': 0 faces detected.", clean_name)
            return EnrollmentResult(
                success=False,
                person_id=None,
                person_name=clean_name,
                embedding_id=None,
                is_new_person=False,
                total_embeddings_for_person=0,
                quality=None,
                detection=None,
                saved_image_path=None,
                message="No face detected in the image. Please provide an image containing a clear face.",
            )

        if len(detections) > 1:
            logger.warning(
                "Enrollment rejected for '%s': %d faces detected.",
                clean_name,
                len(detections),
            )
            return EnrollmentResult(
                success=False,
                person_id=None,
                person_name=clean_name,
                embedding_id=None,
                is_new_person=False,
                total_embeddings_for_person=0,
                quality=None,
                detection=None,
                saved_image_path=None,
                message=f"Multiple faces ({len(detections)}) detected. Please provide an image containing exactly one person.",
            )

        target_detection = detections[0]

        # 4. Perform image and face quality checks
        face_crop = crop_face(img_bgr, target_detection.bbox, margin=0.1)
        quality = check_image_quality(
            face_crop,
            min_face_size=config.min_face_size,
            min_laplacian_var=config.min_laplacian_variance,
        )

        if not quality.is_valid and not skip_quality_check:
            issue_str = "; ".join(quality.issues)
            logger.warning("Enrollment quality check failed for '%s': %s", clean_name, issue_str)
            return EnrollmentResult(
                success=False,
                person_id=None,
                person_name=clean_name,
                embedding_id=None,
                is_new_person=False,
                total_embeddings_for_person=0,
                quality=quality,
                detection=target_detection,
                saved_image_path=None,
                message=f"Face detected, but image quality is insufficient: {issue_str}",
            )

        # 5. Extract 128-d L2-normalized embedding
        try:
            embedding = self.embedder.generate(img_bgr, detection=target_detection)
        except Exception as e:
            logger.error("Embedding generation failed during enrollment for '%s': %s", clean_name, e)
            return EnrollmentResult(
                success=False,
                person_id=None,
                person_name=clean_name,
                embedding_id=None,
                is_new_person=False,
                total_embeddings_for_person=0,
                quality=quality,
                detection=target_detection,
                saved_image_path=None,
                message=f"Biometric embedding generation failed: {e}",
            )

        # 6. Check if person already exists (Multiple Reference Embeddings support)
        existing = self.db.get_person_by_name(clean_name)
        is_new = existing is None
        person_id = self.db.add_person(clean_name)

        # 7. Optionally save reference image for audit/visualization
        saved_img_path = None
        source_name = None
        if allow_save_image:
            person_folder = config.enrolled_dir / clean_name
            person_folder.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"ref_{timestamp}.jpg"
            full_path = person_folder / file_name
            cv2.imwrite(str(full_path), img_bgr)
            saved_img_path = str(full_path)
            source_name = file_name

        # 8. Persist embedding in database
        emb_id = self.db.add_embedding(
            person_id=person_id,
            embedding=embedding,
            source_image_name=source_name,
        )

        # Count total embeddings for this person
        persons_list = self.db.get_persons_with_counts()
        total_for_person = 1
        for p in persons_list:
            if p["id"] == person_id:
                total_for_person = p["embedding_count"]
                break

        action_word = "Enrolled new identity" if is_new else "Added additional reference embedding for"
        msg = f"{action_word} '{clean_name}'. Total stored reference embeddings: {total_for_person}."
        logger.info(msg)

        return EnrollmentResult(
            success=True,
            person_id=person_id,
            person_name=clean_name,
            embedding_id=emb_id,
            is_new_person=is_new,
            total_embeddings_for_person=total_for_person,
            quality=quality,
            detection=target_detection,
            saved_image_path=saved_img_path,
            message=msg,
        )

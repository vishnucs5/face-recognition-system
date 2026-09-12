"""High-level face recognition service coordinating detection, embedding, matching, and visualization."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from app.config import config
from app.database import DatabaseManager
from app.detection.detector import FaceDetection, FaceDetector
from app.recognition.embedder import FaceEmbedder
from app.recognition.matcher import CandidateMatch, FaceMatcher, MatchResult
from app.utils.image_utils import draw_detection_annotations, load_image

logger = logging.getLogger(__name__)


@dataclass
class IdentificationResult:
    """Detailed identification result for an individual detected face."""

    face_index: int
    bbox: Tuple[int, int, int, int]
    detection_confidence: float
    landmarks: np.ndarray
    identity: Optional[str]
    person_id: Optional[int]
    similarity: float
    accepted: bool
    status: str  # "MATCH", "UNKNOWN", or "NO_ENROLLED_FACES"
    threshold_used: float
    top_candidates: List[CandidateMatch] = field(default_factory=list)


@dataclass
class IdentificationOutput:
    """Overall multi-face identification response for an image."""

    num_faces_detected: int
    num_matches: int
    num_unknowns: int
    results: List[IdentificationResult]
    annotated_image: np.ndarray  # BGR
    message: str


class FaceRecognitionService:
    """Service facade for the full face recognition and identification pipeline."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        detector: Optional[FaceDetector] = None,
        embedder: Optional[FaceEmbedder] = None,
        matcher: Optional[FaceMatcher] = None,
    ):
        self.db = db_manager if db_manager is not None else DatabaseManager()
        self.detector = detector if detector is not None else FaceDetector()
        self.embedder = embedder if embedder is not None else FaceEmbedder()
        self.matcher = matcher if matcher is not None else FaceMatcher()

    @property
    def device_info(self) -> str:
        """Return human-readable device configuration."""
        if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            return "CUDA GPU"
        return "CPU"

    def identify(
        self,
        image: Union[str, Path, bytes, np.ndarray],
        threshold: Optional[float] = None,
    ) -> IdentificationOutput:
        """Perform end-to-end multi-face detection, embedding extraction, matching, and unknown rejection.

        Args:
            image: Query image source.
            threshold: Optional override for matching similarity threshold.

        Returns:
            IdentificationOutput: Complete results breakdown and annotated visualization.
        """
        img_bgr = load_image(image)
        thresh = threshold if threshold is not None else self.matcher.match_threshold

        # Step 1: Detect all faces in query image
        detections = self.detector.detect(img_bgr)
        if not detections:
            logger.info("Identification completed: 0 faces found.")
            return IdentificationOutput(
                num_faces_detected=0,
                num_matches=0,
                num_unknowns=0,
                results=[],
                annotated_image=img_bgr.copy(),
                message="No face detected in the image. Please provide an image containing a clear face.",
            )

        # Step 2: Fetch enrolled embeddings matrix from database
        db_matrix, person_ids, person_names = self.db.get_all_embeddings_matrix()

        if len(db_matrix) == 0:
            # Enrolled database is empty
            results: List[IdentificationResult] = []
            for idx, det in enumerate(detections):
                results.append(
                    IdentificationResult(
                        face_index=idx,
                        bbox=det.bbox,
                        detection_confidence=det.confidence,
                        landmarks=det.landmarks,
                        identity=None,
                        person_id=None,
                        similarity=0.0,
                        accepted=False,
                        status="NO_ENROLLED_FACES",
                        threshold_used=thresh,
                        top_candidates=[],
                    )
                )
            annotated = draw_detection_annotations(img_bgr, results)
            return IdentificationOutput(
                num_faces_detected=len(detections),
                num_matches=0,
                num_unknowns=len(detections),
                results=results,
                annotated_image=annotated,
                message="No enrolled identities are available in the database. Please enroll faces first.",
            )

        # Step 3: For each detected face, extract embedding and match
        results = []
        matches_count = 0
        unknowns_count = 0

        for idx, det in enumerate(detections):
            # Generate aligned 128-d L2-normalized embedding
            emb = self.embedder.generate(img_bgr, detection=det)

            # Cosine similarity match against database
            match_res = self.matcher.match(
                query_embedding=emb,
                database_matrix=db_matrix,
                person_ids=person_ids,
                person_names=person_names,
                custom_threshold=thresh,
            )

            if match_res.accepted:
                matches_count += 1
            else:
                unknowns_count += 1

            results.append(
                IdentificationResult(
                    face_index=idx,
                    bbox=det.bbox,
                    detection_confidence=det.confidence,
                    landmarks=det.landmarks,
                    identity=match_res.person_name,
                    person_id=match_res.person_id,
                    similarity=match_res.similarity,
                    accepted=match_res.accepted,
                    status=match_res.status,
                    threshold_used=thresh,
                    top_candidates=match_res.top_candidates,
                )
            )

        # Step 4: Render bounding boxes and status labels
        annotated = draw_detection_annotations(img_bgr, results)

        summary_msg = (
            f"Processed {len(detections)} face(s): {matches_count} matched, {unknowns_count} unknown "
            f"(threshold: {thresh:.2f})."
        )
        logger.info(summary_msg)

        return IdentificationOutput(
            num_faces_detected=len(detections),
            num_matches=matches_count,
            num_unknowns=unknowns_count,
            results=results,
            annotated_image=annotated,
            message=summary_msg,
        )

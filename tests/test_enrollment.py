"""Tests for EnrollmentService and SQLite persistence."""

import cv2
import numpy as np
import pytest
from app.database import DatabaseManager
from app.detection.detector import FaceDetection
from app.enrollment.service import EnrollmentService


class MockDetector:
    """Mock detector returning configurable face detections."""

    def __init__(self, detections_to_return=None):
        self.detections = detections_to_return or []

    def detect(self, img):
        return self.detections


class MockEmbedder:
    """Mock embedder returning deterministic 128-d unit vectors."""

    def generate(self, img, detection=None):
        vec = np.ones(128, dtype=np.float32)
        return vec / np.linalg.norm(vec)


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_faces.db"
    return DatabaseManager(db_path=db_file)


def test_enrollment_invalid_name(temp_db):
    """Enrollment must reject empty or invalid names."""
    service = EnrollmentService(db_manager=temp_db)
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    res = service.enroll("", dummy_img)
    assert not res.success
    assert "Invalid name" in res.message

    res2 = service.enroll("   ", dummy_img)
    assert not res2.success


def test_enrollment_no_face_detected(temp_db):
    """Enrollment must be rejected if 0 faces are found."""
    mock_det = MockDetector(detections_to_return=[])
    service = EnrollmentService(
        db_manager=temp_db, detector=mock_det, embedder=MockEmbedder()
    )

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = service.enroll("Alice", img)

    assert not res.success
    assert "No face detected" in res.message


def test_enrollment_multiple_faces_rejected(temp_db):
    """Single-face policy: Enrollment must reject images with > 1 faces."""
    f1 = FaceDetection((10, 10, 50, 50), 0.9, np.zeros((5, 2)), np.zeros(15))
    f2 = FaceDetection((60, 60, 100, 100), 0.92, np.zeros((5, 2)), np.zeros(15))
    mock_det = MockDetector(detections_to_return=[f1, f2])

    service = EnrollmentService(
        db_manager=temp_db, detector=mock_det, embedder=MockEmbedder()
    )
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    res = service.enroll("Alice", img)

    assert not res.success
    assert "Multiple faces" in res.message


def test_enrollment_success_and_multi_embedding(temp_db):
    """Verify single-person enrollment and adding multiple reference embeddings for same identity."""
    f1 = FaceDetection((10, 10, 100, 100), 0.95, np.zeros((5, 2)), np.zeros(15))
    mock_det = MockDetector(detections_to_return=[f1])
    service = EnrollmentService(
        db_manager=temp_db, detector=mock_det, embedder=MockEmbedder()
    )

    img = np.ones((200, 200, 3), dtype=np.uint8) * 120

    # 1. First enrollment: new person
    res1 = service.enroll("Bob", img, allow_save_image=False, skip_quality_check=True)
    assert res1.success
    assert res1.is_new_person is True
    assert res1.total_embeddings_for_person == 1
    assert temp_db.count_identities() == 1
    assert temp_db.count_embeddings() == 1

    # 2. Second enrollment: existing person adds another reference embedding
    res2 = service.enroll("Bob", img, allow_save_image=False, skip_quality_check=True)
    assert res2.success
    assert res2.is_new_person is False
    assert res2.total_embeddings_for_person == 2
    assert temp_db.count_identities() == 1
    assert temp_db.count_embeddings() == 2


def test_delete_person_cascades_embeddings(temp_db):
    """Deleting a person must remove both the person and all their embeddings."""
    pid = temp_db.add_person("Charlie")
    emb = np.random.randn(128).astype(np.float32)
    temp_db.add_embedding(pid, emb)
    temp_db.add_embedding(pid, emb)

    assert temp_db.count_identities() == 1
    assert temp_db.count_embeddings() == 2

    temp_db.delete_person(pid)
    assert temp_db.count_identities() == 0
    assert temp_db.count_embeddings() == 0


def test_enrollment_webcam_source_and_audit(temp_db):
    """Verify enrollment with webcam source persists 'webcam' metadata in database."""
    f1 = FaceDetection((10, 10, 100, 100), 0.95, np.zeros((5, 2)), np.zeros(15))
    mock_det = MockDetector(detections_to_return=[f1])
    service = EnrollmentService(
        db_manager=temp_db, detector=mock_det, embedder=MockEmbedder()
    )
    img = np.ones((200, 200, 3), dtype=np.uint8) * 150

    # 1. Enroll via webcam
    res_cam = service.enroll(
        "Diana", img, allow_save_image=False, skip_quality_check=True, source_type="webcam"
    )
    assert res_cam.success
    assert res_cam.source_type == "webcam"

    # 2. Add second reference via upload
    res_up = service.enroll(
        "Diana", img, allow_save_image=False, skip_quality_check=True, source_type="upload"
    )
    assert res_up.success
    assert res_up.source_type == "upload"

    # 3. Audit stored embeddings records
    embeddings = temp_db.get_embeddings_for_person(res_cam.person_id)
    assert len(embeddings) == 2
    sources = [e["source"] for e in embeddings]
    assert "webcam" in sources
    assert "upload" in sources


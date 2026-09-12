"""End-to-end tests for the complete identification pipeline."""

import cv2
import numpy as np
import pytest
from app.database import DatabaseManager
from app.detection.detector import FaceDetection
from app.recognition.matcher import FaceMatcher
from app.recognition.recognizer import FaceRecognitionService


class MockDetectorMulti:
    """Mock detector returning configurable faces."""

    def __init__(self, faces):
        self.faces = faces

    def detect(self, img):
        return self.faces


class ControlledEmbedder:
    """Embedder returning controlled vectors for reproducible end-to-end verification."""

    def __init__(self, vector_map=None):
        self.vector_map = vector_map or {}
        self._default = np.zeros(128, dtype=np.float32)
        self._default[0] = 1.0

    def generate(self, img, detection=None):
        idx = getattr(detection, "raw_id", 0) if detection else 0
        return self.vector_map.get(idx, self._default)


@pytest.fixture
def clean_db(tmp_path):
    db_file = tmp_path / "e2e_faces.db"
    return DatabaseManager(db_path=db_file)


def test_end_to_end_known_and_unknown(clean_db):
    """Full pipeline: Enroll Alice, query with Alice (MATCH) and query with Unknown (REJECT)."""
    # 1. Enroll Alice with vector e_alice
    e_alice = np.zeros(128, dtype=np.float32)
    e_alice[0] = 1.0  # Unit vector along dimension 0

    p_id = clean_db.add_person("Alice")
    clean_db.add_embedding(p_id, e_alice)

    # 2. Setup query for Alice: identical vector
    alice_query = np.zeros(128, dtype=np.float32)
    alice_query[0] = 0.95
    alice_query[1] = np.sqrt(1 - 0.95**2)

    # 3. Setup query for Unknown person: orthogonal vector
    unk_query = np.zeros(128, dtype=np.float32)
    unk_query[5] = 1.0  # Orthogonal to dimension 0

    embedder = ControlledEmbedder(
        vector_map={
            1: alice_query,
            2: unk_query,
        }
    )

    f1 = FaceDetection((10, 10, 80, 80), 0.98, np.zeros((5, 2)), np.zeros(15))
    setattr(f1, "raw_id", 1)  # Maps to Alice query
    f2 = FaceDetection((100, 10, 170, 80), 0.96, np.zeros((5, 2)), np.zeros(15))
    setattr(f2, "raw_id", 2)  # Maps to Unknown query

    detector = MockDetectorMulti([f1, f2])
    matcher = FaceMatcher(match_threshold=0.50)

    service = FaceRecognitionService(
        db_manager=clean_db,
        detector=detector,
        embedder=embedder,
        matcher=matcher,
    )

    test_img = np.zeros((200, 300, 3), dtype=np.uint8)
    output = service.identify(test_img, threshold=0.50)

    assert output.num_faces_detected == 2
    assert output.num_matches == 1
    assert output.num_unknowns == 1

    # Check Face #1 (Alice)
    res_alice = output.results[0]
    assert res_alice.accepted is True
    assert res_alice.status == "MATCH"
    assert res_alice.identity == "Alice"
    assert res_alice.similarity == pytest.approx(0.95, abs=1e-3)

    # Check Face #2 (Unknown)
    res_unk = output.results[1]
    assert res_unk.accepted is False
    assert res_unk.status == "UNKNOWN"
    assert res_unk.identity is None
    assert res_unk.similarity == pytest.approx(0.0, abs=1e-3)

    # Verify annotated image generated with correct dimensions
    assert output.annotated_image.shape == test_img.shape


def test_threshold_modification_behavior(clean_db):
    """Test that adjusting the threshold directly changes the acceptance decision."""
    e_enrolled = np.zeros(128, dtype=np.float32)
    e_enrolled[0] = 1.0
    pid = clean_db.add_person("David")
    clean_db.add_embedding(pid, e_enrolled)

    # Query with similarity 0.60
    query_vec = np.zeros(128, dtype=np.float32)
    query_vec[0] = 0.60
    query_vec[1] = np.sqrt(1 - 0.60**2)

    embedder = ControlledEmbedder(vector_map={1: query_vec})
    f1 = FaceDetection((10, 10, 60, 60), 0.99, np.zeros((5, 2)), np.zeros(15))
    setattr(f1, "raw_id", 1)

    service = FaceRecognitionService(
        db_manager=clean_db,
        detector=MockDetectorMulti([f1]),
        embedder=embedder,
        matcher=FaceMatcher(match_threshold=0.50),
    )

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # At threshold 0.50: 0.60 >= 0.50 -> MATCH
    out1 = service.identify(dummy_img, threshold=0.50)
    assert out1.results[0].accepted is True
    assert out1.results[0].status == "MATCH"

    # At threshold 0.75: 0.60 < 0.75 -> UNKNOWN
    out2 = service.identify(dummy_img, threshold=0.75)
    assert out2.results[0].accepted is False
    assert out2.results[0].status == "UNKNOWN"

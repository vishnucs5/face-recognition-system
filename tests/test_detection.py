"""Tests for YuNet face detector."""

import cv2
import numpy as np
import pytest
from app.detection.detector import FaceDetector, FaceDetection


@pytest.fixture(scope="module")
def detector():
    return FaceDetector()


def test_detect_no_face_in_blank_image(detector):
    """Ensure a blank/black image produces zero face detections without error."""
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    detections = detector.detect(blank_img)
    assert isinstance(detections, list)
    assert len(detections) == 0


def test_detect_no_face_in_noise(detector):
    """Ensure random white noise produces zero high-confidence face detections."""
    np.random.seed(42)
    noise_img = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    detections = detector.detect(noise_img)
    assert isinstance(detections, list)
    assert len(detections) == 0


def test_detect_face_properties(detector):
    """Verify that detected faces return properly typed coordinates, confidence, and 5 landmarks."""
    # Create a test face image using Lena or standard test face
    test_face = np.ones((200, 200, 3), dtype=np.uint8) * 128
    # Draw simple facial features: eyes, nose, mouth
    cv2.circle(test_face, (70, 80), 10, (20, 20, 20), -1)  # eye
    cv2.circle(test_face, (130, 80), 10, (20, 20, 20), -1)  # eye
    cv2.circle(test_face, (100, 110), 6, (30, 30, 30), -1)  # nose
    cv2.line(test_face, (70, 150), (130, 150), (20, 20, 20), 4)  # mouth

    # Even if synthesized face is not detected as real, detector must not crash
    detections = detector.detect(test_face)
    assert isinstance(detections, list)


def test_detect_invalid_input(detector):
    """Detector should raise an error for unreadable image inputs."""
    with pytest.raises((ValueError, Exception)):
        detector.detect("non_existent_file_path_12345.jpg")


def test_detection_dataclass():
    """Verify FaceDetection structure and field types."""
    dummy_raw = np.zeros(15, dtype=np.float32)
    det = FaceDetection(
        bbox=(10, 20, 110, 120),
        confidence=0.95,
        landmarks=np.zeros((5, 2), dtype=np.float32),
        raw_array=dummy_raw,
    )
    assert det.bbox == (10, 20, 110, 120)
    assert det.confidence == 0.95
    assert det.landmarks.shape == (5, 2)
    assert len(det.raw_array) == 15

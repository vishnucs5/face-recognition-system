"""Validation logic for biometric inputs, face quality, and identity naming."""

import re
from dataclasses import dataclass, field
from typing import List, Tuple
import cv2
import numpy as np


@dataclass
class QualityAssessment:
    """Biometric image quality assessment report."""

    is_valid: bool
    face_width: int
    face_height: int
    blur_score: float
    brightness: float
    issues: List[str] = field(default_factory=list)


def validate_person_name(name: str) -> str:
    """Validate and sanitize a person's display name for enrollment.

    Args:
        name: Raw input name string.

    Returns:
        str: Sanitized name.

    Raises:
        ValueError: If name is empty, too short, too long, or contains disallowed characters.
    """
    if not isinstance(name, str):
        raise ValueError("Identity name must be a text string.")

    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Identity name cannot be empty or only whitespace.")

    if len(cleaned) < 2:
        raise ValueError("Identity name must be at least 2 characters long.")

    if len(cleaned) > 64:
        raise ValueError("Identity name cannot exceed 64 characters.")

    # Allow letters, numbers, spaces, hyphens, underscores, dots, and apostrophes
    if not re.match(r"^[\w\s\.\-']+$", cleaned, re.UNICODE):
        raise ValueError(
            "Identity name contains invalid characters. Use letters, numbers, spaces, dots, or hyphens."
        )

    return cleaned


def check_image_quality(
    face_img_bgr: np.ndarray,
    min_face_size: int = 60,
    min_laplacian_var: float = 25.0,
    min_brightness: float = 20.0,
    max_brightness: float = 245.0,
) -> QualityAssessment:
    """Evaluate image quality of cropped face to ensure reliable embedding extraction.

    Args:
        face_img_bgr: Cropped face BGR image.
        min_face_size: Minimum width and height required.
        min_laplacian_var: Minimum variance of Laplacian for blur detection.
        min_brightness: Minimum average grayscale intensity.
        max_brightness: Maximum average grayscale intensity.

    Returns:
        QualityAssessment: Assessment details and validation status.
    """
    issues: List[str] = []

    if face_img_bgr is None or face_img_bgr.size == 0:
        return QualityAssessment(
            is_valid=False,
            face_width=0,
            face_height=0,
            blur_score=0.0,
            brightness=0.0,
            issues=["Face crop is empty or corrupted."],
        )

    h, w = face_img_bgr.shape[:2]

    # 1. Dimension check
    if w < min_face_size or h < min_face_size:
        issues.append(
            f"Face resolution too small ({w}x{h} px). Minimum required is {min_face_size}x{min_face_size} px."
        )

    # Convert to grayscale for blur and brightness analysis
    gray = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Blur assessment using variance of the Laplacian
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < min_laplacian_var:
        issues.append(
            f"Image appears blurry or out of focus (sharpness score {lap_var:.1f} < {min_laplacian_var:.1f})."
        )

    # 3. Brightness / exposure assessment
    mean_val = float(np.mean(gray))
    if mean_val < min_brightness:
        issues.append(
            f"Face image is underexposed/too dark (mean brightness {mean_val:.1f} < {min_brightness:.1f})."
        )
    elif mean_val > max_brightness:
        issues.append(
            f"Face image is overexposed/washed out (mean brightness {mean_val:.1f} > {max_brightness:.1f})."
        )

    is_valid = len(issues) == 0
    return QualityAssessment(
        is_valid=is_valid,
        face_width=w,
        face_height=h,
        blur_score=lap_var,
        brightness=mean_val,
        issues=issues,
    )

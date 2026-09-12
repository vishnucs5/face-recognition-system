"""Image manipulation and rendering utilities."""

from io import BytesIO
from pathlib import Path
from typing import Tuple, Union, Optional
import cv2
import numpy as np
from PIL import Image


def load_image(source: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """Load an image from various source formats into a standard BGR numpy array.

    Args:
        source: File path, bytes, numpy array, or PIL Image.

    Returns:
        np.ndarray: BGR image array with shape (H, W, 3).

    Raises:
        ValueError: If image cannot be read, decoded, or has invalid dimensions.
    """
    if isinstance(source, (str, Path)):
        path_str = str(source)
        if not Path(path_str).exists():
            raise ValueError(f"Image file does not exist: {path_str}")
        # Use imdecode to properly handle UTF-8 / non-ASCII paths on Windows
        try:
            with open(path_str, "rb") as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception as e:
            raise ValueError(f"Failed to read image from {path_str}: {e}")
        if img is None:
            raise ValueError(f"Unable to decode image file: {path_str}")
        return img

    elif isinstance(source, bytes):
        file_bytes = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Unable to decode image bytes into a valid image.")
        return img

    elif isinstance(source, Image.Image):
        # PIL loads RGB; convert to BGR
        rgb_array = np.array(source.convert("RGB"))
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    elif isinstance(source, np.ndarray):
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        elif source.ndim == 3:
            if source.shape[2] == 4:
                return cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
            elif source.shape[2] == 3:
                return source.copy()
        raise ValueError(f"Unsupported numpy array image shape: {source.shape}")

    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR OpenCV image to RGB."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def resize_with_aspect_ratio(
    img: np.ndarray, max_dimension: int = 1920
) -> Tuple[np.ndarray, float]:
    """Resize image so its longest side does not exceed max_dimension, preserving aspect ratio.

    Returns:
        Tuple[np.ndarray, float]: Resized image and scale factor.
    """
    h, w = img.shape[:2]
    longest_side = max(h, w)
    if longest_side <= max_dimension:
        return img, 1.0

    scale = max_dimension / float(longest_side)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def crop_face(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.2,
) -> np.ndarray:
    """Crop face from image using bounding box with optional proportional padding.

    Args:
        img: Input image (BGR).
        bbox: (x1, y1, x2, y2) coordinates.
        margin: Fractional margin to add around the face.

    Returns:
        np.ndarray: Cropped face image.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    fw = x2 - x1
    fh = y2 - y1

    dx = int(fw * margin)
    dy = int(fh * margin)

    cx1 = max(0, x1 - dx)
    cy1 = max(0, y1 - dy)
    cx2 = min(w, x2 + dx)
    cy2 = min(h, y2 + dy)

    cropped = img[cy1:cy2, cx1:cx2]
    if cropped.size == 0:
        return img[y1:y2, x1:x2]
    return cropped


def draw_detection_annotations(
    img_bgr: np.ndarray,
    results: list,
) -> np.ndarray:
    """Render bounding boxes and information tags on the image.

    Args:
        img_bgr: Base image in BGR format.
        results: List of identification results or detections.

    Returns:
        np.ndarray: Annotated BGR image.
    """
    annotated = img_bgr.copy()
    h_img, w_img = annotated.shape[:2]

    for res in results:
        # Check if result is IdentificationResult or FaceDetection
        bbox = getattr(res, "bbox", None)
        if bbox is None and isinstance(res, dict):
            bbox = res.get("bbox")
        if not bbox:
            continue

        x1, y1, x2, y2 = bbox
        x1 = max(0, min(w_img - 1, int(x1)))
        y1 = max(0, min(h_img - 1, int(y1)))
        x2 = max(0, min(w_img - 1, int(x2)))
        y2 = max(0, min(h_img - 1, int(y2)))

        is_match = getattr(res, "accepted", False)
        status = getattr(res, "status", "DETECTED")
        identity = getattr(res, "identity", None)
        similarity = getattr(res, "similarity", None)
        confidence = getattr(res, "detection_confidence", None)

        if is_match:
            # Bright Emerald Green for MATCH
            box_color = (0, 200, 0)
            status_text = f"{identity or 'KNOWN'}"
            detail_text = f"Sim: {similarity:.2f}" if similarity is not None else ""
        elif status == "UNKNOWN":
            # Vivid Amber/Red for UNKNOWN
            box_color = (30, 30, 220)
            status_text = "UNKNOWN"
            detail_text = (
                f"Sim: {similarity:.2f} (Rejected)" if similarity is not None else "Rejected"
            )
        else:
            # Cyan for plain detection
            box_color = (255, 200, 0)
            status_text = f"Face ({confidence:.2f})" if confidence else "Face"
            detail_text = ""

        # Draw bounding box rectangle
        thickness = max(2, int(min(h_img, w_img) / 300))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

        # Draw 5-point landmarks if available
        landmarks = getattr(res, "landmarks", None)
        if landmarks is not None and len(landmarks) == 5:
            landmark_colors = [
                (255, 0, 0),    # right eye (blue)
                (0, 0, 255),    # left eye (red)
                (0, 255, 0),    # nose tip (green)
                (255, 255, 0),  # right mouth (cyan)
                (0, 255, 255),  # left mouth (yellow)
            ]
            for idx, pt in enumerate(landmarks):
                px, py = int(pt[0]), int(pt[1])
                if 0 <= px < w_img and 0 <= py < h_img:
                    cv2.circle(annotated, (px, py), max(2, thickness), landmark_colors[idx], -1)

        # Prepare header banner
        label = status_text
        if detail_text:
            label += f" | {detail_text}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(h_img, w_img) / 1000.0)
        font_thickness = max(1, int(font_scale * 2))

        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        banner_h = th + baseline + 10
        banner_y1 = max(0, y1 - banner_h)
        banner_y2 = y1 if banner_y1 < y1 else y1 + banner_h

        # Draw filled banner background
        cv2.rectangle(
            annotated,
            (x1, banner_y1),
            (min(w_img, x1 + tw + 12), banner_y2),
            box_color,
            -1,
        )

        # Draw text in contrasting white or black
        text_color = (255, 255, 255)
        text_y = banner_y1 + th + 4
        cv2.putText(
            annotated,
            label,
            (x1 + 6, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    return annotated

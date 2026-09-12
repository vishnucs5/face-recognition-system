"""Biometric acquisition container and input normalization for Upload and Webcam frames."""

import datetime
from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np


@dataclass
class RecognitionInput:
    """Normalized biometric acquisition container for both Upload and Webcam sources."""

    image: Union[bytes, np.ndarray, str]
    source: str = "upload"  # "upload" or "webcam"
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    filename: Optional[str] = None
    quality_score: Optional[float] = None

    @property
    def is_webcam(self) -> bool:
        """Check if source is a live webcam frame."""
        return str(self.source).lower() == "webcam"

    @property
    def is_upload(self) -> bool:
        """Check if source is an uploaded image file."""
        return str(self.source).lower() == "upload"

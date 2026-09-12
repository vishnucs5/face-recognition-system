"""Face recognition package containing embedder, matcher, and service."""

from app.recognition.embedder import FaceEmbedder
from app.recognition.matcher import FaceMatcher, MatchResult, CandidateMatch
from app.recognition.recognizer import (
    FaceRecognitionService,
    IdentificationResult,
    IdentificationOutput,
)

__all__ = [
    "FaceEmbedder",
    "FaceMatcher",
    "MatchResult",
    "CandidateMatch",
    "FaceRecognitionService",
    "IdentificationResult",
    "IdentificationOutput",
]

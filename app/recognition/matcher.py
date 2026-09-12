"""Similarity matching and unknown rejection logic."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np

from app.config import config

logger = logging.getLogger(__name__)


@dataclass
class CandidateMatch:
    """Individual candidate score summary."""

    person_id: int
    person_name: str
    similarity: float


@dataclass
class MatchResult:
    """Decision output for a single queried face embedding against enrolled database."""

    person_id: Optional[int]
    person_name: Optional[str]
    similarity: float
    accepted: bool
    status: Literal["MATCH", "UNKNOWN", "NO_ENROLLED_FACES"]
    threshold_used: float
    top_candidates: List[CandidateMatch] = field(default_factory=list)


class FaceMatcher:
    """Performs cosine similarity calculation, multi-embedding aggregation, and unknown rejection."""

    def __init__(
        self,
        match_threshold: Optional[float] = None,
        strategy: Optional[Literal["max", "mean"]] = None,
    ):
        self.match_threshold = (
            match_threshold if match_threshold is not None else config.match_threshold
        )
        self.strategy = strategy if strategy is not None else config.matching_strategy

    @staticmethod
    def compute_cosine_similarity(
        query: np.ndarray,
        database_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between 1D query vector and 2D database matrix.

        Args:
            query: (D,) float32 vector.
            database_matrix: (N, D) float32 matrix.

        Returns:
            np.ndarray: (N,) similarities in range [-1.0, 1.0].
        """
        if database_matrix.size == 0:
            return np.empty((0,), dtype=np.float32)

        q_norm = np.linalg.norm(query)
        if q_norm < 1e-7:
            return np.zeros((database_matrix.shape[0],), dtype=np.float32)

        # Normalize query vector
        q = query / q_norm

        # Norm of each database row
        db_norms = np.linalg.norm(database_matrix, axis=1)
        db_norms = np.where(db_norms < 1e-7, 1.0, db_norms)

        # Vectorized dot product divided by norms
        similarities = np.dot(database_matrix, q) / db_norms
        return np.clip(similarities, -1.0, 1.0)

    def match(
        self,
        query_embedding: np.ndarray,
        database_matrix: np.ndarray,
        person_ids: List[int],
        person_names: Dict[int, str],
        custom_threshold: Optional[float] = None,
    ) -> MatchResult:
        """Compare a query embedding against all enrolled embeddings and apply rejection threshold.

        Args:
            query_embedding: (128,) float32 query vector.
            database_matrix: (N, 128) matrix of all stored embeddings.
            person_ids: (N,) person IDs matching rows of database_matrix.
            person_names: Mapping of person_id -> display name.
            custom_threshold: Optional threshold override.

        Returns:
            MatchResult: Structured biometric decision.
        """
        threshold = (
            custom_threshold if custom_threshold is not None else self.match_threshold
        )

        # Case 1: Empty database
        if database_matrix is None or len(database_matrix) == 0:
            return MatchResult(
                person_id=None,
                person_name=None,
                similarity=0.0,
                accepted=False,
                status="NO_ENROLLED_FACES",
                threshold_used=threshold,
                top_candidates=[],
            )

        # Calculate cosine similarity against all stored embeddings
        sims = self.compute_cosine_similarity(query_embedding, database_matrix)

        # Group similarities by person_id according to configured strategy
        person_scores: Dict[int, List[float]] = {}
        for p_id, sim in zip(person_ids, sims):
            if p_id not in person_scores:
                person_scores[p_id] = []
            person_scores[p_id].append(float(sim))

        # Aggregate scores per person
        aggregated: List[CandidateMatch] = []
        for p_id, score_list in person_scores.items():
            if self.strategy == "mean":
                agg_score = float(np.mean(score_list))
            else:  # 'max' strategy (default)
                agg_score = float(np.max(score_list))

            name = person_names.get(p_id, f"Person-{p_id}")
            aggregated.append(
                CandidateMatch(person_id=p_id, person_name=name, similarity=agg_score)
            )

        # Sort candidates by similarity descending
        aggregated.sort(key=lambda c: c.similarity, reverse=True)

        best_candidate = aggregated[0]
        top_k = aggregated[: config.top_k_candidates]

        # UNKNOWN REJECTION LOGIC
        # A face is NOT forced to match merely because it is the nearest neighbor.
        if best_candidate.similarity >= threshold:
            logger.info(
                "Known identity verified: '%s' (similarity=%.3f >= threshold=%.3f)",
                best_candidate.person_name,
                best_candidate.similarity,
                threshold,
            )
            return MatchResult(
                person_id=best_candidate.person_id,
                person_name=best_candidate.person_name,
                similarity=best_candidate.similarity,
                accepted=True,
                status="MATCH",
                threshold_used=threshold,
                top_candidates=top_k,
            )
        else:
            logger.info(
                "Unknown face rejected: closest was '%s' (similarity=%.3f < threshold=%.3f)",
                best_candidate.person_name,
                best_candidate.similarity,
                threshold,
            )
            return MatchResult(
                person_id=None,
                person_name=None,
                similarity=best_candidate.similarity,
                accepted=False,
                status="UNKNOWN",
                threshold_used=threshold,
                top_candidates=top_k,
            )

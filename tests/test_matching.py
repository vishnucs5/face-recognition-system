"""Tests for cosine similarity matching and unknown rejection mechanism."""

import numpy as np
import pytest
from app.recognition.matcher import FaceMatcher


@pytest.fixture
def matcher():
    return FaceMatcher(match_threshold=0.50, strategy="max")


def test_cosine_similarity_identical_vectors():
    """Identical unit vectors must yield a cosine similarity of 1.0."""
    v1 = np.random.randn(128).astype(np.float32)
    v1 /= np.linalg.norm(v1)

    db_matrix = np.array([v1])
    sim = FaceMatcher.compute_cosine_similarity(v1, db_matrix)
    assert sim[0] == pytest.approx(1.0, rel=1e-4)


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal vectors must yield a cosine similarity of 0.0."""
    v1 = np.zeros(128, dtype=np.float32)
    v2 = np.zeros(128, dtype=np.float32)
    v1[0] = 1.0
    v2[1] = 1.0

    db_matrix = np.array([v2])
    sim = FaceMatcher.compute_cosine_similarity(v1, db_matrix)
    assert sim[0] == pytest.approx(0.0, abs=1e-5)


def test_unknown_rejection_when_below_threshold(matcher):
    """When best similarity is below threshold, status must be UNKNOWN, accepted must be False."""
    query = np.zeros(128, dtype=np.float32)
    query[0] = 1.0

    # Enrolled embedding has low similarity (e.g., 0.2)
    enrolled = np.zeros(128, dtype=np.float32)
    enrolled[0] = 0.2
    enrolled[1] = np.sqrt(1 - 0.2**2)  # Unit norm
    db_matrix = np.array([enrolled])

    res = matcher.match(
        query_embedding=query,
        database_matrix=db_matrix,
        person_ids=[1],
        person_names={1: "Alice"},
        custom_threshold=0.50,
    )

    assert not res.accepted
    assert res.status == "UNKNOWN"
    assert res.person_id is None
    assert res.person_name is None
    assert res.similarity == pytest.approx(0.20, abs=1e-3)


def test_known_match_when_above_threshold(matcher):
    """When best similarity exceeds threshold, status must be MATCH and accepted must be True."""
    query = np.zeros(128, dtype=np.float32)
    query[0] = 1.0

    # Enrolled embedding has high similarity (0.85)
    enrolled = np.zeros(128, dtype=np.float32)
    enrolled[0] = 0.85
    enrolled[1] = np.sqrt(1 - 0.85**2)
    db_matrix = np.array([enrolled])

    res = matcher.match(
        query_embedding=query,
        database_matrix=db_matrix,
        person_ids=[1],
        person_names={1: "Alice"},
        custom_threshold=0.50,
    )

    assert res.accepted
    assert res.status == "MATCH"
    assert res.person_id == 1
    assert res.person_name == "Alice"
    assert res.similarity == pytest.approx(0.85, abs=1e-3)


def test_multi_embedding_max_vs_mean_strategy():
    """Verify max and mean aggregation across multiple reference embeddings of the same identity."""
    query = np.zeros(128, dtype=np.float32)
    query[0] = 1.0

    # Person 1 has two embeddings: one with sim=0.8, one with sim=0.4
    e1 = np.zeros(128, dtype=np.float32)
    e1[0] = 0.8
    e1[1] = np.sqrt(1 - 0.8**2)

    e2 = np.zeros(128, dtype=np.float32)
    e2[0] = 0.4
    e2[1] = np.sqrt(1 - 0.4**2)

    db_matrix = np.vstack([e1, e2])
    person_ids = [1, 1]
    person_names = {1: "Alice"}

    # Max strategy
    max_matcher = FaceMatcher(match_threshold=0.50, strategy="max")
    res_max = max_matcher.match(query, db_matrix, person_ids, person_names)
    assert res_max.similarity == pytest.approx(0.8, abs=1e-3)
    assert res_max.accepted is True

    # Mean strategy: mean is (0.8 + 0.4)/2 = 0.6
    mean_matcher = FaceMatcher(match_threshold=0.50, strategy="mean")
    res_mean = mean_matcher.match(query, db_matrix, person_ids, person_names)
    assert res_mean.similarity == pytest.approx(0.6, abs=1e-3)


def test_empty_database_rejection(matcher):
    """Querying against empty database must return NO_ENROLLED_FACES and accepted=False."""
    query = np.random.randn(128).astype(np.float32)
    empty_matrix = np.empty((0, 128), dtype=np.float32)

    res = matcher.match(query, empty_matrix, [], {})
    assert not res.accepted
    assert res.status == "NO_ENROLLED_FACES"
    assert res.similarity == 0.0

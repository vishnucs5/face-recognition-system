"""Tests for SFace embedding extractor."""

import numpy as np
import pytest
from app.config import config
from app.recognition.embedder import FaceEmbedder


@pytest.fixture(scope="module")
def embedder():
    return FaceEmbedder()


def test_embedding_shape_and_type(embedder):
    """Verify generated embedding has exactly 128 dimensions and float32 type."""
    sample_img = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
    emb = embedder.generate(sample_img)

    assert isinstance(emb, np.ndarray)
    assert emb.shape == (config.embedding_dimension,)
    assert emb.dtype == np.float32


def test_embedding_normalization(embedder):
    """Verify that embeddings are strictly L2-normalized (unit vector)."""
    sample_img = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
    emb = embedder.generate(sample_img)

    norm = float(np.linalg.norm(emb))
    assert norm == pytest.approx(1.0, rel=1e-4)


def test_embedding_no_nan_or_inf(embedder):
    """Verify generated vector contains zero NaN or Inf values."""
    sample_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    emb = embedder.generate(sample_img)

    assert np.all(np.isfinite(emb))


def test_embedding_deterministic(embedder):
    """Verify that identical input images produce identical embedding vectors."""
    sample_img = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
    emb1 = embedder.generate(sample_img)
    emb2 = embedder.generate(sample_img)

    np.testing.assert_allclose(emb1, emb2, rtol=1e-5)

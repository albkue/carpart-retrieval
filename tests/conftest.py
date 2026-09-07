import sys
import numpy as np
import pytest
from unittest.mock import MagicMock


def _ensure_mock_module(name):
    """Inject a mock module into sys.modules if not present."""
    if name not in sys.modules:
        sys.modules[name] = MagicMock()


class MockFAISSIndex:
    """Minimal mock FAISS index for unit tests."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.ntotal = 0
        self._vectors = []
        self.is_trained = False
        self.hnsw = MagicMock()
        self.hnsw.efConstruction = 40
        self.hnsw.efSearch = 16

    def add(self, vectors):
        self._vectors.append(vectors)
        self.ntotal += len(vectors)

    def search(self, query, k):
        # Return dummy distances and indices
        n_queries = len(query)
        distances = np.zeros((n_queries, k), dtype='float32')
        indices = np.arange(k, dtype='int64').reshape(1, k).repeat(n_queries, axis=0)
        return distances, indices

    def train(self, vectors):
        self.is_trained = True


class MockFAISS:
    """Minimal mock faiss module."""

    METRIC_L2 = 1
    METRIC_INNER_PRODUCT = 2

    class IndexIVFFlat:
        def __init__(self, *args, **kwargs):
            self.ntotal = 0
            self.is_trained = False

    @staticmethod
    def IndexFlatL2(dimension):
        return MockFAISSIndex(dimension)

    @staticmethod
    def IndexFlatIP(dimension):
        return MockFAISSIndex(dimension)

    @staticmethod
    def IndexHNSWFlat(dimension, m, metric):
        return MockFAISSIndex(dimension)

    @staticmethod
    def write_index(index, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(b'DUMMY_FAISS_INDEX')

    @staticmethod
    def read_index(path):
        idx = MockFAISSIndex(512)
        idx.ntotal = 3
        return idx


@pytest.fixture(autouse=True)
def mock_ml_models(monkeypatch):
    """Automatically mock heavy ML models for all tests unless marked as integration."""

    # Ensure mock modules exist so monkeypatch works even when libs aren't installed
    _ensure_mock_module("ultralytics")
    _ensure_mock_module("transformers")
    _ensure_mock_module("transformers.CLIPModel")
    _ensure_mock_module("transformers.CLIPProcessor")
    _ensure_mock_module("sentence_transformers")
    _ensure_mock_module("paddleocr")

    # Inject mock faiss module
    if "faiss" not in sys.modules:
        sys.modules["faiss"] = MockFAISS()

    # Mock torch.compile to avoid wrapping MagicMock into a function
    import torch
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model, raising=False)

    # Mock YOLO
    mock_yolo = MagicMock()
    mock_yolo.return_value = []
    monkeypatch.setattr("ultralytics.YOLO", lambda x: mock_yolo)

    # Mock CLIP
    mock_clip = MagicMock()
    # Ensure get_image_features returns a mock with proper .cpu().numpy() chain
    mock_outputs = MagicMock()
    mock_outputs.cpu.return_value.numpy.return_value = np.random.randn(1, 512).astype('float32')
    mock_clip.get_image_features.return_value = mock_outputs
    mock_clip.get_text_features.return_value = mock_outputs
    monkeypatch.setattr("transformers.CLIPModel.from_pretrained", lambda x: mock_clip)
    monkeypatch.setattr("transformers.CLIPProcessor.from_pretrained", lambda x: MagicMock())

    # Mock SentenceTransformers (BGE-M3)
    mock_st = MagicMock()
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", lambda *args, **kwargs: mock_st)

    # Mock PaddleOCR
    mock_paddle = MagicMock()
    monkeypatch.setattr("paddleocr.PaddleOCR", lambda *args, **kwargs: mock_paddle)

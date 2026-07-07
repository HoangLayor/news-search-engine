"""Test factory chọn backend + import-guard cho BGE-m3 và Milvus.

Các backend nặng (FlagEmbedding/sentence-transformers, pymilvus) KHÔNG cài trong
môi trường test -> phải fail-fast bằng RuntimeError rõ ràng, KHÔNG im lặng rơi
về local. Backend local phải luôn dùng được.
"""

from __future__ import annotations

import importlib.util

import pytest

from news_search.config import Settings
from news_search.index.embeddings import HashingEmbedder, get_embedder
from news_search.index.lexical import LexicalIndex, get_lexical_index
from news_search.index.vector import VectorIndex, get_vector_index


def _installed(module: str) -> bool:
    """Kiểm tra package có cài KHÔNG import nó (tránh nạp module nặng torch/pymilvus)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def test_get_embedder_hash_mac_dinh():
    emb = get_embedder(Settings(embedder="hash", embedding_dim=128))
    assert isinstance(emb, HashingEmbedder) and emb.dim == 128


def _bge_available() -> bool:
    """True nếu có ÍT NHẤT một backend BGE (BGEEmbedder dùng được cả hai)."""
    return _installed("FlagEmbedding") or _installed("sentence_transformers")


def test_get_embedder_bge_import_guard():
    """EMBEDDER=bge nhưng thiếu CẢ hai thư viện -> RuntimeError (fail-fast)."""
    if _bge_available():
        pytest.skip("Đã cài thư viện BGE — bỏ qua test import-guard")
    with pytest.raises(RuntimeError):
        get_embedder(Settings(embedder="bge"))


def test_get_embedder_gia_tri_la():
    with pytest.raises(ValueError):
        get_embedder(Settings(embedder="khong-ton-tai"))


def test_get_vector_index_local():
    idx = get_vector_index(Settings(vector_backend="local"), dim=64)
    assert isinstance(idx, VectorIndex) and idx.dim == 64


def test_get_vector_index_milvus_import_guard():
    """VECTOR_BACKEND=milvus nhưng thiếu pymilvus -> RuntimeError."""
    if _installed("pymilvus"):  # find_spec: không import module protobuf nặng
        pytest.skip("Đã cài pymilvus — bỏ qua test import-guard")
    with pytest.raises(RuntimeError):
        get_vector_index(Settings(vector_backend="milvus"), dim=1024)


def test_get_vector_index_gia_tri_la():
    with pytest.raises(ValueError):
        get_vector_index(Settings(vector_backend="khong-ton-tai"), dim=64)


def test_get_lexical_index_local():
    assert isinstance(get_lexical_index(Settings(lexical_backend="local")), LexicalIndex)


def test_get_lexical_index_opensearch_guard():
    """LEXICAL_BACKEND=opensearch thiếu opensearch-py -> RuntimeError."""
    try:
        import opensearchpy  # noqa: F401
        pytest.skip("Đã cài opensearch-py — bỏ qua test import-guard")
    except ImportError:
        pass
    with pytest.raises(RuntimeError):
        get_lexical_index(Settings(lexical_backend="opensearch"))


def test_get_lexical_index_gia_tri_la():
    with pytest.raises(ValueError):
        get_lexical_index(Settings(lexical_backend="khong-ton-tai"))

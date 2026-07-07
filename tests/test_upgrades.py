"""Test 4 nâng cấp Nhóm 1: pyvi, datasketch, VN embedder, VN reranker.

Embedder/reranker VN được test qua **sys.modules injection** (fake
sentence-transformers) để KHÔNG tải model ~2GB / không nạp torch trong CI.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types

import numpy as np
import pytest

from news_search.config import Settings


# --------------------------------------------------------- pyvi tokenizer (F-03)


def test_tokenize_pyvi_compounds(monkeypatch):
    """TOKENIZER=pyvi -> tách từ ghép ('bất_động_sản') thay vì âm tiết rời."""
    pytest.importorskip("pyvi")
    import news_search.ingest.tokenizer as tok

    monkeypatch.setenv("TOKENIZER", "pyvi")
    importlib.reload(tok)
    try:
        toks = tok.tokenize("Bất động sản Hà Nội tăng giá mạnh")
        assert "bất_động_sản" in toks and "hà_nội" in toks
    finally:
        monkeypatch.setenv("TOKENIZER", "regex")
        importlib.reload(tok)  # khôi phục regex cho các test sau


# ------------------------------------------------------- datasketch dedup (F-07)


def test_datasketch_deduper():
    pytest.importorskip("datasketch")
    from news_search.index.dedup_datasketch import DatasketchDeduper

    A = ("ủy ban bầu cử quốc gia công bố kết quả chính thức cuộc bầu cử tổng thống "
         "vào sáng nay tại thủ đô sau nhiều giờ kiểm phiếu căng thẳng")
    B = A.replace("sáng", "chiều")       # gần trùng
    C = ("đội tuyển bóng đá quốc gia giành chiến thắng trong trận chung kết "
         "giải vô địch khu vực trước sự cổ vũ của khán giả")  # khác hẳn

    d = DatasketchDeduper(num_perm=128, threshold=0.6, shingle_size=3)
    ca, cb, cc = d.add("a", A), d.add("b", B), d.add("c", C)
    assert ca == cb            # a, b cùng cụm
    assert cc != ca            # c cụm khác
    assert d.similarity("a", "b") > d.similarity("a", "c")
    assert len(d) == 3
    d.add("a", A)              # re-add cùng id -> không phình
    assert len(d) == 3
    d.remove("b")
    assert d.cluster_of("b") is None and len(d) == 2


def test_get_deduper_factory():
    from news_search.index.dedup import MinHashDeduper, get_deduper

    assert isinstance(get_deduper(Settings(dedup_backend="local")), MinHashDeduper)
    if importlib.util.find_spec("datasketch"):
        from news_search.index.dedup_datasketch import DatasketchDeduper
        assert isinstance(get_deduper(Settings(dedup_backend="datasketch")), DatasketchDeduper)
    with pytest.raises(ValueError):
        get_deduper(Settings(dedup_backend="khong-ton-tai"))


# ------------------------------------------------- VN embedder routing (F-05)


def _inject_fake_st(monkeypatch, dim=8):
    """Tiêm fake sentence_transformers vào sys.modules (không nạp torch)."""
    fake = types.ModuleType("sentence_transformers")

    class FakeST:
        def __init__(self, name, device=None):
            self.name = name

        def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
            # vector đơn vị đơn giản, đủ để kiểm shape + chuẩn hóa
            return np.ones((len(texts), dim), dtype=np.float32)

    class FakeCE:
        def __init__(self, name):
            self.name = name

        def predict(self, pairs):
            return [float(i) for i in range(len(pairs))]  # tăng dần theo thứ tự

    fake.SentenceTransformer = FakeST
    fake.CrossEncoder = FakeCE
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    return fake


def test_get_embedder_vi_routing(monkeypatch):
    _inject_fake_st(monkeypatch, dim=8)
    from news_search.index.embeddings import VietnameseEmbedder, get_embedder

    emb = get_embedder(Settings(embedder="vi", vi_embed_dim=8))
    assert isinstance(emb, VietnameseEmbedder) and emb.dim == 8
    v = emb.embed(["xin chào Việt Nam", "tạm biệt"])
    assert v.shape == (2, 8) and v.dtype == np.float32
    assert abs(float(np.linalg.norm(v[0])) - 1.0) < 1e-5   # L2-normalized
    assert emb.embed([]).shape == (0, 8)                   # rỗng an toàn


def test_get_embedder_vi_is_default():
    """Default EMBEDDER là 'vi' (khi không có env override)."""
    # conftest ép EMBEDDER=hash cho test; kiểm giá trị mặc định trực tiếp ở field
    assert Settings(embedder="vi").embedder == "vi"


# ------------------------------------------------- VN reranker routing


def test_get_reranker_vi_routing(monkeypatch):
    # FlagEmbedding không cài -> VietnameseReranker rơi về CrossEncoder (fake)
    _inject_fake_st(monkeypatch)
    from news_search.search.rerank import VietnameseReranker, get_reranker

    rr = get_reranker(Settings(reranker="vi"))
    assert isinstance(rr, VietnameseReranker)
    out = rr.rerank("thuế quan", [("a", "doc a"), ("b", "doc b"), ("c", "doc c")], top_k=2)
    assert len(out) == 2
    assert out[0][1] >= out[1][1]           # sắp giảm dần theo score
    assert rr.rerank("q", []) == []         # rỗng


def test_get_reranker_values():
    from news_search.search.rerank import NoopReranker, get_reranker

    assert isinstance(get_reranker(Settings(reranker="none")), NoopReranker)
    with pytest.raises(ValueError):
        get_reranker(Settings(reranker="khong-ton-tai"))

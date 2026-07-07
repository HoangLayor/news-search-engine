"""Test module semantic (F-05, F-10): embeddings.py + vector.py.

Toàn bộ test deterministic: HashingEmbedder chỉ dùng zlib.crc32, không phụ
thuộc thời gian thực hay random seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from news_search.config import Settings
from news_search.index.embeddings import HashingEmbedder, get_embedder
from news_search.index.vector import VectorIndex

# ---------------------------------------------------------------------------
# HashingEmbedder
# ---------------------------------------------------------------------------


def test_hashing_embedder_deterministic():
    """Embed 2 lần (và với instance mới) phải cho kết quả GIỐNG HỆT."""
    texts = ["Giá vàng trong nước tăng mạnh", "Đội tuyển bóng đá thắng trận"]
    emb = HashingEmbedder(dim=128)
    a = emb.embed(texts)
    b = emb.embed(texts)
    c = HashingEmbedder(dim=128).embed(texts)  # instance mới, cùng dim
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_hashing_embedder_shape_dtype_norm():
    """Shape (n, dim), dtype float32, mỗi hàng khác rỗng có L2-norm ~ 1."""
    emb = HashingEmbedder()  # dim mặc định 256
    texts = ["Giá vàng hôm nay tăng mạnh", "thời tiết Hà Nội", "abc"]
    mat = emb.embed(texts)
    assert mat.shape == (3, 256)
    assert mat.dtype == np.float32
    norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_hashing_embedder_zero_safe():
    """Text rỗng / quá ngắn (không có n-gram) -> zero vector, không chia 0."""
    emb = HashingEmbedder(dim=64)
    mat = emb.embed(["", "   ", "a"])  # "a" chỉ 1 ký tự -> không có 2-gram
    assert mat.shape == (3, 64)
    assert np.array_equal(mat, np.zeros((3, 64), dtype=np.float32))


def test_hashing_embedder_fold_diacritics():
    """Text có dấu và bản không dấu (đã lowercase) phải embed giống hệt."""
    emb = HashingEmbedder(dim=128)
    mat = emb.embed(["Giá VÀNG tăng  đột biến", "gia vang tang dot bien"])
    assert np.array_equal(mat[0], mat[1])


def test_similar_texts_have_higher_cosine():
    """2 văn bản gần nghĩa từ vựng (chung nhiều n-gram) -> cosine cao hơn."""
    emb = HashingEmbedder(dim=256)
    mat = emb.embed(
        [
            "giá vàng trong nước tăng mạnh phiên sáng",
            "giá vàng trong nước tăng nhẹ phiên chiều",
            "đội tuyển bóng đá quốc gia thắng trận chung kết",
        ]
    )
    # Vector đã L2-normalize -> cosine = dot product
    sim_close = float(mat[0] @ mat[1])
    sim_far = float(mat[0] @ mat[2])
    assert sim_close > sim_far
    assert sim_close > 0.5  # chung nhiều n-gram -> phải khá cao


# ---------------------------------------------------------------------------
# VectorIndex
# ---------------------------------------------------------------------------


def _unit(v: list[float]) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def test_vector_index_search_nearest():
    """search trả đúng láng giềng gần nhất, sắp giảm dần theo cosine."""
    idx = VectorIndex(dim=4)
    idx.add("a", np.array([1.0, 0.0, 0.0, 0.0]))
    idx.add("b", np.array([0.9, 0.1, 0.0, 0.0]))
    idx.add("c", np.array([0.0, 0.0, 1.0, 0.0]))
    results = idx.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=2)
    assert [rid for rid, _ in results] == ["a", "b"]
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)
    assert results[0][1] >= results[1][1]


def test_vector_index_allowed_ids_filter():
    """allowed_ids != None -> chỉ trả doc trong tập đó."""
    idx = VectorIndex(dim=4)
    idx.add("a", np.array([1.0, 0.0, 0.0, 0.0]))
    idx.add("b", np.array([0.9, 0.1, 0.0, 0.0]))
    idx.add("c", np.array([0.0, 1.0, 0.0, 0.0]))
    results = idx.search(
        np.array([1.0, 0.0, 0.0, 0.0]), top_k=10, allowed_ids={"b", "c"}
    )
    ids = [rid for rid, _ in results]
    assert "a" not in ids
    assert ids[0] == "b"  # b gần query nhất trong tập allowed
    assert set(ids) == {"b", "c"}


def test_vector_index_remove():
    """remove gỡ đúng id; id không tồn tại -> no-op không raise."""
    idx = VectorIndex(dim=4)
    idx.add("a", np.array([1.0, 0.0, 0.0, 0.0]))
    idx.add("b", np.array([0.0, 1.0, 0.0, 0.0]))
    assert len(idx) == 2
    idx.remove("a")
    assert len(idx) == 1
    assert idx.get("a") is None
    ids = [rid for rid, _ in idx.search(np.array([1.0, 0.0, 0.0, 0.0]), top_k=10)]
    assert "a" not in ids
    idx.remove("khong-ton-tai")  # no-op
    assert len(idx) == 1


def test_vector_index_add_replace_no_growth():
    """add trùng id -> replace, index không phình; get trả bản normalize mới."""
    idx = VectorIndex(dim=4)
    idx.add("a", np.array([1.0, 0.0, 0.0, 0.0]))
    idx.add("a", np.array([0.0, 2.0, 0.0, 0.0]))  # replace, chưa normalize
    assert len(idx) == 1
    got = idx.get("a")
    assert got is not None
    assert np.allclose(got, _unit([0.0, 2.0, 0.0, 0.0]), atol=1e-6)
    # search phản ánh vector MỚI
    results = idx.search(np.array([0.0, 1.0, 0.0, 0.0]), top_k=1)
    assert results[0][0] == "a"
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_vector_index_zero_query_and_empty():
    """Query vector zero -> []; index rỗng -> []; get thiếu id -> None."""
    idx = VectorIndex(dim=4)
    assert idx.search(np.array([1.0, 0.0, 0.0, 0.0])) == []
    idx.add("a", np.array([1.0, 0.0, 0.0, 0.0]))
    assert idx.search(np.zeros(4)) == []
    assert idx.get("khong-co") is None
    assert len(idx) == 1


def test_vector_index_stores_normalized():
    """add lưu bản L2-normalized (norm = 1)."""
    idx = VectorIndex(dim=3)
    idx.add("a", np.array([3.0, 4.0, 0.0]))
    got = idx.get("a")
    assert got is not None
    assert got.dtype == np.float32
    assert np.linalg.norm(got) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(got, [0.6, 0.8, 0.0], atol=1e-6)


def test_embedder_with_vector_index_end_to_end():
    """Embed bài viết -> add vào index -> query gần nghĩa trả đúng láng giềng."""
    emb = HashingEmbedder(dim=256)
    docs = {
        "vang": "Giá vàng trong nước hôm nay tăng mạnh lên đỉnh mới",
        "bongda": "Đội tuyển bóng đá Việt Nam thắng trận chung kết khu vực",
        "thoitiet": "Dự báo thời tiết Hà Nội có mưa rào và dông rải rác",
    }
    idx = VectorIndex(dim=256)
    ids = list(docs.keys())
    mat = emb.embed([docs[i] for i in ids])
    for i, aid in enumerate(ids):
        idx.add(aid, mat[i])
    query_vec = emb.embed(["giá vàng tăng mạnh"])[0]
    results = idx.search(query_vec, top_k=3)
    assert results[0][0] == "vang"


# ---------------------------------------------------------------------------
# get_embedder
# ---------------------------------------------------------------------------


def test_get_embedder_hash():
    """settings.embedder == 'hash' -> HashingEmbedder với dim từ settings."""
    settings = Settings(embedder="hash", embedding_dim=64)
    emb = get_embedder(settings)
    assert isinstance(emb, HashingEmbedder)
    assert emb.dim == 64


def test_get_embedder_openai_missing_raises_runtime_error():
    """'openai' nhưng thiếu package hoặc thiếu key -> RuntimeError khi __init__."""
    settings = Settings(embedder="openai", openai_api_key="")
    with pytest.raises(RuntimeError):
        get_embedder(settings)


def test_get_embedder_unknown_raises_value_error():
    """Giá trị embedder không hỗ trợ -> ValueError."""
    settings = Settings(embedder="khong-ho-tro")
    with pytest.raises(ValueError):
        get_embedder(settings)

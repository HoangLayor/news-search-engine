"""Test guard dim của MilvusVectorIndex — chạy hermetic (fake client, không cần Milvus).

Bối cảnh: collection tạo bằng embedder cũ (vd hash, dim 256) mà đổi sang embedder
mới (vd vi, dim 1024) thì Milvus TỪ CHỐI mọi insert. Trước đây lỗi này bị nuốt
(reindex `except: pass`) -> "0 bài" không rõ nguyên nhân. Guard phải báo lỗi rõ.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from news_search.config import Settings
from news_search.index.milvus_vector import MilvusVectorIndex

# Đủ thuộc tính mà _create_collection dùng
_DATATYPE = SimpleNamespace(VARCHAR=21, FLOAT_VECTOR=101)


class _FakeSchema:
    def add_field(self, *a, **k):
        pass


class _FakeIndexParams:
    def add_index(self, **k):
        self.kw = k


class _FakeClient:
    def __init__(self, dim: int | None = 256, exists: bool = True):
        self._dim, self._exists = dim, exists
        self.dropped = self.created = self.loaded = False

    def has_collection(self, name):
        return self._exists

    def describe_collection(self, name):
        return {"fields": [
            {"name": "article_id", "params": {"max_length": 256}},
            {"name": "vector", "params": ({"dim": self._dim} if self._dim else {})},
        ]}

    def drop_collection(self, name):
        self.dropped, self._exists = True, False

    def create_schema(self, **k):
        return _FakeSchema()

    def prepare_index_params(self):
        return _FakeIndexParams()

    def create_collection(self, **k):
        self.created = True

    def load_collection(self, name):
        self.loaded = True


def _index(dim: int, client: _FakeClient) -> MilvusVectorIndex:
    """Dựng MilvusVectorIndex KHÔNG chạy __init__ (không cần pymilvus/Milvus)."""
    obj = MilvusVectorIndex.__new__(MilvusVectorIndex)
    obj._client, obj.dim = client, dim
    obj.collection, obj.metric = "news_articles", "COSINE"
    return obj


def _settings(recreate: bool) -> Settings:
    return Settings(vector_backend="milvus", embedder="vi",
                    milvus_recreate_on_dim_mismatch=recreate)


def test_dim_mismatch_bao_loi_ro_rang():
    """dim khác + KHÔNG bật cờ -> RuntimeError nêu rõ 256 vs 1024 + cách xử lý."""
    client = _FakeClient(dim=256)
    idx = _index(1024, client)
    with pytest.raises(RuntimeError) as exc:
        idx._ensure_collection(_DATATYPE, _settings(recreate=False))
    msg = str(exc.value)
    assert "256" in msg and "1024" in msg
    assert "MILVUS_RECREATE_ON_DIM_MISMATCH" in msg
    assert not client.dropped and not client.created  # KHÔNG phá dữ liệu


def test_dim_mismatch_recreate_khi_bat_co():
    """dim khác + bật cờ -> drop + tạo lại + load."""
    client = _FakeClient(dim=256)
    _index(1024, client)._ensure_collection(_DATATYPE, _settings(recreate=True))
    assert client.dropped and client.created and client.loaded


def test_dim_khop_khong_dung_toi_collection():
    """dim khớp -> không drop, không tạo lại, chỉ load."""
    client = _FakeClient(dim=1024)
    _index(1024, client)._ensure_collection(_DATATYPE, _settings(recreate=False))
    assert not client.dropped and not client.created and client.loaded


def test_collection_chua_ton_tai_thi_tao_moi():
    client = _FakeClient(exists=False)
    _index(1024, client)._ensure_collection(_DATATYPE, _settings(recreate=False))
    assert client.created and client.loaded and not client.dropped

"""Test độ bền reindex/index (Fix 2-4): nguyên tử, tái dùng model, không nuốt lỗi."""

from __future__ import annotations

import logging

import pytest

from news_search.config import Settings
from news_search.index.manager import IndexManager
from news_search.search.pipeline import SearchPipeline
from news_search.search.rerank import NoopReranker
from news_search.service.search_service import SearchService


def _settings():
    return Settings(embedder="hash", vector_backend="local", tokenizer="regex",
                    dedup_backend="local", reranker="none")


def _art(aid="a1"):
    return {"article_id": aid, "title": "lạm phát", "body": "lạm phát tăng cao",
            "url": f"/{aid}", "published_at": "2026-07-01T00:00:00+07:00"}


# --------------------------------------------------- Fix 4: index_article nguyên tử


def test_index_article_rollback_khi_vector_loi():
    """vector.add ném -> rollback SẠCH: bài không còn ở bất kỳ chỉ mục nào; generation giữ nguyên."""
    m = IndexManager(_settings())
    m.index_article(_art("ok"))
    gen = m.generation

    def boom(*a, **k):
        raise RuntimeError("Milvus từ chối insert")

    m.vector.add = boom  # ép bước dễ lỗi nhất thất bại
    with pytest.raises(RuntimeError):
        m.index_article(_art("bad"))

    # KHÔNG lệch pha: không sót "bad" ở chỉ mục nào
    assert m.store.get("bad") is None
    assert m.deduper.cluster_of("bad") is None
    assert m.kg.entities_for_article("bad") == []
    assert m.generation == gen          # thất bại -> không tăng generation
    assert m.store.get("ok") is not None  # bài cũ nguyên vẹn


# --------------------------------------------------- Fix 2: tái dùng model


def test_embedder_injection():
    m1 = IndexManager(_settings())
    m2 = IndexManager(_settings(), embedder=m1.embedder)
    assert m2.embedder is m1.embedder   # tái dùng, không nạp lại


def test_reranker_injection():
    m = IndexManager(_settings())
    rr = NoopReranker()
    p = SearchPipeline(m, m.settings, reranker=rr)
    assert p.reranker is rr


def test_reindex_tai_dung_embedder_va_reranker():
    svc = SearchService(_settings())   # SOURCE=sample (conftest)
    emb_before = svc.manager.embedder
    rr_before = svc.pipeline.reranker
    n = svc.reindex()
    assert n >= 20
    assert svc.manager.embedder is emb_before   # KHÔNG nạp lại embedder (~GB)
    assert svc.pipeline.reranker is rr_before    # KHÔNG nạp lại reranker (~GB)


# --------------------------------------------------- Fix 3: không nuốt lỗi


class _MixedSource:
    """Nguồn giả: 1 bài hợp lệ + 1 bài lỗi (thiếu field bắt buộc)."""

    def fetch_all(self, batch):
        yield _art("good")
        yield {"article_id": "bad"}  # thiếu title/body/... -> normalize raise

    def close(self):
        pass


class _AllBadSource:
    def fetch_all(self, batch):
        yield {"article_id": "x"}

    def close(self):
        pass


def test_reindex_dem_va_log_bai_loi(caplog):
    svc = SearchService(_settings())
    with caplog.at_level(logging.WARNING):
        n = svc.reindex(source=_MixedSource())
    assert n == 1                                   # chỉ 'good' index được
    assert "BỎ QUA 1 bài" in caplog.text            # có log cảnh báo (không nuốt)
    assert "bad" in caplog.text                      # kèm mẫu bài lỗi


def test_reindex_ca_dot_loi_log_error(caplog):
    svc = SearchService(_settings())
    with caplog.at_level(logging.ERROR):
        n = svc.reindex(source=_AllBadSource())
    assert n == 0
    assert "TẤT CẢ" in caplog.text                   # lộ nguyên nhân thay vì im lặng

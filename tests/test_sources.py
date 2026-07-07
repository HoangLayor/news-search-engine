"""Test tầng nguồn dữ liệu: SampleJsonSource, factory, và ánh xạ PostgresCMSSource.

Ánh xạ Postgres được test KHÔNG cần DB (dựng instance qua __new__ rồi gọi _map),
nên chạy được offline/CI.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from news_search.config import Settings
from news_search.sources import SampleJsonSource, get_source
from news_search.sources.postgres import PostgresCMSSource


def test_sample_source_basic():
    src = SampleJsonSource()
    n = src.count()
    assert n >= 20
    allrows = list(src.fetch_all())
    assert len(allrows) == n
    # dict thô tương thích normalize_article
    assert {"article_id", "title", "body", "url", "published_at"} <= set(allrows[0].keys())


def test_sample_source_fetch_by_ids_and_since():
    src = SampleJsonSource()
    got = list(src.fetch_by_ids(["eco-cpi-01", "khong-ton-tai"]))
    assert [r["article_id"] for r in got] == ["eco-cpi-01"]
    since = datetime(2026, 7, 7, tzinfo=timezone(__import__("datetime").timedelta(hours=7)))
    recent = list(src.fetch_since(since))
    assert all(r["published_at"] >= "2026-07-07" for r in recent)
    assert len(recent) < src.count()


def test_get_source_factory():
    assert isinstance(get_source(Settings(source="sample")), SampleJsonSource)
    with pytest.raises(ValueError):
        get_source(Settings(source="khong-ton-tai"))


def _pg_stub() -> PostgresCMSSource:
    """Dựng PostgresCMSSource KHÔNG kết nối DB để test _map."""
    src = PostgresCMSSource.__new__(PostgresCMSSource)
    src._base_url = ""
    return src


def test_postgres_map_day_du():
    src = _pg_stub()
    dt = datetime(2026, 7, 7, 9, 0, tzinfo=timezone.utc)
    row = {
        "id": 123, "title": "Tiêu đề bài", "brief_title": "bt",
        "lead": "Đoạn sapo.", "content": "<p>Nội dung <b>chính</b>.</p>",
        "slug": "tieu-de-bai-123", "source": "TTXVN",
        "publish_date": dt, "created_at": dt,
        "category": "Kinh tế", "author": "Nguyễn Văn A", "tags": ["vàng", "cpi"],
    }
    m = src._map(row)
    assert m["article_id"] == "123"
    assert m["title"] == "Tiêu đề bài"
    assert "Đoạn sapo." in m["body"] and "Nội dung" in m["body"]
    assert m["url"] == "/tieu-de-bai-123"
    assert m["published_at"] == dt
    assert m["category"] == "Kinh tế" and m["author"] == "Nguyễn Văn A"
    assert m["tags"] == ["vàng", "cpi"]
    assert m["status"] == "published"


def test_postgres_map_body_fallback_va_base_url():
    src = _pg_stub()
    # content/lead rỗng -> body lấy title (bản tin chỉ có tiêu đề)
    m = src._map({"id": 9, "title": "BẢN TIN CUỐI NGÀY", "brief_title": None,
                  "lead": None, "content": None, "slug": "ban-tin-9",
                  "publish_date": None, "created_at": datetime(2026, 6, 1, tzinfo=timezone.utc),
                  "category": None, "author": None, "source": None, "tags": None})
    assert m["body"] == "BẢN TIN CUỐI NGÀY"
    assert m["published_at"] is not None  # fallback created_at
    assert m["tags"] == []
    # base_url dựng URL tuyệt đối
    src._base_url = "https://baomoi.vn"
    m2 = src._map({"id": 5, "title": "T", "brief_title": None, "lead": "x", "content": "y",
                   "slug": "bai-5", "publish_date": datetime(2026, 6, 1, tzinfo=timezone.utc),
                   "created_at": None, "category": None, "author": None, "source": None, "tags": []})
    assert m2["url"] == "https://baomoi.vn/bai-5"

"""Test e2e SearchPipeline trên bộ dữ liệu mẫu (phủ checklist §7.3).

now cố định 2026-07-07 12:00 +07:00 -> kết quả deterministic.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from news_search.config import Settings
from news_search.index.manager import IndexManager
from news_search.models import SearchFilters, SearchQuery
from news_search.search.pipeline import SearchPipeline

NOW = datetime(2026, 7, 7, 12, 0, 0, tzinfo=timezone(timedelta(hours=7)))
DATA = Path(__file__).parent.parent / "data" / "sample_articles.json"


@pytest.fixture()
def env() -> tuple[IndexManager, SearchPipeline]:
    settings = Settings.from_env()
    manager = IndexManager(settings)
    raws = json.loads(DATA.read_text(encoding="utf-8"))
    manager.bulk_index(raws)
    return manager, SearchPipeline(manager, settings)


def _ids(results) -> list[str]:
    return [r.article_id for r in results]


def test_keyword_lam_phat(env):
    """"lạm phát tháng 6" -> bài CPI tháng 6 đứng đầu."""
    _, pipeline = env
    results = pipeline.search(SearchQuery(text="lạm phát tháng 6", top_k=5, now=NOW))
    assert "eco-cpi-01" in _ids(results)
    assert results[0].article_id == "eco-cpi-01"


def test_natural_query_gia_vang(env):
    """Câu hỏi tự nhiên -> bài giải thích nguyên nhân giá vàng đứng đầu."""
    _, pipeline = env
    results = pipeline.search(
        SearchQuery(text="vì sao giá vàng tăng mạnh tuần này", top_k=5, now=NOW)
    )
    assert results[0].article_id == "eco-gold-01"


def test_freshness_dong_dat(env):
    """"động đất": bài mới 07/07 đứng TRƯỚC bài nền 26/06 nhờ time-decay."""
    _, pipeline = env
    results = pipeline.search(SearchQuery(text="động đất", top_k=10, now=NOW))
    ids = _ids(results)
    assert "news-quake-new" in ids and "news-quake-old" in ids
    assert ids.index("news-quake-new") < ids.index("news-quake-old")


def test_filter_category(env):
    """Lọc chuyên mục Kinh tế -> chỉ trả bài Kinh tế (F-12)."""
    manager, pipeline = env
    results = pipeline.search(
        SearchQuery(
            text="giá xăng", top_k=10, now=NOW,
            filters=SearchFilters(category="Kinh tế"),
        )
    )
    ids = _ids(results)
    assert "eco-gas-01" in ids  # bài giá xăng thuộc Kinh tế
    assert "soc-gas-02" not in ids  # bài giá xăng thuộc Xã hội bị loại
    assert all(manager.store.get(i).category == "Kinh tế" for i in ids)


def test_dedup_bau_cu(env):
    """"kết quả bầu cử": 3 bài gần trùng gom 1 cụm, top-10 không có >=2 bài cùng cụm."""
    manager, pipeline = env
    # Tiền đề: 3 bản tin cùng nội dung phải chung một cụm sự kiện
    c1 = manager.deduper.cluster_of("elect-dup-1")
    assert c1 == manager.deduper.cluster_of("elect-dup-2") == manager.deduper.cluster_of("elect-dup-3")

    results = pipeline.search(SearchQuery(text="kết quả bầu cử", top_k=10, now=NOW))
    ids = _ids(results)
    # Không quá 1 bài từ cụm bầu cử gần trùng
    dup_hits = [i for i in ids if manager.deduper.cluster_of(i) == c1]
    assert len(dup_hits) <= 1
    # Nhưng chủ đề bầu cử vẫn xuất hiện (1 bản tin đại diện + bài phân tích khác cụm)
    assert any(i.startswith("elect") for i in ids)
    # Tổng quát: không cụm nào xuất hiện quá 1 lần trong kết quả
    clusters = [manager.deduper.cluster_of(i) for i in ids]
    assert len(clusters) == len(set(clusters))


def test_output_schema_5_truong(env):
    """Mọi kết quả .to_dict() có ĐÚNG 5 khóa quy định."""
    _, pipeline = env
    results = pipeline.search(SearchQuery(text="kinh tế", top_k=10, now=NOW))
    assert results
    for r in results:
        assert set(r.to_dict().keys()) == {
            "article_id", "title", "url", "published_at", "snippet"
        }


def test_query_rong_raise(env):
    """Truy vấn rỗng -> ValueError."""
    _, pipeline = env
    with pytest.raises(ValueError):
        pipeline.search(SearchQuery(text="   ", now=NOW))


def test_query_khong_token_tra_rong(env):
    """Truy vấn chỉ gồm dấu câu (không token từ nào) -> [] (không trả nhiễu)."""
    _, pipeline = env
    assert pipeline.search(SearchQuery(text="@@@ ### !!!", now=NOW)) == []


def test_unpublish_bien_mat(env):
    """F-08: bài status=unpublished KHÔNG hiển thị trong public search (lọc theo
    status), và DELETE mới gỡ hẳn khỏi mọi chỉ mục."""
    manager, pipeline = env
    before = _ids(pipeline.search(SearchQuery(text="giá vàng", top_k=10, now=NOW)))
    assert "eco-gold-01" in before

    manager.index_article({
        "article_id": "eco-gold-01",
        "title": "Giá vàng lập đỉnh: ba nguyên nhân chính đẩy giá đi lên",
        "body": "nội dung không còn hiển thị",
        "url": "/kinh-te/gia-vang-lap-dinh-8905.html",
        "published_at": "2026-07-07T09:05:00+07:00",
        "status": "unpublished",
    })
    # Public search (mặc định lọc status=published) không còn thấy bài
    after = _ids(pipeline.search(SearchQuery(text="giá vàng", top_k=10, now=NOW)))
    assert "eco-gold-01" not in after
    # Thiết kế lọc-theo-status: bài vẫn nằm trong chỉ mục với status=unpublished
    art = manager.store.get("eco-gold-01")
    assert art is not None and art.status == "unpublished"

    # DELETE mới thực sự gỡ khỏi MỌI chỉ mục
    manager.remove_article("eco-gold-01")
    assert manager.store.get("eco-gold-01") is None
    assert "eco-gold-01" not in manager.lexical
    assert manager.vector.get("eco-gold-01") is None
    assert manager.deduper.cluster_of("eco-gold-01") is None

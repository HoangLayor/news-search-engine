"""Test cache (GĐ5): TTL + LRU + factory + tích hợp pipeline (hit + invalidation)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from news_search.config import Settings
from news_search.index.manager import IndexManager
from news_search.models import SearchQuery
from news_search.search.pipeline import SearchPipeline
from news_search.service.cache import InMemoryTTLCache, get_cache

NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone(timedelta(hours=7)))


def _art(aid, title, body, pub="2026-07-01T00:00:00+07:00"):
    return {"article_id": aid, "title": title, "body": body, "url": f"/{aid}",
            "published_at": pub}


def test_ttl_expiry():
    clk = [0.0]
    c = InMemoryTTLCache(max_size=10, ttl_seconds=10, clock=lambda: clk[0])
    c.set("k", "v")
    assert c.get("k") == "v" and c.hits == 1
    clk[0] = 11.0
    assert c.get("k") is None and c.misses == 1  # hết hạn


def test_lru_eviction():
    clk = [0.0]
    c = InMemoryTTLCache(max_size=2, ttl_seconds=100, clock=lambda: clk[0])
    c.set("a", 1)
    c.set("b", 2)
    c.get("a")           # a vừa dùng -> mới hơn b
    c.set("c", 3)        # vượt max -> loại b (cũ nhất)
    assert c.get("b") is None
    assert c.get("a") == 1 and c.get("c") == 3


def test_factory():
    assert get_cache(Settings(cache_enabled=False)) is None
    assert isinstance(get_cache(Settings(cache_enabled=True, cache_backend="memory")),
                      InMemoryTTLCache)


def test_pipeline_cache_hit_and_invalidation():
    s = Settings(embedder="hash", vector_backend="local", cache_enabled=True)
    m = IndexManager(s)
    m.index_article(_art("a1", "lạm phát", "lạm phát tăng"))
    p = SearchPipeline(m, s)

    r1 = p.search(SearchQuery("lạm phát", now=NOW))
    r2 = p.search(SearchQuery("lạm phát", now=NOW))  # cache hit
    assert p.cache.hits >= 1
    assert [x.article_id for x in r1] == [x.article_id for x in r2]

    # Thay đổi chỉ mục -> generation tăng -> khóa cache đổi -> KHÔNG hit cache cũ
    hits_before = p.cache.hits
    m.index_article(_art("a2", "lạm phát cao", "lạm phát cao", "2026-07-02T00:00:00+07:00"))
    p.search(SearchQuery("lạm phát", now=NOW))
    assert p.cache.hits == hits_before  # miss vì generation đã đổi


def test_pipeline_cache_bypass_fresh():
    s = Settings(embedder="hash", vector_backend="local", cache_enabled=True,
                 cache_bypass_fresh=True)
    m = IndexManager(s)
    m.index_article(_art("a1", "động đất", "động đất hôm nay"))
    p = SearchPipeline(m, s)
    p.search(SearchQuery("động đất hôm nay", now=NOW))  # fresh_intent -> không cache
    p.search(SearchQuery("động đất hôm nay", now=NOW))
    assert p.cache.hits == 0  # bỏ qua cache cho truy vấn tin nóng

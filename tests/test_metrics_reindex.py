"""Test metrics/alerts (GĐ5) + snapshot/restore + reindex blue-green."""

from __future__ import annotations

from news_search.config import Settings
from news_search.index.manager import IndexManager
from news_search.service.metrics import MetricsCollector, render_dashboard
from news_search.service.search_service import SearchService


# ------------------------------------------------------------------ metrics


def test_metrics_collector():
    mc = MetricsCollector()
    for i in range(30):
        mc.record_search(10.0 + i, num_results=(0 if i < 3 else 5), cache_hit=(i % 2 == 0))
    snap = mc.snapshot()
    assert snap["searches"] == 30
    assert snap["zero_results"] == 3
    assert 0.0 < snap["cache_hit_rate"] <= 1.0
    assert snap["latency_p95_ms"] >= snap["latency_p50_ms"]
    assert "nse_searches 30" in mc.prometheus_text()


def test_alerts_fire_on_high_latency():
    mc = MetricsCollector()
    for _ in range(25):
        mc.record_search(1000.0, num_results=5)
    alerts = mc.check_alerts(p95_ms=500.0, error_rate=0.5, zero_rate=0.9)
    assert any("p95" in a for a in alerts)


def test_alerts_need_enough_samples():
    mc = MetricsCollector()
    for _ in range(5):  # < 20 mẫu -> chưa cảnh báo
        mc.record_search(9999.0, num_results=0)
    assert mc.check_alerts(1.0, 0.0, 0.0) == []


def test_dashboard_html():
    html = render_dashboard({"searches": 3}, {"ctr": 0.5}, {"articles": 10})
    assert "<html" in html and "Dashboard" in html and "CTR" in html


# --------------------------------------------------------- snapshot/restore


def _art(aid, title, body, **extra):
    d = {"article_id": aid, "title": title, "body": body, "url": f"/{aid}",
         "published_at": "2026-07-01T00:00:00+07:00"}
    d.update(extra)
    return d


def test_snapshot_restore(tmp_path):
    s = Settings(embedder="hash", vector_backend="local")
    m = IndexManager(s)
    m.index_article(_art("a1", "Lạm phát", "lạm phát tăng", category="Kinh tế", tags=["cpi"]))
    m.index_article(_art("a2", "Giá vàng", "giá vàng lập đỉnh"))
    path = str(tmp_path / "snap.jsonl")
    assert m.snapshot(path) == 2

    m2 = IndexManager(s)
    assert m2.restore(path) == 2
    assert m2.store.get("a1").category == "Kinh tế"
    assert m2.store.get("a1").tags == ["cpi"]
    assert m2.vector.get("a1") is not None and m2.vector.get("a2") is not None


# ----------------------------------------------------------- reindex swap


def test_search_service_reindex_swap():
    s = Settings(embedder="hash", vector_backend="local", source="sample")
    svc = SearchService(s)
    old_manager = svc.manager
    assert len(svc.manager.store) == 0  # chưa index gì

    n = svc.reindex()
    assert n >= 20
    assert svc.manager is not old_manager      # đã hoán đổi (blue-green)
    assert len(svc.manager.store) == n
    assert svc.pipeline.manager is svc.manager  # pipeline trỏ chỉ mục mới


def test_generation_increments():
    s = Settings(embedder="hash", vector_backend="local")
    m = IndexManager(s)
    g0 = m.generation
    m.index_article(_art("a1", "x", "nội dung x"))
    assert m.generation == g0 + 1
    m.remove_article("a1")
    assert m.generation == g0 + 2

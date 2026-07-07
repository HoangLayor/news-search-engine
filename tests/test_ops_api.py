"""Test API tính năng vận hành (GĐ1/GĐ5): feedback, metrics, dashboard, experiment, admin."""

from __future__ import annotations

from fastapi.testclient import TestClient

from news_search.api.app import create_app
from news_search.config import Settings

_ART = {"article_id": "a1", "title": "Lạm phát tháng 6", "body": "chỉ số lạm phát tăng",
        "url": "/a1", "published_at": "2026-07-01T00:00:00+07:00"}


def _client(**over) -> TestClient:
    base = dict(embedder="hash", vector_backend="local")
    base.update(over)
    c = TestClient(create_app(Settings(**base)))
    c.post("/articles", json=_ART)
    return c


def test_feedback_flow(tmp_path):
    c = _client(feedback_enabled=True, feedback_backend="jsonl",
                feedback_path=str(tmp_path / "e.jsonl"))
    r = c.get("/search", params={"q": "lạm phát"})
    sid = r.headers.get("x-search-id")
    assert sid  # có header correlate
    cr = c.post("/events/click",
                json={"search_id": sid, "article_id": "a1", "position": 1, "dwell_ms": 2000})
    assert cr.status_code == 200 and cr.json()["logged"] is True
    # dashboard đọc được CTR từ log
    d = c.get("/dashboard")
    assert d.status_code == 200 and "CTR" in d.text


def test_feedback_disabled_no_header():
    c = _client()  # feedback tắt mặc định
    r = c.get("/search", params={"q": "lạm phát"})
    assert "x-search-id" not in r.headers
    # endpoint click vẫn nhận nhưng no-op
    assert c.post("/events/click",
                  json={"search_id": "x", "article_id": "a1", "position": 1}).json()["logged"] is False


def test_click_bad_payload_422():
    c = _client(feedback_enabled=True)
    assert c.post("/events/click", json={"search_id": "x"}).status_code == 422


def test_metrics_endpoint():
    c = _client()
    c.get("/search", params={"q": "lạm phát"})
    m = c.get("/metrics")
    assert m.status_code == 200 and "nse_searches" in m.text


def test_metrics_disabled_404():
    c = _client(metrics_enabled=False)
    assert c.get("/metrics").status_code == 404


def test_experiment_variant_header():
    c = _client(experiment_enabled=True)
    r = c.get("/search", params={"q": "lạm phát"})
    assert r.headers.get("x-variant") in ("A", "B")


def test_admin_reindex_disabled_and_token():
    c = _client()  # ADMIN_TOKEN rỗng -> tắt
    assert c.post("/admin/reindex", json={}).status_code == 404

    c2 = _client(admin_token="secret", source="sample")
    assert c2.post("/admin/reindex", json={}, headers={"X-Admin-Token": "wrong"}).status_code == 403
    ok = c2.post("/admin/reindex", json={"limit": 10}, headers={"X-Admin-Token": "secret"})
    assert ok.status_code == 200 and ok.json()["reindexed"] >= 1


def test_healthz_features():
    c = _client()
    body = c.get("/healthz").json()
    assert "features" in body and body["features"]["cache"] is False

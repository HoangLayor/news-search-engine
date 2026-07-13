"""Test HTTP API (FastAPI TestClient) — CONTRACTS.md §16.

Kiểm: ingest + search hoạt động; response array đúng 5 trường; q rỗng -> 400;
thiếu field bắt buộc -> 422; DELETE gỡ bài; /healthz.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from news_search.api.app import create_app
from news_search.config import Settings

DATA = Path(__file__).parent.parent / "data" / "sample_articles.json"
FIVE_FIELDS = {"article_id", "title", "url", "published_at", "snippet"}


@pytest.fixture()
def client() -> TestClient:
    app = create_app(Settings())
    c = TestClient(app)
    raws = json.loads(DATA.read_text(encoding="utf-8"))
    resp = c.post("/articles/bulk", json=raws)
    assert resp.status_code == 200 and resp.json()["indexed"] == len(raws)
    return c


def test_healthz(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["articles"] == 29 and body["vectors"] == 29


def test_search_tra_ve_array_5_truong(client):
    resp = client.get("/search", params={"q": "lạm phát tháng 6", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list) and data
    for item in data:  # mỗi phần tử ĐÚNG 5 trường, không lộ score/entity
        assert set(item.keys()) == FIVE_FIELDS
    assert data[0]["article_id"] == "eco-cpi-01"


def test_search_khong_dau(client):
    """Truy vấn KHÔNG DẤU vẫn tìm ra bài có dấu (match folded)."""
    data = client.get("/search", params={"q": "lam phat"}).json()
    assert any(item["article_id"].startswith("eco-cpi") for item in data)


def test_search_q_rong_400(client):
    assert client.get("/search", params={"q": ""}).status_code == 400
    assert client.get("/search", params={"q": "   "}).status_code == 400
    assert client.get("/search").status_code == 400  # thiếu q


def test_search_mode_khong_hop_le_400(client):
    assert client.get("/search", params={"q": "kinh tế", "mode": "bogus"}).status_code == 400


def test_search_filter_category(client):
    data = client.get(
        "/search", params={"q": "giá xăng", "category": "Kinh tế", "top_k": 10}
    ).json()
    ids = [d["article_id"] for d in data]
    assert "eco-gas-01" in ids and "soc-gas-02" not in ids


def test_index_thieu_field_422(client):
    """Thiếu field bắt buộc (body) -> 422 với message rõ."""
    resp = client.post("/articles", json={
        "article_id": "x1", "title": "Thiếu body", "url": "/x1",
        "published_at": "2026-07-07T00:00:00+07:00",
    })
    assert resp.status_code == 422
    assert "body" in resp.json()["detail"]


def test_index_roi_delete(client):
    """POST bài mới -> search thấy -> DELETE -> search không còn (F-08)."""
    new = {
        "article_id": "new-001",
        "title": "Ra mắt vệ tinh viễn thám thế hệ mới",
        "body": "Vệ tinh viễn thám thế hệ mới vừa được phóng thành công lên quỹ đạo phục vụ quan trắc.",
        "url": "/khoa-hoc/ve-tinh-001.html",
        "category": "Khoa học",
        "published_at": "2026-07-07T10:00:00+07:00",
    }
    r = client.post("/articles", json=new)
    assert r.status_code == 201 and r.json() == {"indexed": True, "article_id": "new-001"}

    found = client.get("/search", params={"q": "vệ tinh viễn thám"}).json()
    assert any(d["article_id"] == "new-001" for d in found)

    assert client.delete("/articles/new-001").json() == {"removed": True}
    after = client.get("/search", params={"q": "vệ tinh viễn thám"}).json()
    assert all(d["article_id"] != "new-001" for d in after)


def test_article_entities_endpoint(client):
    """GET /articles/{id}/entities trả list; bài không tồn tại -> 404."""
    ents = client.get("/articles/elect-analysis/entities").json()
    assert isinstance(ents, list)
    assert all({"name", "type"} == set(e.keys()) for e in ents)
    assert client.get("/articles/khong-ton-tai/entities").status_code == 404

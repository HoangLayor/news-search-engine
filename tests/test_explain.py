"""Test chế độ explain (giải thích từng bước) + endpoint /explain."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from news_search.api.app import create_app
from news_search.config import Settings
from news_search.index.manager import IndexManager
from news_search.models import SearchQuery
from news_search.search.pipeline import SearchPipeline

NOW = datetime(2026, 7, 7, 12, tzinfo=timezone(timedelta(hours=7)))


def _pipeline():
    s = Settings(embedder="hash", vector_backend="local", tokenizer="regex", dedup_backend="local")
    m = IndexManager(s)
    for i, (t, b) in enumerate([
        ("Lạm phát tháng 6", "chỉ số lạm phát tăng cao"),
        ("Giá vàng lập đỉnh", "giá vàng tăng do lãi suất"),
        ("Bóng đá quốc gia", "đội tuyển thắng trận"),
    ]):
        m.index_article({"article_id": f"a{i}", "title": t, "body": b,
                         "url": f"/a{i}", "published_at": "2026-07-01T00:00:00+07:00"})
    return SearchPipeline(m, s)


def test_explain_hybrid_has_all_steps():
    p = _pipeline()
    out = p.explain(SearchQuery("lạm phát", mode="hybrid", top_k=3, now=NOW))
    assert {"query", "mode", "top_k", "results", "trace"} <= set(out)
    keys = [st["key"] for st in out["trace"]]
    for k in ("understand", "filter", "fusion",
              "decay", "rerank", "dedup", "mmr", "final"):
        assert k in keys, f"thiếu bước {k}"
    # step đánh số tăng dần 1..n
    assert [st["step"] for st in out["trace"]] == list(range(1, len(out["trace"]) + 1))
    # mỗi bước có title; bước truy hồi có items kèm điểm
    fusion = next(st for st in out["trace"] if st["key"] == "fusion")
    assert fusion["items"] and "score" in fusion["items"][0] and "title" in fusion["items"][0]


def test_explain_matches_search():
    p = _pipeline()
    q = SearchQuery("lạm phát", mode="hybrid", top_k=3, now=NOW)
    explained = [r["article_id"] for r in p.explain(q)["results"]]
    searched = [r.article_id for r in p.search(q)]
    assert explained == searched  # explain KHÔNG đổi kết quả


def test_explain_lexical_only_branch():
    p = _pipeline()
    keys = [st["key"] for st in p.explain(SearchQuery("lạm phát", mode="lexical", now=NOW))["trace"]]
    assert "lexical" in keys
    assert "dense" not in keys and "fusion" not in keys


def test_explain_api():
    c = TestClient(create_app(Settings(
        embedder="hash", vector_backend="local", tokenizer="regex",
        dedup_backend="local", source="sample", autoload_sample=True)))
    r = c.get("/explain", params={"q": "lạm phát tháng 6", "mode": "hybrid", "top_k": 5})
    assert r.status_code == 200
    d = r.json()
    assert d["results"] and d["trace"]
    assert d["trace"][0]["key"] == "understand"
    assert c.get("/explain", params={"q": ""}).status_code == 400
    assert c.get("/explain", params={"q": "x", "mode": "bogus"}).status_code == 400

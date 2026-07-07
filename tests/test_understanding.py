"""Test query understanding (GĐ3): spell-correct + expansion + tích hợp pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from news_search.config import Settings
from news_search.index.manager import IndexManager
from news_search.models import SearchQuery
from news_search.search.pipeline import SearchPipeline
from news_search.search.query import parse_query
from news_search.search.understanding import QueryUnderstander, SpellCorrector

NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone(timedelta(hours=7)))


def test_spell_corrector():
    c = SpellCorrector({"lam": 10, "phat": 8, "kinh": 5, "te": 5}, max_edit_distance=2)
    assert c.correct("phatt") == "phat"     # thừa 1 ký tự -> sửa
    assert c.correct("phat") == "phat"      # đã đúng
    assert c.correct("zzzzz") == "zzzzz"    # không có ứng viên -> giữ nguyên


def test_understander_disabled_noop():
    u = QueryUnderstander(Settings(), vocab_provider=lambda: {"lam": 1})
    assert not u.enabled
    out = u.understand(parse_query("lạm phát"))
    assert out.corrections == {} and out.expansions == []


def test_understander_spellcorrect():
    u = QueryUnderstander(Settings(qu_spellcorrect=True),
                          vocab_provider=lambda: {"lam": 5, "phat": 5})
    assert u.enabled
    out = u.understand(parse_query("lam phatt"))
    assert out.corrections.get("phatt") == "phat"
    assert "phat" in out.effective_text


def test_understander_expansion():
    u = QueryUnderstander(Settings(qu_expansion=True), vocab_provider=lambda: {})
    out = u.understand(parse_query("tin tphcm"))
    assert any("ho chi minh" in e or "sai gon" in e for e in out.expansions)


def test_pipeline_spellcorrect_finds_typo():
    """Với QU_SPELLCORRECT bật, truy vấn gõ sai vẫn tìm ra bài đúng."""
    s = Settings(embedder="hash", vector_backend="local", qu_spellcorrect=True)
    m = IndexManager(s)
    m.index_article({"article_id": "a1", "title": "Lạm phát tháng 6",
                     "body": "chỉ số lạm phát tăng cao", "url": "/a1",
                     "published_at": "2026-07-01T00:00:00+07:00"})
    p = SearchPipeline(m, s)
    # "phatt" gõ thừa 1 ký tự; nhờ spell-correct -> khớp "phát"
    res = p.search(SearchQuery("lam phatt", mode="lexical", now=NOW))
    assert [r.article_id for r in res] == ["a1"]

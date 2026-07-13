"""Test query understanding (GĐ3): spell-correct + expansion + tích hợp pipeline."""

from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone

from news_search.config import Settings
from news_search.index.manager import IndexManager
from news_search.models import SearchQuery
from news_search.search.pipeline import SearchPipeline
from news_search.search.query import parse_query
from news_search.search.understanding import QueryUnderstander, SpellCorrector

NOW = datetime(2026, 7, 7, 12, 0, tzinfo=timezone(timedelta(hours=7)))


def _inject_fake_openai(monkeypatch, response_content="{}", raise_error=False, init_calls=None):
    """Tiêm fake `openai` vào sys.modules (không gọi API thật, không cần cài package).

    Mô phỏng client.chat.completions.create(...) trả JSON string `response_content`,
    hoặc raise lỗi nếu `raise_error=True` (để test đường fallback về local).
    """
    fake = types.ModuleType("openai")

    class FakeMessage:
        def __init__(self, content):
            self.content = content

    class FakeChoice:
        def __init__(self, content):
            self.message = FakeMessage(content)

    class FakeResponse:
        def __init__(self, content):
            self.choices = [FakeChoice(content)]

    class FakeCompletions:
        def create(self, **kwargs):
            if raise_error:
                raise RuntimeError("simulated OpenAI failure")
            return FakeResponse(response_content)

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, api_key=None):
            if init_calls is not None:
                init_calls.append(api_key)
            self.chat = FakeChat()

    fake.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)
    return fake


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
    assert "phat" in out.semantic_text


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


def test_understander_llm_understand_success(monkeypatch):
    payload = json.dumps({
        "corrected_query": "lam phat",
        "corrections": {"phatt": "phat"},
        "key_phrases": ["lam phat"],
        "expansions": ["gia ca"],
        "metadata": {},
    })
    _inject_fake_openai(monkeypatch, payload)
    u = QueryUnderstander(
        Settings(openai_api_key="sk-test", qu_spellcorrect=True, qu_expansion=True),
        vocab_provider=lambda: {},
    )
    out = u.understand(parse_query("lam phatt"), now=NOW)
    assert out.corrections == {"phatt": "phat"}
    assert out.expansions == ["gia ca"]
    assert out.key_phrases == ["lam phat"]
    assert out.semantic_text == "lam phat"          # sạch, KHÔNG chứa expansion
    assert "gia ca" in out.lexical_text              # lexical_text CÓ chứa expansion
    assert "lam phat" in out.lexical_text


def test_understander_llm_understand_error_falls_back(monkeypatch):
    _inject_fake_openai(monkeypatch, raise_error=True)
    u = QueryUnderstander(
        Settings(openai_api_key="sk-test", qu_spellcorrect=True),
        vocab_provider=lambda: {"lam": 5, "phat": 5},
    )
    out = u.understand(parse_query("lam phatt"))
    # OpenAI lỗi -> rơi về local Norvig, vẫn sửa được lỗi chính tả
    assert out.corrections.get("phatt") == "phat"


def test_understander_no_key_skips_llm(monkeypatch):
    init_calls = []
    _inject_fake_openai(monkeypatch, init_calls=init_calls)
    u = QueryUnderstander(
        Settings(qu_spellcorrect=True),  # openai_api_key mặc định rỗng
        vocab_provider=lambda: {"lam": 5, "phat": 5},
    )
    out = u.understand(parse_query("lam phatt"))
    assert init_calls == []  # OpenAI() không được khởi tạo khi thiếu key
    assert out.corrections.get("phatt") == "phat"  # vẫn dùng local fallback


def test_understander_llm_metadata_date_range(monkeypatch):
    payload = json.dumps({
        "corrected_query": "tin hom nay",
        "metadata": {"date_from": "2026-07-06", "date_to": "2026-07-06"},
    })
    _inject_fake_openai(monkeypatch, payload)
    u = QueryUnderstander(
        Settings(openai_api_key="sk-test", qu_spellcorrect=True),
        vocab_provider=lambda: {},
    )
    out = u.understand(parse_query("tin hom nay"), now=NOW)
    assert out.metadata.date_from == datetime(2026, 7, 6, 0, 0, tzinfo=NOW.tzinfo)
    assert out.metadata.date_to == datetime(2026, 7, 6, 23, 59, 59, 999999, tzinfo=NOW.tzinfo)


def test_understander_llm_malformed_date_degrades(monkeypatch):
    payload = json.dumps({
        "corrected_query": "lam phat",
        "corrections": {"phatt": "phat"},
        "metadata": {"date_from": "not-a-date"},
    })
    _inject_fake_openai(monkeypatch, payload)
    u = QueryUnderstander(
        Settings(openai_api_key="sk-test", qu_spellcorrect=True),
        vocab_provider=lambda: {},
    )
    out = u.understand(parse_query("lam phatt"), now=NOW)
    # Ngày hỏng -> None, nhưng phần còn lại của kết quả LLM vẫn giữ nguyên
    assert out.metadata.date_from is None
    assert out.corrections == {"phatt": "phat"}
    assert out.semantic_text == "lam phat"


def test_understander_llm_category_entities_informational_only(monkeypatch):
    payload = json.dumps({
        "corrected_query": "lam phat thang 6",
        "metadata": {"category": "kinh te", "entities": ["Ngan hang Nha nuoc"]},
    })
    _inject_fake_openai(monkeypatch, payload)
    u = QueryUnderstander(
        Settings(openai_api_key="sk-test", qu_spellcorrect=True),
        vocab_provider=lambda: {},
    )
    out = u.understand(parse_query("lam phat thang 6"), now=NOW)
    assert out.metadata.category == "kinh te"
    assert out.metadata.entities == ["Ngan hang Nha nuoc"]
    # category/entities KHÔNG rò rỉ vào lexical_text/semantic_text
    assert "kinh te" not in out.lexical_text
    assert "Ngan hang Nha nuoc" not in out.lexical_text

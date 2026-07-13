# OpenAI Query Understanding — Prompt Redesign + Metadata Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the OpenAI query-understanding prompt to also extract `key_phrases` and structured `metadata` (category/entities/date range), replace the naive text-merge with two purpose-built query strings (clean `semantic_text` for embeddings, enriched `lexical_text` for BM25/hybrid), and route the extracted date range into the pipeline's existing `SearchFilters` mechanism instead of dumping everything into free text.

**Architecture:** `QueryUnderstander.understand()`/`_llm_understand()` in `news_search/search/understanding.py` return a restructured `UnderstoodQuery` (new fields: `lexical_text`, `semantic_text`, `key_phrases`, `metadata: QueryMetadata`). `SearchPipeline._run()`/`_retrieve()` in `news_search/search/pipeline.py` consume the two text variants separately per retrieval branch, and auto-fill `SearchQuery.filters.date_from`/`date_to` from `metadata` only when the caller left them unset. `category`/`entities` stay informational-only (surfaced in `explain()` trace notes, never used to filter).

**Tech Stack:** Python 3.12, existing `openai>=1.30` client (already installed), pytest 9.1.1 + `monkeypatch`.

## Global Constraints

- `effective_text` is fully replaced by `lexical_text` + `semantic_text` — confirmed via repo-wide grep it's referenced only in `news_search/search/understanding.py`, `news_search/search/pipeline.py`, and `tests/test_understanding.py`; all three are updated in this plan.
- `category`/`entities` metadata NEVER populate `SearchQuery.filters` — only `date_from`/`date_to` do, and only when the corresponding filter field is currently `None`.
- No new `QU_*` settings flag — `key_phrases`/`metadata` extraction is requested in the same OpenAI call whenever `_llm_understand` already fires under the existing `QU_SPELLCORRECT`/`QU_EXPANSION`/`QU_LLM_REWRITE` flags.
- A malformed/unparseable date string from the LLM must degrade to `None` for that field alone — never crash, never discard the rest of a good LLM result.
- Design doc: `docs/superpowers/specs/2026-07-13-openai-query-metadata-flow-design.md` (read for full rationale/tradeoffs).
- Do not commit — leave the working tree for the user to review, per this session's established preference.

---

### Task 1: Redesign `UnderstoodQuery`/`_llm_understand` in `understanding.py`

**Files:**
- Modify: `news_search/search/understanding.py`
- Test: `tests/test_understanding.py`

**Interfaces:**
- Consumes: nothing new — same `Settings` fields (`openai_api_key`, `qu_spellcorrect`, `qu_expansion`, `qu_llm_rewrite`, `qu_max_edit_distance`, `synonyms_path`) and `ParsedQuery` (`raw`, `normalized`, `folded`, `tokens`, `fresh_intent`) as today.
- Produces (consumed by Task 2): `QueryMetadata(category: Optional[str], entities: list[str], date_from: Optional[datetime], date_to: Optional[datetime])`; `UnderstoodQuery(lexical_text: str, semantic_text: str, corrections: dict[str,str], expansions: list[str], key_phrases: list[str], metadata: QueryMetadata, rewritten: bool)`; `QueryUnderstander.understand(self, parsed: ParsedQuery, now: Optional[datetime] = None) -> UnderstoodQuery`.

- [ ] **Step 1: Update the 3 existing LLM-path tests for the new response shape and field names**

In `tests/test_understanding.py`, replace `test_understander_llm_understand_success`:
```python
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
```

Leave `test_understander_llm_understand_error_falls_back` and `test_understander_no_key_skips_llm` bodies as-is (they only assert on `.corrections`, unaffected by the field rename) — no edit needed for those two.

- [ ] **Step 2: Run the updated test to verify it fails against the current code**

Run: `.venv/bin/pytest tests/test_understanding.py::test_understander_llm_understand_success -v`
Expected: FAIL with `AttributeError: 'UnderstoodQuery' object has no attribute 'semantic_text'` (or similar — old `UnderstoodQuery` only has `effective_text`).

- [ ] **Step 3: Also update the two other places `.effective_text` is asserted**

In `tests/test_understanding.py`, change `test_understander_spellcorrect`:
```python
    assert "phat" in out.effective_text
```
to:
```python
    assert "phat" in out.semantic_text
```

- [ ] **Step 4: Add the new metadata-focused tests**

Append to `tests/test_understanding.py` (after `test_understander_llm_understand_success`):
```python
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
```

- [ ] **Step 5: Run all new/updated tests to verify they still fail (RED) against current code**

Run: `.venv/bin/pytest tests/test_understanding.py -v -k "llm or spellcorrect"`
Expected: multiple `FAIL`s (`AttributeError` on `.semantic_text`/`.lexical_text`/`.metadata`, or `TypeError: understand() got an unexpected keyword argument 'now'`).

- [ ] **Step 6: Rewrite `news_search/search/understanding.py` with the new data model, prompt, and merge/date-parsing logic**

Replace the entire file content with:
```python
"""(GĐ3) Query understanding: sửa lỗi chính tả + mở rộng đồng nghĩa + rewrite LLM.

Tất cả bật/tắt qua cờ (mặc định TẮT). Nguyên tắc "responsible":
- Chỉ SỬA token OOV (không có trong vocab corpus), giữ nguyên token đúng -> ít
  rủi ro làm hỏng truy vấn tốt.
- Sửa lỗi làm trên bản FOLDED (khớp chỉ mục không dấu); token đúng giữ nguyên chữ
  có dấu để snippet/BGE không mất thông tin.
- Thiếu OpenAI/khóa -> bỏ qua rewrite (degrade), không crash.
- metadata (category/entities) chỉ mang tính tham khảo — KHÔNG auto-filter; chỉ
  date_from/date_to auto-fill vào SearchFilters khi người dùng CHƯA khai báo.

Corrector: thuật toán Norvig (edit-distance 1..2) trên từ điển tần suất lấy từ
chỉ mục BM25 -> không cần từ điển ngoài, tự thích ứng theo kho bài.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from news_search.config import Settings
from news_search.ingest.tokenizer import fold_diacritics
from news_search.models import ParsedQuery

_log = logging.getLogger(__name__)


# Ký tự dùng sinh biến thể (token đã fold -> chỉ còn a-z0-9)
_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"

# Từ đồng nghĩa/alias báo chí VN (dạng folded) — mở rộng truy vấn khi QU_EXPANSION=on
_BUILTIN_SYNONYMS: dict[str, list[str]] = {
    "covid": ["corona", "sars cov 2", "dich covid"],
    "corona": ["covid"],
    "tphcm": ["tp hcm", "ho chi minh", "sai gon"],
    "hcm": ["ho chi minh", "sai gon", "tphcm"],
    "ha noi": ["hn", "thu do"],
    "oto": ["xe hoi", "o to"],
    "bds": ["bat dong san", "nha dat"],
    "ntd": ["nguoi tieu dung"],
    "usd": ["do la", "dong bac xanh"],
    "lai suat": ["lai vay"],
}


def _edits1(word: str) -> set[str]:
    """Tập biến thể cách 1 phép sửa (Norvig)."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in _ALPHABET]
    inserts = [a + c + b for a, b in splits for c in _ALPHABET]
    return set(deletes + transposes + replaces + inserts)


def _dedup_merge(base: str, candidates: list[str]) -> str:
    """Nối thêm các cụm CHƯA xuất hiện trong base (so sánh không phân biệt hoa/thường)."""
    text = base
    for c in candidates:
        if c and c.lower() not in text.lower():
            text = f"{text} {c}"
    return text.strip()


def _parse_date_bound(date_str: Optional[str], now: datetime, end_of_day: bool) -> Optional[datetime]:
    """Chuyển 'YYYY-MM-DD' (LLM trả) -> datetime có tzinfo khớp SearchFilters; None nếu rỗng/sai định dạng."""
    if not date_str:
        return None
    try:
        d = datetime.fromisoformat(date_str).date()
    except ValueError:
        return None
    if end_of_day:
        return datetime(d.year, d.month, d.day, 23, 59, 59, 999999, tzinfo=now.tzinfo)
    return datetime(d.year, d.month, d.day, tzinfo=now.tzinfo)


class SpellCorrector:
    """Sửa lỗi chính tả trên vocab tần suất (folded). Chỉ sửa từ OOV."""

    def __init__(self, vocab: dict[str, int], max_edit_distance: int = 2) -> None:
        self.vocab = vocab
        self.max_edit_distance = max_edit_distance

    def _known(self, words) -> set[str]:
        return {w for w in words if w in self.vocab}

    def correct(self, word: str) -> str:
        """Trả từ đúng nhất (theo tần suất) trong vocab; giữ nguyên nếu đã đúng/không sửa được."""
        if not word or word in self.vocab:
            return word
        candidates = self._known(_edits1(word))
        if not candidates and self.max_edit_distance >= 2:
            candidates = self._known(
                e2 for e1 in _edits1(word) for e2 in _edits1(e1)
            )
        if not candidates:
            return word
        return max(candidates, key=lambda w: self.vocab.get(w, 0))


@dataclass
class QueryMetadata:
    """Metadata trích xuất từ truy vấn — CHỈ mang tính tham khảo (không auto-filter),
    trừ date_from/date_to (auto-fill vào SearchFilters nếu người dùng chưa khai báo)."""

    category: Optional[str] = None
    entities: list[str] = field(default_factory=list)
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


@dataclass
class UnderstoodQuery:
    """Kết quả hiểu truy vấn."""

    lexical_text: str                                           # dùng cho BM25/lexical + phần sparse của hybrid
    semantic_text: str                                          # dùng cho embedding (sạch, không pha loãng)
    corrections: dict[str, str] = field(default_factory=dict)  # token -> sửa thành
    expansions: list[str] = field(default_factory=list)        # token thêm vào
    key_phrases: list[str] = field(default_factory=list)       # cụm từ quan trọng trong câu đã sửa
    metadata: QueryMetadata = field(default_factory=QueryMetadata)
    rewritten: bool = False


class QueryUnderstander:
    """Điều phối hiểu truy vấn theo cờ; vocab lấy lười (lazy) từ chỉ mục."""

    def __init__(
        self,
        settings: Settings,
        vocab_provider: Optional[Callable[[], dict[str, int]]] = None,
    ) -> None:
        self.settings = settings
        self._vocab_provider = vocab_provider
        self._corrector: Optional[SpellCorrector] = None
        self.synonyms = dict(_BUILTIN_SYNONYMS)
        if settings.synonyms_path:
            try:
                with open(settings.synonyms_path, encoding="utf-8") as f:
                    for k, v in json.load(f).items():
                        self.synonyms[fold_diacritics(k).lower()] = [
                            fold_diacritics(s).lower() for s in v
                        ]
            except Exception:  # file lỗi -> bỏ qua, dùng builtin
                pass

    @property
    def enabled(self) -> bool:
        s = self.settings
        return s.qu_spellcorrect or s.qu_expansion or s.qu_llm_rewrite

    def reset(self) -> None:
        """Buộc dựng lại corrector (gọi sau reindex khi vocab đổi)."""
        self._corrector = None

    def _get_corrector(self) -> Optional[SpellCorrector]:
        if not self.settings.qu_spellcorrect or self._vocab_provider is None:
            return None
        if self._corrector is None:
            self._corrector = SpellCorrector(
                self._vocab_provider(), self.settings.qu_max_edit_distance
            )
        return self._corrector

    def understand(self, parsed: ParsedQuery, now: Optional[datetime] = None) -> UnderstoodQuery:
        """Sinh truy vấn hiệu dụng từ ParsedQuery theo các cờ đang bật."""
        now = now or datetime.now(timezone.utc)

        # Ưu tiên gọi OpenAI API nếu có API key và bật các cờ tương ứng
        if self.settings.openai_api_key and (
            self.settings.qu_spellcorrect or self.settings.qu_expansion or self.settings.qu_llm_rewrite
        ):
            llm_res = self._llm_understand(parsed.normalized, now)
            if llm_res is not None:
                return llm_res

        # Fallback về local Norvig + synonyms khi không có OpenAI key hoặc API gọi lỗi
        corrector = self._get_corrector()
        out_tokens: list[str] = []
        corrections: dict[str, str] = {}
        expansions: list[str] = []

        for tok in parsed.tokens:
            folded = fold_diacritics(tok).lower()
            if corrector is not None and folded not in corrector.vocab:
                fixed = corrector.correct(folded)
                if fixed != folded:
                    corrections[tok] = fixed
                    out_tokens.append(fixed)  # dùng bản đã sửa (folded)
                else:
                    out_tokens.append(tok)
            else:
                out_tokens.append(tok)  # giữ nguyên chữ có dấu
            if self.settings.qu_expansion and folded in self.synonyms:
                for syn in self.synonyms[folded]:
                    expansions.append(syn)

        semantic_text = " ".join(out_tokens).strip() or parsed.normalized
        lexical_text = " ".join(out_tokens + expansions).strip() or parsed.normalized
        rewritten = False
        if self.settings.qu_llm_rewrite:
            new_text = self._llm_rewrite(lexical_text)
            if new_text:
                lexical_text = new_text
                semantic_text = new_text
                rewritten = True

        return UnderstoodQuery(
            lexical_text=lexical_text, semantic_text=semantic_text,
            corrections=corrections, expansions=expansions, rewritten=rewritten,
        )

    def _llm_understand(self, text: str, now: datetime) -> Optional[UnderstoodQuery]:
        """Sử dụng OpenAI để sửa lỗi chính tả, mở rộng truy vấn, trích cụm từ khóa
        và metadata (chủ đề/thực thể/khoảng ngày) trong một API call."""
        try:
            from openai import OpenAI

            key = self.settings.openai_api_key
            if not key:
                return None
            client = OpenAI(api_key=key)

            tasks = []
            if self.settings.qu_spellcorrect:
                tasks.append("- Phát hiện và sửa lỗi chính tả, lỗi gõ phím tiếng Việt; chỉ sửa từ sai, giữ nguyên từ đã đúng.")
            if self.settings.qu_expansion:
                tasks.append("- Đề xuất tối đa 5 từ hoặc cụm từ đồng nghĩa/viết tắt tương đương để mở rộng phạm vi tìm kiếm.")
            if self.settings.qu_llm_rewrite:
                tasks.append("- Viết lại truy vấn cho rõ ràng, mạch lạc hơn, giữ nguyên ý định tìm kiếm ban đầu.")

            if not tasks:
                return None

            tasks.append("- Trích xuất tối đa 5 cụm từ khóa quan trọng (từ 2 từ trở lên) có sẵn trong câu đã sửa — ưu tiên tên riêng, thuật ngữ chuyên ngành.")
            tasks.append("- Nhận diện metadata nếu có: chủ đề/thể loại tin tức, thực thể (người/tổ chức/địa điểm), khoảng thời gian được nhắc đến.")

            prompt_instructions = "\n".join(tasks)
            system_prompt = (
                "Bạn là trợ lý tối ưu hóa truy vấn tìm kiếm tin tức tiếng Việt.\n"
                "Với câu truy vấn của người dùng, hãy thực hiện các nhiệm vụ sau:\n"
                f"{prompt_instructions}\n\n"
                f"Hôm nay là {now.date().isoformat()}. Dùng mốc này để quy đổi thời gian tương đối "
                "(vd: \"hôm nay\", \"tuần trước\", \"tháng này\") sang ngày cụ thể.\n\n"
                "Trả về DUY NHẤT một đối tượng JSON đúng cấu trúc sau, không giải thích thêm:\n"
                "{\n"
                "  \"corrected_query\": \"câu truy vấn sau khi sửa lỗi chính tả (giữ nguyên nếu không có lỗi)\",\n"
                "  \"corrections\": {\"từ_gõ_sai\": \"từ_gõ_đúng\"},\n"
                "  \"key_phrases\": [\"cụm từ quan trọng trong câu đã sửa\"],\n"
                "  \"expansions\": [\"từ/cụm đồng nghĩa hoặc viết tắt tương đương\"],\n"
                "  \"metadata\": {\n"
                "    \"category\": \"chủ đề tin tức nếu rõ ràng, null nếu không chắc\",\n"
                "    \"entities\": [\"tên người/tổ chức/địa điểm được nhắc đến\"],\n"
                "    \"date_from\": \"YYYY-MM-DD hoặc null\",\n"
                "    \"date_to\": \"YYYY-MM-DD hoặc null\"\n"
                "  }\n"
                "}\n"
                "Trường không áp dụng để trống (\"\", [], null) — không bịa dữ liệu."
            )

            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                max_tokens=384,
                temperature=0.0,
                response_format={"type": "json_object"}
            )

            res_content = (resp.choices[0].message.content or "").strip()
            data = json.loads(res_content)

            corrected_query = (data.get("corrected_query") or text).strip()
            corrections = data.get("corrections") or {}
            key_phrases = [p for p in (data.get("key_phrases") or []) if p]
            expansions = [e for e in (data.get("expansions") or []) if e]
            meta_raw = data.get("metadata") or {}

            semantic_text = corrected_query
            lexical_text = _dedup_merge(_dedup_merge(semantic_text, key_phrases), expansions)

            metadata = QueryMetadata(
                category=meta_raw.get("category") or None,
                entities=[e for e in (meta_raw.get("entities") or []) if e],
                date_from=_parse_date_bound(meta_raw.get("date_from"), now, end_of_day=False),
                date_to=_parse_date_bound(meta_raw.get("date_to"), now, end_of_day=True),
            )

            return UnderstoodQuery(
                lexical_text=lexical_text,
                semantic_text=semantic_text,
                corrections=corrections,
                expansions=expansions,
                key_phrases=key_phrases,
                metadata=metadata,
                rewritten=True,
            )
        except Exception as e:
            _log.error(f"Lỗi gọi OpenAI cho Query Understanding: {e}")
            return None

    def _llm_rewrite(self, text: str) -> Optional[str]:  # pragma: no cover - cần API
        """Viết lại truy vấn bằng LLM (OpenAI); thiếu package/khóa -> None (degrade)."""
        try:
            from openai import OpenAI

            key = self.settings.openai_api_key
            if not key:
                return None
            client = OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Viết lại truy vấn tìm kiếm tiếng Việt "
                     "cho rõ ràng, giữ nguyên ý định. Chỉ trả về truy vấn, không giải thích."},
                    {"role": "user", "content": text},
                ],
                max_tokens=60,
                temperature=0.0,
            )
            return (resp.choices[0].message.content or "").strip() or None
        except Exception:
            return None
```

- [ ] **Step 7: Run the full understanding test file to verify all pass**

Run: `.venv/bin/pytest tests/test_understanding.py -v`
Expected: `test_understander_llm_understand_success`, `test_understander_spellcorrect`, `test_understander_llm_metadata_date_range`, `test_understander_llm_malformed_date_degrades`, `test_understander_llm_category_entities_informational_only`, plus all pre-existing tests — ALL PASS. `test_pipeline_spellcorrect_finds_typo` will likely FAIL at this point (pipeline.py not yet updated for the new field names) — that's expected here, fixed in Task 2.

---

### Task 2: Wire `pipeline.py` to the new `lexical_text`/`semantic_text`/`metadata` shape

**Files:**
- Modify: `news_search/search/pipeline.py`
- Test: `tests/test_understanding.py`

**Interfaces:**
- Consumes: `UnderstoodQuery`/`QueryMetadata` from Task 1 (`news_search.search.understanding`).
- Produces: no new public interface — `SearchPipeline.search()`/`.explain()` signatures unchanged.

- [ ] **Step 1: Confirm Task 1 broke the pipeline test (sanity check before editing)**

Run: `.venv/bin/pytest tests/test_understanding.py::test_pipeline_spellcorrect_finds_typo -v`
Expected: FAIL (pipeline.py still calls `uinfo.effective_text`, which no longer exists after Task 1).

- [ ] **Step 2: Add the new pipeline-level date-autofill test**

In `tests/test_understanding.py`, update the import line:
```python
from news_search.models import SearchQuery
```
to:
```python
from news_search.models import SearchFilters, SearchQuery
```
and:
```python
from news_search.search.understanding import QueryUnderstander, SpellCorrector
```
to:
```python
from news_search.search.understanding import QueryMetadata, QueryUnderstander, SpellCorrector, UnderstoodQuery
```

Then append at the end of the file:
```python
def test_pipeline_date_metadata_autofills_when_unset(monkeypatch):
    s = Settings(embedder="hash", vector_backend="local", qu_spellcorrect=True)
    m = IndexManager(s)
    m.index_article({"article_id": "old", "title": "Lam phat cu",
                     "body": "tin cu ve kinh te", "url": "/old",
                     "published_at": "2020-01-01T00:00:00+07:00"})
    m.index_article({"article_id": "new", "title": "Lam phat moi",
                     "body": "tin moi ve kinh te", "url": "/new",
                     "published_at": "2026-07-07T00:00:00+07:00"})
    p = SearchPipeline(m, s)

    fixed_date_from = datetime(2026, 7, 6, 0, 0, tzinfo=timezone(timedelta(hours=7)))
    fixed_date_to = datetime(2026, 7, 7, 23, 59, 59, 999999, tzinfo=timezone(timedelta(hours=7)))
    fake_result = UnderstoodQuery(
        lexical_text="lam phat", semantic_text="lam phat",
        metadata=QueryMetadata(date_from=fixed_date_from, date_to=fixed_date_to),
    )
    monkeypatch.setattr(p.understander, "understand", lambda parsed, now=None: fake_result)

    # Chưa khai báo date_from/date_to -> LLM tự điền -> chỉ còn bài "new"
    res = p.search(SearchQuery("lam phat", mode="lexical", now=NOW))
    assert [r.article_id for r in res] == ["new"]

    # Đã khai báo date_from riêng -> giữ nguyên, LLM KHÔNG ghi đè
    user_date_from = datetime(2019, 1, 1, tzinfo=timezone(timedelta(hours=7)))
    res2 = p.search(SearchQuery(
        "lam phat", mode="lexical", now=NOW,
        filters=SearchFilters(date_from=user_date_from),
    ))
    assert {r.article_id for r in res2} == {"old", "new"}
```

- [ ] **Step 3: Run the new test to verify it fails (pipeline.py not updated yet)**

Run: `.venv/bin/pytest tests/test_understanding.py::test_pipeline_date_metadata_autofills_when_unset -v`
Expected: FAIL (`AttributeError: 'UnderstoodQuery' object has no attribute 'effective_text'` inside `pipeline.py`, or the lambda's `now=` kwarg being rejected by the old `understand(self, parsed)` call site).

- [ ] **Step 4: Update imports in `news_search/search/pipeline.py`**

Change:
```python
from __future__ import annotations

from datetime import datetime, timezone

from news_search.config import Settings
```
to:
```python
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from news_search.config import Settings
```

- [ ] **Step 5: Update `_cache_key`**

Change:
```python
    def _cache_key(self, query: SearchQuery, effective_text: str) -> str:
        """Khóa cache: gồm generation chỉ mục -> mọi thay đổi index tự vô hiệu cache."""
        f = query.filters
        return "|".join(str(x) for x in (
            self.manager.generation, query.mode, query.top_k, effective_text,
            f.author, f.category, f.source, f.date_from, f.date_to,
        ))
```
to:
```python
    def _cache_key(self, query: SearchQuery, lexical_text: str, semantic_text: str) -> str:
        """Khóa cache: gồm generation chỉ mục -> mọi thay đổi index tự vô hiệu cache."""
        f = query.filters
        return "|".join(str(x) for x in (
            self.manager.generation, query.mode, query.top_k, lexical_text, semantic_text,
            f.author, f.category, f.source, f.date_from, f.date_to,
        ))
```

- [ ] **Step 6: Update the start of `_run` — hoist `now`, split text, add date auto-fill step**

Change:
```python
    def _run(self, query: SearchQuery, trace: list | None = None) -> list[SearchResultItem]:
        """Pipeline dùng chung cho search() và explain(). trace != None -> ghi bước."""
        cfg = self.settings
        explain = trace is not None

        # 1. Query understanding (rỗng -> ValueError, propagate lên caller)
        print("[DEBUG] pipeline._run: step 1 query.text ", query.text, flush=True)
        parsed = parse_query(query.text)
        if not parsed.tokens:
            if explain:
                self._append(trace, "understand", "1. Hiểu truy vấn",
                             "Không có token hợp lệ trong truy vấn -> trả rỗng")
            return []

        # 1b. Query understanding (mặc định tắt -> giữ nguyên query.text)
        print("[DEBUG] pipeline._run: step 1b understander.enabled ", self.understander.enabled, flush=True)
        effective_text = query.text
        try:
            uinfo = self.understander.understand(parsed) if self.understander.enabled else None
            print("[DEBUG] pipeline._run: uinfo =", uinfo, flush=True)
        except Exception as e:
            print("[DEBUG] pipeline._run: understander error:", e, flush=True)
            raise
        if uinfo is not None:
            effective_text = uinfo.effective_text
        if explain:
            note = f"tokens={parsed.tokens} · fresh_intent={parsed.fresh_intent}"
            if uinfo is not None:
                note += (f" · sửa lỗi={uinfo.corrections or '∅'} · mở rộng={uinfo.expansions or '∅'}"
                         f" · truy vấn hiệu dụng='{effective_text}'")
            else:
                note += " · (query-understanding tắt)"
            self._append(trace, "understand", "1. Hiểu truy vấn", note)

        # 1c. Cache (CHỈ khi không explain -> explain luôn tính mới để quan sát)
        cache_key = None
        if not explain and self.cache is not None and not (
            parsed.fresh_intent and cfg.cache_bypass_fresh
        ):
            cache_key = self._cache_key(query, effective_text)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return [SearchResultItem(**d) for d in cached]
```
to:
```python
    def _run(self, query: SearchQuery, trace: list | None = None) -> list[SearchResultItem]:
        """Pipeline dùng chung cho search() và explain(). trace != None -> ghi bước."""
        cfg = self.settings
        explain = trace is not None
        now = query.now or datetime.now(timezone.utc)

        # 1. Query understanding (rỗng -> ValueError, propagate lên caller)
        print("[DEBUG] pipeline._run: step 1 query.text ", query.text, flush=True)
        parsed = parse_query(query.text)
        if not parsed.tokens:
            if explain:
                self._append(trace, "understand", "1. Hiểu truy vấn",
                             "Không có token hợp lệ trong truy vấn -> trả rỗng")
            return []

        # 1b. Query understanding (mặc định tắt -> giữ nguyên query.text)
        print("[DEBUG] pipeline._run: step 1b understander.enabled ", self.understander.enabled, flush=True)
        lexical_text = query.text
        semantic_text = query.text
        try:
            uinfo = self.understander.understand(parsed, now=now) if self.understander.enabled else None
            print("[DEBUG] pipeline._run: uinfo =", uinfo, flush=True)
        except Exception as e:
            print("[DEBUG] pipeline._run: understander error:", e, flush=True)
            raise
        if uinfo is not None:
            lexical_text = uinfo.lexical_text
            semantic_text = uinfo.semantic_text
        if explain:
            note = f"tokens={parsed.tokens} · fresh_intent={parsed.fresh_intent}"
            if uinfo is not None:
                note += (f" · sửa lỗi={uinfo.corrections or '∅'} · cụm từ={uinfo.key_phrases or '∅'}"
                         f" · mở rộng={uinfo.expansions or '∅'}"
                         f" · truy vấn lexical='{lexical_text}' · truy vấn semantic='{semantic_text}'")
                md = uinfo.metadata
                if md.category or md.entities or md.date_from or md.date_to:
                    note += (f" · metadata: chủ_đề={md.category or '∅'}, thực_thể={md.entities or '∅'}"
                             f", từ_ngày={md.date_from or '∅'}, đến_ngày={md.date_to or '∅'}")
            else:
                note += " · (query-understanding tắt)"
            self._append(trace, "understand", "1. Hiểu truy vấn", note)

        # 1c. Áp khoảng ngày do LLM suy luận — CHỈ khi người dùng CHƯA khai báo
        if uinfo is not None and (uinfo.metadata.date_from or uinfo.metadata.date_to):
            new_date_from = query.filters.date_from or uinfo.metadata.date_from
            new_date_to = query.filters.date_to or uinfo.metadata.date_to
            if new_date_from != query.filters.date_from or new_date_to != query.filters.date_to:
                query = replace(query, filters=replace(
                    query.filters, date_from=new_date_from, date_to=new_date_to
                ))
                if explain:
                    self._append(trace, "understand_dates", "1c. Khoảng ngày suy luận",
                                 f"date_from={new_date_from}, date_to={new_date_to} "
                                 "(LLM suy luận, người dùng chưa khai báo)")

        # 1d. Cache (CHỈ khi không explain -> explain luôn tính mới để quan sát)
        cache_key = None
        if not explain and self.cache is not None and not (
            parsed.fresh_intent and cfg.cache_bypass_fresh
        ):
            cache_key = self._cache_key(query, lexical_text, semantic_text)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return [SearchResultItem(**d) for d in cached]
```

- [ ] **Step 7: Update the retrieval call site in `_run`**

Change:
```python
        # 3-4. Retrieval + fusion (ghi bước bên trong _retrieve)
        print("[DEBUG] pipeline._run: step 3-4 effective_text ", effective_text, flush=True)
        base_scores = self._retrieve(query, effective_text, allowed_ids, trace)
```
to:
```python
        # 3-4. Retrieval + fusion (ghi bước bên trong _retrieve)
        print("[DEBUG] pipeline._run: step 3-4 lexical_text ", lexical_text, "semantic_text", semantic_text, flush=True)
        base_scores = self._retrieve(query, lexical_text, semantic_text, allowed_ids, trace)
```

- [ ] **Step 8: Remove the now-redundant `now` recomputation at the time-decay step**

Change:
```python
        # 5. Time-decay / QDF (F-13)
        time_decay_enabled = query.time_decay if query.time_decay is not None else cfg.time_decay_enabled
        if time_decay_enabled:
            now = query.now or datetime.now(timezone.utc)
            half_life = cfg.fresh_half_life_days if parsed.fresh_intent else cfg.half_life_days
```
to:
```python
        # 5. Time-decay / QDF (F-13)
        time_decay_enabled = query.time_decay if query.time_decay is not None else cfg.time_decay_enabled
        if time_decay_enabled:
            half_life = cfg.fresh_half_life_days if parsed.fresh_intent else cfg.half_life_days
```

- [ ] **Step 9: Update `_retrieve`'s signature and body**

Change:
```python
    def _retrieve(
        self, query: SearchQuery, parsed_text: str, allowed_ids: set[str] | None,
        trace: list | None = None,
    ) -> dict[str, float]:
        """Sinh bảng điểm ứng viên theo chế độ (lexical/semantic/hybrid).

        trace != None -> ghi các bước truy hồi (lexical/dense) và hợp nhất RRF.
        """
        cfg = self.settings
        mode = query.mode
        k = cfg.candidates_k
        explain = trace is not None
        embed_name = type(self.manager.embedder).__name__

        if mode == "lexical":
            hits = self.manager.vector.search_lexical(parsed_text, top_k=k, allowed_ids=allowed_ids)
            if explain:
                self._append(trace, "lexical", "3. Truy hồi từ khóa (Sparse/BM25)",
                             f"{len(hits)} ứng viên", self._trace_items(hits), len(hits))
            return dict(hits)

        if mode == "semantic":
            qvec = self.manager.embedder.embed([parsed_text])[0]
            hits = self.manager.vector.search_semantic(qvec, top_k=k, allowed_ids=allowed_ids)
            if explain:
                self._append(trace, "dense", "3. Truy hồi ngữ nghĩa (Dense Vector)",
                             f"embedder={embed_name}, {len(hits)} ứng viên",
                             self._trace_items(hits), len(hits))
            return dict(hits)

        if mode == "hybrid":
            qvec = self.manager.embedder.embed([parsed_text])[0]
            fused = self.manager.vector.search_hybrid(parsed_text, qvec, top_k=k, allowed_ids=allowed_ids)
            fused_dict = dict(fused)
            if explain:
                self._append(trace, "fusion", "3. Truy hồi Kết hợp (Milvus Hybrid Search)",
                             f"{len(fused_dict)} bài trả về từ Milvus RRF",
                             self._trace_items(sorted(fused_dict.items(), key=lambda kv: -kv[1])), len(fused_dict))
            return fused_dict

        raise ValueError(f"mode không hợp lệ: {query.mode!r} (lexical|semantic|hybrid)")
```
to:
```python
    def _retrieve(
        self, query: SearchQuery, lexical_text: str, semantic_text: str,
        allowed_ids: set[str] | None, trace: list | None = None,
    ) -> dict[str, float]:
        """Sinh bảng điểm ứng viên theo chế độ (lexical/semantic/hybrid).

        trace != None -> ghi các bước truy hồi (lexical/dense) và hợp nhất RRF.
        """
        cfg = self.settings
        mode = query.mode
        k = cfg.candidates_k
        explain = trace is not None
        embed_name = type(self.manager.embedder).__name__

        if mode == "lexical":
            hits = self.manager.vector.search_lexical(lexical_text, top_k=k, allowed_ids=allowed_ids)
            if explain:
                self._append(trace, "lexical", "3. Truy hồi từ khóa (Sparse/BM25)",
                             f"{len(hits)} ứng viên", self._trace_items(hits), len(hits))
            return dict(hits)

        if mode == "semantic":
            qvec = self.manager.embedder.embed([semantic_text])[0]
            hits = self.manager.vector.search_semantic(qvec, top_k=k, allowed_ids=allowed_ids)
            if explain:
                self._append(trace, "dense", "3. Truy hồi ngữ nghĩa (Dense Vector)",
                             f"embedder={embed_name}, {len(hits)} ứng viên",
                             self._trace_items(hits), len(hits))
            return dict(hits)

        if mode == "hybrid":
            qvec = self.manager.embedder.embed([semantic_text])[0]
            fused = self.manager.vector.search_hybrid(lexical_text, qvec, top_k=k, allowed_ids=allowed_ids)
            fused_dict = dict(fused)
            if explain:
                self._append(trace, "fusion", "3. Truy hồi Kết hợp (Milvus Hybrid Search)",
                             f"{len(fused_dict)} bài trả về từ Milvus RRF",
                             self._trace_items(sorted(fused_dict.items(), key=lambda kv: -kv[1])), len(fused_dict))
            return fused_dict

        raise ValueError(f"mode không hợp lệ: {query.mode!r} (lexical|semantic|hybrid)")
```

- [ ] **Step 10: Run the full test_understanding.py file**

Run: `.venv/bin/pytest tests/test_understanding.py -v`
Expected: all tests PASS, including `test_pipeline_spellcorrect_finds_typo` and `test_pipeline_date_metadata_autofills_when_unset`.

- [ ] **Step 11: Run the full test suite to confirm no regressions**

Run: `.venv/bin/pytest -q`
Expected: `0 failed` — same baseline pass count as before this change, plus the new tests added in Task 1 and Task 2.

- [ ] **Step 12: Live smoke test with an invalid key (confirms graceful degrade end-to-end)**

Run:
```bash
.venv/bin/python -c "
from news_search.config import Settings
from news_search.search.understanding import QueryUnderstander
from news_search.search.query import parse_query

u = QueryUnderstander(Settings(openai_api_key='sk-invalid-test-key', qu_spellcorrect=True, qu_expansion=True), vocab_provider=lambda: {})
out = u.understand(parse_query('lam phatt hom nay'))
print('lexical_text:', out.lexical_text)
print('semantic_text:', out.semantic_text)
print('metadata:', out.metadata)
print('no crash - degraded gracefully as expected with an invalid key')
"
```
Expected: a logged 401 error from the (real, installed) `openai` client, followed by `lexical_text`/`semantic_text` both equal to the original query text, `metadata` at its all-`None`/empty defaults, and the final print line — confirming the whole redesigned flow still degrades to the local fallback without crashing.

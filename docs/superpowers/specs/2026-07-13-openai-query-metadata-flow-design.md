# OpenAI Query Understanding — Prompt Redesign + Metadata Flow

## Overview

`QueryUnderstander._llm_understand` (in `news_search/search/understanding.py`) currently asks OpenAI for `corrected_query` + `corrections` + `expansions`, then builds one `effective_text` string by naively space-joining `corrected_query` and any expansion words not already present. That single string is fed identically into both lexical/BM25 and semantic/embedding retrieval in `pipeline.py`.

This redesign: (1) rewrites the prompt to also extract `key_phrases` (important multi-word terms already present in the corrected query) and structured `metadata` (category, entities, date range), in professional/concise/clear Vietnamese wording; (2) replaces the naive text-merge with two purpose-built query strings — a clean one for semantic search, an enriched one for lexical search; (3) routes the extracted date range into the pipeline's existing `SearchFilters` mechanism instead of dumping it into free text; (4) leaves category/entities as informational metadata only (surfaced in `explain()`, never auto-filtered).

## Goals

- Prompt asks for `key_phrases` and `metadata` (`category`, `entities`, `date_from`, `date_to`) alongside the existing `corrected_query`/`corrections`/`expansions`.
- Two query strings instead of one: `semantic_text` (clean, for embeddings) and `lexical_text` (enriched with key phrases + expansions, for BM25/hybrid).
- LLM-derived date range auto-fills `SearchQuery.filters.date_from`/`date_to` **only when the caller left them unset**.
- Category/entities are exposed in `explain()` trace notes for visibility, never used to filter results.
- Local (non-LLM) fallback path keeps working unchanged in spirit — it already naturally separates corrected tokens from synonym expansions, so it can produce both text variants with no new logic.
- Everything continues to degrade gracefully: a bad/missing API key, a malformed date string, or an OpenAI error must never crash the request — only narrow what's lost (e.g. an unparseable date just leaves `date_from`/`date_to` as `None`, not the whole LLM result).

## Non-Goals (explicitly out of scope for this change)

- Auto-applying `category` or `entities` as hard filters — no reliable taxonomy validation exists, and a wrong guess would zero out good results (violates this module's existing "never risk breaking a good query" principle).
- Cross-checking LLM-extracted entities against the existing ingest-time NER (`index/entities.py`) or its `KnowledgeGraph` — worth a follow-up, not here.
- Building phrase-quoted/boosted BM25 queries — the lexical backend's `search_lexical(text, top_k, allowed_ids)` takes plain text; `key_phrases` just get merged into `lexical_text` as extra terms, same mechanism as expansions.
- Fixing the pre-existing `rewritten` field bug (`_llm_understand` always sets `rewritten=True` even when only spellcheck/expansion ran) — confirmed dead/unused downstream in an earlier investigation; still out of scope.
- Any new `QU_*` settings flag — `key_phrases`/`metadata` extraction happens automatically whenever the LLM call already fires (same call, no extra cost), gated by the existing `QU_SPELLCORRECT`/`QU_EXPANSION`/`QU_LLM_REWRITE` flags exactly as today.

## Data Model (`news_search/search/understanding.py`)

```python
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
    corrections: dict[str, str] = field(default_factory=dict)
    expansions: list[str] = field(default_factory=list)
    key_phrases: list[str] = field(default_factory=list)
    metadata: QueryMetadata = field(default_factory=QueryMetadata)
    rewritten: bool = False
```

`effective_text` is removed (renamed/split into `lexical_text` + `semantic_text`). Confirmed via repo-wide grep that only `pipeline.py` and `tests/test_understanding.py` reference `.effective_text` — both are updated as part of this change, so nothing else breaks.

## Prompt Redesign

System prompt (built in `_llm_understand`), professional/concise/clear Vietnamese, grounded with the caller-supplied `now` so relative time expressions resolve to real dates:

```python
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
```

`max_tokens` raised from `256` to `384` to comfortably fit the larger schema without truncation risk.

## Merge Logic (replaces "phần gộp chữ hiện tại")

```python
def _dedup_merge(base: str, candidates: list[str]) -> str:
    """Nối thêm các cụm CHƯA xuất hiện trong base (so sánh không phân biệt hoa/thường)."""
    text = base
    for c in candidates:
        if c and c.lower() not in text.lower():
            text = f"{text} {c}"
    return text.strip()
```

In `_llm_understand`, after parsing the response:
```python
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
    lexical_text=lexical_text, semantic_text=semantic_text,
    corrections=corrections, expansions=expansions,
    key_phrases=key_phrases, metadata=metadata, rewritten=True,
)
```

`_parse_date_bound` (module-level helper, mirrors `_edits1`'s placement style):
```python
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
```
A malformed date string degrades to `None` for that field alone — it never raises, so it can't collapse the rest of a good LLM result via the outer `except Exception` in `_llm_understand`.

## Method Signature Changes

- `QueryUnderstander.understand(self, parsed: ParsedQuery, now: Optional[datetime] = None) -> UnderstoodQuery` — resolves `now = now or datetime.now(timezone.utc)` at the top, passes it to `_llm_understand`.
- `QueryUnderstander._llm_understand(self, text: str, now: datetime) -> Optional[UnderstoodQuery]`.
- `_llm_rewrite` signature unchanged.

## Local Fallback Path (unchanged logic, new output shape)

The existing loop already keeps `out_tokens` (corrected/kept tokens) separate from `expansions` (synonym additions) — that maps directly onto the new split with no new logic:
```python
semantic_text = " ".join(out_tokens).strip() or parsed.normalized
lexical_text = " ".join(out_tokens + expansions).strip() or parsed.normalized
rewritten = False
if self.settings.qu_llm_rewrite:
    new_text = self._llm_rewrite(lexical_text)
    if new_text:
        lexical_text = new_text
        semantic_text = new_text
        rewritten = True

return UnderstoodQuery(lexical_text, semantic_text, corrections, expansions)
```
`key_phrases`/`metadata` stay at their dataclass defaults (`[]`/`QueryMetadata()`) — the local path has no way to extract them without the LLM.

## Pipeline Wiring (`news_search/search/pipeline.py`)

- Add `from dataclasses import replace` to imports.
- Hoist `now = query.now or datetime.now(timezone.utc)` to the top of `_run` (currently computed only inline at the time-decay step); reuse that same `now` for both `understander.understand(parsed, now=now)` and the time-decay step (removes a duplicate `datetime.now()` call, keeps one consistent instant for the whole request).
- Step 1b: `lexical_text`/`semantic_text` replace the single `effective_text` local; default to `query.text` for both when understanding is disabled/unavailable. Explain note extended to show `key_phrases` and, when present, the metadata summary (chủ đề/thực thể/khoảng ngày).
- **New step 1c** (before cache-key computation, so caching stays correct): if `uinfo.metadata.date_from`/`date_to` is set AND the corresponding `query.filters` field is currently `None`, rebuild `query` via `dataclasses.replace(query, filters=dataclasses.replace(query.filters, date_from=..., date_to=...))`. User-supplied filters always win — the LLM only fills gaps.
- `_cache_key(self, query, lexical_text, semantic_text)` — both strings included in the key tuple (the date auto-fill is already covered since it's merged into `query.filters` before this runs, and `f.date_from`/`f.date_to` are already part of the key).
- `_retrieve(self, query, lexical_text, semantic_text, allowed_ids, trace)`:
  - `mode == "lexical"` → `search_lexical(lexical_text, ...)`
  - `mode == "semantic"` → `embedder.embed([semantic_text])`
  - `mode == "hybrid"` → `qvec = embedder.embed([semantic_text])[0]`; `search_hybrid(lexical_text, qvec, ...)` — no backend signature changes needed, `search_hybrid` already takes a separate text arg and vector arg.
- Reranker (step 6, uses `query.text`) and snippet highlighting (step 9, uses `parsed.tokens`) are untouched — neither reads `effective_text` today.

## Testing Plan (`tests/test_understanding.py`)

- Update the 3 existing LLM-path tests (`test_understander_llm_understand_success`, `..._error_falls_back`, `..._no_key_skips_llm`) for the new response shape: mocked payload now includes `key_phrases` and `metadata`; assertions check `semantic_text` excludes expansions/key-phrases while `lexical_text` includes them.
- New test: metadata date range parses into tz-aware `datetime` bounds (`date_from` at 00:00:00, `date_to` at 23:59:59.999999) matching the `now` passed in.
- New test: a malformed `metadata.date_from` string (e.g. `"not-a-date"`) degrades to `None` without discarding `corrected_query`/`corrections`/`expansions` from the same response.
- New test: `category`/`entities` are populated on `UnderstoodQuery.metadata` but this alone doesn't change `lexical_text`/`semantic_text`.
- `tests/test_pipeline.py` (or a new case in `test_understanding.py`'s pipeline-level test): a `SearchQuery` with no `date_from`/`date_to` set, run through a pipeline whose understander is monkeypatched to return a fixed `UnderstoodQuery` with `metadata.date_from` set — assert the effective filters used for retrieval include that date, while a `SearchQuery` that already specifies `date_from` keeps its own value untouched.

## Risks / Tradeoffs

- Larger prompt (~2x the JSON fields) increases OpenAI token cost slightly per call and touches `max_tokens` (256→384) — acceptable given this already only fires when `QU_SPELLCORRECT`/`QU_EXPANSION`/`QU_LLM_REWRITE` is explicitly enabled.
- `key_phrases` merged into `lexical_text` could occasionally duplicate what `expansions` already adds if the LLM's two lists overlap — `_dedup_merge` running twice (once per list, cumulative `base`) already prevents literal duplicates across both.
- Splitting `lexical_text`/`semantic_text` changes `_retrieve`'s public-ish signature inside `pipeline.py` — internal to the class, no external callers found via grep, so this is safe within this repo.

# OpenAI Query Understanding — Enable & Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the already-drafted OpenAI-backed spell-correction + query-expansion path in `news_search/search/understanding.py` (`QueryUnderstander._llm_understand`) actually runnable, and add test coverage for it.

**Architecture:** No new architecture — `QueryUnderstander.understand()` already routes to `_llm_understand()` (one OpenAI `gpt-4o-mini` chat-completion call, JSON response with `corrected_query`/`corrections`/`expansions`) whenever `settings.openai_api_key` is set and at least one of `QU_SPELLCORRECT`/`QU_EXPANSION`/`QU_LLM_REWRITE` is enabled, falling back to the local Norvig+synonym path on any failure. This plan only (1) makes the `openai` package actually importable in the dev venv, and (2) adds regression tests for the previously-uncovered LLM path, using this repo's existing `monkeypatch` + fake-module pattern (see `tests/test_upgrades.py::_inject_fake_st`) so tests never hit the real network.

**Tech Stack:** Python 3.12, pytest 9.1.1 + pytest `monkeypatch` fixture, `openai` Python SDK (>=1.30, optional extra), existing `.venv` (editable install).

## Global Constraints

- Do not touch `pyproject.toml` — the `openai` optional extra (`openai = ["openai>=1.30"]`) is already correctly declared there.
- Do not modify `_llm_understand`'s behavior (including the known `rewritten=True`-always quirk) — out of scope per user decision; local fallback and prompt logic stay as-is.
- Tests must not perform real network calls or require a real `OPENAI_API_KEY` — mock via `sys.modules` injection, matching the existing `_inject_fake_st` idiom in `tests/test_upgrades.py`.
- Do not commit any changes — leave the working tree staged/modified for the user to review and commit themselves.
- Never write the real `OPENAI_API_KEY` value into any file — the user adds it to their local `.env` themselves.

---

### Task 1: Install the `openai` extra into the dev venv

**Files:**
- Modify: `requirements.txt:18`

**Interfaces:**
- Consumes: nothing (infra-only task)
- Produces: a working `import openai` in `.venv`, required by Task 2's `_llm_understand` execution path (though Task 2's tests mock `sys.modules["openai"]`, so they don't strictly depend on this — this task is for real end-to-end use, not test-passing).

- [x] **Step 1: Uncomment the openai line in requirements.txt**

Current (`requirements.txt:14-21`):
```
# Tùy chọn — production backends (cài khi cần, code tự phát hiện qua import guard)
# opensearch-py>=2.6        # LEXICAL_BACKEND=opensearch
# pymilvus>=2.4             # VECTOR_BACKEND=milvus (chỉ mục HNSW)
# neo4j>=5.20               # Knowledge graph production
# openai>=1.30              # EMBEDDER=openai (text-embedding-3-large)
# FlagEmbedding>=1.2        # EMBEDDER=bge (BGE-m3) và/hoặc RERANKER=bge (bge-reranker-v2-m3)
# sentence-transformers>=3  # EMBEDDER=bge (fallback nếu không dùng FlagEmbedding)
# underthesea>=6.8          # Tách từ + NER tiếng Việt nâng cao
```

Change line 18 to:
```
openai>=1.30               # EMBEDDER=openai (text-embedding-3-large) và QU_* (spellcheck/expansion/rewrite qua GPT)
```

- [x] **Step 2: Install the extra into the project's venv**

Run: `.venv/bin/pip install -e ".[openai]"`
Expected: pip resolves and installs `openai>=1.30` (plus its transitive deps like `httpx`, `pydantic`, `tqdm`, `distro`) with no errors, ending in `Successfully installed ... news-search-engine`.

- [x] **Step 3: Verify the import works**

Run: `.venv/bin/python -c "import openai; print(openai.__version__)"`
Expected: prints a version string like `1.5x.x` with no `ModuleNotFoundError`.

---

### Task 2: Add regression tests for `QueryUnderstander._llm_understand`

**Files:**
- Modify: `tests/test_understanding.py`

**Interfaces:**
- Consumes: `QueryUnderstander(settings, vocab_provider)` and `.understand(parsed: ParsedQuery) -> UnderstoodQuery` from `news_search/search/understanding.py` (already implemented — fields `effective_text: str`, `corrections: dict[str,str]`, `expansions: list[str]`, `rewritten: bool`). `parse_query(text: str) -> ParsedQuery` from `news_search/search/query.py`. `Settings` fields `openai_api_key: str`, `qu_spellcorrect: bool`, `qu_expansion: bool` from `news_search/config.py`.
- Produces: nothing consumed by later tasks — this is the terminal task.

- [x] **Step 1: Add imports needed for module-injection mocking**

At the top of `tests/test_understanding.py`, change:
```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from news_search.config import Settings
```
to:
```python
from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timedelta, timezone

from news_search.config import Settings
```

- [x] **Step 2: Add the fake-openai injection helper**

Append after the `NOW = ...` line (before `test_spell_corrector`):
```python
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
```

- [x] **Step 3: Write the failing/new tests**

Append at the end of `tests/test_understanding.py`:
```python
def test_understander_llm_understand_success(monkeypatch):
    payload = json.dumps({
        "corrected_query": "lam phat",
        "corrections": {"phatt": "phat"},
        "expansions": ["gia ca"],
    })
    _inject_fake_openai(monkeypatch, payload)
    u = QueryUnderstander(
        Settings(openai_api_key="sk-test", qu_spellcorrect=True, qu_expansion=True),
        vocab_provider=lambda: {},
    )
    out = u.understand(parse_query("lam phatt"))
    assert out.corrections == {"phatt": "phat"}
    assert out.expansions == ["gia ca"]
    assert "lam phat" in out.effective_text
    assert "gia ca" in out.effective_text


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
```

- [x] **Step 4: Run the new tests to verify they pass**

Run: `.venv/bin/pytest tests/test_understanding.py -v`
Expected: all 8 tests pass (5 pre-existing + 3 new): `test_spell_corrector`, `test_understander_disabled_noop`, `test_understander_spellcorrect`, `test_understander_expansion`, `test_pipeline_spellcorrect_finds_typo`, `test_understander_llm_understand_success`, `test_understander_llm_understand_error_falls_back`, `test_understander_no_key_skips_llm` — `8 passed`.

- [x] **Step 5: Run the full test suite to confirm no regressions**

Run: `.venv/bin/pytest -q`
Expected: same pass count as before this change, plus the 3 new tests, `0 failed`.

---

## Execution Notes

Both tasks completed inline; full suite (`.venv/bin/pytest -q`) is green: 173 passed, 4 skipped (pre-existing, unrelated), 0 failed.

**Bug found and fixed during execution (not originally in this plan):** `understand()` called `self._llm_understand(parsed.text)`, but `ParsedQuery` has no `.text` field (its fields are `raw`/`normalized`/`folded`/`tokens`/`fresh_intent` — see `news_search/models.py:117-124`). This crashed with an unhandled `AttributeError` on every call, outside `_llm_understand`'s own try/except, so the LLM path never degraded gracefully as the module's docstring promises — it always crashed. Root-caused via superpowers:systematic-debugging and fixed at `news_search/search/understanding.py`: `parsed.text` → `parsed.normalized`, matching this same method's existing fallback convention (`... or parsed.normalized`). Verified via a live smoke test with an intentionally invalid API key: the real `openai` client now reaches the network (401 response), and `understand()` degrades cleanly to the local fallback with no crash.

## Post-Plan Manual Step (not automated — requires a real secret)

To actually activate the OpenAI-backed path against live traffic:
1. Add `OPENAI_API_KEY=sk-...` to the local `.env` (not `.env.example` — never commit a real key).
2. Confirm `QU_SPELLCORRECT=true` and/or `QU_EXPANSION=true` are set in `.env` (already true today).
3. Restart the app; `QueryUnderstander.understand()` will now route through `_llm_understand()` automatically for every query where those flags apply.

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

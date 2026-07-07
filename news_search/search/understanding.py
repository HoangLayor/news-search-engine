"""(GĐ3) Query understanding: sửa lỗi chính tả + mở rộng đồng nghĩa + rewrite LLM.

Tất cả bật/tắt qua cờ (mặc định TẮT). Nguyên tắc "responsible":
- Chỉ SỬA token OOV (không có trong vocab corpus), giữ nguyên token đúng -> ít
  rủi ro làm hỏng truy vấn tốt.
- Sửa lỗi làm trên bản FOLDED (khớp chỉ mục không dấu); token đúng giữ nguyên chữ
  có dấu để snippet/BGE không mất thông tin.
- Thiếu OpenAI/khóa -> bỏ qua rewrite (degrade), không crash.

Corrector: thuật toán Norvig (edit-distance 1..2) trên từ điển tần suất lấy từ
chỉ mục BM25 -> không cần từ điển ngoài, tự thích ứng theo kho bài.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from news_search.config import Settings
from news_search.ingest.tokenizer import fold_diacritics
from news_search.models import ParsedQuery

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
class UnderstoodQuery:
    """Kết quả hiểu truy vấn."""

    effective_text: str                       # truy vấn dùng để truy hồi
    corrections: dict[str, str] = field(default_factory=dict)  # token -> sửa thành
    expansions: list[str] = field(default_factory=list)        # token thêm vào
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

    def understand(self, parsed: ParsedQuery) -> UnderstoodQuery:
        """Sinh truy vấn hiệu dụng từ ParsedQuery theo các cờ đang bật."""
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

        effective = " ".join(out_tokens + expansions).strip() or parsed.normalized
        rewritten = False
        if self.settings.qu_llm_rewrite:
            new_text = self._llm_rewrite(effective)
            if new_text:
                effective, rewritten = new_text, True

        return UnderstoodQuery(effective, corrections, expansions, rewritten)

    def _llm_rewrite(self, text: str) -> Optional[str]:  # pragma: no cover - cần API
        """Viết lại truy vấn bằng LLM (OpenAI); thiếu package/khóa -> None (degrade)."""
        try:
            from openai import OpenAI

            key = self.settings.openai_api_key
            if not key:
                return None
            client = OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model="gpt-4.1-mini",
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

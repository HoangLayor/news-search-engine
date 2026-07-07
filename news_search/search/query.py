"""Query Understanding (CONTRACTS.md §7) — phân tích truy vấn + nhận diện ý định tin mới (QDF).

Đầu ra là :class:`news_search.models.ParsedQuery`; ``fresh_intent`` = True khi
truy vấn chứa hint "tin mới" (phục vụ F-13: chọn half-life ngắn hơn).
"""

from __future__ import annotations

import re

from news_search.ingest.tokenizer import fold_diacritics, normalize_text, tokenize
from news_search.models import ParsedQuery

# Các hint "tin mới" ở dạng FOLDED (không dấu, lowercase).
# Token đơn và cụm nhiều từ đều được match theo ranh giới từ (\b) trên chuỗi folded.
FRESH_HINT_TOKENS: frozenset[str] = frozenset(
    {
        "hom nay",
        "moi nhat",
        "moi",
        "vua",
        "vua xay ra",
        "dang",
        "truc tiep",
        "nong",
        "breaking",
        "sang nay",
        "chieu nay",
        "toi nay",
        "tuan nay",
        "live",
    }
)

# Ngoại lệ nhập nhằng sau khi bỏ dấu: "moi" đứng ngay trước "truong"
# (tức "môi trường") KHÔNG phải hint tin mới -> chặn bằng negative lookahead.
_HINT_EXCEPTIONS: dict[str, str] = {
    "moi": r"\bmoi\b(?!\s+truong\b)",
}

# Regex biên dịch sẵn cho từng hint; sorted() để thứ tự duyệt deterministic.
_HINT_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(_HINT_EXCEPTIONS.get(hint, r"\b" + re.escape(hint) + r"\b"))
    for hint in sorted(FRESH_HINT_TOKENS)
)


def _has_fresh_hint(folded: str) -> bool:
    """True nếu chuỗi folded chứa hint tin mới (token đơn hoặc cụm nhiều từ)."""
    return any(pattern.search(folded) for pattern in _HINT_PATTERNS)


def parse_query(text: str) -> ParsedQuery:
    """Phân tích truy vấn thô thành :class:`ParsedQuery`.

    - ``normalized``: NFC, lowercase, gọn khoảng trắng.
    - ``folded``: normalized + bỏ dấu (đ -> d).
    - ``tokens``: kết quả ``tokenize`` (lowercase, còn dấu).
    - ``fresh_intent``: có hint tin mới trong ``folded``.

    Raises:
        ValueError: nếu ``text`` rỗng hoặc toàn whitespace.
    """
    if not text or not text.strip():
        raise ValueError("Truy vấn rỗng")
    normalized = normalize_text(text)
    folded = fold_diacritics(normalized)
    return ParsedQuery(
        raw=text,
        normalized=normalized,
        folded=folded,
        tokens=tokenize(text),
        fresh_intent=_has_fresh_hint(folded),
    )

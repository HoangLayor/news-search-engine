"""Làm sạch HTML & chuẩn hóa bài viết thô thành Article (F-02).

Thuần stdlib: regex + html.unescape, KHÔNG cần bs4. Datetime luôn
timezone-aware; chuỗi/datetime naive được gán UTC+7 (Asia/Ho_Chi_Minh).
"""

from __future__ import annotations

import html as _html
import re
from datetime import datetime, timedelta, timezone

from news_search.models import REQUIRED_FIELDS, Article

# Múi giờ Việt Nam (UTC+7) — dùng timezone cố định theo CONTRACTS.md
TZ_VN = timezone(timedelta(hours=7))

# <script>/<style> cùng toàn bộ nội dung bên trong
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
# Comment HTML <!-- ... -->
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
# Tag tạo ngắt đoạn: <p>, </p>, <div>, </div>, <br>, <br/> (kể cả có thuộc tính)
_BLOCK_BREAK_RE = re.compile(r"</?\s*(?:p|div|br)\b[^>]*/?\s*>", re.IGNORECASE)
# Mọi tag còn lại
_ANY_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
# Phát hiện body có chứa tag HTML hay không
_HAS_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>|<!--")
# Whitespace trong 1 dòng (không gồm \n vì đã split theo dòng)
_INLINE_WS_RE = re.compile(r"\s+")


def clean_html(html: str) -> str:
    """Bỏ script/style + mọi tag, unescape entity, gọn khoảng trắng GIỮ ngắt đoạn.

    <p>, <br>, <div> (mở/đóng) -> "\\n"; các dòng rỗng bị loại.
    """
    if not html:
        return ""
    text = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _COMMENT_RE.sub(" ", text)
    text = _BLOCK_BREAK_RE.sub("\n", text)
    text = _ANY_TAG_RE.sub(" ", text)
    # Unescape SAU khi bỏ tag để "&lt;b&gt;" không bị hiểu nhầm là tag
    text = _html.unescape(text)
    lines = []
    for line in text.split("\n"):
        line = _INLINE_WS_RE.sub(" ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _parse_published_at(value: object) -> datetime:
    """Parse published_at: datetime hoặc chuỗi ISO-8601; naive -> gán UTC+7."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError(
                f"published_at không phải chuỗi ISO-8601 hợp lệ: {value!r}"
            ) from exc
    else:
        raise ValueError(
            "published_at phải là datetime hoặc chuỗi ISO-8601, "
            f"nhận được {type(value).__name__}"
        )
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=TZ_VN)  # naive -> mặc định giờ Việt Nam
    return dt


def _strip_or_none(value: object) -> str | None:
    """Strip chuỗi; rỗng/None -> None (cho field tùy chọn)."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_article(raw: dict) -> Article:
    """Chuẩn hóa dict thô từ CMS/API thành Article.

    - Thiếu field bắt buộc (REQUIRED_FIELDS) -> ValueError nêu rõ field.
    - body chứa tag HTML -> clean_html.
    - published_at: datetime/chuỗi ISO; naive -> UTC+7.
    - title/author/category/source: strip khoảng trắng thừa.
    - status mặc định "published"; tags mặc định [].
    """
    missing = [f for f in REQUIRED_FIELDS if raw.get(f) in (None, "")]
    if missing:
        raise ValueError(f"Thiếu field bắt buộc: {', '.join(missing)}")

    body = str(raw["body"])
    if _HAS_TAG_RE.search(body):
        body = clean_html(body)
    else:
        body = body.strip()

    tags_raw = raw.get("tags") or []
    tags = [str(t).strip() for t in tags_raw if str(t).strip()]

    return Article(
        article_id=str(raw["article_id"]).strip(),
        title=str(raw["title"]).strip(),
        body=body,
        url=str(raw["url"]).strip(),
        published_at=_parse_published_at(raw["published_at"]),
        author=_strip_or_none(raw.get("author")),
        category=_strip_or_none(raw.get("category")),
        source=_strip_or_none(raw.get("source")),
        status=(str(raw.get("status")).strip() if raw.get("status") else "published"),
        tags=tags,
    )

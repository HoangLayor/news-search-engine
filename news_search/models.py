"""Mô hình dữ liệu cốt lõi — hợp đồng chung cho mọi module.

Mọi module khác import từ đây; KHÔNG thêm phụ thuộc ngoài stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# Metadata bắt buộc của một bài viết (BaoCao-ThietKe.md §8.2)
REQUIRED_FIELDS = ("article_id", "title", "body", "url", "published_at")

ENTITY_TYPES = ("PER", "ORG", "LOC", "EVT", "MISC")

SEARCH_MODES = ("lexical", "semantic", "hybrid")


@dataclass
class Article:
    """Một bài viết đã chuẩn hóa (đầu ra của F-02)."""

    article_id: str
    title: str
    body: str
    url: str
    published_at: datetime  # PHẢI timezone-aware
    author: Optional[str] = None
    category: Optional[str] = None
    source: Optional[str] = None
    status: str = "published"  # published | unpublished
    tags: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Văn bản dùng để index (tiêu đề + nội dung)."""
        return f"{self.title}\n{self.body}"


@dataclass
class SearchFilters:
    """Bộ lọc metadata (F-12). So sánh chuỗi không phân biệt hoa/thường."""

    author: Optional[str] = None
    category: Optional[str] = None
    source: Optional[str] = None
    date_from: Optional[datetime] = None  # timezone-aware nếu có
    date_to: Optional[datetime] = None

    def is_empty(self) -> bool:
        return not any(
            (self.author, self.category, self.source, self.date_from, self.date_to)
        )

    def matches(self, article: Article) -> bool:
        if self.author and (article.author or "").strip().lower() != self.author.strip().lower():
            return False
        if self.category and (article.category or "").strip().lower() != self.category.strip().lower():
            return False
        if self.source and (article.source or "").strip().lower() != self.source.strip().lower():
            return False
        if self.date_from and article.published_at < self.date_from:
            return False
        if self.date_to and article.published_at > self.date_to:
            return False
        return True


@dataclass
class SearchQuery:
    """Truy vấn đầu vào của pipeline."""

    text: str
    filters: SearchFilters = field(default_factory=SearchFilters)
    top_k: int = 10
    mode: str = "hybrid"  # xem SEARCH_MODES
    now: Optional[datetime] = None  # override thời điểm "hiện tại" (phục vụ test freshness)


@dataclass
class SearchResultItem:
    """Một kết quả trả về — ĐÚNG 5 trường theo BaoCao-ThietKe.md §2.3."""

    article_id: str
    title: str
    url: str
    published_at: str  # chuỗi ISO-8601
    snippet: str

    def to_dict(self) -> dict:
        """Trả về dict có ĐÚNG 5 khóa — không score, không embedding, không entity."""
        return {
            "article_id": self.article_id,
            "title": self.title,
            "url": self.url,
            "published_at": self.published_at,
            "snippet": self.snippet,
        }


@dataclass(frozen=True)
class Entity:
    """Thực thể trích xuất từ bài viết (F-06)."""

    name: str
    type: str  # một trong ENTITY_TYPES


@dataclass
class ParsedQuery:
    """Kết quả của tầng Query Understanding (search/query.py)."""

    raw: str
    normalized: str  # NFC, lowercase, gọn khoảng trắng
    folded: str  # normalized + bỏ dấu (đ -> d)
    tokens: list[str]
    fresh_intent: bool  # True nếu truy vấn thiên về tin nóng (QDF)

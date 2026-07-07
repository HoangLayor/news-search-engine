"""Holder điều phối tìm kiếm + reindex blue-green (GĐ5).

Giữ ``manager`` + ``pipeline`` hiện hành cùng ``metrics`` và ``events`` (feedback).
``reindex`` dựng chỉ mục MỚI hoàn chỉnh rồi HOÁN ĐỔI nguyên tử (blue-green): truy
vấn trong lúc build vẫn phục vụ từ chỉ mục cũ, không gián đoạn.
"""

from __future__ import annotations

from threading import Lock
from typing import Optional

from news_search.config import Settings
from news_search.feedback import get_event_logger
from news_search.index.manager import IndexManager
from news_search.search.pipeline import SearchPipeline
from news_search.service.metrics import MetricsCollector
from news_search.sources import get_source


class SearchService:
    """Bọc manager+pipeline, hỗ trợ hoán đổi chỉ mục an toàn + metrics + feedback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metrics = MetricsCollector() if settings.metrics_enabled else None
        self.events = get_event_logger(settings)
        self._lock = Lock()
        self.manager = IndexManager(settings)
        self.pipeline = SearchPipeline(self.manager, settings)

    def reindex(self, source=None, limit: Optional[int] = None) -> int:
        """Blue-green: build chỉ mục mới từ nguồn rồi swap. Trả số bài đã index."""
        new_manager = IndexManager(self.settings)
        src = source or get_source(self.settings)
        n = 0
        try:
            for raw in src.fetch_all(self.settings.ingest_batch_size):
                try:
                    new_manager.index_article(raw)
                    n += 1
                except Exception:
                    pass  # bài lỗi -> bỏ qua, không làm hỏng cả đợt reindex
                if limit and n >= limit:
                    break
        finally:
            if source is None:
                src.close()
        with self._lock:  # hoán đổi nguyên tử
            self.manager = new_manager
            self.pipeline = SearchPipeline(new_manager, self.settings)
        return n

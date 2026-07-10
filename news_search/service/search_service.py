"""Holder điều phối tìm kiếm + reindex blue-green (GĐ5).

Giữ ``manager`` + ``pipeline`` hiện hành cùng ``metrics`` và ``events`` (feedback).
``reindex`` dựng chỉ mục MỚI hoàn chỉnh rồi HOÁN ĐỔI nguyên tử (blue-green): truy
vấn trong lúc build vẫn phục vụ từ chỉ mục cũ, không gián đoạn.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import Optional

from news_search.config import Settings
from news_search.feedback import get_event_logger
from news_search.index.manager import IndexManager
from news_search.search.pipeline import SearchPipeline
from news_search.service.metrics import MetricsCollector
from news_search.sources import get_source

_log = logging.getLogger(__name__)


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
        """Blue-green: build chỉ mục mới từ nguồn rồi swap. Trả số bài đã index.

        - Mở nguồn TRƯỚC khi cấp phát chỉ mục mới (nguồn hỏng -> không phí RAM).
        - TÁI DÙNG embedder/reranker hiện có (tránh nạp lại model ~GB -> OOM).
        - Lỗi index từng bài được ĐẾM + LOG (không nuốt im lặng); nếu cả đợt lỗi
          -> log ERROR để lộ nguyên nhân (vd dim mismatch, Milvus từ chối).
        """
        # Mở nguồn trước (raise sớm nếu psycopg thiếu / DB không kết nối được)
        src = source or get_source(self.settings)
        try:
            # Tái dùng model nặng của chỉ mục hiện hành -> không nhân đôi RAM
            new_manager = IndexManager(self.settings, embedder=self.manager.embedder)
            n = skipped = 0
            samples: list[str] = []
            for raw in src.fetch_all(self.settings.ingest_batch_size):
                try:
                    new_manager.index_article(raw)
                    n += 1
                except Exception as exc:
                    skipped += 1
                    if len(samples) < 5:
                        samples.append(f"{raw.get('article_id')}: {type(exc).__name__}: {exc}")
                if limit and n >= limit:
                    break
        finally:
            if source is None:
                src.close()

        if skipped:
            _log.warning("Reindex: index %d bài, BỎ QUA %d bài lỗi. Ví dụ: %s",
                         n, skipped, "; ".join(samples))
        if n == 0 and skipped > 0:
            _log.error("Reindex: TẤT CẢ %d bài đều lỗi — kiểm tra dim collection/kết nối "
                       "(vd EMBEDDER đổi dim nhưng collection Milvus cũ). Mẫu lỗi: %s",
                       skipped, "; ".join(samples))

        with self._lock:  # hoán đổi nguyên tử, tái dùng reranker của pipeline cũ
            self.manager = new_manager
            self.pipeline = SearchPipeline(new_manager, self.settings,
                                           reranker=self.pipeline.reranker)
        return n

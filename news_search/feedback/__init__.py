"""(GĐ1) Ghi log hành vi tìm kiếm & click/dwell — nền tảng đánh giá + học xếp hạng.

Bật/tắt bằng ``FEEDBACK_ENABLED`` (mặc định TẮT). Khi tắt -> ``NullEventLogger``
(no-op). Khi bật, backend ``jsonl`` (bền vững, append-only) hoặc ``redis``; nếu
Redis không dùng được -> tự động degrade về jsonl (không mất dữ liệu, không crash).
"""

from __future__ import annotations

import warnings

from news_search.config import Settings
from news_search.feedback.base import ClickEvent, EventLogger, SearchEvent
from news_search.feedback.logger import JsonlEventLogger, NullEventLogger

__all__ = [
    "EventLogger", "SearchEvent", "ClickEvent",
    "JsonlEventLogger", "NullEventLogger", "get_event_logger",
]


def get_event_logger(settings: Settings | None = None) -> EventLogger:
    """Factory chọn logger theo cấu hình (responsible: mặc định tắt, degrade an toàn)."""
    settings = settings or Settings.from_env()
    if not settings.feedback_enabled:
        return NullEventLogger()
    if settings.feedback_backend == "redis":
        try:
            from news_search.feedback.logger import RedisEventLogger

            return RedisEventLogger(settings.redis_url)
        except Exception as exc:  # thiếu redis / không kết nối -> degrade về jsonl
            warnings.warn(f"Feedback Redis không dùng được ({exc}); dùng jsonl thay thế.")
    return JsonlEventLogger(settings.feedback_path)

"""Mô hình sự kiện + giao diện EventLogger."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Protocol


@dataclass
class SearchEvent:
    """Một lượt tìm kiếm (auto-log ở tầng API)."""

    search_id: str
    query: str
    mode: str
    top_k: int
    result_ids: list[str]
    num_results: int
    zero_result: bool
    variant: Optional[str] = None  # phục vụ A/B (F: experiment)
    ts: Optional[str] = None       # ISO-8601 UTC; None -> logger tự stamp

    def payload(self) -> dict:
        return asdict(self)


@dataclass
class ClickEvent:
    """Một lượt click (kèm dwell time nếu có) trên kết quả tìm kiếm."""

    search_id: str
    article_id: str
    position: int
    dwell_ms: Optional[int] = None
    ts: Optional[str] = None

    def payload(self) -> dict:
        return asdict(self)


class EventLogger(Protocol):
    """Giao diện ghi sự kiện (append-only)."""

    def log_search(self, event: SearchEvent) -> None: ...
    def log_click(self, event: ClickEvent) -> None: ...
    def close(self) -> None: ...

"""Các backend ghi sự kiện: Null (no-op), JSONL (bền vững), Redis (import-guard)."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

from news_search.feedback.base import ClickEvent, SearchEvent


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class NullEventLogger:
    """Không ghi gì — dùng khi FEEDBACK_ENABLED=false (không tốn chi phí)."""

    def log_search(self, event: SearchEvent) -> None:
        pass

    def log_click(self, event: ClickEvent) -> None:
        pass

    def close(self) -> None:
        pass


class JsonlEventLogger:
    """Ghi mỗi sự kiện 1 dòng JSON vào file append-only (bền vững, dễ phân tích).

    An toàn đa luồng bằng lock. Tự tạo thư mục cha nếu chưa có.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.Lock()

    def _append(self, kind: str, payload: dict) -> None:
        payload = dict(payload)
        payload["kind"] = kind
        payload["ts"] = payload.get("ts") or _now_iso()
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def log_search(self, event: SearchEvent) -> None:
        self._append("search", event.payload())

    def log_click(self, event: ClickEvent) -> None:
        self._append("click", event.payload())

    def close(self) -> None:
        pass


class RedisEventLogger:  # pragma: no cover - cần Redis
    """Đẩy sự kiện vào Redis Stream (phục vụ pipeline tiêu thụ realtime).

    Import-guard: thiếu ``redis`` hoặc không kết nối -> raise (factory sẽ degrade).
    """

    def __init__(self, url: str, stream_prefix: str = "nse:events") -> None:
        import redis  # type: ignore[import-not-found]

        self._client = redis.Redis.from_url(url)
        self._client.ping()  # fail sớm nếu không kết nối được
        self._prefix = stream_prefix

    def _push(self, kind: str, payload: dict) -> None:
        flat = {k: ("" if v is None else json.dumps(v, ensure_ascii=False)
                    if isinstance(v, (list, dict)) else str(v))
                for k, v in payload.items()}
        flat["kind"] = kind
        flat["ts"] = payload.get("ts") or _now_iso()
        self._client.xadd(f"{self._prefix}:{kind}", flat)

    def log_search(self, event: SearchEvent) -> None:
        self._push("search", event.payload())

    def log_click(self, event: ClickEvent) -> None:
        self._push("click", event.payload())

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

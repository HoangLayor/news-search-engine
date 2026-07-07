"""Cache kết quả truy vấn (GĐ5) — giảm chi phí rerank/embedding cho truy vấn nóng.

Bật bằng ``CACHE_ENABLED`` (mặc định TẮT). Backend ``memory`` (TTL + LRU) hoặc
``redis`` (import-guard; thiếu -> degrade về memory). "Responsible":
- **Invalidation theo generation**: khóa cache gồm generation của chỉ mục -> mọi
  thay đổi index (thêm/gỡ bài) tự động vô hiệu cache cũ.
- **TTL ngắn** (mặc định 60s) giới hạn độ "cũ" do time-decay.
- **Bypass truy vấn tin nóng** (fresh_intent) để không phục vụ kết quả cũ cho tin nóng.
"""

from __future__ import annotations

import json
import time
import warnings
from collections import OrderedDict
from threading import Lock
from typing import Any, Optional

from news_search.config import Settings


class InMemoryTTLCache:
    """Cache TTL + giới hạn kích thước (LRU), an toàn đa luồng."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 60,
                 clock=time.monotonic) -> None:
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._clock = clock  # tiêm được để test deterministic
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.misses += 1
                return None
            expiry, value = item
            if self._clock() >= expiry:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)  # LRU: vừa dùng -> mới nhất
            self.hits += 1
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            expiry = self._clock() + (ttl if ttl is not None else self.ttl)
            self._data[key] = (expiry, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)  # bỏ mục cũ nhất

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class RedisCache:  # pragma: no cover - cần Redis
    """Cache trên Redis (JSON). Thiếu redis/không kết nối -> raise (factory degrade)."""

    def __init__(self, url: str, ttl_seconds: int = 60, prefix: str = "nse:cache:") -> None:
        import redis  # type: ignore[import-not-found]

        self._r = redis.Redis.from_url(url)
        self._r.ping()
        self.ttl = ttl_seconds
        self.prefix = prefix
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        raw = self._r.get(self.prefix + key)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(raw)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self._r.set(self.prefix + key, json.dumps(value, ensure_ascii=False),
                    ex=ttl if ttl is not None else self.ttl)

    def clear(self) -> None:
        for k in self._r.scan_iter(self.prefix + "*"):
            self._r.delete(k)


def get_cache(settings: Settings):
    """Factory cache theo cấu hình; ``CACHE_ENABLED=false`` -> None (tắt hẳn)."""
    if not settings.cache_enabled:
        return None
    if settings.cache_backend == "redis":
        try:
            return RedisCache(settings.redis_url, settings.cache_ttl_seconds)
        except Exception as exc:
            warnings.warn(f"Cache Redis không dùng được ({exc}); dùng memory thay thế.")
    return InMemoryTTLCache(settings.cache_max_size, settings.cache_ttl_seconds)

"""Nguồn dữ liệu offline: đọc bài mẫu từ ``data/sample_articles.json``.

Dùng cho dev/test/demo — cùng interface với nguồn Postgres nên pipeline không cần
biết dữ liệu đến từ file hay DB.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Iterator, Optional

from news_search.config import BASE_DIR
from news_search.sources.base import RawArticle

_DEFAULT_PATH = os.path.join(BASE_DIR, "data", "sample_articles.json")


class SampleJsonSource:
    """Nguồn từ file JSON (list các dict thô)."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path or _DEFAULT_PATH
        with open(self.path, encoding="utf-8") as f:
            self._rows: list[RawArticle] = json.load(f)

    def count(self) -> int:
        return sum(1 for r in self._rows if r.get("status", "published") == "published")

    def fetch_all(self, batch_size: int = 500) -> Iterator[RawArticle]:
        yield from self._rows

    def fetch_since(self, since: datetime, batch_size: int = 500) -> Iterator[RawArticle]:
        for r in self._rows:
            pub = r.get("published_at")
            if isinstance(pub, str):
                try:
                    pub = datetime.fromisoformat(pub)
                except ValueError:
                    pub = None
            if pub is None or pub >= since:
                yield r

    def fetch_by_ids(self, ids: list[str]) -> Iterator[RawArticle]:
        wanted = set(ids)
        for r in self._rows:
            if str(r.get("article_id")) in wanted:
                yield r

    def close(self) -> None:  # không giữ tài nguyên
        pass

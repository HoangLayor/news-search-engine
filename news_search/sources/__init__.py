"""Tầng nguồn dữ liệu (data source) — trừu tượng hóa nơi bài viết đến từ đâu.

Mọi nguồn (Postgres CMS, file JSON mẫu, tương lai: API/Kafka) đều tuân theo
:class:`ArticleSource` và trả về **dict thô** tương thích
:func:`news_search.ingest.cleaner.normalize_article`. Nhờ vậy tầng index/search
hoàn toàn không phụ thuộc nguồn — thêm nguồn mới = thêm 1 file + 1 nhánh factory.
"""

from __future__ import annotations

from news_search.config import Settings
from news_search.sources.base import ArticleSource, RawArticle
from news_search.sources.sample import SampleJsonSource

__all__ = ["ArticleSource", "RawArticle", "SampleJsonSource", "get_source"]


def get_source(settings: Settings | None = None) -> ArticleSource:
    """Factory chọn nguồn theo ``settings.source``.

    - ``"sample"``   -> :class:`SampleJsonSource` (data/sample_articles.json, offline)
    - ``"postgres"`` -> ``PostgresCMSSource`` (CMS thật; cần psycopg + DB)
    - khác           -> ``ValueError``
    """
    settings = settings or Settings.from_env()
    if settings.source == "sample":
        return SampleJsonSource()
    if settings.source == "postgres":
        from news_search.sources.postgres import PostgresCMSSource  # lazy: chỉ cần khi dùng

        return PostgresCMSSource(settings)
    raise ValueError(f"Nguồn dữ liệu không hỗ trợ: {settings.source!r}")

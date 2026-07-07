"""Nguồn dữ liệu Postgres CMS (bảng ``public.articles`` + bảng liên quan).

Ánh xạ schema CMS thực tế -> dict thô cho pipeline:
    articles.id                        -> article_id
    articles.title / brief_title       -> title
    articles.desc + articles.content   -> body (HTML, tầng ingest làm sạch)
    articles.slug                      -> url (ARTICLE_BASE_URL + slug)
    articles.publish_date              -> published_at (tz-aware Asia/HCM)
    article_authors.full_name          -> author (gộp nhiều tác giả)
    categories.name (is_major)         -> category
    tags.name                          -> tags[]
    articles.source                    -> source

Chỉ lấy bài ĐÃ XUẤT BẢN, CHƯA XÓA: ``deleted_at IS NULL AND publish_date IS NOT NULL``.

Chi tiết kỹ thuật:
- Import guard ``psycopg`` (v3): thiếu -> RuntimeError khi __init__.
- **Keyset pagination theo ``a.id``** (không dùng server-side cursor) để tương
  thích pgbouncer (cổng 6432, transaction pooling) và chịu được kho lớn.
- ``deleted_ids_since`` trả id các bài đã gỡ để đồng bộ xóa (F-08) khi index tăng dần.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterator, Optional

from news_search.config import Settings
from news_search.sources.base import RawArticle

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Mệnh đề SELECT + JOIN dựng sẵn (schema chèn sau khi đã kiểm định là identifier).
_SELECT = """
SELECT
    a.id, a.title, a.brief_title, a.desc AS lead, a.content, a.slug, a.source,
    a.publish_date, a.created_at, a.updated_at,
    cat.name    AS category,
    au.authors  AS author,
    tg.tags     AS tags
FROM {s}.articles a
LEFT JOIN LATERAL (
    SELECT c.name
    FROM {s}.article_categories ac
    JOIN {s}.categories c ON c.id = ac.category_id AND c.deleted_at IS NULL
    WHERE ac.article_id = a.id AND ac.deleted_at IS NULL
    ORDER BY ac.is_major DESC NULLS LAST, ac.display_order ASC NULLS LAST
    LIMIT 1
) cat ON TRUE
LEFT JOIN LATERAL (
    SELECT string_agg(x.full_name, ', ' ORDER BY x.order_index NULLS LAST) AS authors
    FROM (
        SELECT full_name, order_index
        FROM {s}.article_authors
        WHERE article_id = a.id AND deleted_at IS NULL AND full_name IS NOT NULL
    ) x
) au ON TRUE
LEFT JOIN LATERAL (
    SELECT array_agg(t.name) AS tags
    FROM {s}.article_tags atg
    JOIN {s}.tags t ON t.id = atg.tag_id AND t.deleted_at IS NULL
    WHERE atg.article_id = a.id AND atg.deleted_at IS NULL
) tg ON TRUE
WHERE a.deleted_at IS NULL AND a.publish_date IS NOT NULL
"""


class PostgresCMSSource:
    """Đọc bài viết đã xuất bản từ Postgres CMS, ánh xạ về dict thô."""

    def __init__(self, settings: Settings) -> None:
        try:
            import psycopg  # noqa: F401
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
            raise RuntimeError(
                "PostgresCMSSource cần package 'psycopg' (chưa cài) — "
                "cài 'psycopg[binary]' hoặc dùng SOURCE=sample"
            ) from exc

        if not _IDENT_RE.match(settings.pg_schema):
            raise ValueError(f"PG_SCHEMA không hợp lệ: {settings.pg_schema!r}")

        self.settings = settings
        self.schema = settings.pg_schema
        self._dict_row = dict_row
        self._base_url = settings.article_base_url.rstrip("/")
        self._select = _SELECT.format(s=self.schema)
        # Một kết nối dùng chung; autocommit để tránh treo transaction với pgbouncer.
        self._conn = psycopg.connect(autocommit=True, **settings.pg_conninfo())

    # ------------------------------------------------------------------ mapping
    def _map(self, row: dict) -> RawArticle:
        parts = [p for p in (row.get("lead"), row.get("content")) if p and str(p).strip()]
        body = "\n\n".join(str(p) for p in parts).strip()
        title = (row.get("title") or row.get("brief_title") or "").strip()
        if not body:  # bản tin chỉ có tiêu đề (content rỗng) -> dùng title làm body
            body = title
        slug = row.get("slug")
        if slug:
            url = f"{self._base_url}/{slug}" if self._base_url else f"/{slug}"
        else:
            url = f"{self._base_url}/article/{row['id']}" if self._base_url else f"/article/{row['id']}"
        return {
            "article_id": str(row["id"]),
            "title": title,
            "body": body,
            "url": url,
            "published_at": row.get("publish_date") or row.get("created_at"),
            "author": row.get("author"),
            "category": row.get("category"),
            "source": row.get("source"),
            "status": "published",
            "tags": list(row.get("tags") or []),
        }

    # ------------------------------------------------------------------ helpers
    def _run(self, sql: str, params: dict) -> list[dict]:
        with self._conn.cursor(row_factory=self._dict_row) as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def _keyset_iter(
        self, extra_where: str, params: dict, batch_size: int
    ) -> Iterator[RawArticle]:
        """Duyệt theo keyset ``a.id`` (ổn định, không cần server-side cursor)."""
        last_id = 0
        sql = (
            self._select
            + extra_where
            + " AND a.id > %(last_id)s ORDER BY a.id ASC LIMIT %(limit)s"
        )
        while True:
            rows = self._run(sql, {**params, "last_id": last_id, "limit": batch_size})
            if not rows:
                return
            for row in rows:
                yield self._map(row)
            last_id = rows[-1]["id"]
            if len(rows) < batch_size:
                return

    # ------------------------------------------------------------------ API
    def count(self) -> int:
        sql = (
            f"SELECT count(*) AS n FROM {self.schema}.articles a "
            "WHERE a.deleted_at IS NULL AND a.publish_date IS NOT NULL"
        )
        return int(self._run(sql, {})[0]["n"])

    def fetch_all(self, batch_size: int = 500) -> Iterator[RawArticle]:
        yield from self._keyset_iter("", {}, batch_size)

    def fetch_since(self, since: datetime, batch_size: int = 500) -> Iterator[RawArticle]:
        yield from self._keyset_iter(
            " AND (a.publish_date >= %(since)s OR a.updated_at >= %(since)s)",
            {"since": since},
            batch_size,
        )

    def fetch_by_ids(self, ids: list[str]) -> Iterator[RawArticle]:
        int_ids = [int(i) for i in ids if str(i).isdigit()]
        if not int_ids:
            return
        rows = self._run(self._select + " AND a.id = ANY(%(ids)s)", {"ids": int_ids})
        for row in rows:
            yield self._map(row)

    def deleted_ids_since(self, since: datetime) -> list[str]:
        """Id các bài bị GỠ (deleted hoặc bỏ xuất bản) từ ``since`` — để đồng bộ xóa (F-08)."""
        sql = (
            f"SELECT a.id FROM {self.schema}.articles a "
            "WHERE a.updated_at >= %(since)s "
            "AND (a.deleted_at IS NOT NULL OR a.publish_date IS NULL)"
        )
        return [str(r["id"]) for r in self._run(sql, {"since": since})]

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is not None and not conn.closed:
            conn.close()

    def __enter__(self) -> "PostgresCMSSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

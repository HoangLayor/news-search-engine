"""Điều phối index (F-01 khởi tạo, F-08 đồng bộ) — CONTRACTS.md §14.

``IndexManager`` là điểm ghi duy nhất: mỗi bài viết được nạp một lần rồi phân
phối xuống mọi chỉ mục con (lexical BM25, vector, dedup MinHash, knowledge
graph) + kho bài. ``ArticleStore`` giữ bản Article gốc phục vụ lọc metadata
(F-12) và dựng snippet.

F-08 (đồng bộ gỡ/cập nhật) là yêu cầu nghiệm thu quan trọng: bài status khác
"published" (gỡ/ẩn) hoặc gọi ``remove_article`` PHẢI biến mất khỏi MỌI chỉ mục.
"""

from __future__ import annotations

import json
import os

from news_search.config import Settings
from news_search.index.dedup import get_deduper
from news_search.index.embeddings import get_embedder
from news_search.index.entities import KnowledgeGraph, extract_entities
from news_search.index.lexical import get_lexical_index
from news_search.index.vector import get_vector_index
from news_search.ingest.cleaner import normalize_article
from news_search.models import Article, SearchFilters


class ArticleStore:
    """Kho bài viết gốc (in-memory) phục vụ lọc metadata + dựng snippet."""

    def __init__(self) -> None:
        self._articles: dict[str, Article] = {}

    def put(self, article: Article) -> None:
        self._articles[article.article_id] = article

    def get(self, article_id: str) -> Article | None:
        return self._articles.get(article_id)

    def delete(self, article_id: str) -> None:
        self._articles.pop(article_id, None)

    def filter_ids(self, filters: SearchFilters) -> set[str] | None:
        """Trả tập id thỏa bộ lọc; None nếu filters rỗng (không thu hẹp)."""
        if filters.is_empty():
            return None
        return {
            aid for aid, art in self._articles.items() if filters.matches(art)
        }

    def __len__(self) -> int:
        return len(self._articles)


class IndexManager:
    """Nạp & đồng bộ bài viết xuống toàn bộ chỉ mục (F-01, F-08)."""

    def __init__(self, settings: Settings | None = None, embedder=None) -> None:
        self.settings = settings or Settings.from_env()
        # F-01: khởi tạo sẵn mọi kho lưu trữ trước khi index.
        # ``embedder`` có thể được TIÊM để tái dùng (tránh nạp lại model ~GB khi
        # reindex blue-green — xem SearchService.reindex).
        self.store = ArticleStore()
        self.lexical = get_lexical_index(self.settings)
        self.embedder = embedder if embedder is not None else get_embedder(self.settings)
        self.vector = get_vector_index(self.settings, self.embedder.dim)
        self.deduper = get_deduper(self.settings)
        self.kg = KnowledgeGraph()
        # Bộ đếm thế hệ: tăng mỗi khi chỉ mục đổi -> dùng cho invalidation cache.
        self.generation = 0

    # ------------------------------------------------------------------ index

    def index_article(self, raw: dict | Article) -> Article:
        """Chuẩn hóa (nếu cần) rồi ghi bài vào mọi chỉ mục — NGUYÊN TỬ.

        - ``dict`` -> ``normalize_article`` (F-02); ``Article`` dùng trực tiếp.
        - Tính vector + thực thể TRƯỚC khi chạm chỉ mục; nếu bất kỳ bước ghi nào
          lỗi (vd Milvus từ chối) -> rollback toàn bộ để chỉ mục KHÔNG lệch pha.
        - Re-index cùng id an toàn: mọi chỉ mục con đều replace theo id.
        """
        article = raw if isinstance(raw, Article) else normalize_article(raw)
        aid = article.article_id

        # Phần nặng/dễ lỗi tính TRƯỚC, ngoài vùng ghi (không để lại rác nếu lỗi ở đây)
        vector = self.embedder.embed([article.text])[0]
        entities = extract_entities(article.text)

        try:
            self.store.put(article)
            self.lexical.add(article)
            self.vector.add(aid, vector)          # dễ lỗi nhất (Milvus dim/kết nối)
            self.deduper.add(aid, article.text)
            self.kg.add_article(aid, entities)
        except Exception:
            self._purge(aid)                      # dọn phần đã ghi -> không lệch pha
            raise
        self.generation += 1
        return article

    def bulk_index(self, raws: list[dict | Article]) -> int:
        """Nạp một lô bài viết, trả số bài được index (published)."""
        indexed = 0
        for raw in raws:
            article = self.index_article(raw)
            if article.status == "published":
                indexed += 1
        return indexed

    def _purge(self, article_id: str) -> None:
        """Gỡ bài khỏi mọi chỉ mục con KHÔNG tăng generation (dùng nội bộ + rollback)."""
        self.store.delete(article_id)
        self.lexical.remove(article_id)
        self.vector.remove(article_id)
        self.deduper.remove(article_id)
        self.kg.remove_article(article_id)

    def remove_article(self, article_id: str) -> None:
        """Gỡ bài khỏi MỌI chỉ mục (F-08) — store, lexical, vector, dedup, kg."""
        self._purge(article_id)
        self.generation += 1

    # -------------------------------------------------------------- bền vững
    def snapshot(self, path: str) -> int:
        """Lưu toàn bộ bài (dạng thô) ra file JSONL để phục hồi — chống mất dữ liệu.

        Chỉ lưu Article gốc; các chỉ mục dẫn xuất (BM25/vector/dedup/kg) được dựng
        lại khi ``restore`` (không phụ thuộc backend/embedder cụ thể). Trả số bài.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        arts = self.store._articles  # noqa: SLF001 - snapshot nội bộ
        with open(path, "w", encoding="utf-8") as f:
            for art in arts.values():
                f.write(json.dumps({
                    "article_id": art.article_id, "title": art.title, "body": art.body,
                    "url": art.url, "published_at": art.published_at.isoformat(),
                    "author": art.author, "category": art.category, "source": art.source,
                    "status": art.status, "tags": art.tags,
                }, ensure_ascii=False) + "\n")
        return len(arts)

    def restore(self, path: str) -> int:
        """Nạp lại bài từ file snapshot JSONL (dựng lại mọi chỉ mục). Trả số bài."""
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.index_article(json.loads(line))
                    n += 1
        return n

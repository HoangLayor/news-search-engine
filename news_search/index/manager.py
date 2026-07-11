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
    """Kho bài viết gốc phục vụ lọc metadata + dựng snippet.

    Hai chế độ:
    - **Authoritative** (mặc định, backend local): giữ toàn bộ Article trong RAM;
      ``filter_ids`` duyệt được toàn store.
    - **Lazy** (khi có ``loader``, vd backend OpenSearch bền vững): RAM chỉ là
      CACHE; ``get`` miss -> nạp theo id từ nguồn bền vững; ``filter_ids`` trả None
      (không duyệt được toàn bộ) -> pipeline lọc post-retrieval trên tập ứng viên.
      Nhờ vậy app khởi động lại KHÔNG cần nạp lại toàn bộ RAM / reindex.
    """

    def __init__(self, loader=None, batch_loader=None) -> None:
        self._articles: dict[str, Article] = {}
        self._loader = loader              # Callable[[str], Article | None]
        self._batch_loader = batch_loader  # Callable[[list[str]], dict[str, Article]]

    @property
    def lazy(self) -> bool:
        return self._loader is not None

    def put(self, article: Article) -> None:
        self._articles[article.article_id] = article

    def get(self, article_id: str) -> Article | None:
        art = self._articles.get(article_id)
        if art is None and self._loader is not None:
            art = self._loader(article_id)   # lazy: nạp từ backend bền vững
            if art is not None:
                self._articles[article_id] = art  # cache lại
        return art

    def preload(self, ids) -> None:
        """Nạp trước (batch) các id còn thiếu vào cache — tránh N round-trip khi lọc."""
        if self._batch_loader is None:
            return
        missing = [i for i in ids if i not in self._articles]
        if missing:
            for aid, art in self._batch_loader(missing).items():
                if art is not None:
                    self._articles[aid] = art

    def delete(self, article_id: str) -> None:
        self._articles.pop(article_id, None)

    def filter_ids(self, filters: SearchFilters) -> set[str] | None:
        """Tập id thỏa lọc; None nếu filters rỗng HOẶC store lazy (không duyệt được)."""
        if self._loader is not None or filters.is_empty():
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
        print("  -> Đang khởi tạo Lexical Index (BM25)...")
        self.lexical = get_lexical_index(self.settings)

        # ArticleStore LAZY nếu backend lexical bền vững (OpenSearch) hỗ trợ get/mget
        # -> app khởi động lại không cần nạp lại RAM. Local -> in-memory như cũ.
        loader = getattr(self.lexical, "get", None)
        batch_loader = getattr(self.lexical, "mget", None)
        self.store = ArticleStore(
            loader=loader if callable(loader) else None,
            batch_loader=batch_loader if callable(batch_loader) else None,
        )

        if embedder is not None:
            self.embedder = embedder
        else:
            print(f"  -> Đang tải mô hình Embedding ({self.settings.embedder}) - Có thể mất thời gian nếu tải lần đầu...")
            self.embedder = get_embedder(self.settings)
            
        print(f"  -> Đang kết nối Vector Database ({self.settings.vector_backend})...")
        self.vector = get_vector_index(self.settings, self.embedder.dim)
        
        print("  -> Đang khởi tạo Deduper...")
        self.deduper = get_deduper(self.settings)
        
        print("  -> Đang khởi tạo Knowledge Graph...")
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
        # 0. Chuẩn hóa bài viết
        try:
            article = raw if isinstance(raw, Article) else normalize_article(raw)
            aid = article.article_id
        except Exception as exc:
            print(f"  [LỖI] Chuẩn hóa bài viết thất bại: {exc}")
            raise

        # 1. Tính toán vector embedding
        try:
            vector = self.embedder.embed([article.text])[0]
        except Exception as exc:
            print(f"  [LỖI] Tính vector embedding thất bại cho bài {aid}: {exc}")
            raise

        # 2. Trích xuất thực thể (NER)
        try:
            entities = extract_entities(article.text)
        except Exception as exc:
            print(f"  [LỖI] Trích xuất thực thể thất bại cho bài {aid}: {exc}")
            raise

        # 3. Ghi vào các chỉ mục con
        try:
            self.store.put(article)
            self.lexical.add(article)
            self.vector.add(aid, vector)          # dễ lỗi nhất (Milvus dim/kết nối)
            self.deduper.add(aid, article.text)
            self.kg.add_article(aid, entities)
        except Exception as exc:
            print(f"  [LỖI] Ghi dữ liệu chỉ mục thất bại cho bài {aid}: {exc}")
            self._purge(aid)                      # dọn phần đã ghi -> không lệch pha
            raise
        self.generation += 1
        return article

    def bulk_index(self, raws: list[dict | Article]) -> int:
        """Nạp một lô bài viết song song/tối ưu hơn — NGUYÊN TỬ CHO CẢ LÔ.

        Tính embedding gộp một lần và ghi theo lô (nếu backend hỗ trợ).
        """
        if not raws:
            return 0

        # 1. Chuẩn hóa tất cả bài viết trong lô
        try:
            articles = [
                raw if isinstance(raw, Article) else normalize_article(raw)
                for raw in raws
            ]
            article_ids = [art.article_id for art in articles]
        except Exception as exc:
            print(f"  [LỖI LÔ] Chuẩn hóa lô bài viết thất bại: {exc}")
            raise

        # 2. Tạo vector embedding hàng loạt (tận dụng tối ưu song song hóa trên CPU/GPU)
        texts = [art.text for art in articles]
        try:
            vectors = self.embedder.embed(texts)
        except Exception as exc:
            print(f"  [LỖI LÔ] Tạo vector embedding hàng loạt thất bại: {exc}")
            raise

        # 3. Trích xuất thực thể
        try:
            entities_list = [extract_entities(art.text) for art in articles]
        except Exception as exc:
            print(f"  [LỖI LÔ] Trích xuất thực thể hàng loạt thất bại: {exc}")
            raise

        # 4. Lưu trữ và lập chỉ mục con cục bộ (In-memory)
        try:
            for art, ent in zip(articles, entities_list):
                self.store.put(art)
                self.lexical.add(art)
                self.deduper.add(art.article_id, art.text)
                self.kg.add_article(art.article_id, ent)
        except Exception as exc:
            print(f"  [LỖI LÔ] Ghi các chỉ mục in-memory thất bại: {exc}")
            for aid in article_ids:
                self._purge(aid)
            raise

        # 5. Ghi lô vector (Milvus hoặc local)
        try:
            if hasattr(self.vector, "add_batch"):
                self.vector.add_batch(article_ids, vectors)
            else:
                for aid, vec in zip(article_ids, vectors):
                    self.vector.add(aid, vec)
        except Exception as exc:
            print(f"  [LỖI LÔ] Ghi lô vector (Milvus/local) thất bại: {exc}")
            for aid in article_ids:
                self._purge(aid)
            raise

        self.generation += 1
        return sum(1 for art in articles if art.status == "published")

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

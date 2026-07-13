"""Chỉ mục vector local (F-05, F-10) — brute-force cosine bằng numpy.

``VectorIndex`` lưu vector đã L2-normalize trong dict id -> vector, cache
ma trận (n, dim) để tính cosine hàng loạt (dot product vì đã normalize).

GHI CHÚ PRODUCTION: brute-force O(n*dim) mỗi truy vấn chỉ phù hợp
dev/test/demo; production thay bằng Milvus hoặc chỉ mục HNSW
(``VECTOR_BACKEND=milvus``) — giữ nguyên interface này.
"""

from __future__ import annotations

import numpy as np


class VectorIndex:
    """ANN local: brute-force cosine similarity trên ma trận numpy."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError("dim phải > 0")
        self.dim = dim
        # id -> vector đã L2-normalize (float32, shape (dim,))
        self._vectors: dict[str, np.ndarray] = {}
        # Cache ma trận (n, dim) + thứ tự id — invalidate khi add/remove
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []
        
        # Local metadata store & lexical index để tương thích interface Milvus
        self._metadata: dict[str, Article] = {}
        from news_search.index.lexical import LexicalIndex
        self._lexical = LexicalIndex()

    # ------------------------------------------------------------------ utils
    def _as_normalized(self, vector: np.ndarray) -> np.ndarray:
        """Ép về float32 shape (dim,) rồi L2-normalize (zero-safe)."""
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(
                f"Vector dim {vec.shape[0]} không khớp index dim {self.dim}"
            )
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return vec

    def _ensure_matrix(self) -> None:
        """Rebuild cache ma trận nếu đã bị invalidate."""
        if self._matrix is None:
            self._ids = list(self._vectors.keys())
            if self._ids:
                self._matrix = np.vstack([self._vectors[i] for i in self._ids])
            else:
                self._matrix = np.zeros((0, self.dim), dtype=np.float32)

    # -------------------------------------------------------------------- API Store
    def get_article(self, article_id: str) -> Article | None:
        return self._metadata.get(article_id)

    def mget_articles(self, article_ids: list[str]) -> dict[str, Article]:
        return {aid: self._metadata[aid] for aid in article_ids if aid in self._metadata}

    # -------------------------------------------------------------------- API Vector
    def add(self, article_id: str, vector: np.ndarray) -> None:
        """Thêm vector; trùng id -> replace (không phình index). Lưu bản L2-normalized."""
        self._vectors[article_id] = self._as_normalized(vector)
        self._matrix = None

    def add_batch(self, article_ids: list[str], vectors: np.ndarray) -> None:
        """Thêm nhiều vector cùng lúc; trùng id -> replace. Lưu bản L2-normalized."""
        for aid, vec in zip(article_ids, vectors):
            self._vectors[aid] = self._as_normalized(vec)
        self._matrix = None
        
    def add_hybrid(self, article: Article, vector: np.ndarray) -> None:
        self.add(article.article_id, vector)
        self._metadata[article.article_id] = article
        self._lexical.add(article)
        
    def add_batch_hybrid(self, articles: list[Article], vectors: list[np.ndarray]) -> None:
        for art, vec in zip(articles, vectors):
            self.add(art.article_id, vec)
            self._metadata[art.article_id] = art
            self._lexical.add(art)

    def remove(self, article_id: str) -> None:
        """Gỡ vector; id không tồn tại -> no-op."""
        if self._vectors.pop(article_id, None) is not None:
            self._matrix = None
        self._metadata.pop(article_id, None)
        self._lexical.remove(article_id)

    def get(self, article_id: str) -> np.ndarray | None:
        """Trả vector đã normalize (bản copy) hoặc None nếu không có."""
        vec = self._vectors.get(article_id)
        return None if vec is None else vec.copy()

    def search_semantic(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Tìm top_k láng giềng theo cosine similarity, sắp giảm dần."""
        if top_k <= 0 or not self._vectors:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"Vector dim {q.shape[0]} không khớp index dim {self.dim}")
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return []
        q = q / q_norm
        self._ensure_matrix()
        sims = self._matrix @ q
        pairs = [
            (aid, float(score))
            for aid, score in zip(self._ids, sims)
            if allowed_ids is None or aid in allowed_ids
        ]
        pairs.sort(key=lambda p: (-p[1], p[0]))
        return pairs[:top_k]

    def search_lexical(
        self, query_text: str, top_k: int = 10, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        return self._lexical.search(query_text, top_k, allowed_ids)

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 10,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Alias backward compatibility cho test."""
        return self.search_semantic(vector, top_k, allowed_ids)

    def search_hybrid(
        self, query_text: str, vector: np.ndarray, top_k: int = 10, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        from news_search.search.fusion import rrf
        lex = dict(self.search_lexical(query_text, top_k, allowed_ids))
        sem = dict(self.search_semantic(vector, top_k, allowed_ids))
        fused = rrf([list(lex.keys()), list(sem.keys())], k=60)
        return [(k, float(v)) for k, v in list(fused.items())[:top_k]]

    def __len__(self) -> int:
        return len(self._vectors)


def get_vector_index(settings, dim: int):
    """Factory chọn vector backend theo ``settings.vector_backend``.

    - ``"local"``  -> :class:`VectorIndex` (brute-force, offline)
    - ``"milvus"`` -> ``MilvusVectorIndex`` (HNSW, cần Milvus server + pymilvus)
    - khác         -> ``ValueError``
    """
    backend = settings.vector_backend
    if backend == "local":
        return VectorIndex(dim)
    if backend == "milvus":
        from news_search.index.milvus_vector import MilvusVectorIndex

        return MilvusVectorIndex(dim, settings)
    raise ValueError(f"Vector backend không hỗ trợ: {backend!r}")

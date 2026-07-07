"""Chỉ mục từ khóa BM25 local (F-04, F-09).

Inverted index trên token ĐÃ FOLD (không dấu, lowercase) để match không phân
biệt dấu. Tiêu đề được nhân trọng số ``title_weight`` (lặp token). IDF theo
công thức BM25 chuẩn (Robertson): ``ln((N - df + 0.5)/(df + 0.5) + 1)``.

Production thay bằng OpenSearch/Elasticsearch; bản local này thuần stdlib.
"""

from __future__ import annotations

import math
from collections import Counter

from news_search.ingest.tokenizer import fold_tokens, tokenize
from news_search.models import Article


class LexicalIndex:
    """Chỉ mục BM25 in-memory: add/update/remove/search theo CONTRACTS.md §3."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, title_weight: int = 3):
        self.k1 = k1
        self.b = b
        self.title_weight = title_weight
        # token -> {doc_id: term_freq}
        self._postings: dict[str, dict[str, int]] = {}
        # doc_id -> term_freq từng token của doc (phục vụ remove nhanh)
        self._doc_tf: dict[str, dict[str, int]] = {}
        # doc_id -> tổng số token (đã nhân trọng số tiêu đề)
        self._doc_len: dict[str, int] = {}
        self._total_len: int = 0  # tổng độ dài mọi doc (tính avgdl)

    # ------------------------------------------------------------------ CRUD

    def add(self, article: Article) -> None:
        """Thêm bài vào chỉ mục; re-add cùng id = replace, không trùng lặp."""
        self.remove(article.article_id)
        title_tokens = fold_tokens(tokenize(article.title))
        body_tokens = fold_tokens(tokenize(article.body))
        # Doc tokens = title * title_weight + body (theo contract)
        tokens = title_tokens * self.title_weight + body_tokens
        tf = dict(Counter(tokens))
        doc_id = article.article_id
        for token, freq in tf.items():
            self._postings.setdefault(token, {})[doc_id] = freq
        self._doc_tf[doc_id] = tf
        self._doc_len[doc_id] = len(tokens)
        self._total_len += len(tokens)

    def update(self, article: Article) -> None:
        """Cập nhật bài = remove + add."""
        self.remove(article.article_id)
        self.add(article)

    def remove(self, article_id: str) -> None:
        """Gỡ bài khỏi chỉ mục; id không tồn tại -> no-op."""
        tf = self._doc_tf.pop(article_id, None)
        if tf is None:
            return
        for token in tf:
            postings = self._postings.get(token)
            if postings is not None:
                postings.pop(article_id, None)
                if not postings:  # dọn posting list rỗng để df chính xác
                    del self._postings[token]
        self._total_len -= self._doc_len.pop(article_id, 0)

    # ---------------------------------------------------------------- Search

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        allowed_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Chấm điểm BM25, trả list (article_id, score>0) giảm dần theo score.

        - Match không phân biệt dấu/hoa thường (token folded).
        - ``allowed_ids`` khác None -> chỉ chấm doc trong tập đó.
        - Query rỗng hoặc index rỗng -> [].
        """
        query_tokens = fold_tokens(tokenize(query_text))
        n_docs = len(self._doc_len)
        if not query_tokens or n_docs == 0:
            return []
        avgdl = (self._total_len / n_docs) or 1.0  # tránh chia 0 khi mọi doc rỗng
        scores: dict[str, float] = {}
        for token in query_tokens:  # token lặp trong query -> cộng dồn (qtf)
            postings = self._postings.get(token)
            if not postings:
                continue
            df = len(postings)
            # IDF Robertson: luôn > 0
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for doc_id, tf in postings.items():
                if allowed_ids is not None and doc_id not in allowed_ids:
                    continue
                dl = self._doc_len[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (self.k1 + 1.0) / denom
        # Chỉ giữ score > 0; sắp giảm dần, tie-break theo id cho deterministic
        ranked = [(doc_id, score) for doc_id, score in scores.items() if score > 0.0]
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked[:top_k]

    # ------------------------------------------------------------- Protocols

    def vocab_frequencies(self) -> dict[str, int]:
        """Từ điển tần suất {token folded: tổng term-frequency} — phục vụ spell-correct."""
        return {token: sum(postings.values()) for token, postings in self._postings.items()}

    def __contains__(self, article_id: str) -> bool:
        return article_id in self._doc_tf

    def __len__(self) -> int:
        return len(self._doc_tf)


def get_lexical_index(settings):
    """Factory chọn backend lexical theo ``settings.lexical_backend``.

    - ``"local"``      -> :class:`LexicalIndex` (BM25 thuần Python, offline)
    - ``"opensearch"`` -> ``OpenSearchLexicalIndex`` (cần opensearch-py + server)
    - khác             -> ``ValueError``
    """
    backend = settings.lexical_backend
    if backend == "local":
        return LexicalIndex()
    if backend == "opensearch":
        from news_search.index.opensearch_lexical import OpenSearchLexicalIndex

        return OpenSearchLexicalIndex(settings)
    raise ValueError(f"Lexical backend không hỗ trợ: {backend!r}")

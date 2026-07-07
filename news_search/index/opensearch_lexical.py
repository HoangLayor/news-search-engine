"""Backend lexical production: OpenSearch (BM25 + analyzer tiếng Việt).

Cùng interface với :class:`news_search.index.lexical.LexicalIndex` để thay thế
trong suốt qua ``LEXICAL_BACKEND=opensearch``. Import-guard ``opensearch-py``:
thiếu package / không kết nối -> ``RuntimeError`` khi __init__.

LƯU Ý: cần OpenSearch server (`docker run -p 9200:9200 opensearchproject/opensearch`).
Chưa có test offline. Tiêu đề nhân trọng số bằng ``title^3`` trong multi_match.
"""

from __future__ import annotations

from news_search.config import Settings
from news_search.models import Article

# Ánh xạ (mapping) + analyzer: dùng phân tích unicode/ICU nếu có; mặc định standard
# đã đủ tách theo khoảng trắng cho tiếng Việt (âm tiết rời).
_INDEX_BODY = {
    "settings": {
        "index": {"number_of_shards": 1, "number_of_replicas": 1},
        "analysis": {
            "analyzer": {
                "vi_analyzer": {"type": "custom", "tokenizer": "standard",
                                "filter": ["lowercase", "asciifolding"]}
            }
        },
    },
    "mappings": {
        "properties": {
            "title": {"type": "text", "analyzer": "vi_analyzer"},
            "body": {"type": "text", "analyzer": "vi_analyzer"},
            "published_at": {"type": "date"},
            "author": {"type": "keyword"},
            "category": {"type": "keyword"},
            "source": {"type": "keyword"},
            "status": {"type": "keyword"},
        }
    },
}


class OpenSearchLexicalIndex:
    """Chỉ mục BM25 trên OpenSearch, khớp interface LexicalIndex."""

    def __init__(self, settings: Settings) -> None:
        try:
            from opensearchpy import OpenSearch  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "OpenSearchLexicalIndex cần 'opensearch-py' (chưa cài) — "
                "cài opensearch-py hoặc dùng LEXICAL_BACKEND=local"
            ) from exc
        self.index = settings.opensearch_index
        auth = None
        if settings.opensearch_user:
            auth = (settings.opensearch_user, settings.opensearch_password)
        try:
            self._os = OpenSearch(hosts=[settings.opensearch_url], http_auth=auth,
                                  timeout=10, max_retries=2, retry_on_timeout=True)
            if not self._os.indices.exists(self.index):
                self._os.indices.create(self.index, body=_INDEX_BODY)
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Không kết nối/khởi tạo được OpenSearch: {exc}") from exc

    def add(self, article: Article) -> None:
        self._os.index(index=self.index, id=article.article_id, body={
            "title": article.title, "body": article.body,
            "published_at": article.published_at.isoformat(),
            "author": article.author, "category": article.category,
            "source": article.source, "status": article.status,
        }, refresh=True)

    def update(self, article: Article) -> None:
        self.add(article)

    def remove(self, article_id: str) -> None:
        try:
            self._os.delete(index=self.index, id=article_id, refresh=True)
        except Exception:
            pass  # không tồn tại -> no-op

    def search(self, query_text: str, top_k: int = 10,
               allowed_ids: set[str] | None = None) -> list[tuple[str, float]]:
        if not query_text.strip():
            return []
        must = {"multi_match": {"query": query_text, "fields": ["title^3", "body"]}}
        query: dict = {"bool": {"must": must}}
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            query["bool"]["filter"] = {"ids": {"values": list(allowed_ids)}}
        resp = self._os.search(index=self.index, body={
            "size": top_k, "query": query, "_source": False})
        return [(h["_id"], float(h["_score"])) for h in resp["hits"]["hits"]]

    def vocab_frequencies(self) -> dict[str, int]:
        # Không rút vocab từ OpenSearch (tốn kém) -> spell-correct sẽ no-op (degrade).
        return {}

    def __contains__(self, article_id: str) -> bool:
        try:
            return bool(self._os.exists(index=self.index, id=article_id))
        except Exception:
            return False

    def __len__(self) -> int:
        try:
            return int(self._os.count(index=self.index)["count"])
        except Exception:
            return 0

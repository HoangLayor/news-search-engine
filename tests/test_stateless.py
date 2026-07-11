"""Test đường đọc STATELESS: ArticleStore lazy + post-filter + OpenSearch get/mget.

Mục tiêu: app khởi động lại KHÔNG cần nạp lại RAM/reindex — store đọc bài theo id
từ backend bền vững (OpenSearch), và lọc metadata (status) vẫn đúng khi lazy.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from news_search.config import Settings
from news_search.index.manager import ArticleStore, IndexManager
from news_search.index.opensearch_lexical import OpenSearchLexicalIndex, _doc_to_article
from news_search.models import Article, SearchFilters, SearchQuery
from news_search.search.pipeline import SearchPipeline
from news_search.search.rerank import NoopReranker

NOW = datetime(2026, 7, 7, 12, tzinfo=timezone(timedelta(hours=7)))


def _article(aid, title, body, status="published", **extra):
    return Article(article_id=aid, title=title, body=body, url=f"/{aid}",
                   published_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
                   status=status, **extra)


# ------------------------------------------------------------ ArticleStore lazy


def test_article_store_lazy_load_and_cache():
    backing = {"a": _article("a", "t", "nội dung")}
    calls: list[str] = []
    store = ArticleStore(
        loader=lambda i: (calls.append(i), backing.get(i))[1],
        batch_loader=lambda ids: {i: backing[i] for i in ids if i in backing},
    )
    assert store.lazy is True
    # lazy -> filter_ids KHÔNG duyệt được toàn bộ -> None (pipeline lọc post-retrieval)
    assert store.filter_ids(SearchFilters(category="X")) is None

    assert store.get("a") is not None and calls == ["a"]
    store.get("a")                      # lần 2 -> lấy từ cache, loader không gọi lại
    assert calls == ["a"]
    store.preload(["a", "b"])           # a đã cache, b không có -> an toàn
    assert store.get("b") is None


def test_article_store_authoritative_khi_khong_loader():
    store = ArticleStore()              # local -> không lazy
    assert store.lazy is False
    store.put(_article("a", "t", "b", category="Kinh tế"))
    store.put(_article("b", "t", "b", category="Thể thao"))
    ids = store.filter_ids(SearchFilters(category="Kinh tế"))
    assert ids == {"a"}                 # authoritative -> duyệt & lọc được


# ------------------------------------------------------------ pipeline lazy no-leak


def test_pipeline_lazy_khong_ro_ri_unpublished():
    """Store lazy: bài unpublished KHÔNG rò rỉ ra public search (post-filter status)."""
    from news_search.ingest.cleaner import normalize_article

    s = Settings(embedder="hash", vector_backend="local", tokenizer="regex",
                 dedup_backend="local", reranker="none")
    m = IndexManager(s)
    raws = [
        _article("pub1", "giá vàng", "giá vàng tăng mạnh"),
        _article("pub2", "giá vàng cao", "giá vàng lập đỉnh"),
        _article("unpub", "giá vàng ẩn", "giá vàng nội dung đã ẩn", status="unpublished"),
    ]
    for a in raws:
        m.index_article(a)             # index vào lexical/vector (kể cả unpub)

    # Chuyển store sang LAZY (mô phỏng khởi động lại: RAM rỗng, đọc theo id)
    backing = {a.article_id: a for a in raws}
    m.store = ArticleStore(
        loader=lambda i: backing.get(i),
        batch_loader=lambda ids: {i: backing[i] for i in ids if i in backing},
    )
    p = SearchPipeline(m, s, reranker=NoopReranker())
    res = [r.article_id for r in p.search(SearchQuery("giá vàng", mode="lexical", top_k=10, now=NOW))]
    assert "unpub" not in res          # <-- KHÔNG rò rỉ unpublished
    assert "pub1" in res and "pub2" in res


# ------------------------------------------------------------ OpenSearch get/mget


class _FakeOS:
    def __init__(self):
        self.docs: dict[str, dict] = {}

    def index(self, index, id, body, refresh=True):
        self.docs[id] = body

    def get(self, index, id):
        return {"found": id in self.docs, "_source": self.docs.get(id, {})}

    def mget(self, index, body):
        return {"docs": [{"_id": i, "found": i in self.docs, "_source": self.docs.get(i, {})}
                         for i in body["ids"]]}


def _os_index() -> OpenSearchLexicalIndex:
    idx = OpenSearchLexicalIndex.__new__(OpenSearchLexicalIndex)
    idx._os = _FakeOS()
    idx.index = "news_articles"
    return idx


def test_doc_to_article_reconstruct():
    art = _doc_to_article("id1", {
        "title": "T", "body": "B", "url": "/u", "published_at": "2026-07-01T00:00:00+07:00",
        "author": "A", "category": "Kinh tế", "status": "published", "tags": ["vàng"]})
    assert art.article_id == "id1" and art.url == "/u" and art.category == "Kinh tế"
    assert art.tags == ["vàng"] and art.status == "published"
    assert art.published_at.isoformat().startswith("2026-07-01")


def test_opensearch_add_get_mget():
    idx = _os_index()
    a = Article("a1", "Tiêu đề", "Nội dung chính", "/a1",
                datetime(2026, 7, 1, tzinfo=timezone.utc), category="Kinh tế", tags=["vàng"])
    idx.add(a)
    # doc lưu ĐỦ field để dựng lại (kể cả url/tags)
    assert idx._os.docs["a1"]["url"] == "/a1" and idx._os.docs["a1"]["tags"] == ["vàng"]

    got = idx.get("a1")
    assert got is not None and got.title == "Tiêu đề" and got.url == "/a1"
    assert got.category == "Kinh tế" and got.tags == ["vàng"]
    assert idx.get("missing") is None

    batch = idx.mget(["a1", "missing"])
    assert set(batch) == {"a1"} and batch["a1"].body == "Nội dung chính"
    assert idx.mget([]) == {}

"""Pipeline tìm kiếm phễu nhiều tầng (CONTRACTS.md §15).

Ghép các module thành luồng 9 bước:
    parse -> lọc metadata -> retrieval (lexical/semantic/hybrid) -> RRF
          -> time-decay (QDF) -> rerank -> collapse cụm -> MMR -> snippet
Đầu ra là list :class:`SearchResultItem` (đúng 5 trường khi ``.to_dict()``).
"""

from __future__ import annotations

from datetime import datetime, timezone

from news_search.config import Settings
from news_search.index.manager import IndexManager
from news_search.models import SearchQuery, SearchResultItem
from news_search.search.diversify import collapse_clusters, mmr
from news_search.search.fusion import rrf
from news_search.search.query import parse_query
from news_search.search.ranking import apply_time_decay
from news_search.search.rerank import get_reranker
from news_search.search.snippet import make_snippet
from news_search.search.understanding import QueryUnderstander
from news_search.service.cache import get_cache


class SearchPipeline:
    """Điều phối tìm kiếm end-to-end trên các chỉ mục của IndexManager."""

    def __init__(self, manager: IndexManager, settings: Settings | None = None) -> None:
        self.manager = manager
        self.settings = settings or manager.settings
        self.reranker = get_reranker(self.settings)
        # (GĐ3) Query understanding — vocab lấy lười từ chỉ mục BM25.
        self.understander = QueryUnderstander(
            self.settings, vocab_provider=lambda: self.manager.lexical.vocab_frequencies()
        )
        # (GĐ5) Cache truy vấn nóng — None nếu CACHE_ENABLED=false.
        self.cache = get_cache(self.settings)

    def _cache_key(self, query: SearchQuery, effective_text: str) -> str:
        """Khóa cache: gồm generation chỉ mục -> mọi thay đổi index tự vô hiệu cache."""
        f = query.filters
        return "|".join(str(x) for x in (
            self.manager.generation, query.mode, query.top_k, effective_text,
            f.author, f.category, f.source, f.date_from, f.date_to,
        ))

    def search(self, query: SearchQuery) -> list[SearchResultItem]:
        """Ánh xạ truy vấn -> danh sách bài viết xếp hạng (5 trường)."""
        cfg = self.settings
        # 1. Query understanding (rỗng -> ValueError, để propagate lên caller)
        parsed = parse_query(query.text)
        # Truy vấn không có token từ nào (vd chỉ dấu câu "@@@ ###") -> không có gì
        # để tìm; tránh nhánh vector băm nhiễu trên n-gram dấu câu.
        if not parsed.tokens:
            return []

        # 1b. (GĐ3) Query understanding — chỉ đổi truy vấn khi có cờ bật (mặc định
        # tắt -> giữ nguyên query.text để hành vi/kết quả không đổi).
        effective_text = query.text
        if self.understander.enabled:
            effective_text = self.understander.understand(parsed).effective_text

        # 1c. (GĐ5) Cache: tra trước khi tính; bỏ qua cho truy vấn tin nóng.
        cache_key = None
        if self.cache is not None and not (
            parsed.fresh_intent and self.settings.cache_bypass_fresh
        ):
            cache_key = self._cache_key(query, effective_text)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return [SearchResultItem(**d) for d in cached]

        # 2. Lọc metadata (F-12): None = không thu hẹp
        allowed_ids = self.manager.store.filter_ids(query.filters)
        if allowed_ids is not None and not allowed_ids:
            return []  # bộ lọc loại hết -> không có kết quả

        # 3-4. Retrieval + fusion tùy chế độ (dùng truy vấn hiệu dụng)
        base_scores = self._retrieve(query, parsed_text=effective_text, allowed_ids=allowed_ids)
        if not base_scores:
            return []

        # 5. Time-decay / QDF (F-13): truy vấn tin nóng -> half-life ngắn hơn
        now = query.now or datetime.now(timezone.utc)
        half_life = cfg.fresh_half_life_days if parsed.fresh_intent else cfg.half_life_days
        decayed = apply_time_decay(
            base_scores,
            published_lookup=lambda aid: self.manager.store.get(aid).published_at,
            now=now,
            half_life_days=half_life,
            floor=cfg.decay_floor,
        )

        # 6. Rerank tinh trên top-N theo điểm đã decay
        ranked_by_decay = sorted(decayed.items(), key=lambda kv: (-kv[1], kv[0]))
        top_for_rerank = ranked_by_decay[: cfg.rerank_top_n]
        docs = [(aid, self.manager.store.get(aid).text) for aid, _ in top_for_rerank]
        reranked = self.reranker.rerank(query.text, docs, top_k=cfg.rerank_top_n)
        rerank_score = dict(reranked)
        reranked_ids = [aid for aid, _ in reranked]

        # 7. Gom cụm sự kiện (F-07/F-14): mỗi cụm giữ tối đa max_per_cluster bài
        collapsed = collapse_clusters(
            reranked_ids, self.manager.deduper.cluster_of, cfg.max_per_cluster
        )

        # 8. Đa dạng hóa MMR trên pool rồi lấy top_k
        pool = [(aid, rerank_score.get(aid, 0.0)) for aid in collapsed[: cfg.mmr_pool]]
        final_ids = mmr(
            pool, self.manager.vector.get, lambda_=cfg.mmr_lambda, top_k=query.top_k
        )

        # 9. Dựng kết quả 5 trường + snippet bôi đậm (F-15)
        results: list[SearchResultItem] = []
        for aid in final_ids:
            article = self.manager.store.get(aid)
            if article is None:  # phòng khi bài bị gỡ giữa chừng
                continue
            snippet = make_snippet(article.body, parsed.tokens, cfg.snippet_max_len)
            results.append(
                SearchResultItem(
                    article_id=article.article_id,
                    title=article.title,
                    url=article.url,
                    published_at=article.published_at.isoformat(),
                    snippet=snippet,
                )
            )
        # 1c'. Lưu cache (nếu bật) — lưu dạng dict để dùng chung memory/redis.
        if cache_key is not None:
            self.cache.set(cache_key, [r.to_dict() for r in results])
        return results

    # ------------------------------------------------------------------ helpers

    def _retrieve(
        self, query: SearchQuery, parsed_text: str, allowed_ids: set[str] | None
    ) -> dict[str, float]:
        """Sinh bảng điểm ứng viên theo chế độ (lexical/semantic/hybrid)."""
        cfg = self.settings
        mode = query.mode
        k = cfg.candidates_k

        if mode == "lexical":
            hits = self.manager.lexical.search(parsed_text, top_k=k, allowed_ids=allowed_ids)
            return dict(hits)

        if mode == "semantic":
            qvec = self.manager.embedder.embed([parsed_text])[0]
            hits = self.manager.vector.search(qvec, top_k=k, allowed_ids=allowed_ids)
            return dict(hits)

        if mode == "hybrid":
            lex = self.manager.lexical.search(parsed_text, top_k=k, allowed_ids=allowed_ids)
            qvec = self.manager.embedder.embed([parsed_text])[0]
            vec = self.manager.vector.search(qvec, top_k=k, allowed_ids=allowed_ids)
            lex_ids = [aid for aid, _ in lex]
            vec_ids = [aid for aid, _ in vec]
            return rrf([lex_ids, vec_ids], k=cfg.rrf_k)

        raise ValueError(f"mode không hợp lệ: {query.mode!r} (lexical|semantic|hybrid)")

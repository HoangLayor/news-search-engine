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

    def __init__(self, manager: IndexManager, settings: Settings | None = None,
                 reranker=None) -> None:
        self.manager = manager
        self.settings = settings or manager.settings
        # ``reranker`` có thể được TIÊM để tái dùng (tránh nạp lại model VN ~GB
        # khi reindex blue-green — xem SearchService.reindex).
        self.reranker = reranker if reranker is not None else get_reranker(self.settings)
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
        return self._run(query, trace=None)

    def explain(self, query: SearchQuery) -> dict:
        """Chạy pipeline và GHI LẠI từng bước + kết quả trung gian (bỏ qua cache).

        Trả dict ``{query, mode, top_k, results, trace}``; ``trace`` là danh sách
        các bước, mỗi bước ``{step, key, title, note, items:[{article_id,title,
        score,...}], count}`` — để UI quan sát truy vấn được xử lý qua từng tầng.
        """
        trace: list[dict] = []
        results = self._run(query, trace=trace)
        return {
            "query": query.text, "mode": query.mode, "top_k": query.top_k,
            "results": [r.to_dict() for r in results], "trace": trace,
        }

    # ------------------------------------------------------------------ core

    def _append(self, trace, key, title, note="", items=None, count=None) -> None:
        trace.append({"step": len(trace) + 1, "key": key, "title": title,
                      "note": note, "items": items or [], "count": count})

    def _trace_items(self, scored, limit: int = 8, with_cluster: bool = False) -> list[dict]:
        """[(article_id, score)] -> danh sách item hiển thị (kèm tiêu đề, điểm)."""
        out = []
        for aid, score in list(scored)[:limit]:
            art = self.manager.store.get(aid)
            item = {"article_id": aid, "title": art.title if art else aid,
                    "score": round(float(score), 4)}
            if with_cluster:
                item["cluster"] = self.manager.deduper.cluster_of(aid)
            out.append(item)
        return out

    def _run(self, query: SearchQuery, trace: list | None = None) -> list[SearchResultItem]:
        """Pipeline dùng chung cho search() và explain(). trace != None -> ghi bước."""
        cfg = self.settings
        explain = trace is not None

        # 1. Query understanding (rỗng -> ValueError, propagate lên caller)
        parsed = parse_query(query.text)
        if not parsed.tokens:
            if explain:
                self._append(trace, "understand", "1. Hiểu truy vấn",
                             "Không có token hợp lệ trong truy vấn -> trả rỗng")
            return []

        # 1b. Query understanding (mặc định tắt -> giữ nguyên query.text)
        effective_text = query.text
        uinfo = self.understander.understand(parsed) if self.understander.enabled else None
        if uinfo is not None:
            effective_text = uinfo.effective_text
        if explain:
            note = f"tokens={parsed.tokens} · fresh_intent={parsed.fresh_intent}"
            if uinfo is not None:
                note += (f" · sửa lỗi={uinfo.corrections or '∅'} · mở rộng={uinfo.expansions or '∅'}"
                         f" · truy vấn hiệu dụng='{effective_text}'")
            else:
                note += " · (query-understanding tắt)"
            self._append(trace, "understand", "1. Hiểu truy vấn", note)

        # 1c. Cache (CHỈ khi không explain -> explain luôn tính mới để quan sát)
        cache_key = None
        if not explain and self.cache is not None and not (
            parsed.fresh_intent and cfg.cache_bypass_fresh
        ):
            cache_key = self._cache_key(query, effective_text)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return [SearchResultItem(**d) for d in cached]

        # 2. Lọc metadata (F-12)
        allowed_ids = self.manager.store.filter_ids(query.filters)
        if allowed_ids is not None and not allowed_ids:
            if explain:
                self._append(trace, "filter", "2. Lọc metadata", "Bộ lọc loại hết bài -> rỗng")
            return []
        if explain:
            self._append(trace, "filter", "2. Lọc metadata",
                         "Không áp bộ lọc" if allowed_ids is None
                         else f"{len(allowed_ids)} bài thỏa bộ lọc")

        # 3-4. Retrieval + fusion (ghi bước bên trong _retrieve)
        base_scores = self._retrieve(query, effective_text, allowed_ids, trace)
        if not base_scores:
            if explain:
                self._append(trace, "empty", "Kết quả", "Không có ứng viên nào khớp")
            return []

        # 4b. Store LAZY (vd OpenSearch): preload ứng viên (batch) rồi lọc TRÊN TẬP
        # ỨNG VIÊN — (a) bỏ id "mồ côi" không dựng lại được (tránh crash time-decay),
        # (b) áp bộ lọc metadata (status/category...) mà filter_ids không pre-lọc
        # được khi lazy -> KHÔNG rò rỉ bài unpublished. Local (không lazy) bỏ qua.
        if self.manager.store.lazy:
            self.manager.store.preload(list(base_scores))
            active = not query.filters.is_empty()
            kept: dict[str, float] = {}
            for aid, s in base_scores.items():
                art = self.manager.store.get(aid)
                if art is None:
                    continue  # (có trong Milvus/BM25 nhưng không dựng lại được)
                if active and not query.filters.matches(art):
                    continue
                kept[aid] = s
            if explain:
                self._append(trace, "filter_post", "2b. Lọc metadata (post-retrieval, store lazy)",
                             f"{len(base_scores)} → {len(kept)} bài (bỏ mồ côi + lọc metadata)",
                             self._trace_items(sorted(kept.items(), key=lambda kv: -kv[1])), len(kept))
            base_scores = kept
            if not base_scores:
                return []

        # 5. Time-decay / QDF (F-13)
        now = query.now or datetime.now(timezone.utc)
        half_life = cfg.fresh_half_life_days if parsed.fresh_intent else cfg.half_life_days
        decayed = apply_time_decay(
            base_scores,
            published_lookup=lambda aid: self.manager.store.get(aid).published_at,
            now=now, half_life_days=half_life, floor=cfg.decay_floor,
        )
        ranked_by_decay = sorted(decayed.items(), key=lambda kv: (-kv[1], kv[0]))
        if explain:
            self._append(trace, "decay", "5. Ưu tiên độ mới (time-decay/QDF)",
                         f"half-life={half_life} ngày (fresh_intent={parsed.fresh_intent}), floor={cfg.decay_floor}",
                         self._trace_items(ranked_by_decay), len(decayed))

        # 6. Rerank tinh trên top-N theo điểm đã decay
        top_for_rerank = ranked_by_decay[: cfg.rerank_top_n]
        docs = [(aid, self.manager.store.get(aid).text) for aid, _ in top_for_rerank]
        reranked = self.reranker.rerank(query.text, docs, top_k=cfg.rerank_top_n)
        rerank_score = dict(reranked)
        reranked_ids = [aid for aid, _ in reranked]
        if explain:
            self._append(trace, "rerank", "6. Rerank tinh",
                         f"reranker={type(self.reranker).__name__} (top {cfg.rerank_top_n})",
                         self._trace_items(reranked), len(reranked))

        # 7. Gom cụm sự kiện (F-07/F-14)
        collapsed = collapse_clusters(reranked_ids, self.manager.deduper.cluster_of, cfg.max_per_cluster)
        if explain:
            self._append(trace, "dedup", "7. Gom cụm trùng lặp",
                         f"{len(reranked_ids)} → {len(collapsed)} (bỏ {len(reranked_ids) - len(collapsed)} bài gần trùng, tối đa {cfg.max_per_cluster}/cụm)",
                         self._trace_items([(a, rerank_score.get(a, 0.0)) for a in collapsed], with_cluster=True),
                         len(collapsed))

        # 8. Đa dạng hóa MMR
        pool = [(aid, rerank_score.get(aid, 0.0)) for aid in collapsed[: cfg.mmr_pool]]
        final_ids = mmr(pool, self.manager.vector.get, lambda_=cfg.mmr_lambda, top_k=query.top_k)
        if explain:
            self._append(trace, "mmr", "8. Đa dạng hóa MMR",
                         f"lambda={cfg.mmr_lambda}, pool={len(pool)} → top_k={query.top_k}",
                         self._trace_items([(a, rerank_score.get(a, 0.0)) for a in final_ids]),
                         len(final_ids))

        # 9. Dựng kết quả 5 trường + snippet bôi đậm (F-15)
        results: list[SearchResultItem] = []
        for aid in final_ids:
            article = self.manager.store.get(aid)
            if article is None:  # phòng khi bài bị gỡ giữa chừng
                continue
            snippet = make_snippet(article.body, parsed.tokens, cfg.snippet_max_len)
            results.append(SearchResultItem(
                article_id=article.article_id, title=article.title, url=article.url,
                published_at=article.published_at.isoformat(), snippet=snippet,
            ))
        if explain:
            self._append(trace, "final", "9. Kết quả cuối (JSON 5 trường)",
                         f"{len(results)} bài trả về",
                         [{"article_id": r.article_id, "title": r.title, "score": None} for r in results],
                         len(results))
        # Lưu cache (chỉ path search thường)
        if cache_key is not None:
            self.cache.set(cache_key, [r.to_dict() for r in results])
        return results

    # ------------------------------------------------------------------ helpers

    def _retrieve(
        self, query: SearchQuery, parsed_text: str, allowed_ids: set[str] | None,
        trace: list | None = None,
    ) -> dict[str, float]:
        """Sinh bảng điểm ứng viên theo chế độ (lexical/semantic/hybrid).

        trace != None -> ghi các bước truy hồi (lexical/dense) và hợp nhất RRF.
        """
        cfg = self.settings
        mode = query.mode
        k = cfg.candidates_k
        explain = trace is not None
        embed_name = type(self.manager.embedder).__name__

        if mode == "lexical":
            hits = self.manager.lexical.search(parsed_text, top_k=k, allowed_ids=allowed_ids)
            if explain:
                self._append(trace, "lexical", "3. Truy hồi từ khóa (BM25)",
                             f"{len(hits)} ứng viên", self._trace_items(hits), len(hits))
            return dict(hits)

        if mode == "semantic":
            qvec = self.manager.embedder.embed([parsed_text])[0]
            hits = self.manager.vector.search(qvec, top_k=k, allowed_ids=allowed_ids)
            if explain:
                self._append(trace, "dense", "3. Truy hồi ngữ nghĩa (vector)",
                             f"embedder={embed_name}, {len(hits)} ứng viên",
                             self._trace_items(hits), len(hits))
            return dict(hits)

        if mode == "hybrid":
            lex = self.manager.lexical.search(parsed_text, top_k=k, allowed_ids=allowed_ids)
            qvec = self.manager.embedder.embed([parsed_text])[0]
            vec = self.manager.vector.search(qvec, top_k=k, allowed_ids=allowed_ids)
            lex_ids = [aid for aid, _ in lex]
            vec_ids = [aid for aid, _ in vec]
            fused = rrf([lex_ids, vec_ids], k=cfg.rrf_k)
            if explain:
                self._append(trace, "lexical", "3a. Truy hồi từ khóa (BM25)",
                             f"{len(lex)} ứng viên", self._trace_items(lex), len(lex))
                self._append(trace, "dense", "3b. Truy hồi ngữ nghĩa (vector)",
                             f"embedder={embed_name}, {len(vec)} ứng viên",
                             self._trace_items(vec), len(vec))
                self._append(trace, "fusion", "4. Hợp nhất RRF (Reciprocal Rank Fusion)",
                             f"k={cfg.rrf_k}, {len(fused)} bài sau hợp nhất",
                             self._trace_items(sorted(fused.items(), key=lambda kv: -kv[1])), len(fused))
            return fused

        raise ValueError(f"mode không hợp lệ: {query.mode!r} (lexical|semantic|hybrid)")

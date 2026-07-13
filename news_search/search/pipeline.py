"""Pipeline tìm kiếm phễu nhiều tầng (CONTRACTS.md §15).

Ghép các module thành luồng 9 bước:
    parse -> lọc metadata -> retrieval (lexical/semantic/hybrid) -> RRF
          -> time-decay (QDF) -> rerank -> collapse cụm -> MMR -> snippet
Đầu ra là list :class:`SearchResultItem` (đúng 5 trường khi ``.to_dict()``).
"""

from __future__ import annotations

from dataclasses import replace
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
        # (GĐ3) Query understanding — tắt vocab_provider vì không còn local BM25 index.
        self.understander = QueryUnderstander(
            self.settings, vocab_provider=lambda: {}
        )
        # (GĐ5) Cache truy vấn nóng — None nếu CACHE_ENABLED=false.
        self.cache = get_cache(self.settings)

    def _cache_key(self, query: SearchQuery, lexical_text: str, semantic_text: str) -> str:
        """Khóa cache: gồm generation chỉ mục -> mọi thay đổi index tự vô hiệu cache."""
        f = query.filters
        return "|".join(str(x) for x in (
            self.manager.generation, query.mode, query.top_k, lexical_text, semantic_text,
            f.author, f.category, f.source, f.date_from, f.date_to,
        ))

    def search(self, query: SearchQuery) -> list[SearchResultItem]:
        """Ánh xạ truy vấn -> danh sách bài viết xếp hạng (5 trường)."""
        print(f"[PIPELINE DEBUG] Entered search with query: {query.text}", flush=True)
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
        now = query.now or datetime.now(timezone.utc)

        # 1. Query understanding (rỗng -> ValueError, propagate lên caller)
        print("[DEBUG] pipeline._run: step 1 query.text ", query.text, flush=True)
        parsed = parse_query(query.text)
        if not parsed.tokens:
            if explain:
                self._append(trace, "understand", "1. Hiểu truy vấn",
                             "Không có token hợp lệ trong truy vấn -> trả rỗng")
            return []

        # 1b. Query understanding (mặc định tắt -> giữ nguyên query.text)
        print("[DEBUG] pipeline._run: step 1b understander.enabled ", self.understander.enabled, flush=True)
        lexical_text = query.text
        semantic_text = query.text
        try:
            uinfo = self.understander.understand(parsed, now=now) if self.understander.enabled else None
            print("[DEBUG] pipeline._run: uinfo =", uinfo, flush=True)
        except Exception as e:
            print("[DEBUG] pipeline._run: understander error:", e, flush=True)
            raise
        if uinfo is not None:
            lexical_text = uinfo.lexical_text
            semantic_text = uinfo.semantic_text
        if explain:
            note = f"tokens={parsed.tokens} · fresh_intent={parsed.fresh_intent}"
            if uinfo is not None:
                note += (f" · sửa lỗi={uinfo.corrections or '∅'} · cụm từ={uinfo.key_phrases or '∅'}"
                         f" · mở rộng={uinfo.expansions or '∅'}"
                         f" · truy vấn lexical='{lexical_text}' · truy vấn semantic='{semantic_text}'")
                md = uinfo.metadata
                if md.category or md.entities or md.date_from or md.date_to:
                    note += (f" · metadata: chủ_đề={md.category or '∅'}, thực_thể={md.entities or '∅'}"
                             f", từ_ngày={md.date_from or '∅'}, đến_ngày={md.date_to or '∅'}")
            else:
                note += " · (query-understanding tắt)"
            self._append(trace, "understand", "1. Hiểu truy vấn", note)

        # 1c. Áp khoảng ngày do LLM suy luận — CHỈ khi người dùng CHƯA khai báo
        if uinfo is not None and (uinfo.metadata.date_from or uinfo.metadata.date_to):
            new_date_from = query.filters.date_from or uinfo.metadata.date_from
            new_date_to = query.filters.date_to or uinfo.metadata.date_to
            if new_date_from != query.filters.date_from or new_date_to != query.filters.date_to:
                query = replace(query, filters=replace(
                    query.filters, date_from=new_date_from, date_to=new_date_to
                ))
                if explain:
                    self._append(trace, "understand_dates", "1c. Khoảng ngày suy luận",
                                 f"date_from={new_date_from}, date_to={new_date_to} "
                                 "(LLM suy luận, người dùng chưa khai báo)")

        # 1d. Cache (CHỈ khi không explain -> explain luôn tính mới để quan sát)
        cache_key = None
        if not explain and self.cache is not None and not (
            parsed.fresh_intent and cfg.cache_bypass_fresh
        ):
            cache_key = self._cache_key(query, lexical_text, semantic_text)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return [SearchResultItem(**d) for d in cached]

        # 2. Lọc metadata (F-12)
        print("[DEBUG] pipeline._run: step 2 filter_ids query.filters", query.filters, flush=True)
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
        print("[DEBUG] pipeline._run: step 3-4 lexical_text ", lexical_text, "semantic_text", semantic_text, flush=True)
        base_scores = self._retrieve(query, lexical_text, semantic_text, allowed_ids, trace)
        print("[DEBUG] pipeline._run: retrieved", len(base_scores), flush=True)
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
        time_decay_enabled = query.time_decay if query.time_decay is not None else cfg.time_decay_enabled
        if time_decay_enabled:
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
        else:
            ranked_by_decay = sorted(base_scores.items(), key=lambda kv: (-kv[1], kv[0]))
            if explain:
                self._append(trace, "decay", "5. Ưu tiên độ mới (time-decay/QDF) - Tắt",
                             "Bỏ qua bước này",
                             self._trace_items(ranked_by_decay), len(ranked_by_decay))

        # 6. Rerank tinh trên top-N theo điểm đã decay
        top_for_rerank = ranked_by_decay[: cfg.rerank_top_n]
        print("[DEBUG] pipeline._run: step 6 top_for_rerank ", top_for_rerank, flush=True)
        docs = [(aid, self.manager.store.get(aid).text) for aid, _ in top_for_rerank]
        reranked = self.reranker.rerank(query.text, docs, top_k=cfg.rerank_top_n)
        rerank_score = dict(reranked)
        reranked_ids = [aid for aid, _ in reranked]
        if explain:
            self._append(trace, "rerank", "6. Rerank tinh",
                         f"reranker={type(self.reranker).__name__} (top {cfg.rerank_top_n})",
                         self._trace_items(reranked), len(reranked))

        # 7. Gom cụm sự kiện (F-07/F-14)
        print("[DEBUG] pipeline._run: step 7 reranked_ids ", reranked_ids, flush=True)
        dedup_enabled = query.dedup if query.dedup is not None else cfg.dedup_enabled
        if dedup_enabled:
            collapsed = collapse_clusters(reranked_ids, self.manager.deduper.cluster_of, cfg.max_per_cluster)
            if explain:
                self._append(trace, "dedup", "7. Gom cụm trùng lặp",
                             f"{len(reranked_ids)} → {len(collapsed)} (bỏ {len(reranked_ids) - len(collapsed)} bài gần trùng, tối đa {cfg.max_per_cluster}/cụm)",
                             self._trace_items([(a, rerank_score.get(a, 0.0)) for a in collapsed], with_cluster=True),
                             len(collapsed))
        else:
            collapsed = reranked_ids
            if explain:
                self._append(trace, "dedup", "7. Gom cụm trùng lặp - Tắt",
                             "Bỏ qua bước này",
                             self._trace_items([(a, rerank_score.get(a, 0.0)) for a in collapsed]),
                             len(collapsed))

        # 8. Đa dạng hóa MMR
        print("[DEBUG] pipeline._run: step 8 collapsed ", collapsed, flush=True)
        mmr_enabled = query.mmr if query.mmr is not None else cfg.mmr_enabled
        if mmr_enabled:
            pool = [(aid, rerank_score.get(aid, 0.0)) for aid in collapsed[: cfg.mmr_pool]]
            final_ids = mmr(pool, self.manager.vector.get, lambda_=cfg.mmr_lambda, top_k=query.top_k)
            if explain:
                self._append(trace, "mmr", "8. Đa dạng hóa MMR",
                             f"lambda={cfg.mmr_lambda}, pool={len(pool)} → top_k={query.top_k}",
                             self._trace_items([(a, rerank_score.get(a, 0.0)) for a in final_ids]),
                             len(final_ids))
        else:
            final_ids = collapsed[:query.top_k]
            if explain:
                self._append(trace, "mmr", "8. Đa dạng hóa MMR - Tắt",
                             f"Bỏ qua bước này, lấy thẳng top_k={query.top_k}",
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
        self, query: SearchQuery, lexical_text: str, semantic_text: str,
        allowed_ids: set[str] | None, trace: list | None = None,
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
            hits = self.manager.vector.search_lexical(lexical_text, top_k=k, allowed_ids=allowed_ids)
            if explain:
                self._append(trace, "lexical", "3. Truy hồi từ khóa (Sparse/BM25)",
                             f"{len(hits)} ứng viên", self._trace_items(hits), len(hits))
            return dict(hits)

        if mode == "semantic":
            qvec = self.manager.embedder.embed([semantic_text])[0]
            hits = self.manager.vector.search_semantic(qvec, top_k=k, allowed_ids=allowed_ids)
            if explain:
                self._append(trace, "dense", "3. Truy hồi ngữ nghĩa (Dense Vector)",
                             f"embedder={embed_name}, {len(hits)} ứng viên",
                             self._trace_items(hits), len(hits))
            return dict(hits)

        if mode == "hybrid":
            qvec = self.manager.embedder.embed([semantic_text])[0]
            fused = self.manager.vector.search_hybrid(lexical_text, qvec, top_k=k, allowed_ids=allowed_ids)
            fused_dict = dict(fused)
            if explain:
                self._append(trace, "fusion", "3. Truy hồi Kết hợp (Milvus Hybrid Search)",
                             f"{len(fused_dict)} bài trả về từ Milvus RRF",
                             self._trace_items(sorted(fused_dict.items(), key=lambda kv: -kv[1])), len(fused_dict))
            return fused_dict

        raise ValueError(f"mode không hợp lệ: {query.mode!r} (lexical|semantic|hybrid)")

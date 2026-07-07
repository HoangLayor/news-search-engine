"""Rerank tinh (GĐ 3, CONTRACTS.md §10) — Protocol Reranker + Noop + BGE (import-guard)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from news_search.config import Settings


@runtime_checkable
class Reranker(Protocol):
    """Hợp đồng reranker: nhận (query, docs) -> [(article_id, score)] giảm dần theo score."""

    def rerank(
        self,
        query: str,
        docs: list[tuple[str, str]],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """``docs = [(article_id, doc_text)]`` theo thứ tự hiện tại."""
        ...


class NoopReranker:
    """Không xếp lại: giữ nguyên thứ tự vào, score = 1/rank (rank bắt đầu = 1)."""

    def rerank(
        self,
        query: str,
        docs: list[tuple[str, str]],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        selected = docs if top_k is None else docs[:top_k]
        return [
            (article_id, 1.0 / rank)
            for rank, (article_id, _text) in enumerate(selected, start=1)
        ]


class BGEReranker:
    """Cross-encoder BGE qua FlagEmbedding; thiếu package -> RuntimeError khi khởi tạo."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        try:  # import guard: FlagEmbedding là thư viện nặng, tùy chọn
            from FlagEmbedding import FlagReranker  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "BGEReranker cần package FlagEmbedding (chưa cài) — "
                "cài FlagEmbedding hoặc dùng RERANKER=none"
            ) from exc
        self._model = FlagReranker(model_name, use_fp16=True)

    def rerank(
        self,
        query: str,
        docs: list[tuple[str, str]],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        if not docs:
            return []
        pairs = [[query, text] for _article_id, text in docs]
        raw = self._model.compute_score(pairs)
        if not isinstance(raw, (list, tuple)):  # 1 cặp -> có thể trả float đơn
            raw = [raw]
        scored = [
            (article_id, float(score))
            for (article_id, _text), score in zip(docs, raw)
        ]
        # sort ổn định: score bằng nhau giữ thứ tự vào
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored if top_k is None else scored[:top_k]


class VietnameseReranker:
    """Cross-encoder tiếng Việt CHUYÊN BIỆT (``AITeamVN/Vietnamese_Reranker``).

    Fine-tune từ bge-reranker-v2-m3 cho tiếng Việt -> top-k chính xác hơn rõ.
    Import-guard hai tầng: ``FlagEmbedding.FlagReranker`` (native) rồi
    ``sentence_transformers.CrossEncoder``; thiếu cả hai -> RuntimeError.
    """

    def __init__(self, model_name: str = "AITeamVN/Vietnamese_Reranker") -> None:
        self.model_name = model_name
        self._backend = ""
        self._model = None
        try:  # 1) FlagEmbedding (native, khớp kiến trúc bge-reranker)
            from FlagEmbedding import FlagReranker  # type: ignore[import-not-found]

            self._model = FlagReranker(model_name, use_fp16=True)
            self._backend = "flagembedding"
            return
        except ImportError:
            pass
        try:  # 2) sentence-transformers CrossEncoder
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]

            self._model = CrossEncoder(model_name)
            self._backend = "cross-encoder"
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "VietnameseReranker cần 'FlagEmbedding' hoặc 'sentence-transformers' "
                "(chưa cài) — cài một trong hai hoặc dùng RERANKER=none"
            ) from exc

    def rerank(
        self,
        query: str,
        docs: list[tuple[str, str]],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        if not docs:
            return []
        pairs = [[query, text] for _aid, text in docs]
        if self._backend == "flagembedding":
            raw = self._model.compute_score(pairs)
            if not isinstance(raw, (list, tuple)):
                raw = [raw]
        else:  # cross-encoder
            raw = self._model.predict(pairs)
        scored = [(aid, float(score)) for (aid, _text), score in zip(docs, raw)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored if top_k is None else scored[:top_k]


def get_reranker(settings: Settings) -> Reranker:
    """Factory theo ``settings.reranker``.

    "none" -> Noop; "vi" -> VietnameseReranker (AITeamVN, khuyến nghị VN);
    "bge" -> BGEReranker; khác -> ValueError.
    """
    name = settings.reranker.strip().lower()
    if name == "none":
        return NoopReranker()
    if name == "vi":
        return VietnameseReranker(settings.vi_rerank_model)
    if name == "bge":
        return BGEReranker()
    raise ValueError(f"Reranker không hỗ trợ: {settings.reranker!r}")

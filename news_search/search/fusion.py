"""Reciprocal Rank Fusion (F-11, CONTRACTS.md §8) — trộn nhiều ranking thành một bảng điểm."""

from __future__ import annotations

from typing import Sequence


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> dict[str, float]:
    """RRF: ``score(id) = tổng 1/(k + rank)`` trên mọi ranking chứa id.

    - ``rank`` bắt đầu từ 1 (vị trí đầu ranking).
    - Ranking rỗng bị bỏ qua (không đóng góp điểm).
    - Trả dict ``{id: score}`` cộng dồn giữa các ranking.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores

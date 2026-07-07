"""Gom cụm & đa dạng hóa kết quả (F-14): collapse cụm + MMR.

- ``collapse_clusters``: mỗi cụm sự kiện chỉ giữ tối đa ``max_per_cluster`` bài,
  duyệt theo thứ tự ranking hiện có.
- ``mmr``: Maximal Marginal Relevance — cân bằng độ liên quan (đã min-max
  normalize) với độ đa dạng (cosine similarity so với các bài đã chọn).
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def collapse_clusters(ranked_ids: list[str],
                      cluster_of: Callable[[str], str | None],
                      max_per_cluster: int = 1) -> list[str]:
    """Duyệt theo thứ tự, mỗi cluster giữ tối đa ``max_per_cluster`` id.

    ``cluster_of`` trả None -> coi như cụm riêng của chính id đó (không gộp).
    """
    counts: dict[tuple[str, str], int] = {}
    kept: list[str] = []
    for article_id in ranked_ids:
        cluster_id = cluster_of(article_id)
        if cluster_id is None:
            key = ("solo", article_id)  # cụm riêng, không gộp với bài khác
        else:
            key = ("cluster", cluster_id)
        seen = counts.get(key, 0)
        if seen < max_per_cluster:
            counts[key] = seen + 1
            kept.append(article_id)
    return kept


def _cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity có guard chia 0 (vector zero -> 0.0)."""
    a = np.asarray(vec_a, dtype=np.float64).ravel()
    b = np.asarray(vec_b, dtype=np.float64).ravel()
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def mmr(ranked: list[tuple[str, float]],
        vector_lookup: Callable[[str], np.ndarray | None],
        lambda_: float = 0.7, top_k: int = 10) -> list[str]:
    """Maximal Marginal Relevance (CONTRACTS.md §12).

    Chọn lặp id có ``lambda_*rel_norm - (1-lambda_)*max_sim_với_đã_chọn`` lớn
    nhất. ``rel_norm`` = score min-max normalize về [0, 1] (mọi score bằng
    nhau -> 1.0). ``vector_lookup`` trả None -> sim = 0 với mọi doc.
    Trả list id theo thứ tự chọn, tối đa ``top_k``.
    """
    if top_k <= 0 or not ranked:
        return []

    # Khử id trùng lặp, giữ lần xuất hiện đầu (score đầu tiên).
    seen: set[str] = set()
    items: list[tuple[str, float]] = []
    for article_id, score in ranked:
        if article_id not in seen:
            seen.add(article_id)
            items.append((article_id, float(score)))

    # Min-max normalize score về [0, 1]; mọi score bằng nhau -> 1.0.
    scores = [s for _, s in items]
    lo, hi = min(scores), max(scores)
    if hi - lo <= 1e-12:
        rel_norm = {article_id: 1.0 for article_id, _ in items}
    else:
        rel_norm = {aid: (s - lo) / (hi - lo) for aid, s in items}

    # Lấy vector 1 lần cho mỗi id (None -> sim 0 với mọi doc).
    vectors = {article_id: vector_lookup(article_id) for article_id, _ in items}

    selected: list[str] = []
    remaining = [article_id for article_id, _ in items]
    while remaining and len(selected) < top_k:
        best_id = remaining[0]
        best_score = -float("inf")
        for article_id in remaining:
            vec = vectors[article_id]
            max_sim = 0.0
            if vec is not None:
                for chosen_id in selected:
                    chosen_vec = vectors[chosen_id]
                    if chosen_vec is None:
                        continue  # None -> sim = 0, không ảnh hưởng max
                    sim = _cosine(vec, chosen_vec)
                    if sim > max_sim:
                        max_sim = sim
            mmr_score = lambda_ * rel_norm[article_id] - (1.0 - lambda_) * max_sim
            if mmr_score > best_score:  # dùng ">" để hòa điểm chọn id đứng trước
                best_id, best_score = article_id, mmr_score
        selected.append(best_id)
        remaining.remove(best_id)
    return selected

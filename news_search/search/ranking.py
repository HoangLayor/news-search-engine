"""Time-decay / QDF (F-13, CONTRACTS.md §9) — ưu tiên bài mới bằng hệ số suy giảm theo tuổi."""

from __future__ import annotations

from datetime import datetime
from typing import Callable


def time_decay_multiplier(
    published_at: datetime,
    now: datetime,
    half_life_days: float,
    floor: float = 0.3,
) -> float:
    """Hệ số suy giảm theo tuổi bài: ``floor + (1 - floor) * 0.5 ** (age / half_life)``.

    - ``age_days`` kẹp về 0 nếu âm (bài "tương lai") -> multiplier = 1.0.
    - ``half_life_days <= 0`` -> trả 1.0 (tắt decay).
    - Cả hai datetime PHẢI timezone-aware; trộn naive/aware sẽ raise (để nguyên).
    """
    if half_life_days <= 0:
        return 1.0
    age_days = max(0.0, (now - published_at).total_seconds() / 86400.0)
    return floor + (1.0 - floor) * 0.5 ** (age_days / half_life_days)


def apply_time_decay(
    scores: dict[str, float],
    published_lookup: Callable[[str], datetime],
    now: datetime,
    half_life_days: float,
    floor: float = 0.3,
) -> dict[str, float]:
    """Nhân từng score với multiplier tương ứng; trả dict MỚI (không sửa dict vào)."""
    return {
        doc_id: score
        * time_decay_multiplier(published_lookup(doc_id), now, half_life_days, floor)
        for doc_id, score in scores.items()
    }

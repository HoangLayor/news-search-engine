"""Phân tích log sự kiện JSONL -> chỉ số vận hành (CTR, position bias, zero-result).

Đây là cầu nối giữa log hành vi (GĐ1) và đánh giá/giám sát (GĐ5): dùng để tính
CTR cho dashboard hoặc làm nhãn ngầm (implicit feedback) cho học xếp hạng.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict


def read_events(path: str) -> list[dict]:
    """Đọc file JSONL sự kiện; không tồn tại -> []."""
    if not os.path.exists(path):
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def compute_metrics(path: str) -> dict:
    """Tổng hợp chỉ số từ log: CTR, CTR theo vị trí, zero-result rate, dwell TB."""
    events = read_events(path)
    searches = [e for e in events if e.get("kind") == "search"]
    clicks = [e for e in events if e.get("kind") == "click"]

    n_search = len(searches)
    n_zero = sum(1 for e in searches if e.get("zero_result"))
    searches_with_click = {c["search_id"] for c in clicks if c.get("search_id")}

    ctr_by_pos: dict[int, int] = defaultdict(int)
    dwell_vals: list[int] = []
    for c in clicks:
        pos = c.get("position")
        if isinstance(pos, int):
            ctr_by_pos[pos] += 1
        if isinstance(c.get("dwell_ms"), int):
            dwell_vals.append(c["dwell_ms"])

    return {
        "searches": n_search,
        "clicks": len(clicks),
        "zero_result_rate": (n_zero / n_search) if n_search else 0.0,
        # CTR = tỷ lệ lượt tìm có ít nhất 1 click (đã khử trùng theo search_id)
        "ctr": (len(searches_with_click & {s["search_id"] for s in searches}) / n_search)
        if n_search else 0.0,
        "clicks_by_position": dict(sorted(ctr_by_pos.items())),
        "avg_dwell_ms": (sum(dwell_vals) / len(dwell_vals)) if dwell_vals else 0.0,
    }

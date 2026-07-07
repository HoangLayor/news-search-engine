"""Giám sát (GĐ5): thu thập metrics trong tiến trình + kiểm ngưỡng cảnh báo.

Bật bằng ``METRICS_ENABLED`` (mặc định BẬT — chỉ là bộ đếm nhẹ, không đổi kết quả).
Cung cấp số liệu cho ``/metrics`` (Prometheus text) và ``/dashboard`` (HTML), và
kiểm ngưỡng alert (p95 latency, error rate, zero-result rate).
"""

from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Optional


def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(p * len(sorted_vals)))
    return sorted_vals[idx]


class MetricsCollector:
    """Bộ đếm nhẹ, an toàn đa luồng cho search latency/error/zero-result/cache."""

    def __init__(self, window: int = 2000) -> None:
        self._lock = Lock()
        self._latencies: deque[float] = deque(maxlen=window)
        self.searches = 0
        self.errors = 0
        self.zero_results = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def record_search(self, latency_ms: float, num_results: int,
                      cache_hit: bool = False, error: bool = False) -> None:
        with self._lock:
            self.searches += 1
            self._latencies.append(latency_ms)
            if error:
                self.errors += 1
            if num_results == 0 and not error:
                self.zero_results += 1
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self._latencies)
            n = self.searches
            cache_total = self.cache_hits + self.cache_misses
            return {
                "searches": n,
                "errors": self.errors,
                "error_rate": (self.errors / n) if n else 0.0,
                "zero_results": self.zero_results,
                "zero_result_rate": (self.zero_results / n) if n else 0.0,
                "latency_p50_ms": round(_percentile(lat, 0.50), 2),
                "latency_p95_ms": round(_percentile(lat, 0.95), 2),
                "latency_p99_ms": round(_percentile(lat, 0.99), 2),
                "cache_hits": self.cache_hits,
                "cache_hit_rate": (self.cache_hits / cache_total) if cache_total else 0.0,
            }

    def prometheus_text(self) -> str:
        s = self.snapshot()
        lines = []
        for key, val in s.items():
            lines.append(f"nse_{key} {val}")
        return "\n".join(lines) + "\n"

    def check_alerts(self, p95_ms: float, error_rate: float, zero_rate: float) -> list[str]:
        """Trả danh sách cảnh báo đang kích hoạt theo ngưỡng."""
        s = self.snapshot()
        alerts = []
        if s["searches"] >= 20:  # cần đủ mẫu mới cảnh báo
            if s["latency_p95_ms"] > p95_ms:
                alerts.append(f"p95 latency {s['latency_p95_ms']}ms > {p95_ms}ms")
            if s["error_rate"] > error_rate:
                alerts.append(f"error rate {s['error_rate']:.2%} > {error_rate:.2%}")
            if s["zero_result_rate"] > zero_rate:
                alerts.append(f"zero-result rate {s['zero_result_rate']:.2%} > {zero_rate:.2%}")
        return alerts


def render_dashboard(metrics: dict, feedback: Optional[dict], counts: dict) -> str:
    """Trang HTML tối giản hiển thị metrics + feedback + số lượng index."""
    def rows(d: dict) -> str:
        return "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in d.items())

    fb_section = ""
    if feedback:
        fb_section = f"<h2>Feedback (CTR)</h2><table>{rows(feedback)}</table>"
    return f"""<!doctype html><html lang="vi"><head><meta charset="utf-8">
<title>News Search — Dashboard</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:2rem;color:#222}}
table{{border-collapse:collapse;margin:.5rem 0 1.5rem}}td{{border:1px solid #ddd;padding:.35rem .8rem}}
td:first-child{{color:#555}}h1{{margin-top:0}}h2{{margin-top:1.2rem}}</style></head><body>
<h1>News Search Engine — Dashboard</h1>
<h2>Chỉ mục</h2><table>{rows(counts)}</table>
<h2>Truy vấn (metrics)</h2><table>{rows(metrics)}</table>
{fb_section}
<p style="color:#888">Tự làm mới: <code>location.reload()</code> · Prometheus: <code>/metrics</code></p>
</body></html>"""

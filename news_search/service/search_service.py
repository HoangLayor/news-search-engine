"""Holder điều phối tìm kiếm + reindex blue-green (GĐ5).

Giữ ``manager`` + ``pipeline`` hiện hành cùng ``metrics`` và ``events`` (feedback).
``reindex`` dựng chỉ mục MỚI hoàn chỉnh rồi HOÁN ĐỔI nguyên tử (blue-green): truy
vấn trong lúc build vẫn phục vụ từ chỉ mục cũ, không gián đoạn.
"""

from __future__ import annotations

import json
import logging
import itertools
import time
from pathlib import Path
from threading import Lock
from typing import Optional

from news_search.config import Settings
from news_search.feedback import get_event_logger
from news_search.index.manager import IndexManager
from news_search.search.pipeline import SearchPipeline
from news_search.service.metrics import MetricsCollector
from news_search.sources import get_source

_log = logging.getLogger(__name__)


class SearchService:
    """Bọc manager+pipeline, hỗ trợ hoán đổi chỉ mục an toàn + metrics + feedback."""

    def __init__(self, settings: Settings, lazy_init: bool = False) -> None:
        self.settings = settings
        self.metrics = MetricsCollector() if settings.metrics_enabled else None
        self.events = get_event_logger(settings)
        self._lock = Lock()
        self.manager = IndexManager(settings, lazy_init=lazy_init)
        self.pipeline = None
        self._state_file = Path("data/reindex_progress.json")
        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize default state if not exists
        if not self._state_file.exists():
            self._write_reindex_state({
                "status": "idle",
                "processed": 0,
                "total": 0,
                "skipped": 0,
                "error_msg": None,
                "elapsed_seconds": 0.0,
                "speed": 0.0
            })
        else:
            state = self._read_reindex_state()
            if state.get("status") == "indexing":
                state["status"] = "failed"
                state["error_msg"] = "Tiến trình bị gián đoạn do hệ thống khởi động lại."
                self._write_reindex_state(state)

        if not lazy_init:
            self.load_resources()

    def is_ready(self) -> bool:
        """Kiểm tra xem hệ thống đã nạp xong các model AI chưa."""
        return self.pipeline is not None

    def load_resources(self) -> None:
        """Nạp các mô hình AI (embedding, reranker) ở chế độ chạy ngầm."""
        self.manager.load_models()
        with self._lock:
            if self.pipeline is None:
                self.pipeline = SearchPipeline(self.manager, self.settings)

    def _read_reindex_state(self) -> dict:
        try:
            return json.loads(self._state_file.read_text(encoding="utf-8"))
        except Exception:
            return {"status": "idle", "processed": 0, "total": 0, "skipped": 0, "error_msg": None, "elapsed_seconds": 0.0, "speed": 0.0}

    def _write_reindex_state(self, state: dict) -> None:
        try:
            # Atomic write via temp file (safe across workers)
            tmp_file = self._state_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            tmp_file.replace(self._state_file)
        except Exception as exc:
            _log.error(f"Lỗi ghi trạng thái reindex: {exc}")

    @property
    def reindex_progress(self) -> dict:
        return self._read_reindex_state()

    def reindex(self, source=None, limit: Optional[int] = None) -> int:
        """Blue-green: build chỉ mục mới từ nguồn rồi swap. Trả số bài đã index.

        - Mở nguồn TRƯỚC khi cấp phát chỉ mục mới (nguồn hỏng -> không phí RAM).
        - TÁI DÙNG embedder/reranker hiện có (tránh nạp lại model ~GB -> OOM).
        - Lỗi index từng bài được ĐẾM + LOG (không nuốt im lặng); nếu cả đợt lỗi
          -> log ERROR để lộ nguyên nhân (vd dim mismatch, Milvus từ chối).
        """
        state = self._read_reindex_state()
        state.update({
            "status": "indexing",
            "processed": 0,
            "total": 0,
            "skipped": 0,
            "error_msg": None,
            "elapsed_seconds": 0.0,
            "speed": 0.0
        })
        self._write_reindex_state(state)
        t0 = time.perf_counter()

        # Mở nguồn trước (raise sớm nếu psycopg thiếu / DB không kết nối được)
        try:
            src = source or get_source(self.settings)
        except Exception as exc:
            state = self._read_reindex_state()
            state.update({
                "status": "failed",
                "error_msg": f"Không kết nối được nguồn dữ liệu: {type(exc).__name__}: {exc}"
            })
            self._write_reindex_state(state)
            raise

        try:
            # Thử lấy tổng số bài để phục vụ phần trăm tiến độ
            try:
                total_count = src.count()
                state = self._read_reindex_state()
                state["total"] = limit if (limit and limit < total_count) else total_count
                self._write_reindex_state(state)
            except Exception:
                pass

            # Tái dùng model nặng của chỉ mục hiện hành -> không nhân đôi RAM
            new_manager = IndexManager(self.settings, embedder=self.manager.embedder)
            n = skipped = 0
            samples: list[str] = []

            def chunked(iterable, n):
                it = iter(iterable)
                while True:
                    chunk = list(itertools.islice(it, n))
                    if not chunk:
                        break
                    yield chunk

            # Gom lô nhỏ (chunk) để tính embedding song song tối ưu và bulk insert
            # Giảm xuống 16 bài để tránh lỗi CUDA Out of Memory trên GPU có VRAM nhỏ (4GB)
            ingest_chunk_size = 16
            for chunk in chunked(src.fetch_all(self.settings.ingest_batch_size), ingest_chunk_size):
                if limit and n + len(chunk) > limit:
                    chunk = chunk[:(limit - n)]
                    if not chunk:
                        break

                try:
                    n_added = new_manager.bulk_index(chunk)
                    n += n_added
                except Exception:
                    # Fallback: chạy từng bài trong lô để bỏ qua bài lỗi và đếm chi tiết lỗi
                    for raw in chunk:
                        try:
                            new_manager.index_article(raw)
                            n += 1
                        except Exception as exc:
                            skipped += 1
                            if len(samples) < 5:
                                samples.append(f"{raw.get('article_id')}: {type(exc).__name__}: {exc}")

                # Cập nhật trạng thái tiến độ thời gian thực
                elapsed = time.perf_counter() - t0
                speed = n / elapsed if elapsed > 0 else 0.0
                state = self._read_reindex_state()
                state.update({
                    "processed": n,
                    "skipped": skipped,
                    "elapsed_seconds": elapsed,
                    "speed": speed
                })
                self._write_reindex_state(state)
                total_target = state.get("total") or "unknown"
                _log.info(f"  -> [Reindex] Đã xử lý: {n}/{total_target} bài | Tốc độ: {speed:.1f} bài/giây (bỏ qua: {skipped})")

                if limit and n >= limit:
                    break
        except Exception as exc:
            state = self._read_reindex_state()
            state.update({
                "status": "failed",
                "error_msg": f"Lỗi trong quá trình index: {type(exc).__name__}: {exc}"
            })
            self._write_reindex_state(state)
            raise
        finally:
            if source is None:
                src.close()

        if skipped:
            _log.warning("Reindex: index %d bài, BỎ QUA %d bài lỗi. Ví dụ: %s",
                         n, skipped, "; ".join(samples))
        if n == 0 and skipped > 0:
            _log.error("Reindex: TẤT CẢ %d bài đều lỗi — kiểm tra dim collection/kết nối "
                       "(vd EMBEDDER đổi dim nhưng collection Milvus cũ). Mẫu lỗi: %s",
                       skipped, "; ".join(samples))

        with self._lock:  # hoán đổi nguyên tử, tái dùng reranker của pipeline cũ
            self.manager = new_manager
            if self.pipeline is not None:
                self.pipeline = SearchPipeline(new_manager, self.settings,
                                               reranker=self.pipeline.reranker)
            else:
                self.pipeline = SearchPipeline(new_manager, self.settings)
        
        state = self._read_reindex_state()
        state.update({
            "status": "success",
            "processed": n,
            "skipped": skipped,
            "elapsed_seconds": time.perf_counter() - t0
        })
        self._write_reindex_state(state)
        return n

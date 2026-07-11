"""Holder điều phối tìm kiếm + reindex blue-green (GĐ5).

Giữ ``manager`` + ``pipeline`` hiện hành cùng ``metrics`` và ``events`` (feedback).
``reindex`` dựng chỉ mục MỚI hoàn chỉnh rồi HOÁN ĐỔI nguyên tử (blue-green): truy
vấn trong lúc build vẫn phục vụ từ chỉ mục cũ, không gián đoạn.
"""

from __future__ import annotations

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
import itertools
import time
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

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.metrics = MetricsCollector() if settings.metrics_enabled else None
        self.events = get_event_logger(settings)
        self._lock = Lock()
        self.manager = IndexManager(settings)
        self.pipeline = SearchPipeline(self.manager, settings)
        # Dictionary lưu trạng thái tiến trình phục vụ UI/API polling
        self.reindex_progress = {
            "status": "idle",  # "idle" | "indexing" | "success" | "failed"
            "processed": 0,
            "total": 0,
            "skipped": 0,
            "error_msg": None,
            "elapsed_seconds": 0.0,
            "speed": 0.0
        }

    def reindex(self, source=None, limit: Optional[int] = None) -> int:
        """Blue-green: build chỉ mục mới từ nguồn rồi swap. Trả số bài đã index.

        - Mở nguồn TRƯỚC khi cấp phát chỉ mục mới (nguồn hỏng -> không phí RAM).
        - TÁI DÙNG embedder/reranker hiện có (tránh nạp lại model ~GB -> OOM).
        - Lỗi index từng bài được ĐẾM + LOG (không nuốt im lặng); nếu cả đợt lỗi
          -> log ERROR để lộ nguyên nhân (vd dim mismatch, Milvus từ chối).
        """
        self.reindex_progress.update({
            "status": "indexing",
            "processed": 0,
            "total": 0,
            "skipped": 0,
            "error_msg": None,
            "elapsed_seconds": 0.0,
            "speed": 0.0
        })
        t0 = time.perf_counter()

        # Mở nguồn trước (raise sớm nếu psycopg thiếu / DB không kết nối được)
        try:
            src = source or get_source(self.settings)
        except Exception as exc:
            self.reindex_progress.update({
                "status": "failed",
                "error_msg": f"Không kết nối được nguồn dữ liệu: {type(exc).__name__}: {exc}"
            })
            raise

        try:
            # Thử lấy tổng số bài để phục vụ phần trăm tiến độ
            try:
                total_count = src.count()
                self.reindex_progress["total"] = limit if (limit and limit < total_count) else total_count
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
                self.reindex_progress.update({
                    "processed": n,
                    "skipped": skipped,
                    "elapsed_seconds": elapsed,
                    "speed": n / elapsed if elapsed > 0 else 0.0
                })

                if limit and n >= limit:
                    break
        except Exception as exc:
            self.reindex_progress.update({
                "status": "failed",
                "error_msg": f"Lỗi trong quá trình index: {type(exc).__name__}: {exc}"
            })
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
            self.pipeline = SearchPipeline(new_manager, self.settings,
                                           reranker=self.pipeline.reranker)
        
        self.reindex_progress.update({
            "status": "success",
            "processed": n,
            "skipped": skipped,
            "elapsed_seconds": time.perf_counter() - t0
        })
        return n

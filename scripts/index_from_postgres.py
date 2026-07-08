"""Nạp bài viết từ nguồn dữ liệu (Postgres CMS hoặc mẫu) vào chỉ mục & tìm thử.

Mặc định dùng cấu hình từ ``.env`` (SOURCE, VECTOR_BACKEND, EMBEDDER...). Có cờ
override để chạy nhanh/ổn định khi thử nghiệm.

Ví dụ:
    # Nạp 200 bài đầu từ Postgres bằng backend local + embedder hash (nhanh), rồi tìm thử
    python scripts/index_from_postgres.py --source postgres --limit 200 \\
        --vector-backend local --embedder hash --query "kinh tế"

    # Nạp toàn bộ theo đúng .env (Postgres + Milvus + BGE)
    python scripts/index_from_postgres.py --source postgres
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from news_search.config import Settings
from news_search.index.manager import IndexManager
from news_search.models import SearchQuery
from news_search.search.pipeline import SearchPipeline
from news_search.sources import get_source

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nạp bài từ nguồn dữ liệu vào chỉ mục.")
    p.add_argument("--source", help="sample | postgres (mặc định theo .env)")
    p.add_argument("--limit", type=int, default=100, help="giới hạn số bài (0 = tất cả)")
    p.add_argument("--batch-size", type=int, default=0, help="cỡ lô fetch (0 = theo cấu hình)")
    p.add_argument("--since", help="chỉ nạp bài từ mốc ISO này (index tăng dần)")
    p.add_argument("--embedder", help="override EMBEDDER (hash|bge|openai)")
    p.add_argument("--vector-backend", help="override VECTOR_BACKEND (local|milvus)")
    p.add_argument("--query", help="chạy 1 truy vấn thử sau khi nạp")
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()
    if args.source:
        settings.source = args.source
    if args.embedder:
        settings.embedder = args.embedder
    if args.vector_backend:
        settings.vector_backend = args.vector_backend
    batch = args.batch_size or settings.ingest_batch_size

    print(f"Nguồn={settings.source} | vector={settings.vector_backend} | "
          f"embedder={settings.embedder} | batch={batch}"
          + (f" | limit={args.limit}" if args.limit else ""))

    # Nạp IndexManager TRƯỚC (có thể tải torch/BGE) rồi mới mở psycopg — nạp DLL
    # torch trước giúp tránh xung đột OpenMP torch<->libpq trên Windows.
    manager = IndexManager(settings)  # có thể tải model BGE / kết nối Milvus tùy cấu hình

    try:
        source = get_source(settings)
    except Exception as exc:  # psycopg thiếu / DB không kết nối được
        print(f"[LỖI] Không mở được nguồn: {type(exc).__name__}: {exc}")
        sys.exit(1)
    n_ok = n_skip = 0
    skip_samples: list[str] = []
    t0 = time.perf_counter()
    try:
        try:
            print(f"Tổng bài đủ điều kiện ở nguồn: {source.count()}")
        except Exception:
            pass
        if args.since:
            stream = source.fetch_since(datetime.fromisoformat(args.since), batch)
        else:
            stream = source.fetch_all(batch)
        for raw in stream:
            try:
                manager.index_article(raw)
                n_ok += 1
            except Exception as exc:
                n_skip += 1
                if len(skip_samples) < 5:
                    skip_samples.append(f"{raw.get('article_id')}: {type(exc).__name__}: {exc}")
            if n_ok and n_ok % 100 == 0:
                print(f"  ...đã index {n_ok} bài")
            if args.limit and n_ok >= args.limit:
                break
    finally:
        source.close()

    dt = time.perf_counter() - t0
    print(f"\nXong: index {n_ok} bài, bỏ qua {n_skip} bài, {dt:.1f}s "
          f"({n_ok / dt:.1f} bài/s)" if dt > 0 else f"\nXong: index {n_ok} bài")
    if skip_samples:
        print("Ví dụ bài bỏ qua:")
        for s in skip_samples:
            print("  -", s)

    if args.query:
        pipeline = SearchPipeline(manager, settings)
        print(f"\n=== Tìm thử: {args.query!r} ===")
        for r in pipeline.search(SearchQuery(text=args.query, top_k=args.top_k)):
            print(f"  [{r.published_at[:10]}] {r.article_id:>8}  {r.title}")


if __name__ == "__main__":
    main()
